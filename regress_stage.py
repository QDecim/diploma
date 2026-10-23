#!/usr/bin/env python3
"""
Stage 3b: voice embedding -> AMs, evaluated honestly on a very small number of speakers.

Inputs
  voices.npz     from voice_stage.py   (ids, emb)
  labels.npz     from flame_ams.py labels   (ids, ams, names, keep)
  targets.json   from flame_ams.py analyze  (the non-redundant subset, used for the PCs and the summary)
  dataset/meta.csv   "<id>,<name>,<sex>,<clip>" without header (only id and sex are used)

Model: ridge regression (dual form, features standardised, targets standardised on the training fold, alpha
chosen by leave-one-out inside every training fold). Evaluation: leave-one-speaker-out, so every speaker is
predicted by a model that never saw him or her.

Why sex needs special care: a speaker embedding separates men from women almost perfectly and AMs differ
between the sexes, so a model can score a decent Pearson r just by "detecting sex". Three numbers per AM:
  r_all     Pearson r(true, predicted) over all speakers
  r_within  the same after subtracting the per-sex means of both true and predicted values
            = does the voice predict anything beyond sex?
  r_sex     plain correlation of the AM with sex (in-sample): how much of the AM sex alone explains
Only r_within is evidence that the voice carries facial information beyond sex. Leave-one-out r is biased below 0
when there is no signal, so judge by the permutation p-values (same procedure on shuffled data), not by raw r.
Two permutation tests back this up (shuffle voices over speakers; shuffle within each sex).

Robustness checks (neither needs new data):
  --exclude 1,2,8   drop these speakers from everything (e.g. the ones with the weakest face labels or the least
                    voice). Answers: does the result survive without the shakiest data points? Without --out the
                    results go to results_excl_<ids>.csv, so the main results.csv is not overwritten.
  --controls faces  negative controls: the same model and the same within-sex permutation test, but the targets are
                    video-condition variables taken from faces/<id>.npz (sharpness, face size in pixels, mean
                    |yaw|, mean |roll|), which cannot depend on the face's anatomy. If the voice predicts these as
                    well as it predicts the AMs, the voice embedding carries recording conditions (channel, camera,
                    language), and the AM result cannot be credited to the face. If it does not, that particular
                    confounder is not visible (age and others are not excluded by this).

  --residualize mean_abs_yaw[,log_sharpness]   regress these video covariates out of every AM before modelling
                    (covariates from --controls DIR). Output goes to results_resid_<covars>.csv, not results.csv.
                    Tests whether the within-sex result survives once head pose / sharpness are removed from the AMs.

Usage:
  python regress_stage.py --voices voices.npz --labels labels.npz --targets targets.json --meta dataset/meta.csv
  python regress_stage.py ... --perm 2000 --controls faces
  python regress_stage.py ... --perm 2000 --exclude 1,2,8
  python regress_stage.py ... --perm 2000 --controls faces --residualize mean_abs_yaw
  python regress_stage.py --selftest
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ALPHAS = np.logspace(-1, 4, 11)


# ------------------------------------------------------------------ model
def loo_predict(X, Y, alphas=ALPHAS):
    """Leave-one-row-out ridge. X (n,d), Y (n,m) -> predictions (n,m) and the alpha chosen in each fold."""
    n = len(X)
    P = np.zeros_like(Y, dtype=np.float64)
    chosen = []
    for i in range(n):
        tr = np.arange(n) != i
        mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-8
        my, sy = Y[tr].mean(0), Y[tr].std(0) + 1e-8
        Xs, Ys = (X[tr] - mx) / sx, (Y[tr] - my) / sy
        s, U = np.linalg.eigh(Xs @ Xs.T)
        s = np.clip(s, 0, None)
        UtY = U.T @ Ys
        best, best_mse = alphas[0], np.inf
        for al in alphas:                       # closed-form leave-one-out error of ridge (hat-matrix trick)
            sh = s / (s + al)
            res = (Ys - U @ (sh[:, None] * UtY)) / (1 - (U ** 2) @ sh)[:, None]
            mse = float(np.mean(res ** 2))
            if mse < best_mse:
                best, best_mse = al, mse
        coef = U @ (UtY / (s + best)[:, None])
        P[i] = (((X[i] - mx) / sx) @ Xs.T @ coef) * sy + my
        chosen.append(best)
    return P, chosen


def sex_baseline(Y, grp):
    """Leave-one-out 'mean of the same sex' predictor (falls back to the overall mean if a sex is missing)."""
    P = np.zeros_like(Y, dtype=np.float64)
    for i in range(len(Y)):
        tr = np.arange(len(Y)) != i
        same = tr & (grp == grp[i])
        P[i] = Y[same].mean(0) if same.any() else Y[tr].mean(0)
    return P


def pearson(A, B):
    return np.array([np.corrcoef(A[:, k], B[:, k])[0, 1] if A[:, k].std() > 0 and B[:, k].std() > 0 else np.nan
                     for k in range(A.shape[1])])


def demean_by_group(A, grp):
    A = A.astype(np.float64).copy()
    for g in np.unique(grp):
        A[grp == g] -= A[grp == g].mean(0)
    return A


def metrics(Y, P, grp):
    return dict(r_all=pearson(Y, P), r_within=pearson(demean_by_group(Y, grp), demean_by_group(P, grp)))


def perm_p(X, Y, grp, observed, within, n_perm, seed=0):
    """p-value that the mean r over targets is this high by chance. within=False shuffles voices over all
    speakers (tests voice->AM incl. the sex effect); within=True shuffles only inside each sex (tests what is left
    after sex is held fixed). Returns (p, mean of the null distribution); leave-one-out r is biased below 0
    when there is no signal, and the null mean shows by how much."""
    rng = np.random.default_rng(seed)
    cnt, null = 0, []
    for _ in range(n_perm):
        idx = np.arange(len(X))
        if within:
            for g in np.unique(grp):
                m = np.where(grp == g)[0]
                idx[m] = rng.permutation(m)
        else:
            idx = rng.permutation(len(X))
        P, _ = loo_predict(X[idx], Y)
        r = float(np.nanmean(metrics(Y, P, grp)["r_within" if within else "r_all"]))
        null.append(r)
        cnt += int(r >= observed)
    return (1 + cnt) / (1 + n_perm), float(np.mean(null))


# ------------------------------------------------------------------ data
def load_meta(path):
    sex = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[0].strip():
                sex.setdefault(row[0].strip(), row[-2].strip().lower())   # id first, sex second to last
    return sex


def load_data(a):
    v = np.load(a.voices, allow_pickle=True)
    lab = np.load(a.labels, allow_pickle=True)
    tg = json.load(open(a.targets, encoding="utf-8")) if a.targets and Path(a.targets).exists() else None
    meta = load_meta(a.meta)
    vid = {str(i): k for k, i in enumerate(v["ids"])}
    names = [str(n) for n in lab["names"]]
    rows, why = [], []
    excl = {x.strip() for x in (getattr(a, "exclude", "") or "").split(",") if x.strip()}
    unknown = excl - {str(i) for i in lab["ids"]} - set(vid)
    if unknown:
        print(f"  WARNING: --exclude ids not found anywhere: {sorted(unknown)}")
    for k, i in enumerate(lab["ids"]):
        i = str(i)
        if i in excl:
            why.append((i, "excluded by --exclude"))
        elif not bool(lab["keep"][k]):
            why.append((i, "fit-quality filter (keep=False)"))
        elif i not in vid:
            why.append((i, "no voice embedding"))
        elif i not in meta:
            why.append((i, "not in meta.csv"))
        else:
            rows.append((i, vid[i], k))
    for i in vid:
        if i not in [str(x) for x in lab["ids"]] and i not in excl:
            why.append((i, "no face labels"))
    for i, w in why:
        print(f"  excluded speaker {i}: {w}")
    ids = [r[0] for r in rows]
    X = np.stack([v["emb"][r[1]] for r in rows]).astype(np.float64)
    Y = np.stack([lab["ams"][r[2]] for r in rows]).astype(np.float64)
    grp = np.array([meta[i] for i in ids])
    sel = [n for n in (tg["selected"] if tg else names) if n in names]
    return ids, X, Y, grp, names, sel


# ------------------------------------------------------------------ negative controls
CONTROL_NAMES = ["log_sharpness", "log_face_px", "mean_abs_yaw", "mean_abs_roll"]


def control_features(path):
    """Video-condition variables of one speaker from faces/<id>.npz (the frames that went into the labels)."""
    d = np.load(path)
    lm = d["landmarks"]                                              # (frames, 478, 3), pixels
    size = (lm[:, :, :2].max(1) - lm[:, :, :2].min(1)).max(1)        # bounding-box side per frame
    return dict(log_sharpness=float(np.log(np.mean(d["sharp"]) + 1e-6)),
                log_face_px=float(np.log(np.mean(size) + 1e-6)),
                mean_abs_yaw=float(np.mean(np.abs(d["yaw"]))),
                mean_abs_roll=float(np.mean(np.abs(d["roll"]))))


def run_controls(ctrl_dir, ids, X, grp, n_perm, am_ref=None):
    """Same model + same within-sex permutation test as the AMs, with video-condition variables as targets.
    am_ref = (mean r_within, permutation p) of the AMs, printed for comparison."""
    rows, C, miss = [], [], []
    for k, i in enumerate(ids):
        p = Path(ctrl_dir) / f"{i}.npz"
        if not p.exists():
            miss.append(i)
            continue
        f = control_features(p)
        rows.append(k)
        C.append([f[c] for c in CONTROL_NAMES])
    if miss:
        print(f"  controls: no {ctrl_dir}/<id>.npz for speakers {miss}; they are left out of this check")
    if len(rows) < 8:
        print("  controls: too few speakers, skipped")
        return None
    rows, C = np.array(rows), np.array(C, dtype=np.float64)
    Xs, gs = X[rows], grp[rows]
    P, _ = loo_predict(Xs, C)
    M = metrics(C, P, gs)
    print(f"\nnegative controls: video-condition variables as targets ({len(rows)} speakers, same model, "
          f"same within-sex permutation test)")
    print(f"{'control':16s} {'r_all':>6s} {'r_within':>9s}")
    for k, c in enumerate(CONTROL_NAMES):
        print(f"{c:16s} {M['r_all'][k]:6.2f} {M['r_within'][k]:9.2f}")
    mean_in = float(np.nanmean(M["r_within"]))
    out = dict(n=len(rows), r_all=M["r_all"], r_within=M["r_within"], mean_r_within=mean_in, p=None)
    line = f"mean r_within over controls: {mean_in:.2f}"
    if n_perm > 0:
        p, nm = perm_p(Xs, C, gs, mean_in, True, n_perm)
        out["p"] = p
        line += f" | permutation p (voices shuffled within sex) = {p:.3f} (null mean {nm:+.2f})"
    print(line)
    if am_ref is not None:
        pa = "n/a" if am_ref[1] is None else f"{am_ref[1]:.3f}"
        print(f"for comparison, the AMs: mean r_within {am_ref[0]:.2f}, permutation p = {pa}")
    print("reading: controls with r_within comparable to the AMs and a small p -> the voice embedding carries "
          "recording conditions, so the AM result cannot be attributed to the face.\n"
          "         controls near the null mean and a large p -> no sign of this confounder (age and other "
          "confounders are not excluded by it). The 4 controls are correlated, so do not count them as 4 tests.")
    return out


# ------------------------------------------------------------------ residualisation (video-condition confounders)
def covariate_matrix(ctrl_dir, ids, cov):
    """Video-condition covariates of the given speakers, from faces/<id>.npz. Raises if any speaker is missing."""
    Z = []
    for i in ids:
        p = Path(ctrl_dir) / f"{i}.npz"
        if not p.exists():
            raise SystemExit(f"--residualize: missing {p} (every speaker in the regression needs a faces file)")
        f = control_features(p)
        Z.append([f[c] for c in cov])
    return np.array(Z, dtype=np.float64)


def residualize(Y, Z):
    """Remove the linear effect of the covariates Z from every AM (intercept + standardised covariates).
    Fitted on the AMs and covariates only; the voice is not used. Caveat: the fit uses all speakers, including the
    one left out in each LOO fold, so this is not strictly leakage-free; a per-fold version would be stricter."""
    Zc = np.column_stack([np.ones(len(Z)), (Z - Z.mean(0)) / (Z.std(0) + 1e-12)])
    beta, *_ = np.linalg.lstsq(Zc, Y, rcond=None)
    return Y - Zc @ beta


def print_residual_effect(names, Y_raw, Y_res, Z, cov):
    """Per AM: Pearson r with each covariate before and after residualisation (after is ~0 by construction)."""
    m = Y_raw.shape[1]
    print(f"\nresidualised on {', '.join(cov)}: AM vs covariate correlation, before -> after")
    print(f"{'AM':22s} " + " ".join(f"{c:>22s}" for c in cov))
    for k in range(m):
        cells = []
        for j in range(len(cov)):
            zj = np.repeat(Z[:, [j]], m, axis=1)
            rb = pearson(Y_raw[:, [k]], zj[:, [0]])[0]
            ra = pearson(Y_res[:, [k]], zj[:, [0]])[0]
            cells.append(f"{rb:+.2f} -> {ra:+.2f}")
        print(f"{names[k]:22s} " + " ".join(f"{c:>22s}" for c in cells))


# ------------------------------------------------------------------ run
def run(a):
    ids, X, Y, grp, names, sel = load_data(a)
    n = len(ids)
    if getattr(a, "exclude", ""):
        print(f"\nSENSITIVITY RUN: --exclude {a.exclude}")
    cov = [c.strip() for c in (getattr(a, "residualize", "") or "").split(",") if c.strip()]
    if cov:
        bad = [c for c in cov if c not in CONTROL_NAMES]
        if bad:
            raise SystemExit(f"--residualize: unknown covariates {bad}; choose from {CONTROL_NAMES}")
        if not a.controls:
            raise SystemExit("--residualize needs --controls DIR (the covariates come from DIR/<id>.npz)")
        Z = covariate_matrix(a.controls, ids, cov)
        Y_raw = Y.copy()
        Y = residualize(Y, Z)
        print_residual_effect(names, Y_raw, Y, Z, cov)
        print("  note: all subsequent numbers are for the residualised AMs (the covariates are removed from the targets).")
    sexes, counts = np.unique(grp, return_counts=True)
    print(f"\n{n} speakers; sex counts: {dict(zip(sexes.tolist(), counts.tolist()))}; embedding dim {X.shape[1]}")
    if len(sexes) < 2 or counts.min() < 3:
        print("  WARNING: a sex has fewer than 3 speakers; the within-sex numbers are not meaningful.")
    print(f"  critical |r| for p<0.05 (two-sided, n={n}): {abs(_rcrit(n)):.2f}  (before any multiple-comparison correction)\n")

    P, al = loo_predict(X, Y)
    M = metrics(Y, P, grp)
    ref = "m" if "m" in sexes else sexes[-1]
    MB = pearson(Y, (grp == ref).astype(np.float64)[:, None].repeat(Y.shape[1], 1))   # AM vs sex, in-sample
    print(f"{'AM':22s} {'r_all':>6s} {'r_within':>9s} {'r_sex(' + ref + ')':>8s}")
    order = np.argsort(-np.nan_to_num(M["r_within"], nan=-9))
    for k in order:
        star = "*" if names[k] in sel else " "
        print(f"{star}{names[k]:21s} {M['r_all'][k]:6.2f} {M['r_within'][k]:9.2f} {MB[k]:8.2f}")
    print(f"  (* = in the non-redundant subset; r_sex = plain correlation of the AM with sex={ref}, i.e. how much of "
          f"it sex alone explains)")

    si = [names.index(s) for s in sel]
    o_all, o_in = float(np.nanmean(M["r_all"][si])), float(np.nanmean(M["r_within"][si]))
    print(f"\nmean over the {len(si)} non-redundant AMs: r_all {o_all:.2f} | r_within {o_in:.2f} | "
          f"mean |r_sex| {float(np.nanmean(np.abs(MB[si]))):.2f}")

    # principal components of the (standardised) non-redundant targets
    Z = (Y[:, si] - Y[:, si].mean(0)) / (Y[:, si].std(0) + 1e-12)
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    k_pc = min(a.n_pc, len(S))
    PC = Z @ Vt[:k_pc].T
    PP, _ = loo_predict(X, PC)
    MP = metrics(PC, PP, grp)
    ev = S ** 2 / np.sum(S ** 2)
    print("\nprincipal components of the targets (computed on all speakers, which uses no voice information):")
    for c in range(k_pc):
        print(f"  PC{c + 1} ({ev[c] * 100:.0f}% var): r_all {MP['r_all'][c]:.2f}  r_within {MP['r_within'][c]:.2f}")

    p_in = None
    if a.perm > 0:
        Ys = Y[:, si]
        p_all, nm_all = perm_p(X, Ys, grp, o_all, False, a.perm)
        p_in, nm_in = perm_p(X, Ys, grp, o_in, True, a.perm)
        print(f"\npermutation tests ({a.perm} shuffles, statistic = mean r over the non-redundant AMs):")
        print(f"  voices shuffled over all speakers  : p = {p_all:.3f}   (null mean r_all {nm_all:+.2f}; includes the sex effect)")
        print(f"  voices shuffled within each sex    : p = {p_in:.3f}   (null mean r_within {nm_in:+.2f}; voice beyond sex)")

    if a.controls:
        run_controls(a.controls, ids, X, grp, a.perm, am_ref=(o_in, p_in))

    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["am", "in_subset", "r_all", "r_within", "r_with_sex"])
            for k in range(len(names)):
                w.writerow([names[k], int(names[k] in sel), f"{M['r_all'][k]:.4f}", f"{M['r_within'][k]:.4f}",
                            f"{MB[k]:.4f}"])
        np.savez(Path(a.out).with_suffix(".npz"), ids=np.array(ids), sex=grp, names=np.array(names), true=Y, pred=P)
        print("\nsaved", a.out, "and", Path(a.out).with_suffix(".npz"))


def _rcrit(n):
    """two-sided 5% critical Pearson r for n samples (Fisher z approximation)."""
    return float(np.tanh(1.96 / np.sqrt(max(n - 3, 1))))


# ------------------------------------------------------------------ self-test (synthetic, numpy only)
def selftest(seed=0):
    rng = np.random.default_rng(seed)
    n, d, m = 40, 64, 5
    grp = np.array(["m", "f"] * (n // 2))
    sexvec = rng.normal(size=d)

    def make(sex_only):
        z = rng.normal(size=(n, 4))                                   # latent "face" factors
        Wf = rng.normal(size=(4, d))
        X = z @ Wf + (grp == "m")[:, None] * sexvec * 1.5 + rng.normal(size=(n, d)) * 0.5
        Wy = rng.normal(size=(4, m))
        Y = (np.zeros((n, m)) if sex_only else z @ Wy) + (grp == "m")[:, None] * rng.choice([-2.0, 2.0], size=m) \
            + rng.normal(size=(n, m)) * 0.3
        return X, Y

    X, Y = make(sex_only=False)                                        # voice really carries the factors
    P, _ = loo_predict(X, Y)
    r1 = metrics(Y, P, grp)
    X2, Y2 = make(sex_only=True)                                       # AMs depend on sex only
    P2, _ = loo_predict(X2, Y2)
    r2 = metrics(Y2, P2, grp)
    base2 = pearson(Y2, sex_baseline(Y2, grp))
    p_in = perm_p(X, Y, grp, float(np.nanmean(r1["r_within"])), True, 30)[0]
    p_null = perm_p(X2, Y2, grp, float(np.nanmean(r2["r_within"])), True, 30)[0]
    print(f"real signal   : r_all {np.nanmean(r1['r_all']):.2f}  r_within {np.nanmean(r1['r_within']):.2f}  "
          f"perm p(within) {p_in:.2f}")
    print(f"sex-only data : r_all {np.nanmean(r2['r_all']):.2f}  r_within {np.nanmean(r2['r_within']):.2f}  "
          f"sex baseline r {np.nanmean(base2):.2f}  perm p(within) {p_null:.2f}")
    ok = (np.nanmean(r1["r_within"]) > 0.5 and p_in < 0.1
          and np.nanmean(r2["r_all"]) > 0.6                            # sex alone gives a good-looking r_all ...
          and abs(np.nanmean(r2["r_within"])) < 0.3 and p_null >= p_in   # ... but nothing within sex
          and np.nanmean(base2) > 0.6)
    ok_pl = _plumbing_test()
    print("plumbing (--exclude, control features, controls run):", "ok" if ok_pl else "FAILED")
    ok = ok and ok_pl
    ok_rs = _residual_test()
    print("residualisation (removes a planted covariate, keeps the voice signal):", "ok" if ok_rs else "FAILED")
    ok = ok and ok_rs
    print("SELFTEST", "PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


def _residual_test(seed=2):
    """Planted confounder: AMs = voice-related signal + a * yaw + noise. After residualising on yaw, the AM-yaw
    correlation must drop to ~0 and the voice prediction within sex must stay (the signal was not on yaw)."""
    rng = np.random.default_rng(seed)
    n, d, m = 40, 48, 4
    grp = np.array(["m", "f"] * (n // 2))
    z = rng.normal(size=(n, 3))
    X = z @ rng.normal(size=(3, d)) + rng.normal(size=(n, d)) * 0.5
    yaw = rng.normal(size=(n, 1))
    Y = z @ rng.normal(size=(3, m)) + 2.0 * yaw + rng.normal(size=(n, m)) * 0.3
    Yr = residualize(Y, yaw)
    r_before = float(np.nanmean(np.abs(pearson(Y, np.repeat(yaw, m, 1)))))
    r_after = float(np.nanmean(np.abs(pearson(Yr, np.repeat(yaw, m, 1)))))
    P_before, _ = loo_predict(X, Y)
    P_after, _ = loo_predict(X, Yr)
    rw_after = float(np.nanmean(metrics(Yr, P_after, grp)["r_within"]))
    print(f"residualisation check: |r(AM,yaw)| {r_before:.2f} -> {r_after:.2f}; r_within after {rw_after:.2f}")
    return r_before > 0.5 and r_after < 0.1 and rw_after > 0.3


def _plumbing_test():
    import tempfile
    from types import SimpleNamespace
    rng = np.random.default_rng(1)
    n = 14
    ids = [str(i) for i in range(1, n + 1)]
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        np.savez(t / "v.npz", ids=np.array(ids), emb=rng.normal(size=(n, 16)))
        np.savez(t / "l.npz", ids=np.array(ids), ams=rng.normal(size=(n, 3)), names=np.array(["a", "b", "c"]),
                 keep=np.array([True] * n))
        (t / "meta.csv").write_text("".join(f"{i},N{i},{'m' if int(i) % 2 else 'f'},x\n" for i in ids), encoding="utf-8")
        fd = t / "faces"
        fd.mkdir()
        for i in ids[:-1]:                                             # the last speaker has no faces file
            np.savez(fd / f"{i}.npz", landmarks=rng.uniform(0, 100, size=(3, 478, 3)).astype(np.float32),
                     sharp=rng.uniform(10, 50, 3), yaw=rng.uniform(-5, 5, 3), roll=rng.uniform(-5, 5, 3))
        a = SimpleNamespace(voices=str(t / "v.npz"), labels=str(t / "l.npz"), targets=None,
                            meta=str(t / "meta.csv"), exclude="2, 5")
        got, X, Y, grp, names, sel = load_data(a)
        ok1 = got == [i for i in ids if i not in ("2", "5")] and X.shape == (n - 2, 16) and Y.shape == (n - 2, 3)
        f = control_features(fd / "1.npz")
        ok2 = list(f) == CONTROL_NAMES and all(np.isfinite(v) for v in f.values())
        res = run_controls(str(fd), got, X, grp, 3)
        ok3 = res is not None and res["n"] == n - 3 and res["p"] is not None and len(res["r_within"]) == 4
    return bool(ok1 and ok2 and ok3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--voices", default="voices.npz")
    ap.add_argument("--labels", default="labels.npz")
    ap.add_argument("--targets", default="targets.json")
    ap.add_argument("--meta", default="dataset/meta.csv")
    ap.add_argument("--n-pc", type=int, default=4)
    ap.add_argument("--perm", type=int, default=200, help="permutation shuffles (0 = skip)")
    ap.add_argument("--out", default=None,
                    help="default results.csv (results_excl_<ids>.csv when --exclude is used)")
    ap.add_argument("--exclude", default="",
                    help="comma-separated speaker ids to leave out of everything (sensitivity check)")
    ap.add_argument("--controls", default=None,
                    help="directory with faces/<id>.npz: also run the negative controls (video conditions as targets)")
    ap.add_argument("--residualize", default="",
                    help="comma-separated video covariates (log_sharpness,log_face_px,mean_abs_yaw,mean_abs_roll) to "
                         "regress out of every AM before the model; needs --controls DIR for the covariates")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.out is None:
        ex = [x.strip() for x in a.exclude.split(",") if x.strip()]
        cov = [x.strip() for x in a.residualize.split(",") if x.strip()]
        name = "results"
        if cov:
            name += "_resid_" + "-".join(cov)
        if ex:
            name += "_excl_" + "-".join(ex)
        a.out = name + ".csv"
    run(a)


if __name__ == "__main__":
    main()