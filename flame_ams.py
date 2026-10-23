#!/usr/bin/env python3
"""
Stage 2b / 4 core: FLAME shape model + craniofacial measurements (AMs).

  MediaPipe landmarks of one person (several frames)
      --fit-->  FLAME shape coefficients beta   (neutral face, pose/expression removed)
      --measure--> AMs: scale-free facial indices (widths, lengths, ratios)       <- training labels
  predicted AMs  --fit-->  beta*  -->  mesh (.obj)                                <- stage 4

Files you need locally (FLAME licence: non-commercial research, do not redistribute):
  generic_model.pkl                         from FLAME 2020
  mediapipe_landmark_embedding.*            "FLAME Mediapipe Landmark Embedding" (.npz or .npy)

Everything is built on landmarks, so no chumpy is needed and no full-mesh forward pass.

Commands:
  python flame_ams.py inspect  --flame generic_model.pkl --emb mediapipe_landmark_embedding.npz
  python flame_ams.py labels   --flame ... --emb ... --lmk faces/ --out labels.npz
  python flame_ams.py reliability --flame ... --emb ... --lmk faces/ --out reliability.csv
  python flame_ams.py analyze  --labels labels.npz --reliability reliability.csv --out targets.json
  python flame_ams.py mesh     --flame ... --emb ... --labels labels.npz --id p001 --out p001.obj
  python flame_ams.py selftest
"""
import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import torch

# ============================================================ FLAME loading (chumpy-free)


class _Stub:
    """Stands in for chumpy / unavailable classes while unpickling."""
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Stub
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub


def _to_np(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    d = getattr(x, "__dict__", {})
    for k in ("x", "r", "_state"):
        if k in d:
            return _to_np(d[k])
    return np.asarray(x)


def load_flame_pkl(path):
    with open(path, "rb") as f:
        raw = _Unpickler(f, encoding="latin1").load()
    out = {"v_template": _to_np(raw["v_template"]).astype(np.float32),
           "shapedirs": _to_np(raw["shapedirs"]).astype(np.float32),
           "faces": _to_np(raw["f"]).astype(np.int64)}
    assert out["v_template"].ndim == 2 and out["shapedirs"].ndim == 3, \
        f"unexpected shapes {out['v_template'].shape} {out['shapedirs'].shape}"
    return out


def load_embedding(path):
    """Barycentric landmark embedding on the FLAME mesh. Returns (face_idx, bary, mp_idx)."""
    d = np.load(path, allow_pickle=True, encoding="latin1")
    if isinstance(d, np.ndarray):
        d = d[()]
    keys = list(d.files if hasattr(d, "files") else d.keys())
    arr = {k: np.asarray(d[k]) for k in keys}

    def find(cands, sub):
        for c in cands:
            if c in arr:
                return arr[c]
        for k in arr:
            if all(s in k.lower() for s in sub):
                return arr[k]
        return None

    fi = find(("lmk_face_idx", "full_lmk_faces_idx"), ("face", "idx"))
    bc = find(("lmk_b_coords", "full_lmk_bary_coords"), ("bary",))
    if bc is None:
        bc = find((), ("b_coords",))
    mp = find(("landmark_indices", "mediapipe_indices"), ("landmark", "ind"))
    if fi is None or bc is None:
        raise ValueError(f"cannot find face-index / barycentric arrays in {path}; keys: {keys}")
    fi = fi.astype(np.int64).reshape(-1)
    bc = bc.astype(np.float32).reshape(len(fi), 3)
    mp = np.arange(len(fi)) if mp is None else mp.astype(np.int64).reshape(-1)
    assert len(mp) == len(fi), f"landmark_indices ({len(mp)}) and face idx ({len(fi)}) differ"
    return fi, bc, mp


class Flame:
    """FLAME shape (+ optional expression) basis restricted to the embedded landmarks."""

    def __init__(self, model, emb, n_shape=100, n_expr=50):
        v, sd, self.faces = model["v_template"], model["shapedirs"], model["faces"]
        fi, bc, self.mp_idx = emb
        S = sd.shape[2]
        self.ns = min(n_shape, 300, S)
        self.ne = min(n_expr, max(0, S - 300))
        self.v_full, self.sd_full = v, sd
        tri = self.faces[fi]                                     # (N,3) vertex ids
        w = bc
        self.lm_t = torch.tensor(np.einsum("nk,nkc->nc", w, v[tri]))
        self.lm_s = torch.tensor(np.einsum("nk,nkcs->ncs", w, sd[tri][..., :self.ns]))
        self.lm_e = (torch.tensor(np.einsum("nk,nkcs->ncs", w, sd[tri][..., 300:300 + self.ne]))
                     if self.ne else None)
        self.row = {int(m): i for i, m in enumerate(self.mp_idx)}

    def lmk(self, beta, psi=None):
        """beta (...,ns), psi (...,ne) -> landmark positions (...,N,3)."""
        p = self.lm_t + torch.einsum("ncs,...s->...nc", self.lm_s, beta)
        if psi is not None and self.lm_e is not None:
            p = p + torch.einsum("ncs,...s->...nc", self.lm_e, psi)
        return p

    def mesh(self, beta):
        b = np.asarray(beta, dtype=np.float32)
        return self.v_full + np.einsum("vcs,s->vc", self.sd_full[..., :self.ns], b)


# ============================================================ AMs (scale-free indices)
def _D(g, a, b):
    return torch.linalg.norm(g(a) - g(b), dim=-1)


# (name, function of g, MediaPipe landmark ids it needs). All are ratios, so camera distance and head size
# drop out. The public FLAME-MediaPipe embedding has only 105 landmarks (inner face: eyes, brows, nose, lips;
# NO face-oval / chin / cheek-contour points such as 152, 234, 454, 172, 397, 1), so every index below is built
# from points that ARE in it, and the scale normaliser is the outer-eye-corner distance (33-263) instead of the
# bizygomatic width (234-454). Face-width / jaw / chin indices are therefore not available from this embedding.
BIOC = lambda g: _D(g, 33, 263)                       # biocular width = scale reference
_mid = lambda g, a, b: 0.5 * (g(a) + g(b))
_N = torch.linalg.norm

AM_SPECS = [
    ("nose_width_idx",    lambda g: _D(g, 129, 358) / BIOC(g),                       {129, 358, 33, 263}),
    ("nose_length_idx",   lambda g: _D(g, 168, 2) / BIOC(g),                         {168, 2, 33, 263}),
    ("nasal_index",       lambda g: _D(g, 129, 358) / _D(g, 168, 2),                 {129, 358, 168, 2}),
    ("mouth_width_idx",   lambda g: _D(g, 61, 291) / BIOC(g),                        {61, 291, 33, 263}),
    ("mouth_nose_ratio",  lambda g: _D(g, 61, 291) / _D(g, 129, 358),                {61, 291, 129, 358}),
    ("intercanthal_idx",  lambda g: _D(g, 133, 362) / BIOC(g),                       {133, 362, 33, 263}),
    ("eye_width_idx",     lambda g: 0.5 * (_D(g, 33, 133) + _D(g, 263, 362)) / BIOC(g),
                                                                                    {33, 133, 263, 362}),
    ("brow_distance_idx", lambda g: _D(g, 105, 334) / BIOC(g),                       {105, 334, 33, 263}),
    ("brow_nose_idx",     lambda g: _N(_mid(g, 105, 334) - g(2), dim=-1) / BIOC(g), {105, 334, 2, 33, 263}),
    ("midface_idx",       lambda g: _D(g, 168, 0) / BIOC(g),                         {168, 0, 33, 263}),
    ("eye_mouth_idx",     lambda g: _N(_mid(g, 33, 263) - g(0), dim=-1) / BIOC(g),   {33, 263, 0}),
    ("philtrum_idx",      lambda g: _D(g, 2, 0) / _D(g, 168, 2),                     {2, 0, 168}),
    ("lip_height_idx",    lambda g: _D(g, 0, 17) / _D(g, 61, 291),                   {0, 17, 61, 291}),
    # added after `inspect` showed ids 4, 13, 14, 144-160, 374-386 etc. are in the embedding
    ("nose_protrusion_idx", lambda g: _N(g(4) - _mid(g, 129, 358), dim=-1) / _D(g, 129, 358),
                                                                                    {4, 129, 358}),
    ("eye_height_idx",   lambda g: 0.5 * (_D(g, 159, 145) / _D(g, 33, 133) + _D(g, 386, 374) / _D(g, 362, 263)),
                                                                                    {159, 145, 33, 133, 386, 374, 362, 263}),
    ("brow_eye_idx",     lambda g: 0.5 * (_D(g, 105, 159) + _D(g, 334, 386)) / BIOC(g),
                                                                                    {105, 159, 334, 386, 33, 263}),
    ("upper_lip_idx",    lambda g: _D(g, 0, 13) / _D(g, 61, 291),                   {0, 13, 61, 291}),
    ("lower_lip_idx",    lambda g: _D(g, 14, 17) / _D(g, 61, 291),                  {14, 17, 61, 291}),
]


def available_ams(mp_idx):
    have = set(int(i) for i in mp_idx)
    return [n for n, _, need in AM_SPECS if need <= have]


def ams_from_points(P, mp_idx, names=None):
    """P (...,N,3) landmarks whose rows correspond to mp_idx. Returns (...,len(names))."""
    row = {int(m): i for i, m in enumerate(mp_idx)}
    g = lambda i: P[..., row[i], :]
    names = names or available_ams(mp_idx)
    if not names:
        raise RuntimeError("none of the AMs can be computed from this landmark embedding; "
                           "run `inspect` to see which MediaPipe ids it contains")
    spec = {n: f for n, f, _ in AM_SPECS}
    return torch.stack([spec[n](g) for n in names], dim=-1)


@torch.no_grad()
def neutral_ams(flame, beta, names=None):
    beta = torch.as_tensor(beta, dtype=torch.float32)
    return ams_from_points(flame.lmk(beta), flame.mp_idx, names).numpy()


# ============================================================ fitting
def rodrigues(r):
    """axis-angle (...,3) -> rotation matrices (...,3,3)"""
    th = torch.linalg.norm(r, dim=-1, keepdim=True).clamp_min(1e-8)
    k = r / th
    K = torch.zeros(*r.shape[:-1], 3, 3, dtype=r.dtype)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th = th[..., None]
    eye = torch.eye(3, dtype=r.dtype).expand_as(K)
    return eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def mediapipe_to_model_frame(lm_px, mp_idx):
    """(K,478,3) pixels (x right, y down, z negative = closer) -> (K,N,3) in a y-up, z-toward-camera frame,
    centred and scaled per frame (rms radius = 1)."""
    p = torch.as_tensor(lm_px, dtype=torch.float32)[:, list(mp_idx), :]
    p = p * torch.tensor([1.0, -1.0, -1.0])
    p = p - p.mean(1, keepdim=True)
    return p / p.pow(2).sum(-1).mean(1).sqrt()[:, None, None]


def fit_identity(flame, lm_px, iters=500, sigma=0.02, z_weight=0.3, lr=0.03):
    """Fit one shared beta + per-frame (rotation, scale, translation, expression) to K frames of landmarks.
    Returns dict(beta, rms, per-frame rms). Residuals are in units where the landmark cloud has rms radius 1."""
    tgt = mediapipe_to_model_frame(lm_px, flame.mp_idx)
    K = tgt.shape[0]
    base = flame.lmk(torch.zeros(flame.ns))
    c0 = base.mean(0)
    s0 = 1.0 / (base - c0).pow(2).sum(-1).mean().sqrt()
    beta = torch.zeros(flame.ns, requires_grad=True)
    psi = torch.zeros(K, max(flame.ne, 1), requires_grad=True)
    rot = torch.zeros(K, 3, requires_grad=True)
    logs = torch.full((K,), float(torch.log(s0)), requires_grad=True)
    t = torch.zeros(K, 3, requires_grad=True)
    w = torch.tensor([1.0, 1.0, z_weight])

    def predict():
        lm = flame.lmk(beta.expand(K, -1), psi[:, :flame.ne] if flame.ne else None)
        lm = lm - c0
        R = rodrigues(rot)
        lm = torch.einsum("kij,knj->kni", R, lm) * logs.exp()[:, None, None] + t[:, None, :]
        return lm

    def loss_fn():
        res = (predict() - tgt) * w
        reg = sigma ** 2 * (beta.pow(2).sum() + psi.pow(2).sum() / K)
        return res.pow(2).sum() / K + reg

    rigid = torch.optim.Adam([rot, logs, t], lr=lr)
    for _ in range(iters // 5):
        rigid.zero_grad(); loss_fn().backward(); rigid.step()
    opt = torch.optim.Adam([beta, psi, rot, logs, t], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for _ in range(iters):
        opt.zero_grad(); loss = loss_fn(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        per = ((predict() - tgt) * w).pow(2).sum(-1).mean(1).sqrt()
    return {"beta": beta.detach().numpy(), "rms": float(per.mean()), "rms_frames": per.numpy()}


def fit_beta_from_ams(flame, names, target, rel_sd=0.05, reg=1e-2, iters=600, lr=0.05):
    """beta* = argmin reg*||beta||^2 + sum_k ((AM_k(beta) - m_k) / sd_k)^2   (stage 4)"""
    target = torch.as_tensor(target, dtype=torch.float32)
    sd = (rel_sd * target.abs()).clamp_min(0.01)
    beta = torch.zeros(flame.ns, requires_grad=True)
    opt = torch.optim.Adam([beta], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        pred = ams_from_points(flame.lmk(beta), flame.mp_idx, names)
        loss = reg * beta.pow(2).sum() + ((pred - target) / sd).pow(2).sum()
        loss.backward(); opt.step()
    return beta.detach().numpy()


def export_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


# ============================================================ commands
def build(args):
    model = load_flame_pkl(args.flame)
    emb = load_embedding(args.emb)
    return Flame(model, emb, args.n_shape, args.n_expr)


def cmd_inspect(a):
    model = load_flame_pkl(a.flame)
    print("FLAME pkl:", {k: v.shape for k, v in model.items()})
    emb = load_embedding(a.emb)
    print("embedding: landmarks", len(emb[0]), "| face idx max", emb[0].max(),
          "| faces in model", len(model["faces"]), "| mp ids range", emb[2].min(), "-", emb[2].max())
    fl = Flame(model, emb, a.n_shape, a.n_expr)
    print(f"shape comps used: {fl.ns}, expression comps: {fl.ne}")
    print("MediaPipe ids in embedding:", sorted(int(i) for i in fl.mp_idx))
    av = available_ams(fl.mp_idx)
    print(f"AMs computable ({len(av)}/{len(AM_SPECS)}):", av)
    miss = [n for n, _, _ in AM_SPECS if n not in av]
    if miss:
        need = sorted(set().union(*[nd for n, _, nd in AM_SPECS if n in miss]) - set(int(i) for i in fl.mp_idx))
        print("not computable:", miss, "| missing MediaPipe ids:", need)
    if not av:
        return
    mean_ams = neutral_ams(fl, np.zeros(fl.ns), av)
    print("AMs of the mean FLAME face:", dict(zip(av, np.round(mean_ams, 3))))


def cmd_labels(a):
    fl = build(a)
    names = available_ams(fl.mp_idx)
    ids, betas, ams, errs = [], [], [], []
    for p in sorted(Path(a.lmk).glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        lm = d["landmarks"]
        if len(lm) < a.min_frames:
            print(f"skip {p.stem}: {len(lm)} frames < {a.min_frames}")
            continue
        r = fit_identity(fl, lm, iters=a.iters)
        ids.append(p.stem); betas.append(r["beta"]); errs.append(r["rms"])
        ams.append(neutral_ams(fl, r["beta"], names))
        print(f"{p.stem}: frames {len(lm)}  fit rms {r['rms']:.4f}")
    errs = np.array(errs)
    keep = errs <= (np.median(errs) + 3 * 1.4826 * np.median(np.abs(errs - np.median(errs))) + 1e-9)
    np.savez(a.out, ids=np.array(ids), beta=np.stack(betas), ams=np.stack(ams), names=np.array(names),
             fit_rms=errs, keep=keep)
    print(f"saved {a.out}: {len(ids)} identities, {keep.sum()} pass the fit-quality filter, {len(names)} AMs")


def split_half_ams(flame, lm, names, iters=300):
    """Fit beta separately on the even / odd frames of one speaker (frames are stored best-first, so the two
    halves have similar quality) and return the AMs of both fits."""
    a = fit_identity(flame, lm[0::2], iters=iters)
    b = fit_identity(flame, lm[1::2], iters=iters)
    return neutral_ams(flame, a["beta"], names), neutral_ams(flame, b["beta"], names)


def reliability_table(A, B, names):
    """A, B: (n_identities, n_AMs) from the two halves. Per AM: Pearson r between halves, the Spearman-Brown
    corrected r (reliability of a fit on all frames), and the mean relative difference between the halves."""
    rows = []
    for k, n in enumerate(names):
        r = float(np.corrcoef(A[:, k], B[:, k])[0, 1])
        sb = 2 * r / (1 + r) if r > -0.999 else float("nan")
        rel = float(np.mean(np.abs(A[:, k] - B[:, k]) / (0.5 * (np.abs(A[:, k]) + np.abs(B[:, k])) + 1e-9)))
        rows.append((n, r, sb, rel))
    return rows


def cmd_reliability(a):
    import csv
    fl = build(a)
    names = available_ams(fl.mp_idx)
    ids, A, B = [], [], []
    for p in sorted(Path(a.lmk).glob("*.npz")):
        lm = np.load(p, allow_pickle=True)["landmarks"]
        if len(lm) < 2 * a.min_half:
            print(f"skip {p.stem}: {len(lm)} frames < {2 * a.min_half}")
            continue
        x, y = split_half_ams(fl, lm, names, a.iters)
        ids.append(p.stem); A.append(x); B.append(y)
        print(f"{p.stem}: {len(lm)} frames -> halves {len(lm[0::2])}/{len(lm[1::2])}")
    if len(ids) < 5:
        raise SystemExit(f"only {len(ids)} speakers usable; need at least 5 for a meaningful correlation")
    rows = reliability_table(np.stack(A), np.stack(B), names)
    print(f"\nsplit-half reliability over {len(ids)} speakers "
          f"(each half uses only ~{a.min_half}+ frames, so r is pessimistic; r_SB is the estimate for the full fit)")
    print(f"{'AM':22s} {'r':>6s} {'r_SB':>6s} {'rel.diff':>9s}  verdict")
    for n, r, sb, rel in sorted(rows, key=lambda t: -t[2]):
        v = "ok" if sb >= 0.7 else ("weak" if sb >= 0.5 else "drop?")
        print(f"{n:22s} {r:6.2f} {sb:6.2f} {rel * 100:8.1f}%  {v}")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["am", "r_half", "r_spearman_brown", "mean_rel_diff"])
            w.writerows(rows)
        print("saved", a.out)


def cmd_analyze(a):
    """Redundancy / dimensionality of the AM labels and a non-redundant target subset (numpy only)."""
    import csv
    import json
    d = np.load(a.labels, allow_pickle=True)
    keep = d["keep"].astype(bool)
    names = [str(n) for n in d["names"]]
    X = d["ams"][keep]
    ids = [str(i) for i in d["ids"][keep]]
    n, p = X.shape
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    C = np.corrcoef(Z.T)
    rel = {}
    if a.reliability and Path(a.reliability).exists():
        with open(a.reliability, encoding="utf-8") as f:
            rel = {r["am"]: float(r["r_spearman_brown"]) for r in csv.DictReader(f)}
    print(f"{n} identities (keep=True), {p} AMs" + (f", reliability from {a.reliability}" if rel else ""))
    if n < 30:
        print(f"  note: n={n} is small; correlations/PCA here are rough guides (PC1 is inflated, |r| has a wide CI)")

    pairs = [(names[i], names[j], float(C[i, j])) for i in range(p) for j in range(i + 1, p)
             if abs(C[i, j]) > a.max_corr]
    print(f"\npairs with |r| > {a.max_corr}:")
    for x, y, c in sorted(pairs, key=lambda t: -abs(t[2])):
        print(f"  {x:22s} {y:22s} {c:+.2f}")
    if not pairs:
        print("  none")

    # greedy subset: most reliable first, skip an AM that is too correlated with one already kept
    order = sorted(range(p), key=lambda i: -rel.get(names[i], 0.0))
    sel, dropped = [], {}
    for i in order:
        clash = [(j, abs(C[i, j])) for j in sel if abs(C[i, j]) > a.max_corr]
        if clash:
            j, c = max(clash, key=lambda t: t[1])
            dropped[names[i]] = f"|r|={c:.2f} with {names[j]}"
        else:
            sel.append(i)
    selected = [names[i] for i in sorted(sel)]
    print(f"\nnon-redundant subset ({len(selected)}/{p}):", selected)
    for k, v in dropped.items():
        print(f"  dropped {k:22s} ({v})")

    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    ev = S ** 2 / np.sum(S ** 2)
    cum = np.cumsum(ev)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    print("\nPCA explained variance:", np.round(ev[:8], 2).tolist(), "| cumulative:", np.round(cum[:8], 2).tolist())
    print(f"components for 90% of the variance: {k90}")
    for c in range(min(4, len(ev))):
        top = np.argsort(-np.abs(Vt[c]))[:3]
        print(f"  PC{c + 1} ({ev[c] * 100:.0f}%): " + ", ".join(f"{names[i]} {Vt[c, i]:+.2f}" for i in top))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"all": names, "selected": selected, "dropped": dropped, "n_identities": n, "ids": ids,
                       "pca_explained": [float(x) for x in ev], "max_corr": a.max_corr}, f, indent=1)
        print("saved", a.out)


def cmd_mesh(a):
    fl = build(a)
    d = np.load(a.labels, allow_pickle=True)
    if a.id == "mean":
        beta = np.zeros(fl.ns, dtype=np.float32)
    else:
        beta = d["beta"][list(d["ids"]).index(a.id)]
    export_obj(a.out, fl.mesh(beta), fl.faces)
    print("wrote", a.out)


# ============================================================ self-test (synthetic model)
def _fake_files(tmp):
    """A made-up 'FLAME' (random linear shape model) in the real file layouts, incl. a chumpy-style pickle."""
    import sys
    import types
    rng = np.random.default_rng(0)
    V, F, N = 800, 2000, 105
    v = np.stack([rng.uniform(-.08, .08, V), rng.uniform(-.11, .11, V), rng.uniform(0, .1, V)], 1).astype(np.float32)
    sd = (rng.normal(size=(V, 3, 400)) * 0.002).astype(np.float32)
    faces = rng.integers(0, V, size=(F, 3))
    mod = types.ModuleType("chumpy"); sub = types.ModuleType("chumpy.ch")

    class Ch:
        def __init__(self, x): self.x = x
        def __getstate__(self): return {"x": self.x, "_dirty": True}
    Ch.__module__ = "chumpy.ch"; Ch.__qualname__ = "Ch"
    sub.Ch = Ch; mod.ch = sub
    sys.modules["chumpy"], sys.modules["chumpy.ch"] = mod, sub
    pkl = Path(tmp) / "generic_model.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"v_template": Ch(v), "shapedirs": Ch(sd), "f": faces}, f, protocol=2)
    del sys.modules["chumpy"], sys.modules["chumpy.ch"]
    fi = rng.integers(0, F, size=N)
    bc = rng.dirichlet([1, 1, 1], size=N).astype(np.float32)
    need = sorted(set().union(*[nd for _, _, nd in AM_SPECS]))
    rest = [i for i in range(468) if i not in need]
    mp = np.sort(np.array(need + list(rng.choice(rest, N - len(need), replace=False))))
    np.savez(Path(tmp) / "emb.npz", lmk_face_idx=fi, lmk_b_coords=bc, landmark_indices=mp)
    np.save(Path(tmp) / "emb2.npy", {"full_lmk_faces_idx": fi, "full_lmk_bary_coords": bc}, allow_pickle=True)
    return pkl, Path(tmp) / "emb.npz", Path(tmp) / "emb2.npy"


def selftest():
    import tempfile
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        pkl, e1, e2 = _fake_files(tmp)
        model = load_flame_pkl(pkl)
        assert model["shapedirs"].shape == (800, 3, 400), model["shapedirs"].shape
        emb = load_embedding(e1); emb2 = load_embedding(e2)
        assert np.array_equal(emb[0], emb2[0]) and len(emb[2]) == len(emb2[2]) == 105
        fl = Flame(model, emb, n_shape=40, n_expr=20)
    names = available_ams(fl.mp_idx)
    print(f"loader OK (chumpy-style pickle, .npz and .npy embeddings); {len(names)} AMs")

    rng = np.random.default_rng(1)
    n_id, K = 10, 6
    true_b = rng.normal(size=(n_id, fl.ns)).astype(np.float32) * 0.8
    true_a = neutral_ams(fl, true_b, names)
    fit_a, fit_b = [], []
    for i in range(n_id):
        frames = []
        for _ in range(K):
            rv = torch.tensor(rng.normal(size=3) * 0.12, dtype=torch.float32)
            R = rodrigues(rv)
            psi = torch.tensor(rng.normal(size=fl.ne) * 0.3, dtype=torch.float32)
            lm = fl.lmk(torch.tensor(true_b[i]), psi)
            lm = (lm - lm.mean(0)) @ R.T * float(rng.uniform(1000, 1600)) + torch.tensor(rng.normal(size=3) * 20, dtype=torch.float32)
            px = lm * torch.tensor([1.0, -1.0, -1.0])                     # model frame -> MediaPipe-like pixels
            px = px + torch.randn_like(px) * torch.tensor([0.7, 0.7, 3.0])  # landmark noise, z noisier
            px[:, :2] += torch.tensor([640.0, 360.0])
            frames.append(px.numpy())
        full = np.zeros((K, 478, 3), dtype=np.float32)
        full[:, fl.mp_idx] = np.stack(frames)
        r = fit_identity(fl, full, iters=400)
        fit_b.append(r["beta"]); fit_a.append(neutral_ams(fl, r["beta"], names))
        if i == 0:
            print(f"fit rms (normalised units): {r['rms']:.4f}")
    fit_a = np.stack(fit_a)
    cors = [np.corrcoef(true_a[:, k], fit_a[:, k])[0, 1] for k in range(len(names))]
    for n, c in zip(names, cors):
        print(f"  {n:22s} corr(true, fitted over identities) = {c:5.2f}")
    print(f"mean AM correlation: {np.mean(cors):.2f}")

    b_star = fit_beta_from_ams(fl, names, true_a[0])
    rec = neutral_ams(fl, b_star, names)
    rel = np.abs(rec - true_a[0]) / np.abs(true_a[0])
    print(f"AMs -> beta* -> AMs round trip: mean rel. error {rel.mean() * 100:.2f}%  max {rel.max() * 100:.2f}%")
    ok = np.mean(cors) > 0.6 and rel.mean() < 0.02
    print("SELFTEST", "PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--flame", required=True)
        p.add_argument("--emb", required=True)
        p.add_argument("--n-shape", type=int, default=100)
        p.add_argument("--n-expr", type=int, default=50)
    p = sub.add_parser("inspect"); common(p); p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("labels"); common(p)
    p.add_argument("--lmk", required=True, help="folder with <id>.npz from face_stage.py")
    p.add_argument("--out", default="labels.npz")
    p.add_argument("--min-frames", type=int, default=3)
    p.add_argument("--iters", type=int, default=500); p.set_defaults(fn=cmd_labels)
    p = sub.add_parser("reliability"); common(p)
    p.add_argument("--lmk", required=True, help="folder with <id>.npz from face_stage.py")
    p.add_argument("--min-half", type=int, default=3, help="min frames per half (speaker needs 2x this)")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--out", default="reliability.csv"); p.set_defaults(fn=cmd_reliability)
    p = sub.add_parser("analyze")
    p.add_argument("--labels", default="labels.npz")
    p.add_argument("--reliability", default="reliability.csv")
    p.add_argument("--max-corr", type=float, default=0.85)
    p.add_argument("--out", default="targets.json"); p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("mesh"); common(p)
    p.add_argument("--labels", required=True); p.add_argument("--id", default="mean")
    p.add_argument("--out", default="face.obj"); p.set_defaults(fn=cmd_mesh)
    p = sub.add_parser("selftest"); p.set_defaults(fn=lambda a: selftest())
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()