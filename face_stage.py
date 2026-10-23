#!/usr/bin/env python3
"""
Stage 2a: pick usable face frames and extract MediaPipe landmarks, one .npz per speaker.

For every speaker folder in --frames (output of collect_dataset.py) it keeps the K frames that have
exactly one large, sharp, near-frontal face with open eyes and a mostly closed mouth, and stores their
478 landmarks (x, y in pixels, z in the same scale). flame_ams.py then fits FLAME to them.

Install:   pip install mediapipe opencv-python numpy
Model:     download once (the Tasks API needs the model file):
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Optional filters (both off by default):
  --others-ratio R   R=0 (default): a frame is rejected unless MediaPipe finds exactly one face.
                     R>0: take the LARGEST face and reject the frame only if another face is >= R x its size
                     (tiny background faces are tolerated, so fewer frames are lost).
  --id-filter        embed every candidate face with facenet (VGGFace2) and keep only the speaker's dominant
                     identity (drops interviewers, guests, wrong person). --id-min-sim is the cosine threshold.
                     Needs:  pip install facenet-pytorch --no-deps pillow requests tqdm   (torch is already there)
                     Prints per-speaker similarity statistics and, with --preview, writes prev/<id>_otherperson.jpg
                     (the rejected faces closest to the threshold, with their similarity) so the threshold can be
                     tuned by eye.

Usage:
  python face_stage.py --frames dataset/frames --out faces --model face_landmarker.task [--k 8] [--preview prev]
  python face_stage.py --selftest
"""
import argparse
import csv
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ------------------------------------------------------------------ geometry
def _d(lm, a, b):
    return float(np.linalg.norm(lm[a] - lm[b]))


def head_angles(lm):
    """(yaw, roll, pitch_proxy) in degrees from the MediaPipe landmarks.
    yaw: cheek-to-cheek vector in the x-z plane; roll: eye-corner line; pitch_proxy: forehead-chin vector in the
    y-z plane (has a person-specific offset, so it is compared against the speaker's own median)."""
    yaw = np.degrees(np.arctan2(lm[454, 2] - lm[234, 2], lm[454, 0] - lm[234, 0]))
    roll = np.degrees(np.arctan2(lm[263, 1] - lm[33, 1], lm[263, 0] - lm[33, 0]))
    pitch = np.degrees(np.arctan2(lm[152, 2] - lm[10, 2], lm[152, 1] - lm[10, 1]))
    return float(yaw), float(roll), float(pitch)


def expression_metrics(lm):
    eye_r = _d(lm, 159, 145) / max(_d(lm, 33, 133), 1e-6)
    eye_l = _d(lm, 386, 374) / max(_d(lm, 362, 263), 1e-6)
    mouth = _d(lm, 13, 14) / max(_d(lm, 61, 291), 1e-6)
    return 0.5 * (eye_r + eye_l), mouth


# ------------------------------------------------------------------ MediaPipe
def make_landmarker(model_path):
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE, num_faces=4,
        min_face_detection_confidence=0.6, min_face_presence_confidence=0.6)
    return vision.FaceLandmarker.create_from_options(opts)


def read_image(path):
    import cv2
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def detect_all(landmarker, bgr):
    """List of (478,3) landmark arrays, one per face MediaPipe found (possibly empty)."""
    import cv2
    import mediapipe as mp
    h, w = bgr.shape[:2]
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = landmarker.detect(img)
    return [np.array([[p.x * w, p.y * h, p.z * w] for p in f], dtype=np.float32) for f in res.face_landmarks]


def face_size(lm):
    """Longer side of the landmark bounding box, pixels (robust to yaw, unlike cheek-to-cheek distance)."""
    return float((lm[:, :2].max(0) - lm[:, :2].min(0)).max())


def pick_main(faces, others_ratio):
    """faces: list of landmark arrays -> (landmarks or None, reject reason or None).
    others_ratio <= 0: exactly one face is required. > 0: the largest face is the subject, and the frame is
    rejected only if the second largest is at least others_ratio times as large."""
    if not faces:
        return None, "no_face"
    if others_ratio <= 0:
        return (faces[0], None) if len(faces) == 1 else (None, "other_faces")
    sizes = [face_size(f) for f in faces]
    order = np.argsort(sizes)[::-1]
    if len(faces) > 1 and sizes[order[1]] >= others_ratio * sizes[order[0]]:
        return None, "other_faces"
    return faces[int(order[0])], None


# ------------------------------------------------------------------ identity filter (facenet)
def face_crop(bgr, lm, out=160, margin=1.15):
    """Square RGB crop around the landmark box (replicate-padded at image borders), resized to out x out."""
    import cv2
    c = (lm[:, :2].min(0) + lm[:, :2].max(0)) / 2
    half = max(face_size(lm) * margin / 2, 8.0)
    pad = int(np.ceil(half)) + 2
    p = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    x0, y0 = int(round(c[0] - half)) + pad, int(round(c[1] - half)) + pad
    x1, y1 = int(round(c[0] + half)) + pad, int(round(c[1] + half)) + pad
    crop = cv2.resize(p[y0:y1, x0:x1], (out, out), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def load_embedder(device):
    from facenet_pytorch import InceptionResnetV1
    return InceptionResnetV1(pretrained="vggface2").eval().to(device)


def embed_crops(model, crops, device, batch=32):
    """List of (160,160,3) RGB uint8 -> (n,512) L2-normalised facenet embeddings."""
    import torch
    out = []
    with torch.no_grad():
        for b in range(0, len(crops), batch):
            x = torch.from_numpy(np.stack(crops[b:b + batch])).permute(0, 3, 1, 2).float()
            e = model(((x - 127.5) / 128.0).to(device)).cpu().numpy()
            out.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9))
    return np.concatenate(out)


def dominant_identity(E, min_sim):
    """E (n,d) L2-normalised embeddings of one speaker's frames -> (keep mask, cosine to the identity centre, info).
    The centre is built in two passes: the medoid (face most similar to the rest) seeds a cluster of faces with
    cosine >= min_sim to it, the cluster mean is the centre, and frames below min_sim to the centre are dropped.
    With fewer than 3 faces there is nothing to compare, so everything is kept (info['decided'] = False)."""
    n = len(E)
    if n < 3:
        return np.ones(n, bool), np.ones(n), dict(decided=False, share=1.0)
    S = E @ E.T
    np.fill_diagonal(S, np.nan)
    medoid = int(np.nanargmax(np.nanmedian(S, axis=1)))
    seed = S[medoid] >= min_sim
    seed[medoid] = True
    keep = seed
    for _ in range(2):
        c = E[keep].mean(0)
        c = c / (np.linalg.norm(c) + 1e-9)
        sim = E @ c
        keep = sim >= min_sim
        if not keep.any():
            keep[medoid] = True
    return keep, sim, dict(decided=True, share=float(keep.mean()))


def sharpness(bgr, lm):
    import cv2
    x0, y0 = np.floor(lm[:, :2].min(0)).astype(int)
    x1, y1 = np.ceil(lm[:, :2].max(0)).astype(int)
    h, w = bgr.shape[:2]
    crop = bgr[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]
    if crop.size == 0:
        return 0.0
    crop = cv2.resize(crop, (192, 192))            # compare faces at a common size
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


# ------------------------------------------------------------------ selection
def select(cands, k, max_yaw, max_roll, max_pitch_dev, min_eye, max_mouth, stats=None):
    """cands: list of dicts with yaw, roll, pitch, eye, mouth, ... -> the best k."""
    if not cands:
        return []
    med_p = float(np.median([c["pitch"] for c in cands]))
    good = []
    for c in cands:
        c["pitch_rel"] = c["pitch"] - med_p
        if stats is not None:
            for key, bad in (("yaw", abs(c["yaw"]) > max_yaw), ("roll", abs(c["roll"]) > max_roll),
                             ("pitch", abs(c["pitch_rel"]) > max_pitch_dev), ("eyes", c["eye"] < min_eye),
                             ("mouth", c["mouth"] > max_mouth)):
                stats[key] = stats.get(key, 0) + int(bad)
        if (abs(c["yaw"]) <= max_yaw and abs(c["roll"]) <= max_roll and abs(c["pitch_rel"]) <= max_pitch_dev
                and c["eye"] >= min_eye and c["mouth"] <= max_mouth):
            c["score"] = float(np.hypot(np.hypot(c["yaw"], c["roll"]), c["pitch_rel"]) + 50 * c["mouth"])
            good.append(c)
    good.sort(key=lambda c: c["score"])
    return good[:k]


def sharp_threshold(sharps, min_sharp, auto, pct, floor):
    """Absolute threshold, or (auto) a per-speaker one: the pct-th percentile of that speaker's own sharpness,
    never above min_sharp (good videos keep the plain rule) and never below floor (hopeless frames stay out)."""
    if not auto or len(sharps) == 0:
        return min_sharp
    return float(max(floor, min(min_sharp, np.percentile(sharps, pct))))


def run(a):
    import cv2
    lmk = make_landmarker(a.model)
    emb_model, device = None, "cpu"
    if a.id_filter:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        emb_model = load_embedder(device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prev = Path(a.preview) if a.preview else None
    if prev:
        prev.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(p for p in Path(a.frames).iterdir() if p.is_dir()):
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        rej = dict(unreadable=0, no_face=0, other_faces=0, small=0, other_person=0, blurry=0)
        det = []                                   # single/main faces that are large enough (images are not kept)
        for f in files:
            bgr = read_image(f)
            if bgr is None:
                rej["unreadable"] += 1
                continue
            lm, why = pick_main(detect_all(lmk, bgr), a.others_ratio)
            if lm is None:
                rej[why] += 1
                continue
            if _d(lm, 234, 454) < a.min_face_px:
                rej["small"] += 1
                continue
            det.append(dict(f=f, lm=lm, sh=sharpness(bgr, lm),
                            crop=face_crop(bgr, lm) if a.id_filter else None, sim=np.nan))
        id_note = ""
        if a.id_filter and det:
            E = embed_crops(emb_model, [x["crop"] for x in det], device)
            keep, sim, info = dominant_identity(E, a.id_min_sim)
            for x, s_ in zip(det, sim):
                x["sim"] = float(s_)
            if info["decided"]:
                ks = sim[keep]
                id_note = (f" | id sim kept: median {np.median(ks):.2f} min {ks.min():.2f}; "
                           f"dropped {int((~keep).sum())}/{len(keep)} (share {info['share']:.0%})")
                if info["share"] < 0.5:
                    id_note += "  !! dominant identity covers < 50% of frames: check the preview"
                if prev and (~keep).any():          # the rejected faces closest to the threshold decide the tuning
                    idx = [i for i in np.argsort(-sim) if not keep[i]][:8]
                    tiles = []
                    for i in idx:
                        t = cv2.cvtColor(det[i]["crop"], cv2.COLOR_RGB2BGR).copy()
                        cv2.putText(t, f"{sim[i]:.2f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        tiles.append(cv2.resize(t, (128, 128)))
                    cv2.imwrite(str(prev / f"{d.name}_otherperson.jpg"), np.hstack(tiles))
            else:
                id_note = " | id filter skipped (< 3 faces)"
            rej["other_person"] += int((~keep).sum())
            det = [x for x, k in zip(det, keep) if k]
        thr = sharp_threshold([x["sh"] for x in det], a.min_sharp, a.auto_sharp, a.sharp_pct, a.sharp_floor)
        cands = []
        for x in det:
            if x["sh"] < thr:
                rej["blurry"] += 1
                continue
            yaw, roll, pitch = head_angles(x["lm"])
            eye, mouth = expression_metrics(x["lm"])
            cands.append(dict(file=x["f"].name, path=x["f"], lm=x["lm"], yaw=yaw, roll=roll, pitch=pitch, eye=eye,
                              mouth=mouth, sharp=x["sh"], sim=x["sim"]))
        stats = {}
        best = select(cands, a.k, a.max_yaw, a.max_roll, a.max_pitch_dev, a.min_eye, a.max_mouth, stats)
        rows.append((d.name, len(files), len(cands), len(best)))
        print(f"{d.name}: {len(files)} frames, {len(cands)} usable faces, kept {len(best)}"
              + (f" | sharp thr {thr:.1f}" if a.auto_sharp else "") + id_note)
        print("   dropped early:", {k: v for k, v in rej.items() if v},
              "| failed pose/expr filter (can overlap):", stats)
        if not best:
            continue
        np.savez(out / f"{d.name}.npz", landmarks=np.stack([c["lm"] for c in best]),
                 files=np.array([c["file"] for c in best]), yaw=np.array([c["yaw"] for c in best]),
                 roll=np.array([c["roll"] for c in best]), pitch_rel=np.array([c["pitch_rel"] for c in best]),
                 sharp=np.array([c["sharp"] for c in best]), id_sim=np.array([c["sim"] for c in best]))
        if prev:
            tiles = []
            for c in best:
                bgr = read_image(c["path"])
                x0, y0 = np.maximum(c["lm"][:, :2].min(0).astype(int) - 20, 0)
                x1, y1 = c["lm"][:, :2].max(0).astype(int) + 20
                tiles.append(cv2.resize(bgr[y0:y1, x0:x1], (128, 160)))
            cv2.imwrite(str(prev / f"{d.name}.jpg"), np.hstack(tiles))
    with open(out / "_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "frames", "usable", "kept"])
        w.writerows(rows)
    ok = sum(1 for r in rows if r[3] >= a.min_keep)
    print(f"done: {ok}/{len(rows)} speakers have >= {a.min_keep} good frames -> {out}")


# ------------------------------------------------------------------ self-test (geometry + selection only)
def selftest():
    base = np.zeros((478, 3), dtype=np.float32)
    pts = {234: (-70, 0, 0), 454: (70, 0, 0), 33: (-40, -30, 5), 263: (40, -30, 5), 10: (0, -90, -10),
           152: (0, 90, 10), 159: (-40, -33, 4), 145: (-40, -27, 4), 386: (40, -33, 4), 374: (40, -27, 4),
           133: (-15, -30, 5), 362: (15, -30, 5), 13: (0, 40, 8), 14: (0, 43, 8), 61: (-25, 42, 6), 291: (25, 42, 6)}
    for k, v in pts.items():
        base[k] = v

    def rot(P, yaw=0, roll=0):
        ty, tr = np.radians(yaw), np.radians(roll)
        Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
        Rz = np.array([[np.cos(tr), -np.sin(tr), 0], [np.sin(tr), np.cos(tr), 0], [0, 0, 1]])
        return (P @ Ry.T @ Rz.T).astype(np.float32)

    y0, r0, p0 = head_angles(base)
    y1, r1, _ = head_angles(rot(base, yaw=20))
    y2, r2, _ = head_angles(rot(base, roll=10))
    print(f"frontal yaw/roll {y0:.1f}/{r0:.1f} | yaw-20 gives yaw {y1:.1f} | roll-10 gives roll {r2:.1f}")
    eye, mouth = expression_metrics(base)
    print(f"eye openness {eye:.2f}, mouth opening {mouth:.2f}")
    cands = [dict(yaw=y, roll=r, pitch=p, eye=0.25, mouth=m) for y, r, p, m in
             [(2, 1, 0, .02), (25, 0, 0, .02), (3, 2, 1, .30), (-4, 3, -2, .03), (1, 0, 40, .02)]]
    best = select(cands, 3, 12, 12, 15, 0.12, 0.15)
    print("kept:", [(c["yaw"], c["roll"], c["pitch"]) for c in best])
    # the sign of yaw/roll depends on the rotation convention; only the magnitude is used for filtering
    ok = (abs(y0) < 1e-3 and abs(r0) < 1e-3 and abs(abs(y1) - 20) < 1.5 and abs(abs(r2) - 10) < 1.5
          and len(best) == 2 and best[0]["yaw"] == 2)
    t_sharp = sharp_threshold([100, 80, 60, 50], 30, True, 40, 8)       # sharp video -> plain min_sharp
    t_blur = sharp_threshold([20, 15, 12, 10, 9], 30, True, 40, 8)      # blurry video -> its own percentile
    t_junk = sharp_threshold([3, 2, 2, 1], 30, True, 40, 8)             # hopeless video -> floor
    t_off = sharp_threshold([3, 2], 30, False, 40, 8)
    print(f"sharp thresholds: sharp {t_sharp}, blurry {t_blur:.1f}, hopeless {t_junk}, off {t_off}")
    ok = ok and t_sharp == 30 and 10 < t_blur < 15 and t_junk == 8 and t_off == 30
    # --- others-ratio
    def box(size):
        L = np.zeros((478, 3), dtype=np.float32)
        L[0, :2], L[1, :2] = (0, 0), (size, size)
        return L
    big, tiny, mid = box(200), box(40), box(80)
    r1 = pick_main([big], 0)[0] is not None
    r2 = pick_main([big, tiny], 0)[1] == "other_faces"                 # strict mode: any second face rejects
    r3 = pick_main([tiny, big], 0.3)[0] is big                         # ratio mode: tiny (0.2) tolerated, big chosen
    r4 = pick_main([big, mid], 0.3)[1] == "other_faces"                # 0.4 >= 0.3 -> rejected
    r5 = pick_main([], 0.3)[1] == "no_face"
    ok_o = r1 and r2 and r3 and r4 and r5
    # --- dominant identity (synthetic unit-norm embeddings, within-identity cosine ~0.7)
    rng = np.random.default_rng(0)
    dim = 128
    vA, vB = rng.normal(size=dim), rng.normal(size=dim)
    vA, vB = vA / np.linalg.norm(vA), vB / np.linalg.norm(vB)
    mk = lambda v, n: np.stack([v + 0.6 * rng.normal(size=dim) / np.sqrt(dim) for _ in range(n)])
    E = np.concatenate([mk(vA, 16), mk(vB, 4)])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    keep, sim, info = dominant_identity(E, 0.4)
    ok_i = keep[:16].all() and not keep[16:].any() and abs(info["share"] - 0.8) < 1e-9
    k2 = dominant_identity(E[:2], 0.4)
    ok_i = ok_i and k2[0].all() and not k2[2]["decided"]
    print(f"others-ratio logic ok: {ok_o} | identity filter: kept {int(keep.sum())}/20, median sim kept "
          f"{np.median(sim[keep]):.2f}, max sim of the other person {sim[16:].max():.2f}")
    ok = ok and ok_o and ok_i
    try:                                                              # face_crop needs cv2 (optional here)
        L = box(100)
        L[1, :2] = (100, 100)
        img = np.full((120, 150, 3), 90, np.uint8)
        L[:, :2] += 60                                                # box reaches (60..160, 60..160): runs off the edge
        ok = ok and face_crop(img, L).shape == (160, 160, 3)
    except ImportError:
        print("(cv2 missing: face_crop not tested)")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--frames")
    ap.add_argument("--out", default="faces")
    ap.add_argument("--model", default="face_landmarker.task")
    ap.add_argument("--preview")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--min-keep", type=int, default=3)
    ap.add_argument("--min-face-px", type=float, default=90)
    ap.add_argument("--min-sharp", type=float, default=30)
    ap.add_argument("--auto-sharp", action="store_true",
                    help="per-speaker sharpness threshold: percentile of the speaker's own frames, capped by --min-sharp")
    ap.add_argument("--sharp-pct", type=float, default=40, help="percentile used by --auto-sharp")
    ap.add_argument("--sharp-floor", type=float, default=8, help="lowest threshold --auto-sharp may use")
    ap.add_argument("--max-yaw", type=float, default=12)
    ap.add_argument("--max-roll", type=float, default=12)
    ap.add_argument("--max-pitch-dev", type=float, default=15)
    ap.add_argument("--min-eye", type=float, default=0.12)
    ap.add_argument("--max-mouth", type=float, default=0.15)
    ap.add_argument("--others-ratio", type=float, default=0.0,
                    help="0 = require exactly one face; R>0 = pick the largest face, reject only if another face is "
                         ">= R x its size")
    ap.add_argument("--id-filter", action="store_true",
                    help="keep only the speaker's dominant identity (facenet embeddings)")
    ap.add_argument("--id-min-sim", type=float, default=0.4,
                    help="cosine threshold for --id-filter (not validated: tune it on the printed statistics/preview)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if not a.frames:
        ap.error("need --frames (or --selftest)")
    run(a)


if __name__ == "__main__":
    main()