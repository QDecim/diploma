#!/usr/bin/env python3
"""
Stage 3a: speaker embeddings, one vector per person (to be regressed onto the AMs from flame_ams.py labels).

Layout expected:  dataset/audio/<id>/*.wav   (<id> = same speaker id as in dataset/frames/<id> and faces/<id>.npz)

Every wav is converted to 16 kHz mono, cut into fixed-length segments (silent ones dropped), and each segment is
embedded with a pretrained ECAPA-TDNN (SpeechBrain, 192-d). Output voices.npz holds
  ids, emb          one L2-normalised mean embedding per speaker  (the regression input)
  seg_emb, seg_id   all segment embeddings and their owner         (for segment-level training / augmentation)
  n_seg, seconds    how much audio each speaker contributed

Install:   pip install speechbrain soundfile      (torch, scipy, numpy are already in the env)
Usage:
  python voice_stage.py --audio dataset/audio --out voices.npz [--seg 4] [--max-seg 40] [--ids 1,2,3]
  python voice_stage.py --selftest                (segmentation + pooling only, no model needed)

Windows note: SpeechBrain downloads the model into --cache and by default tries symlinks, which fail without
admin rights / developer mode. This script asks for copies instead.
"""
import argparse
from math import gcd
from pathlib import Path

import numpy as np

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
SR = 16000


# ------------------------------------------------------------------ audio -> segments (numpy only)
def to_mono_16k(x, sr):
    """x (T,) or (T, C) float -> (T',) mono float32 at 16 kHz."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x


def segment(x, seg_s=4.0, min_s=1.5, rms_thr=0.005):
    """Cut a 16 kHz mono signal into non-overlapping seg_s windows. The tail is kept if it is >= min_s long
    (and zero-padded by repeating nothing: it is simply embedded at its own length by the caller, so here it is
    returned as-is). Segments with rms below rms_thr (silence / digital zero) are dropped."""
    n, m = int(seg_s * SR), int(min_s * SR)
    out = []
    for s in range(0, len(x), n):
        w = x[s:s + n]
        if len(w) < m:
            break
        if float(np.sqrt(np.mean(w ** 2))) >= rms_thr:
            out.append(w)
    return out


def pool(seg_emb):
    """(S, D) segment embeddings -> one L2-normalised speaker vector (mean of normalised segments)."""
    e = np.asarray(seg_emb, dtype=np.float32)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    m = e.mean(0)
    return m / (np.linalg.norm(m) + 1e-9)


# ------------------------------------------------------------------ model
def load_encoder(cache, device):
    try:
        from speechbrain.inference.speaker import EncoderClassifier      # speechbrain >= 1.0
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier             # older versions
    kw = dict(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(cache), run_opts={"device": device})
    try:
        from speechbrain.utils.fetching import LocalStrategy
        kw["local_strategy"] = LocalStrategy.COPY                        # avoid symlinks on Windows
    except Exception:
        pass
    return EncoderClassifier.from_hparams(**kw)


def embed_segments(enc, segs, device, batch_len=None):
    """List of 1-D float32 arrays (may differ in length for the tail) -> (S, 192). Equal-length segments are
    batched; the shorter tail is embedded on its own."""
    import torch
    out = [None] * len(segs)
    groups = {}
    for i, w in enumerate(segs):
        groups.setdefault(len(w), []).append(i)
    with torch.no_grad():
        for L, idx in groups.items():
            for b in range(0, len(idx), 16):
                ii = idx[b:b + 16]
                x = torch.from_numpy(np.stack([segs[i] for i in ii])).to(device)
                e = enc.encode_batch(x).squeeze(1).cpu().numpy()
                for i, v in zip(ii, e):
                    out[i] = v
    return np.stack(out)


def read_audio(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono_16k(x, sr)


# ------------------------------------------------------------------ run
def run(a):
    import torch
    device = "cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)
    enc = load_encoder(a.cache, device)
    root = Path(a.audio)
    want = set(a.ids.split(",")) if a.ids else None
    ids, emb, n_seg, secs = [], [], [], []
    seg_emb, seg_id = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if want and d.name not in want:
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in AUDIO_EXT)
        segs, total = [], 0.0
        for f in files:
            try:
                x = read_audio(f)
            except Exception as e:                                       # one bad file must not stop the run
                print(f"  skip {f.name}: {e}")
                continue
            total += len(x) / SR
            segs += segment(x, a.seg, a.min_seg, a.rms_thr)
        if len(segs) > a.max_seg:                                        # spread the cap evenly over the speaker
            keep = np.linspace(0, len(segs) - 1, a.max_seg).round().astype(int)
            segs = [segs[i] for i in keep]
        if len(segs) < a.min_segments:
            print(f"{d.name}: {len(files)} files, {total:.0f}s audio, {len(segs)} segments -> skipped "
                  f"(< {a.min_segments})")
            continue
        e = embed_segments(enc, segs, device)
        ids.append(d.name); emb.append(pool(e)); n_seg.append(len(segs)); secs.append(total)
        seg_emb.append(e); seg_id += [d.name] * len(e)
        print(f"{d.name}: {len(files)} files, {total:.0f}s audio, {len(segs)} segments")
    if not ids:
        raise SystemExit("no speaker produced any embedding")
    np.savez(a.out, ids=np.array(ids), emb=np.stack(emb), seg_emb=np.concatenate(seg_emb), seg_id=np.array(seg_id),
             n_seg=np.array(n_seg), seconds=np.array(secs))
    print(f"saved {a.out}: {len(ids)} speakers, {len(seg_id)} segments, embedding dim {emb[0].shape[0]}")


# ------------------------------------------------------------------ self-test (no model)
def selftest():
    t = np.arange(SR * 10) / SR
    x = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    x[SR * 4:SR * 8] = 0.0                                               # a silent 4 s stretch
    segs = segment(x, 4.0, 1.5, 0.005)
    # 10 s -> windows [0-4] voiced, [4-8] silent (dropped), [8-10] 2 s tail voiced and >= 1.5 s (kept)
    ok1 = len(segs) == 2 and len(segs[0]) == 4 * SR and len(segs[1]) == 2 * SR
    short = segment(x[:SR * 5 + SR // 2][-SR:], 4.0, 1.5, 0.005)         # 1 s clip is below min_s -> nothing
    ok2 = short == []
    st = np.stack([x[:SR * 2], x[:SR * 2]])
    ok3 = to_mono_16k(np.stack([x[:100], -x[:100]], 1), 16000).shape == (100,) and \
        np.allclose(to_mono_16k(np.stack([x[:100], -x[:100]], 1), 16000), 0)
    y = to_mono_16k(x[:44100], 44100)                                    # 1 s at 44.1 kHz -> 16000 samples
    ok4 = abs(len(y) - SR) <= 1
    rng = np.random.default_rng(0)
    v = rng.normal(size=192)
    e = np.stack([v + rng.normal(size=192) * 0.2 for _ in range(6)])
    p = pool(e)
    ok5 = p.shape == (192,) and abs(np.linalg.norm(p) - 1) < 1e-5 and float(p @ (v / np.linalg.norm(v))) > 0.95
    print(f"segments {len(segs)}, short clip {len(short)}, stereo cancel {ok3}, resample {len(y)}, pool cos "
          f"{float(p @ (v / np.linalg.norm(v))):.3f}")
    ok = ok1 and ok2 and ok3 and ok4 and ok5
    print("SELFTEST", "PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--audio", default="dataset/audio")
    ap.add_argument("--out", default="voices.npz")
    ap.add_argument("--cache", default="pretrained_ecapa")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--seg", type=float, default=4.0, help="segment length, seconds")
    ap.add_argument("--min-seg", type=float, default=1.5, help="shortest tail kept, seconds")
    ap.add_argument("--max-seg", type=int, default=40, help="cap on segments per speaker (keeps speakers balanced)")
    ap.add_argument("--min-segments", type=int, default=3, help="skip speakers with fewer segments")
    ap.add_argument("--rms-thr", type=float, default=0.005, help="drop segments quieter than this (silence)")
    ap.add_argument("--ids", help="comma-separated speaker ids to process (default: all)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    run(a)


if __name__ == "__main__":
    main()
