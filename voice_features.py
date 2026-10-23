#!/usr/bin/env python3
"""
Stage 1: audio preprocessing and acoustic features for the voice -> face pipeline.

  audio -> energy VAD -> fixed-length segments -> log-mel + (F0, formants F1-F4, jitter/shimmer/HNR)

Design notes (for the report):
  * No aggressive denoising and no VTLN / pitch normalisation: they would remove the
    speaker-specific vocal-tract information we want to keep. Only pause/click removal (VAD).
  * Formants come from LPC: the roots of the all-pole filter give the resonances of the
    vocal tract (order = 2 + sr/1000), F0 comes from pYIN.
  * Jitter, shimmer and HNR come from Praat (optional: pip install praat-parselmouth).

Install:  pip install librosa soundfile scipy numpy tqdm   (+ praat-parselmouth, optional)

Usage:
  python voice_features.py --selftest
  python voice_features.py --in wavs/ --out feats/ [--seg-sec 6 --min-sec 3]
"""
import argparse
from pathlib import Path

import numpy as np
import librosa

SR = 16000
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
ACOUSTIC_NAMES = ["f0_mean", "f0_median", "f0_std", "voiced_ratio",
                  "F1", "F2", "F3", "F4", "jitter", "shimmer", "hnr"]


# ------------------------------------------------------------------ VAD / segments
def load_audio(path, sr=SR):
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def vad(y, top_db=30, min_gap=0.3, min_len=0.2, sr=SR):
    """Energy VAD: keep speech intervals, bridge pauses shorter than min_gap, drop clicks."""
    iv = librosa.effects.split(y, top_db=top_db, frame_length=800, hop_length=200)
    merged = []
    for s, e in iv:
        if merged and s - merged[-1][1] < min_gap * sr:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    parts = [y[s:e] for s, e in merged if (e - s) >= min_len * sr]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=y.dtype)


def segment(speech, seg_sec=6.0, min_sec=3.0, sr=SR):
    """Non-overlapping segments of exactly seg_sec. A shorter tail (>= min_sec) is tiled to length."""
    n = int(seg_sec * sr)
    segs = [speech[i:i + n] for i in range(0, len(speech) - n + 1, n)]
    tail = speech[len(segs) * n:]
    if len(tail) >= min_sec * sr:
        segs.append(np.resize(tail, n))
    elif not segs and len(speech) >= min_sec * sr:
        segs.append(np.resize(speech, n))
    return segs


# ------------------------------------------------------------------ features
def log_mel(seg, n_mels=80):
    m = librosa.feature.melspectrogram(y=seg, sr=SR, n_fft=512, win_length=400, hop_length=160,
                                       n_mels=n_mels, fmin=20, fmax=7600)
    return np.log(m + 1e-6).astype(np.float32)


def lpc_formants(seg, n_frames=150, fmin=90.0, max_bw=400.0):
    """Median F1..F4 (Hz) over the louder 30 ms frames, from the roots of the LPC polynomial."""
    order = 2 + SR // 1000
    x = librosa.effects.preemphasis(seg, coef=0.97)
    flen = int(0.03 * SR)
    frames = librosa.util.frame(x, frame_length=flen, hop_length=flen // 2).T
    energy = frames.var(1)
    idx = np.where(energy > np.percentile(energy, 50))[0]
    if len(idx) > n_frames:
        idx = idx[np.linspace(0, len(idx) - 1, n_frames).astype(int)]
    win = np.hamming(flen)
    found = []
    for f in frames[idx]:
        try:
            a = librosa.lpc(f * win, order=order)
        except Exception:
            continue
        if not np.all(np.isfinite(a)):
            continue
        r = np.roots(a)
        r = r[np.imag(r) > 0.01]
        freq = np.arctan2(np.imag(r), np.real(r)) * SR / (2 * np.pi)
        bw = -(SR / np.pi) * np.log(np.abs(r))
        freq = np.sort(freq[(freq > fmin) & (bw < max_bw)])
        if len(freq) >= 4:
            found.append(freq[:4])
    if not found:
        return [np.nan] * 4
    return list(np.median(np.array(found), axis=0))


def f0_stats(seg):
    f0, voiced, _ = librosa.pyin(seg, fmin=60, fmax=500, sr=SR, frame_length=1024, hop_length=256)
    v = f0[voiced & np.isfinite(f0)]
    if len(v) < 5:
        return [np.nan, np.nan, np.nan, 0.0]
    return [float(v.mean()), float(np.median(v)), float(v.std()), float(voiced.mean())]


def praat_quality(seg):
    """Jitter (local), shimmer (local), HNR in dB. NaNs if parselmouth is missing or fails."""
    try:
        import parselmouth
        from parselmouth.praat import call
        snd = parselmouth.Sound(seg.astype(np.float64), sampling_frequency=SR)
        pp = call(snd, "To PointProcess (periodic, cc)", 60, 500)
        jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shi = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr = call(call(snd, "To Harmonicity (cc)", 0.01, 60, 0.1, 1.0), "Get mean", 0, 0)
        return [jit, shi, hnr]
    except Exception:
        return [np.nan, np.nan, np.nan]


def acoustic_vector(seg):
    return np.array(f0_stats(seg) + lpc_formants(seg) + praat_quality(seg), dtype=np.float32)


# ------------------------------------------------------------------ pipeline
def process_file(path, seg_sec=6.0, min_sec=3.0):
    """Returns dict(logmel=(N,80,T), acoustic=(N,11)) or None if there is no usable speech."""
    speech = vad(load_audio(path))
    segs = segment(speech, seg_sec, min_sec)
    if not segs:
        return None
    return {"logmel": np.stack([log_mel(s) for s in segs]),
            "acoustic": np.stack([acoustic_vector(s) for s in segs])}


def run_dir(src, dst, seg_sec, min_sec):
    from tqdm import tqdm
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(src).rglob("*") if p.suffix.lower() in AUDIO_EXT)
    ok = 0
    for p in tqdm(files, desc="audio"):
        out = dst / (p.relative_to(src).with_suffix("").as_posix().replace("/", "__") + ".npz")
        if out.exists():
            ok += 1
            continue
        res = process_file(p, seg_sec, min_sec)
        if res is None:
            print(f"skip (no speech >= {min_sec}s): {p}")
            continue
        np.savez_compressed(out, names=np.array(ACOUSTIC_NAMES), **res)
        ok += 1
    print(f"done: {ok}/{len(files)} files -> {dst}")


# ------------------------------------------------------------------ self-test
def synth_vowel(f0=120.0, formants=(700, 1220, 2600, 3500), bws=(80, 90, 120, 150), sec=8.0):
    """Impulse train through a cascade of 2nd-order resonators, with silence around it."""
    from scipy.signal import lfilter
    n = int(sec * SR)
    x = np.zeros(n)
    x[::int(SR / f0)] = 1.0
    for f, b in zip(formants, bws):
        r = np.exp(-np.pi * b / SR)
        th = 2 * np.pi * f / SR
        x = lfilter([1.0], [1.0, -2 * r * np.cos(th), r * r], x)
    x = x / np.max(np.abs(x)) * 0.5
    x += np.random.default_rng(0).normal(0, 1e-4, n)
    sil = np.random.default_rng(1).normal(0, 1e-4, SR)
    return np.concatenate([sil, x, sil]).astype(np.float32)


def selftest():
    import soundfile as sf
    import tempfile
    truth = [700, 1220, 2600, 3500]
    y = synth_vowel()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "vowel.wav"
        sf.write(p, y, SR)
        res = process_file(p, seg_sec=6.0, min_sec=3.0)
    assert res is not None, "VAD removed everything"
    sp = librosa.effects.split(y, top_db=30)
    print(f"audio {len(y) / SR:.1f}s, speech after VAD {len(vad(y)) / SR:.1f}s (expected ~8s)")
    print("logmel", res["logmel"].shape, "acoustic", res["acoustic"].shape)
    a = dict(zip(ACOUSTIC_NAMES, res["acoustic"][0]))
    for k, v in a.items():
        print(f"  {k:12s} {v:10.3f}")
    ok = abs(a["f0_median"] - 120) < 6
    for i, t in enumerate(truth):
        ok &= abs(a[f"F{i + 1}"] - t) / t < 0.15
    print("SELFTEST", "PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--in", dest="src")
    ap.add_argument("--out", dest="dst")
    ap.add_argument("--seg-sec", type=float, default=6.0)
    ap.add_argument("--min-sec", type=float, default=3.0)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if not (a.src and a.dst):
        ap.error("need --in and --out (or --selftest)")
    run_dir(a.src, a.dst, a.seg_sec, a.min_sec)


if __name__ == "__main__":
    main()
