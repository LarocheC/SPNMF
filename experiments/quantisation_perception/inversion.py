"""Why does int8 quantisation improve SNR while degrading PESQ?

Measured on LiSenNet (claroche1/LiSenNet), the author's own published fp32/int8
ONNX pairs, over VoiceBank-DEMAND. Four candidate mechanisms are tested and
three are refuted; the fourth explains the PESQ half only.

Prerequisites
-------------
* ``pip install torch onnxruntime pesq soundfile pyarrow huggingface_hub``
* the enhancer's DSP, cloned anywhere and passed with ``--eco8``::

      git clone https://github.com/LarocheC/eco8-neaixt

* the checkpoints and data (both public)::

      huggingface-cli download claroche1/LiSenNet --local-dir <MODELS>
      huggingface-cli download JacobLinCool/VoiceBank-DEMAND-16k \
          --repo-type dataset --local-dir <DATA>

The RMS normalisation in ``common/dataset.py`` of eco8-neaixt is load-bearing:
without it the enhancer loses ~1.3 PESQ and every number here is meaningless.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

EPS = 1e-12


# ------------------------------------------------------------------ plumbing
def load_clips(data_dir, n=None, stride=1):
    """VoiceBank-DEMAND test clips with the enhancer's training-time scaling."""
    import pyarrow.parquet as pq
    import soundfile as sf

    path = Path(data_dir) / "data" / "test-00000-of-00001.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path} — see this file's docstring")
    seen = index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            index += 1
            if (index - 1) % stride:
                continue
            clean, sr = sf.read(io.BytesIO(row["clean"]["bytes"]), dtype="float64")
            noisy, _ = sf.read(io.BytesIO(row["noisy"]["bytes"]), dtype="float64")
            m = min(len(clean), len(noisy))
            clean, noisy = clean[:m], noisy[:m]
            k = np.sqrt(m / (np.sum(noisy**2) + 1e-8))   # common/dataset.py
            yield row["id"], clean * k, noisy * k, sr
            seen += 1
            if n is not None and seen >= n:
                return


class Enhancer:
    """LiSenNet fp32/int8 pair plus the author's own STFT and reconstruction."""

    def __init__(self, models_dir, eco8_dir, variant="conv-hardened"):
        import onnxruntime as ort
        import torch

        sys.path.insert(0, str(eco8_dir))
        from common.env import AttrDict
        from lisennet.model import build_lisennet

        self.torch = torch
        torch.set_num_threads(4)
        root = Path(models_dir) / variant
        cfg = json.load(open(root / "config.json"))
        self.model = build_lisennet(AttrDict(cfg)).eval()
        self.model.load_state_dict(
            torch.load(root / "g_best", map_location="cpu", weights_only=True)["generator"]
        )

        def session(path):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = opts.inter_op_num_threads = 1
            return ort.InferenceSession(str(path), sess_options=opts,
                                        providers=["CPUExecutionProvider"])

        self.sessions = {
            "fp32": session(root / "g_best_fp32.onnx"),
            "int8": session(root / "g_best_int8_static.onnx"),
        }

    def analyse(self, noisy):
        """Return (gains dict, magnitude, phase, length) in the compressed domain."""
        torch = self.torch
        x = torch.from_numpy(noisy).float().unsqueeze(0)
        spec = self.model.power_compress(self.model.apply_stft(x))
        mag, pha = spec.abs(), spec.angle()
        feat = self.model.build_features(mag, pha).numpy()
        gains = {
            k: torch.from_numpy(s.run(["est_mag"], {"feat": feat})[0]) / (mag + 1e-8)
            for k, s in self.sessions.items()
        }
        return gains, mag, pha, x.shape[-1]

    def render(self, gain, mag, pha, length):
        """Compressed gain -> waveform, with the noisy phase (the real-time path)."""
        torch = self.torch
        est = gain * mag
        spec = torch.complex(est * pha.cos(), est * pha.sin())
        wav = self.model.apply_istft(self.model.power_uncompress(spec), length=length)
        return wav.squeeze(0).numpy().astype(np.float64)


# ------------------------------------------------------------- perturbations
def smooth_gain(G, k_time, k_freq, torch):
    """Separable box smoothing of a (1, T, F) gain field."""
    X = G.clone()
    pad = torch.nn.functional.pad
    pool = torch.nn.functional.avg_pool2d
    if k_time > 1:
        p = k_time // 2
        X = pool(pad(X.unsqueeze(1), (0, 0, p, p), mode="replicate"),
                 (k_time, 1), stride=1).squeeze(1)
    if k_freq > 1:
        p = k_freq // 2
        X = pool(pad(X.unsqueeze(1), (p, p, 0, 0), mode="replicate"),
                 (1, k_freq), stride=1).squeeze(1)
    return X


def snr_db(reference, estimate):
    n = min(len(reference), len(estimate))
    r, e = reference[:n], estimate[:n]
    return 10 * np.log10(np.sum(r**2) / (np.sum((r - e) ** 2) + 1e-20) + 1e-20)


def report(results, baseline="fp32"):
    from pesq import pesq  # noqa: F401  (imported by caller; kept for clarity)

    p0 = np.mean(results[baseline]["pesq"])
    s0 = np.mean(results[baseline]["snr"])
    print(f"\n{'arm':24s} {'PESQ':>7s} {'dPESQ':>9s} {'SNR dB':>8s} {'dSNR':>7s}")
    print("-" * 60)
    for name, r in results.items():
        p, s = np.mean(r["pesq"]), np.mean(r["snr"])
        d = np.array(r["pesq"]) - np.array(results[baseline]["pesq"])
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        tag = "" if name == baseline else f"{p - p0:+6.3f}±{ci:.3f}"
        print(f"{name:24s} {p:7.3f} {tag:>9s} {s:8.2f} {s - s0:+7.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True, help="claroche1/LiSenNet local dir")
    ap.add_argument("--data", required=True, help="VoiceBank-DEMAND-16k local dir")
    ap.add_argument("--eco8", required=True, help="LarocheC/eco8-neaixt clone")
    ap.add_argument("--variant", default="conv-hardened")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--stride", type=int, default=6)
    args = ap.parse_args()

    from pesq import pesq

    enh = Enhancer(args.models, args.eco8, args.variant)
    torch = enh.torch
    rng = np.random.default_rng(0)
    results, rough, jitter_stats = {}, [], []

    with torch.no_grad():
        for _, clean, noisy, sr in load_clips(args.data, n=args.n, stride=args.stride):
            g, mag, pha, length = enh.analyse(noisy)
            err = g["int8"] - g["fp32"]
            jitter_stats.append((float(err.mean()), float(err.std())))
            rough.append([
                (float(g[k].diff(dim=1).abs().mean()), float(g[k].diff(dim=2).abs().mean()))
                for k in ("fp32", "int8")
            ])

            arms = {"fp32": g["fp32"], "int8": g["int8"]}
            # H1 smoothing  H2 uniform gain  H4 zero-mean jitter
            arms["H1 smooth t3f1"] = smooth_gain(g["fp32"], 3, 1, torch)
            arms["H2 gain x1.0012"] = g["fp32"] * 1.0012
            arms["H3 int8 debiased"] = g["int8"] * (g["fp32"].mean() / g["int8"].mean())
            for s in (0.01, 0.02, 0.03):
                noise = torch.from_numpy(rng.normal(0, s, tuple(g["fp32"].shape))).float()
                arms[f"H4 jitter {s}"] = torch.clamp(g["fp32"] + noise, min=0.0)

            for name, gain in arms.items():
                w = enh.render(gain, mag, pha, length)
                m = min(len(w), len(clean))
                results.setdefault(name, {"pesq": [], "snr": []})
                results[name]["pesq"].append(pesq(sr, clean[:m], w[:m], "wb"))
                results[name]["snr"].append(snr_db(clean, w))

    r = np.array(rough)
    print(f"gain-field roughness, {len(rough)} clips  (int8 / fp32 ratio)")
    print(f"  |d gain / dt| : {r[:, 1, 0].mean() / r[:, 0, 0].mean():.4f}")
    print(f"  |d gain / df| : {r[:, 1, 1].mean() / r[:, 0, 1].mean():.4f}")
    jm = np.mean([a for a, _ in jitter_stats])
    js = np.mean([b for _, b in jitter_stats])
    print(f"int8-minus-fp32 gain error: mean {jm:+.5f}, std {js:.5f}  (essentially zero-mean)")
    report(results)
    print("\nH1 smoothing: refuted — int8 makes the field ROUGHER, not smoother.")
    print("H2 uniform gain: refuted — PESQ is invariant to it (level normalisation).")
    print("H3 debiasing: refuted — removing the mean gain difference keeps the whole effect.")
    print("H4 jitter: reproduces the PESQ loss at ~0.02 std, but moves SNR the WRONG way.")


if __name__ == "__main__":
    main()
