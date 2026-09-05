"""Do compression artefacts have distinguishable perceptual *fingerprints*?

A single MOS says how bad a compressed model got. A dimensional rater can in
principle say WHAT it broke. This renders perturbations of known mechanism
through one enhancer -- changing only the gain field -- and asks a dimensional
rater whether their signatures differ.

Rater: NISQA v2 (Mittag et al., Interspeech 2021), which predicts MOS plus
noisiness / discontinuity / coloration / loudness, runs on CPU, and is the same
rater used to build the corpus targets in LarocheC/llm-sqa.

    git clone https://github.com/gabrielmittag/NISQA   # weights ship in-repo

See SQA_PROTOCOL.md for the audio-LLM version of the same experiment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from inversion import Enhancer, load_clips, smooth_gain  # noqa: E402

DIMS = [("noi_pred", "noisiness"), ("dis_pred", "discontinuity"),
        ("col_pred", "coloration"), ("loud_pred", "loudness")]


def build_conditions(gain, torch, rng, strong):
    """Perturbations of the gain field with known, different mechanisms."""
    if not strong:
        noise = torch.from_numpy(rng.normal(0, 0.02, tuple(gain.shape))).float()
        return {
            "ref": gain,
            "jitter": torch.clamp(gain + noise, min=0.0),
            "smooth": smooth_gain(gain, 3, 1, torch),
        }
    noise = torch.from_numpy(rng.normal(0, 0.12, tuple(gain.shape))).float()
    return {
        "ref": gain,
        "jitter_strong": torch.clamp(gain + noise, min=0.0),
        "smooth_strong": smooth_gain(gain, 11, 5, torch),
        # negative control: a uniform gain is perceptually null once the rater
        # normalises level, so this must produce exactly zero movement.
        "bias_strong": torch.clamp(gain * 0.75, min=0.0),
    }


def render_all(args, strong):
    enh = Enhancer(args.models, args.eco8, args.variant)
    torch = enh.torch
    rng = np.random.default_rng(0)
    out = Path(args.out) / ("strong" if strong else "mild")
    names = None
    with torch.no_grad():
        for cid, _clean, noisy, sr in load_clips(args.data, n=args.n, stride=args.stride):
            g, mag, pha, length = enh.analyse(noisy)
            conds = build_conditions(g["fp32"], torch, rng, strong)
            if not strong:
                conds["int8"] = g["int8"]
            names = list(conds)
            for name, gain in conds.items():
                (out / name).mkdir(parents=True, exist_ok=True)
                w = enh.render(gain, mag, pha, length)
                peak = np.abs(w).max()
                sf.write(out / name / f"{cid}.wav",
                         w / peak * 0.9 if peak > 0.9 else w, sr, subtype="PCM_16")
    return out, names


def rate(nisqa_dir, root, names):
    import pandas as pd

    frames = {}
    for name in names:
        outdir = root / f"_rated_{name}"
        outdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "run_predict.py", "--mode", "predict_dir",
             "--pretrained_model", "weights/nisqa.tar", "--data_dir", str(root / name),
             "--num_workers", "0", "--bs", "8", "--output_dir", str(outdir)],
            cwd=nisqa_dir, check=True, capture_output=True,
        )
        frames[name] = pd.read_csv(outdir / "NISQA_results.csv").set_index("deg").sort_index()
    common = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
    return {k: v.loc[common] for k, v in frames.items()}, common


def analyse(frames, common, label):
    print(f"\n=== {label} ===  {len(common)} clips, NISQA v2 (higher = better)")
    print(f"{'condition':16s} {'MOS':>7s} " + " ".join(f"{n:>14s}" for _, n in DIMS))
    print("-" * 80)
    for name, f in frames.items():
        print(f"{name:16s} {f['mos_pred'].mean():7.3f} "
              + " ".join(f"{f[k].mean():14.3f}" for k, _ in DIMS))

    print("\ndelta vs ref, and the shape of the damage:")
    vecs = {}
    for name, f in frames.items():
        if name == "ref":
            continue
        d = np.array([(f[k] - frames["ref"][k]).mean() for k, _ in DIMS])
        vecs[name] = d / (np.linalg.norm(d) + 1e-12)
        dmos = (f["mos_pred"] - frames["ref"]["mos_pred"]).mean()
        # discontinuity : noisiness -- the axis that separates the mechanisms.
        # Undefined when nothing moved at all (the null control).
        ratio = abs(d[1]) / abs(d[0]) if abs(d[0]) > 1e-4 else float("nan")
        print(f"  {name:16s} dMOS {dmos:+6.3f} | "
              + " ".join(f"{n} {v:+.3f}" for (_, n), v in zip(DIMS, d, strict=True))
              + (" | disc:noise    n/a" if ratio != ratio else f" | disc:noise {ratio:6.2f}"))

    keys = list(vecs)
    if len(keys) > 1:
        print("\ncosine similarity between damage shapes:")
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                print(f"  {keys[i]:15s} vs {keys[j]:15s}: "
                      f"{float(vecs[keys[i]] @ vecs[keys[j]]):+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--eco8", required=True)
    ap.add_argument("--nisqa", required=True, help="gabrielmittag/NISQA clone")
    ap.add_argument("--out", default="/tmp/fingerprint_stimuli")
    ap.add_argument("--variant", default="conv-hardened")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--stride", type=int, default=7)
    args = ap.parse_args()

    for strong, label in ((False, "int8-scale perturbations"), (True, "strong perturbations")):
        root, names = render_all(args, strong)
        frames, common = rate(args.nisqa, root, names)
        analyse(frames, common, label)


if __name__ == "__main__":
    main()
