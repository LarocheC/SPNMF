"""State coherence, not mask accuracy, is what frame-skipping costs.

Dynamic-compute work for audio usually asks which frames can be skipped without
hurting accuracy. On a *stateful* streaming graph that is the wrong question:
most of the damage is not the held mask, it is the recurrence losing its place.

Three arms at matched skip rate, on the int8 streaming LiSenNet graph
(17 FIFO state tensors):

  periodic     skip every k-th frame
  adaptive     skip the lowest-spectral-flux frames (a statistic the front end
               already computes -- no learned gate)
  fresh-state  hold exactly the same masks on exactly the same frames, but keep
               running the network so the FIFO state stays current. Isolates the
               cost of the held mask from the cost of stale state.

Prerequisites and paths: see ../quantisation_perception/inversion.py, whose
clip loader and enhancer wrapper this reuses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "quantisation_perception"))
from inversion import load_clips, snr_db  # noqa: E402,F401


def build_stream(models_dir, variant="conv-hardened"):
    """Frame-by-frame runner over the streaming ONNX graph."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    path = Path(models_dir) / variant / "g_best_streaming_int8_static.onnx"
    sess = ort.InferenceSession(str(path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    state_inputs = [i for i in sess.get_inputs() if i.name != "feat"]
    out_names = [o.name for o in sess.get_outputs()]

    def zeros(shape):
        return np.zeros([d if isinstance(d, int) else 1 for d in shape], np.float32)

    def run(feat, skip, freeze_state=True):
        """feat (1,3,T,257) -> est_mag (1,T,257). skip[t] -> hold last output."""
        states = {i.name: zeros(i.shape) for i in state_inputs}
        T = feat.shape[2]
        out = np.zeros((1, T, 257), np.float32)
        last = None
        for t in range(T):
            frame = feat[:, :, t : t + 1, :]
            held = skip[t] and last is not None
            if not held or not freeze_state:
                res = sess.run(out_names, {"feat": frame, **states})
                for i, v in zip(state_inputs, res[1:], strict=True):
                    states[i.name] = v
                if not held:
                    last = res[0][:, 0, :]
            out[:, t, :] = last
        return out

    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True, help="claroche1/LiSenNet local dir")
    ap.add_argument("--data", required=True, help="VoiceBank-DEMAND-16k local dir")
    ap.add_argument("--eco8", required=True, help="LarocheC/eco8-neaixt clone")
    ap.add_argument("--variant", default="conv-hardened")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.20, 0.30])
    args = ap.parse_args()

    import torch
    from inversion import Enhancer
    from pesq import pesq

    enh = Enhancer(args.models, args.eco8, args.variant)
    stream = build_stream(args.models, args.variant)
    results = {}

    with torch.no_grad():
        for _cid, clean, noisy, sr in load_clips(args.data, n=args.n, stride=args.stride):
            _g, mag, pha, length = enh.analyse(noisy)
            feat = enh.model.build_features(mag, pha).numpy()
            T = feat.shape[2]
            # routing statistic: spectral flux of the noisy magnitude
            m = mag[0].numpy()
            flux = np.concatenate([[np.inf], np.abs(np.diff(m, axis=0)).sum(1)])

            def score(name, est, mag=mag, pha=pha, length=length, clean=clean, sr=sr):
                w = enh.render(torch.from_numpy(est), mag, pha, length)
                n = min(len(w), len(clean))
                results.setdefault(name, []).append(pesq(sr, clean[:n], w[:n], "wb"))

            score("no skip", stream(feat, np.zeros(T, bool)))
            for rate in args.rates:
                k = max(2, int(round(1 / rate)))
                periodic = np.zeros(T, bool)
                periodic[::k] = True
                periodic[0] = False
                adaptive = np.zeros(T, bool)
                adaptive[np.argsort(flux)[: periodic.sum()]] = True
                adaptive[0] = False
                score(f"periodic {rate:.0%}", stream(feat, periodic))
                score(f"adaptive {rate:.0%}", stream(feat, adaptive))
                score(f"adaptive {rate:.0%} fresh-state",
                      stream(feat, adaptive, freeze_state=False))

    base = np.array(results["no skip"])
    print(f"\n{len(base)} clips, streaming int8 graph, {args.variant}\n")
    print(f"{'arm':32s} {'PESQ':>7s} {'dPESQ':>10s}")
    print("-" * 52)
    for name, vals in results.items():
        a = np.array(vals)
        d = a - base
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        tag = "" if name == "no skip" else f"{d.mean():+6.3f}±{ci:.3f}"
        print(f"{name:32s} {a.mean():7.3f} {tag:>10s}")

    for rate in args.rates:
        total = np.mean(results[f"adaptive {rate:.0%}"]) - base.mean()
        fresh = np.mean(results[f"adaptive {rate:.0%} fresh-state"]) - base.mean()
        if abs(total) > 1e-9:
            share = 100 * abs(total - fresh) / abs(total)
            print(f"\nadaptive {rate:.0%}: total {total:+.3f}, held mask alone {fresh:+.3f}, "
                  f"state staleness {total - fresh:+.3f} ({share:.0f}% of the damage)")


if __name__ == "__main__":
    main()
