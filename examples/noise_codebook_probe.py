"""Does a noise codebook beat classical noise tracking? Measured answer: no.

This is the follow-up experiment to the SPNMF diagnosis. If a structured
low-rank non-negative model is not useful for splitting a mixture (see
``ablate_projective.py``), the remaining hope is to put it on the *nuisance*:
a codebook of noise spectra, fitted per frame, used to estimate the noise floor
where speech hides it. That is the classical codebook-based enhancement idea
(Srinivasan, Samuelsson & Kleijn, TASLP 2007) in modern clothing.

The mechanism it needs is real: during speech only some bands show the noise, so
a codebook could read the visible bands and *extrapolate across frequency* to
the hidden ones. A per-band recursive average cannot do that — it can only
extrapolate across *time*.

Measured on VoiceBank-DEMAND (train and test use disjoint noise types), the
cross-frequency extrapolation loses to plain temporal persistence at every
codebook size and every iteration count, including a single atom.

Requires the VoiceBank-DEMAND-16k parquets:

    huggingface-cli download JacobLinCool/VoiceBank-DEMAND-16k \
        --repo-type dataset --local-dir <DATA>

then run with ``--data <DATA>``.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

from spnmf import nmf
from spnmf.divergence import EPS

SR, N_FFT, HOP, N_BANDS = 16000, 512, 256, 32
N_BINS = N_FFT // 2 + 1


# ----------------------------------------------------------------- front end
def band_matrix(n_bands=N_BANDS, n_bins=N_BINS, sr=SR):
    """(n_bands, n_bins) mel-spaced 0/1 partition."""
    def to_hz(m):
        return 700 * (10 ** (m / 2595.0) - 1)

    def to_mel(f):
        return 2595 * np.log10(1 + f / 700.0)

    edges = to_hz(np.linspace(to_mel(50), to_mel(sr / 2), n_bands + 1))
    bins = np.maximum.accumulate(
        np.clip(np.round(edges / (sr / 2) * (n_bins - 1)).astype(int), 0, n_bins - 1)
    )
    M = np.zeros((n_bands, n_bins))
    for b in range(n_bands):
        M[b, bins[b] : max(bins[b + 1], bins[b] + 1)] = 1.0
    return M


def power_spectrogram(x, n_fft=N_FFT, hop=HOP):
    window = np.hanning(n_fft + 1)[:-1]
    n = 1 + max(0, len(x) - n_fft) // hop
    if n < 1:
        x, n = np.pad(x, (0, n_fft - len(x))), 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(x[idx] * window, n=n_fft, axis=1).T) ** 2


def clips(data_dir, split, n=None, stride=1):
    """Yield (id, clean, noisy, noise). VoiceBank-DEMAND is additive, so the
    noise is recovered exactly as noisy - clean."""
    import pyarrow.parquet as pq
    import soundfile as sf

    name = ("test-00000-of-00001" if split == "test" else "train-00000-of-00005")
    path = Path(data_dir) / "data" / f"{name}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path} — see this file's docstring")
    seen = index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            index += 1
            if (index - 1) % stride:
                continue
            clean, _ = sf.read(io.BytesIO(row["clean"]["bytes"]), dtype="float64")
            noisy, _ = sf.read(io.BytesIO(row["noisy"]["bytes"]), dtype="float64")
            m = min(len(clean), len(noisy))
            yield row["id"], clean[:m], noisy[:m], noisy[:m] - clean[:m]
            seen += 1
            if n is not None and seen >= n:
                return


# -------------------------------------------------------------------- arms
def gerkmann_hendriks(Y2, alpha=0.8, xi_h1_db=15.0, init=6):
    """MMSE noise-PSD tracker with speech-presence uncertainty
    (Gerkmann & Hendriks, TASLP 2012). The classical bar."""
    xi = 10 ** (xi_h1_db / 10.0)
    noise = np.maximum(Y2[:, :init].mean(axis=1), EPS)
    out = np.empty_like(Y2)
    for t in range(Y2.shape[1]):
        y2 = Y2[:, t]
        gamma = y2 / (noise + EPS)
        p = 1.0 / (1.0 + (1.0 + xi) * np.exp(-np.clip(gamma * xi / (1.0 + xi), 0, 200)))
        noise = alpha * noise + (1.0 - alpha) * (p * noise + (1.0 - p) * y2)
        noise = np.minimum(noise, np.maximum(y2, EPS) * 8.0)
        out[:, t] = noise
    return np.maximum(out, EPS)


def gated_ema(bY, tau_s, gate, init=8):
    """Per-band recursive average, updated only where the noise is visible and
    held otherwise. Extrapolates across TIME only."""
    a = float(np.exp(-1.0 / (tau_s * SR / HOP)))
    state = bY[:, :init].mean(axis=1)
    out = np.empty_like(bY)
    for t in range(bY.shape[1]):
        state = np.where(gate[:, t], a * state + (1 - a) * bY[:, t], state)
        out[:, t] = state
    return np.maximum(out, EPS)


def codebook_fit(bY, D, gate, n_steps=2):
    """Missing-data IS fit of w per frame on the visible bands, warm-started.
    Extrapolates across FREQUENCY: read visible bands, predict hidden ones."""
    w = np.full(D.shape[1], bY.mean() / max(D.mean() * D.shape[1], EPS))
    out = np.empty_like(bY)
    for t in range(bY.shape[1]):
        m = gate[:, t]
        if m.any():
            Dm, vm = D[m], bY[m, t]
            for _ in range(n_steps):
                ap = np.maximum(Dm @ w, EPS)
                w = np.maximum(
                    w * ((Dm.T @ (vm / (ap * ap))) / ((Dm.T @ (1.0 / ap)) + EPS)), 0.0
                )
        out[:, t] = D @ w
    return np.maximum(out, EPS)


def smooth(X, half=2):
    k = 2 * half + 1
    P = np.pad(X, ((0, 0), (half, half)), mode="edge")
    return np.stack([P[:, i : i + X.shape[1]] for i in range(k)]).mean(0)


def lsd(estimate, target):
    return float(np.mean(np.abs(10 * np.log10(estimate + EPS) - 10 * np.log10(target + EPS))))


# --------------------------------------------------------------------- main
def train_codebooks(data_dir, M, sizes, n_clips=400):
    """IS-NMF over speech-free noise frames from the TRAIN split."""
    frames = []
    for _, clean, _, noise in clips(data_dir, "train", n=n_clips):
        bN, bC = M @ power_spectrogram(noise), M @ power_spectrogram(clean)
        energy = bC.sum(axis=0)
        quiet = energy < 1e-4 * max(energy.max(), EPS)
        if quiet.sum() > 5:
            frames.append(bN[:, quiet])
    V = np.maximum(np.concatenate(frames, axis=1), EPS)
    print(f"trained on {V.shape[1]} speech-free noise frames from the train split")
    books = {}
    for K in sizes:
        W = np.maximum(
            nmf(V, n_components=K, divergence="is", n_iter=300, tol=None,
                random_state=0).W, EPS
        )
        books[K] = W / (np.linalg.norm(W, axis=0, keepdims=True) + EPS)
    return books


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="VoiceBank-DEMAND-16k directory")
    ap.add_argument("--stride", type=int, default=8, help="test subsampling")
    args = ap.parse_args()

    M = band_matrix()
    sizes = (1, 4, 8, 16, 32)
    books = train_codebooks(args.data, M, sizes)

    acc, visible = {}, []
    for _, clean, noisy, noise in clips(args.data, "test", stride=args.stride):
        Y2, N2, C2 = (power_spectrogram(z) for z in (noisy, noise, clean))
        bY, bN, bC = M @ Y2, M @ N2, M @ C2
        target = smooth(bN, 2)          # a PSD, not one chi-squared draw
        energy = bC.sum(axis=0)
        active = energy > 1e-3 * energy.max()
        if active.sum() < 10:
            continue
        gate = bN > bC                  # ORACLE: where the noise is readable
        visible.append(gate[:, active].sum(axis=0).mean())

        def add(name, est, active=active, target=target):
            acc.setdefault(name, []).append(lsd(est[:, active], target[:, active]))

        add("floor (true, unsmoothed)", bN)
        add("Gerkmann-Hendriks (bar)", M @ gerkmann_hendriks(Y2))
        add("gated EMA 0.16 s", gated_ema(bY, 0.16, gate))
        for K in sizes:
            for steps in (2, 50):
                add(f"codebook K={K} ({steps} steps)", codebook_fit(bY, books[K], gate, steps))
        # expressiveness ceiling: fit the TRUE noise with every band visible
        add("codebook K=8 ceiling", codebook_fit(bN, books[8], np.ones_like(gate), 50))

    n = len(acc["floor (true, unsmoothed)"])
    bar = np.array(acc["Gerkmann-Hendriks (bar)"])
    print(f"\n{n} test clips, unseen noise types, speech-active frames only")
    print(f"oracle-visible bands during speech: {np.mean(visible):.1f} of {N_BANDS}\n")
    print(f"{'arm':30s} {'LSD dB':>8s} {'vs bar':>14s}")
    print("-" * 54)
    for name in acc:
        a = np.array(acc[name])
        d = bar - a
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        tag = "" if "bar" in name or "floor" in name else f"{d.mean():+6.2f} ±{ci:4.2f}"
        print(f"{name:30s} {a.mean():8.2f} {tag:>14s}")


if __name__ == "__main__":
    main()
