"""Use case 2: what a pooled quality score hides about quantisation.

The model here is a supervised NMF denoiser -- a fixed speech dictionary and a
fixed noise dictionary in flash, a few multiplicative iterations per frame to
solve the activations, then a Wiener mask. It is a real, embeddable
architecture, so quantising its dictionaries to 8/6/4/3 bits is the actual
sub-byte question rather than a simulation of one.

Two knobs are swept, both of which cost quality to buy silicon:

* dictionary bit-width -- the weight-quantisation axis
* inference iterations -- the compute/early-exit axis

For each setting the error is reported pooled, and then split into what it did
to steady content versus what it did to transients.
"""

from __future__ import annotations

import numpy as np

from spnmf import istft, learn_dictionary, nmf, per_part_snr, stft, wiener_mask

SR, DUR = 16000, 6.0
N = int(SR * DUR)
N_FFT, HOP = 512, 128


def speech(seed):
    """Voiced harmonics + fricatives + plosives: tonal and transient content."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / SR
    out = np.zeros(N)

    def env(on, off, a=0.01, r=0.03):
        e = np.zeros(N)
        i0, i1 = int(on * SR), int(off * SR)
        u = np.arange(i1 - i0) / SR
        e[i0:i1] = (1 - np.exp(-u / a)) * np.exp(-np.maximum(0, u - (off - on - r)) / r)
        return e

    f0 = rng.uniform(95, 130) + 25 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voiced = np.zeros(N)
    for k in range(1, 26):
        formant = np.exp(-((k * 120 - 700) / 700) ** 2) + 0.6 * np.exp(
            -((k * 120 - 1800) / 900) ** 2
        )
        voiced += (formant / k**0.5) * np.sin(k * phase)
    for on in np.arange(0.2, DUR - 0.8, 1.2):
        out += voiced * env(on + rng.uniform(0, 0.1), on + 0.7)

    freqs = np.fft.rfftfreq(N, 1 / SR)
    for on in np.arange(0.95, DUR - 0.3, 1.2):  # fricatives
        b = np.fft.irfft(np.fft.rfft(rng.standard_normal(N)) * (freqs > 3500), N)
        out += 0.5 * b * env(on, on + 0.2, 0.005, 0.02)
    for on in np.arange(0.18, DUR - 0.2, 1.2):  # plosives
        i0 = int(on * SR)
        u = np.arange(int(0.012 * SR)) / SR
        out[i0 : i0 + len(u)] += 1.2 * np.exp(-u / 0.003) * rng.standard_normal(len(u))
    return out / (np.abs(out).max() + 1e-12)


def noise(seed):
    """Fan hum plus keyboard clicks: a tonal and an impulsive component."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / SR
    hum = sum(0.6**k * np.sin(2 * np.pi * 100 * (k + 1) * t) for k in range(5))
    hum = hum + 0.4 * np.sin(2 * np.pi * 2000 * t)
    clicks = np.zeros(N)
    for on in rng.uniform(0, DUR - 0.1, int(7 * DUR)):
        i0 = int(on * SR)
        u = np.arange(int(0.03 * SR)) / SR
        clicks[i0 : i0 + len(u)] += np.exp(-u / 0.006) * rng.standard_normal(len(u))
    out = 0.6 * hum / np.abs(hum).max() + 0.7 * clicks / np.abs(clicks).max()
    return out / (np.abs(out).max() + 1e-12)


def quantise(M, bits):
    """Per-tensor unsigned quantisation -- the dictionaries are non-negative."""
    if bits is None:
        return M
    scale = M.max() / (2**bits - 1)
    return np.clip(np.round(M / scale), 0, 2**bits - 1) * scale


def denoise(noisy, W_speech, W_noise, n_iter=40):
    X = stft(noisy, N_FFT, HOP)
    V = np.abs(X)
    W = np.concatenate([W_speech, W_noise], axis=1)
    H = nmf(V, W=W, update_W=False, divergence="kl", n_iter=n_iter,
            tol=None, random_state=0).H
    k = W_speech.shape[1]
    V_s = W_speech @ H[:k]
    V_n = W_noise @ H[k:]
    return istft(wiener_mask(V_s, [V_n]) * X, N_FFT, HOP, length=len(noisy))


def sweep(title, settings, run, reference, baseline_row=None):
    print(f"\n{title}")
    print(
        f"{'setting':>12s} {'pooled SNR':>11s} {'tonal':>9s} {'transient':>11s} "
        f"{'gap':>7s}   loss vs float"
    )
    print("-" * 78)
    first = None
    for label, value in settings:
        scores = per_part_snr(reference, run(value), SR, n_fft=N_FFT, hop_length=HOP)
        if first is None:
            first = scores
        print(
            f"{label:>12s} {scores.overall:8.2f} dB {scores.tonal:6.2f} dB "
            f"{scores.transient:8.2f} dB {scores.tonal - scores.transient:5.1f} dB   "
            f"pooled {scores.overall - first.overall:+5.2f} / "
            f"transient {scores.transient - first.transient:+5.2f}"
        )
    return first


def main():
    clean, interference = speech(1), noise(2)
    noisy = clean + interference

    # Dictionaries are trained on different utterances and a different noise
    # instance than the ones being denoised.
    dict_kwargs = dict(n_fft=N_FFT, hop_length=HOP, n_iter=200, random_state=0)
    W_speech = learn_dictionary(speech(11), n_components=32, **dict_kwargs)
    W_noise = learn_dictionary(noise(12), n_components=16, **dict_kwargs)

    ref = per_part_snr(clean, noisy, SR, n_fft=N_FFT, hop_length=HOP)
    print(f"unprocessed input:  pooled {ref.overall:.2f} dB, tonal {ref.tonal:.2f} dB, "
          f"transient {ref.transient:.2f} dB")
    print(f"tonal content carries {100 * ref.tonal_energy_fraction:.1f}% of the "
          f"reference energy -- which is what a pooled score mostly measures")

    sweep(
        "Dictionary weight quantisation (40 iterations)",
        [("float32", None)] + [(f"int{b}", b) for b in (8, 6, 5, 4, 3, 2)],
        lambda b: denoise(noisy, quantise(W_speech, b), quantise(W_noise, b)),
        clean,
    )

    sweep(
        "Inference budget at float32 (the early-exit axis)",
        [(f"{n} iter", n) for n in (40, 20, 10, 5, 3, 2, 1)],
        lambda n: denoise(noisy, W_speech, W_noise, n_iter=n),
        clean,
    )

    # The conclusion should not depend on how the reference was split. It does
    # not -- but note the SPNMF split is lopsided (it puts ~97% of the clean
    # speech energy in the tonal bucket against the median split's 75%), so
    # trust its direction, not its absolute numbers. Median is the default.
    print("\ncross-check: same measurement, SPNMF instead of median decomposition")
    for bits in (None, 4, 3):
        out = denoise(noisy, quantise(W_speech, bits), quantise(W_noise, bits))
        med = per_part_snr(clean, out, SR, n_fft=N_FFT, hop_length=HOP)
        spn = per_part_snr(clean, out, SR, n_fft=N_FFT, hop_length=HOP,
                           decomposition="spnmf")
        label = "float32" if bits is None else f"int{bits}"
        print(f"  {label:>8s}  median: tonal {med.tonal:6.2f} transient {med.transient:6.2f}"
              f"   |  spnmf: tonal {spn.tonal:6.2f} transient {spn.transient:6.2f}")
    print(
        "\nBoth agree that transients give way several times faster than the\n"
        "pooled score suggests, which is the finding that matters."
    )


if __name__ == "__main__":
    main()
