"""Separate one audio file, optionally with a trained drum dictionary.

    python examples/separate_file.py song.wav --drums drum_loop.wav

The semi-supervised path is the one that works: without a dictionary, SPNMF
finds a two-part decomposition but has no reason to put the drums in the NMF
part rather than the projective one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spnmf import learn_dictionary, separate
from spnmf.io import read_audio, write_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="audio file to separate")
    parser.add_argument(
        "--drums", type=Path, nargs="*", default=[],
        help="percussive training audio for the dictionary",
    )
    parser.add_argument("--outdir", type=Path, default=Path("separated"))
    parser.add_argument("--divergence", default="kl", choices=["euclidean", "kl", "is"])
    parser.add_argument("--n-iter", type=int, default=200)
    args = parser.parse_args()

    audio, sample_rate = read_audio(args.input)
    print(f"loaded {args.input} ({len(audio) / sample_rate:.1f} s at {sample_rate} Hz)")

    dictionary = None
    if args.drums:
        training = []
        for path in args.drums:
            data, rate = read_audio(path)
            assert rate == sample_rate, f"{path} is {rate} Hz, expected {sample_rate}"
            training.append(data if data.ndim == 1 else data.mean(axis=1))
        dictionary = learn_dictionary(training, n_components=20, random_state=0)
        print(f"trained a {dictionary.shape[1]}-component drum dictionary")

    result = separate(
        audio,
        sample_rate,
        divergence=args.divergence,
        W_p=dictionary,
        n_harmonic=20,
        n_percussive=20,
        n_iter=args.n_iter,
        random_state=0,
    )
    print(f"cost {result.cost[0]:.4g} -> {result.cost[-1]:.4g}")
    if result.swapped:
        print("the NMF part was the more tonal one, so the outputs were swapped")

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, signal in [("harmonic", result.harmonic), ("percussive", result.percussive)]:
        path = args.outdir / f"{args.input.stem}_{name}.wav"
        write_audio(path, signal, sample_rate)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
