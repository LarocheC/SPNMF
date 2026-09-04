"""Command-line interface: ``spnmf separate`` and ``spnmf demo``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__


def _add_stft_options(parser):
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT size (default: 2048)")
    parser.add_argument(
        "--hop-length", type=int, default=512, help="hop size in samples (default: 512)"
    )
    parser.add_argument(
        "--window", default="sqrt_hann", help="analysis window (default: sqrt_hann)"
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="spnmf",
        description="Harmonic/percussive source separation with Structured "
        "Projective NMF.",
    )
    parser.add_argument("--version", action="version", version=f"spnmf {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sep = sub.add_parser(
        "separate", help="split an audio file into harmonic and percussive parts"
    )
    sep.add_argument("input", type=Path, help="input audio file")
    sep.add_argument(
        "-o", "--outdir", type=Path, default=Path("."), help="output directory (default: .)"
    )
    sep.add_argument(
        "--divergence", default="kl", choices=["euclidean", "kl", "is"],
        help="cost function (default: kl)",
    )
    sep.add_argument("--n-harmonic", type=int, default=20, help="rank of the projective part")
    sep.add_argument(
        "--n-percussive", type=int, default=20,
        help="number of NMF components (ignored with --dictionary)",
    )
    sep.add_argument(
        "--dictionary", type=Path, nargs="+", metavar="AUDIO",
        help="percussive training audio; enables the semi-supervised algorithm",
    )
    sep.add_argument(
        "--dict-size", type=int, default=20, help="dictionary size (default: 20)"
    )
    sep.add_argument("--n-iter", type=int, default=200, help="iterations (default: 200)")
    sep.add_argument(
        "--power", type=float, default=1.0,
        help="spectrogram exponent: 1 magnitude, 2 power (default: 1)",
    )
    sep.add_argument(
        "--mask-power", type=float, default=2.0, help="Wiener mask exponent (default: 2)"
    )
    sep.add_argument(
        "--assign", default="auto", choices=["auto", "projective"],
        help="how to map model parts onto outputs (default: auto)",
    )
    sep.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    sep.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    _add_stft_options(sep)

    demo = sub.add_parser("demo", help="run the built-in synthetic benchmark")
    demo.add_argument(
        "-o", "--outdir", type=Path, default=None, help="write WAV files here"
    )
    demo.add_argument("--duration", type=float, default=4.0, help="seconds (default: 4)")
    demo.add_argument("--n-iter", type=int, default=150, help="iterations (default: 150)")

    return parser


def _run_separate(args):
    from .dictionary import learn_dictionary
    from .io import read_audio, write_audio
    from .separation import separate

    audio, sample_rate = read_audio(args.input)
    if not args.quiet:
        channels = 1 if audio.ndim == 1 else audio.shape[1]
        print(
            f"{args.input}: {len(audio) / sample_rate:.1f} s, "
            f"{sample_rate} Hz, {channels} channel(s)"
        )

    dictionary = None
    if args.dictionary:
        training = []
        for path in args.dictionary:
            data, rate = read_audio(path)
            if rate != sample_rate:
                raise SystemExit(
                    f"error: {path} is {rate} Hz but {args.input} is {sample_rate} Hz; "
                    "resample the training audio first"
                )
            training.append(data if data.ndim == 1 else data.mean(axis=1))
        if not args.quiet:
            print(
                f"training a {args.dict_size}-component dictionary "
                f"on {len(training)} file(s)"
            )
        dictionary = learn_dictionary(
            training,
            n_components=args.dict_size,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            window=args.window,
            power=args.power,
            divergence=args.divergence,
            random_state=args.seed,
        )

    result = separate(
        audio,
        sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        window=args.window,
        power=args.power,
        n_harmonic=args.n_harmonic,
        n_percussive=args.n_percussive,
        divergence=args.divergence,
        W_p=dictionary,
        n_iter=args.n_iter,
        mask_power=args.mask_power,
        assign=args.assign,
        random_state=args.seed,
        verbose=not args.quiet,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    harmonic_path = args.outdir / f"{stem}_harmonic.wav"
    percussive_path = args.outdir / f"{stem}_percussive.wav"
    write_audio(harmonic_path, result.harmonic, sample_rate)
    write_audio(percussive_path, result.percussive, sample_rate)

    if not args.quiet:
        factorisation = result.factorisation
        print(
            f"converged after {factorisation.n_iter} iterations "
            f"(cost {factorisation.cost[0]:.4g} -> {factorisation.cost[-1]:.4g})"
        )
        if result.swapped:
            print(
                "note: the NMF part was the more tonal of the two, so the parts "
                "were swapped (--assign projective disables this)"
            )
        print(f"wrote {harmonic_path}")
        print(f"wrote {percussive_path}")
    return 0


def _run_demo(args):
    from .demo import run_demo

    run_demo(outdir=args.outdir, duration=args.duration, n_iter=args.n_iter)
    return 0


def main(argv=None):
    args = _build_parser().parse_args(argv)
    np.seterr(over="ignore", invalid="ignore")
    if args.command == "separate":
        return _run_separate(args)
    if args.command == "demo":
        return _run_demo(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
