# SPNMF — Structured Projective Non-negative Matrix Factorization

Harmonic/percussive source separation by factorising a spectrogram into a
near-orthogonal *projective* part for the tonal source and an ordinary NMF for
the transients.

```
V  ≈  W_h W_hᵀ V  +  W_p H_p
      └────┬────┘   └───┬───┘
      harmonic part   percussive part
```

`W_h W_hᵀ V` is a projective NMF: a rank-`r` non-negative projection of the data
onto itself. Its basis comes out sparse in frequency and stable in time, which
is the shape of a sustained tonal source. `W_p H_p` is unconstrained, free to
model broadband transients. Both are fitted with multiplicative updates under
the Euclidean distance, the Kullback–Leibler divergence, or the Itakura–Saito
divergence.

---

## Where this came from

This repository was created in September 2015 and never received a commit. The
work it was meant to hold ended up scattered across two other repositories:

| | |
|---|---|
| **Algorithm (MATLAB, 2016)** | [`LarocheC/voice-extraction`](https://github.com/LarocheC/voice-extraction) — six `.m` files under `SPNMF/`, buried next to a 100 MB `hits.mat`, a vendored copy of LTFAT and three `sandbox*.m` driver scripts |
| **Journal manuscript** | [`LarocheC/Journal_SPNMF`](https://github.com/LarocheC/Journal_SPNMF) — *Structured Projective Non Negative Matrix Factorization with drum dictionaries for Harmonic/Percussive Source Separation*, Laroche, Kowalski, Papadopoulos & Richard |
| **Earlier conference paper** | *A structured nonnegative matrix factorization for source separation*, EUSIPCO 2015 |

This package is a Python port of that algorithm: no MATLAB, no LTFAT, no
100 MB binary blobs, and no scattering-transform detour. NumPy is the only
required dependency.

## What changed in the port

The 2016 MATLAB and the manuscript's appendix agree with each other, and both
depart from the gradients they claim to implement. The port derives every
update from the β-divergence and checks it against a numerical gradient in
`tests/test_core.py`.

- **Euclidean cross terms carried a spurious factor of 2.** `SPNMF.m` and the
  appendix put `2·V H_pᵀ W_pᵀ W_h` in the denominator of the `W_h` update (and
  `2·W_h W_hᵀ V H_pᵀ`, `2·W_pᵀ W_h W_hᵀ V` in the `W_p`/`H_p` updates). The
  gradient of `‖V − W_h W_hᵀ V − W_p H_p‖²` gives coefficient 1.
- **The KL solver used Euclidean updates for `W_p` and `H_p`.** `SPNMF_KL.m`
  updated them with `V H_pᵀ / V̂ H_pᵀ` — the Euclidean rule — while reporting the
  I-divergence as its cost; the appendix repeats the Euclidean gradients under
  the KL heading. The KL rule uses `Z = V/V̂` against a matrix of ones.
- **`W = W./norm(W)` broke the objective.** `SPNMF.m` rescaled `W_h` by its
  spectral norm every iteration. `W_h W_hᵀ V` is quadratic in `W_h`, so that
  changes the cost being minimised. The port normalises only `W_p`'s columns and
  compensates in `H_p`, which leaves `W_p H_p` — and the cost — untouched.
- **Two cost readouts were wrong.** The Euclidean `e(iter)` used MATLAB's
  matrix `norm`, i.e. the largest singular value rather than the Frobenius norm
  it was minimising; `SPNMF_IS_W2TRAIN.m` had a `sum` nested inside its outer
  `sum`. Diagnostics only, but they made convergence unreadable.
- **Three near-duplicate files per divergence became one code path.** The six
  `.m` files differed mainly in their β exponent, so `spnmf.core.spnmf` takes
  `divergence=` and the fixed-dictionary variants are `W_p=`.

Every divergence now decreases its cost monotonically over the whole run —
asserted in the test suite, not just observed.

## Install

```bash
pip install -e ".[audio]"     # soundfile, for formats beyond PCM WAV
pip install -e ".[audio,dev]" # plus pytest and ruff
```

NumPy is the only hard dependency. Without `soundfile` the CLI still reads and
writes plain PCM WAV through the standard library.

## Use it

```python
from spnmf import separate, learn_dictionary
from spnmf.io import read_audio, write_audio

mix, sr = read_audio("song.wav")

# Semi-supervised: train the percussive dictionary on any drum audio you have.
drums, _ = read_audio("drum_loop.wav")
W_p = learn_dictionary(drums, n_components=20)

out = separate(mix, sr, W_p=W_p, n_harmonic=20, divergence="kl")
write_audio("harmonic.wav", out.harmonic, sr)
write_audio("percussive.wav", out.percussive, sr)
```

`out.harmonic + out.percussive` reconstructs the input sample for sample: the
two Wiener masks partition the mixture.

From the shell:

```bash
spnmf separate song.wav -o stems/ --dictionary drum_loop.wav --divergence kl
spnmf demo -o demo_output/      # synthetic benchmark, writes WAVs
```

## How well does it work

`spnmf demo` builds a mixture whose harmonic and percussive parts are known
exactly, separates it, and scores the result with BSS_EVAL. The drum dictionary
is trained on *different* drums (another tempo, another seed) than the mixture
contains. Numbers are dB; reproduce them with `spnmf demo`.

| method | SDR h | SDR p | SIR h | SIR p |
|---|---|---|---|---|
| median HPSS (baseline) | 20.1 | 6.4 | 22.2 | 23.3 |
| SPNMF unsupervised, Euclidean | 18.9 | 8.6 | 19.7 | 27.0 |
| SPNMF unsupervised, KL | 16.5 | 8.2 | 16.7 | 31.0 |
| SPNMF unsupervised, IS | 14.8 | −15.1 | 14.8 | 9.6 |
| **SPNMF + drum dictionary, Euclidean** | **24.9** | **12.6** | 27.7 | 30.9 |
| **SPNMF + drum dictionary, KL** | **25.6** | **13.0** | 28.7 | 29.9 |
| **SPNMF + drum dictionary, IS** | **25.5** | **12.5** | 29.2 | 27.1 |

This is a synthetic mixture, so treat it as a sanity check and a regression
test — not as evidence about real music. The paper's own evaluation used
SiSEC and ENST-Drums.

Two things the table shows, both of which match the paper:

- **The dictionary is what makes it work.** With one, SPNMF beats the median
  baseline on both sources by 5–6 dB. Without one, it is no better.
- **IS is fragile here.** It is the most sensitive to initialisation and the
  scale of `V`; try `power=2.0` (a power spectrogram) if you want to use it.

### It is not a speech denoiser

Worth stating plainly, because the name invites the assumption. SPNMF separates
by *signal character*, not by *source identity*, and for speech those axes do
not line up — speech is both tonal and transient, and noise is too. Routing
measured on a speech-plus-noise mixture with known components:

| content | → harmonic out | → percussive out | |
|---|---|---|---|
| voiced speech | 99.8% | 0.2% | correct |
| fricatives (speech) | 0.0% | 100.0% | **misrouted** |
| plosives (speech) | 22.2% | 77.8% | **misrouted** |
| tonal noise (fan, whine) | 99.7% | 0.3% | **misrouted** |
| impulsive noise (keys) | 17.5% | 82.5% | correct |

Used as a denoiser it would strip the consonants and keep the hum. Use a
speech-enhancement model for speech enhancement.

### The unsupervised swap

Run unsupervised, SPNMF finds a good two-part decomposition — and then puts the
**drums** in the projective part and the tonal source in the NMF part, the
opposite of what the model intends. The paper reports the same failure and
solves it with the fixed drum dictionary.

`separate(..., assign='auto')`, the default, compares the spectral flatness of
the two parts and calls the flatter one percussive. It is a no-op whenever the
dictionary is doing its job, and it is what keeps the output labels meaningful
when there is no dictionary. `assign='projective'` restores the published
behaviour; `result.swapped` records which way it went.

```
$ python examples/reproduce_paper_figure.py
configuration                                    SDR h   SDR p  source order
------------------------------------------------------------------------------
unsupervised, assign='projective' (as published)    8.16   16.45  SWAPPED
unsupervised, assign='auto'                        16.45    8.16  correct
drum dictionary, assign='projective'               25.58   12.99  correct
drum dictionary, assign='auto'                     25.58   12.99  correct
```

## What it is good for now

It is a training-free separator: no dataset, no GPU, no checkpoint. The 4-second
demo mixture above separates in about a second of CPU time at 150 iterations,
and the whole thing is a few hundred lines of NumPy.

- **A baseline** to put next to a neural separator, or to bootstrap stems for
  training one.
- **A transient/steady front-end** for onset detection, beat tracking or chord
  recognition — music tasks, where the tonal/transient axis is the one you want.
- **Stratifying a noise corpus**: splitting untranscribed real-world noise
  recordings into tonal (fans, HVAC, whine) and impulsive (keys, cutlery,
  door slams) components, with no ground truth needed.
- **The semi-supervised case it was designed for**: you have isolated
  percussion but no paired mixtures, so a supervised model has nothing to learn
  from — but a dictionary is one NMF away.
- **A readable reference implementation** of projective NMF and β-divergence
  multiplicative updates.

## API

| | |
|---|---|
| `separate(x, sr, ...)` | end-to-end separation → `Separation(harmonic, percussive, ...)` |
| `spnmf(V, ...)` | the factorisation itself → `SPNMFResult(W_h, W_p, H_p, cost)` |
| `nmf(V, ...)` | plain β-NMF |
| `learn_dictionary(signals, ...)` | train `W_p` by NMF over percussive audio |
| `stft_dictionary(signals, ...)` | build `W_p` from raw STFT frames instead |
| `median_hpss(x, sr, ...)` | median-filter HPSS baseline (Fitzgerald, 2010) |
| `bss_eval_sources(refs, ests)` | SDR/SIR/SAR with permutation search |
| `si_sdr`, `spectral_flatness` | scale-invariant SDR; tonal-vs-noisy descriptor |
| `stft`, `istft`, `wiener_mask` | exact-reconstruction STFT layer |
| `synthetic_mixture(...)` | harmonic + percussive test signal with ground truth |

Key `separate` arguments: `divergence` (`'euclidean'`/`'kl'`/`'is'`),
`n_harmonic`, `n_percussive`, `W_p`, `n_fft`, `hop_length`, `power`,
`mask_power`, `assign`, `n_iter`, `tol`, `random_state`.

## Layout

```
src/spnmf/
  core.py         SPNMF and NMF multiplicative updates
  divergence.py   β-divergence and its gradient split
  stft.py         STFT/ISTFT (exact round trip) and Wiener masks
  separation.py   end-to-end separation, part assignment
  dictionary.py   percussive dictionary construction
  baselines.py    median-filter HPSS
  metrics.py      BSS_EVAL, SI-SDR, spectral flatness
  signals.py      synthetic harmonic/percussive signals
  io.py           audio I/O (soundfile, else stdlib wave)
  demo.py, cli.py
tests/            88 tests, including numerical-gradient checks
examples/
```

```bash
pytest              # ~8 s
ruff check src tests examples
```

## Citation

The algorithm is from:

```bibtex
@inproceedings{laroche2015structured,
  author    = {Laroche, Cl\'ement and Kowalski, Matthieu and
               Papadopoulos, H\'el\`ene and Richard, Ga\"el},
  title     = {A structured nonnegative matrix factorization for source separation},
  booktitle = {Proc. European Signal Processing Conference (EUSIPCO)},
  year      = {2015}
}
```

extended in the journal manuscript *Structured Projective Non Negative Matrix
Factorization with drum dictionaries for Harmonic/Percussive Source
Separation* (same authors), kept in
[`LarocheC/Journal_SPNMF`](https://github.com/LarocheC/Journal_SPNMF).
