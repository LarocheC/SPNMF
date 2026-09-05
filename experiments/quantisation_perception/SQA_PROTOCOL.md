# Running this with an audio LLM instead of NISQA

The CPU experiment in `fingerprint.py` establishes that compression artefacts
*do* have separable perceptual signatures — jitter and over-smoothing differ by
~9x on the discontinuity:noisiness axis — but that NISQA cannot resolve them at
int8 severity, where the whole effect is 0.02–0.03 MOS.

That is exactly the regime a **calibrated descriptive rater** might reach, and
[`LarocheC/llm-sqa`](https://github.com/LarocheC/llm-sqa) is one: SALMONN
fine-tuned with two-stage LoRA-SFT to emit per-dimension severities
(`noise reverberation bandwidth clipping discontinuity loudness`), a free-text
description, and a MOS with 63 distinct values rather than NISQA's smooth but
low-contrast output.

Needs a 24 GB GPU, which is why it is a protocol and not a result.

## Stimuli

`fingerprint.py` already writes them. Both sets, unchanged:

```bash
python fingerprint.py --models <LiSenNet> --data <VBD> --eco8 <eco8-neaixt> \
    --nisqa <NISQA> --out ./stimuli --n 200 --stride 1
```

`stimuli/mild/{ref,int8,jitter,smooth}` and `stimuli/strong/{...}`. Every
condition is the same enhancer on the same input; only the gain field differs.

## The three questions, in order of how much they are worth

**1. Does the rater separate mechanisms at strong severity?** The positive
control. NISQA does (disc:noise 0.95 for jitter vs 0.11 for smoothing). If the
SQA model cannot reproduce that separation, it is not a sharper instrument for
this and questions 2–3 are moot. Report per-dimension severities on
`stimuli/strong`, and check `bias_strong` returns *no* degradation — it is a
perceptually null change and any model that penalises it is reporting level, not
quality.

**2. Does it resolve int8?** The actual question. On `stimuli/mild`, is the
paired `int8 − ref` difference significant on any dimension, and does its shape
match `jitter − ref` rather than `smooth − ref`? The prior from this repo says
the mechanism is jitter, so `discontinuity` is the dimension to watch. Effect
sizes are ~0.02 MOS, so use paired per-clip tests and at least 200 clips.

**3. Does the free text say something the scalars do not?** The part no
conventional metric can do. Collect descriptions for matched-severity jitter and
smoothing and check whether the vocabulary separates ("warbling", "fluttering",
"musical noise" vs "muffled", "dull", "over-suppressed") even where the numeric
dimensions do not. A rater that names the artefact correctly at a severity where
its own scalars are ambiguous would be the interesting result.

## What would make this publishable rather than a curiosity

Deployment engineers currently choose a compression setting from one scalar, and
this repo has already shown two scalars **disagree in sign** about whether int8
hurt (PESQ −0.018 with 68% of clips worse, pooled SNR +0.29 dB). An instrument
that reports *what* a compression step broke, validated against mechanisms whose
ground truth is known by construction, is a different contribution from another
MOS predictor — and the validation is what makes it a contribution rather than a
demo. The mechanisms here are known by construction, which is the hard part of
that validation already solved.

## Honest expectations

The mild-condition effect may simply be below any current rater's resolution,
including a fine-tuned SALMONN. That is a publishable negative too, and cheaper
to establish than to assume: question 1 alone is a day of GPU time and settles
whether the instrument has the dynamic range before anything is built on it.
