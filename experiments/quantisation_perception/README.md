# What int8 quantisation actually does to a speech enhancer

A study of one reproducible anomaly: **int8 quantisation improves pooled SNR
while degrading PESQ**, on real deployed models, consistently enough that the
choice of metric flips the sign of the conclusion.

Everything here is measured on
[`claroche1/LiSenNet`](https://huggingface.co/claroche1/LiSenNet) — published
fp32/int8 ONNX pairs of a 36k-parameter enhancer deployed on an STM32N6
Neural-ART NPU — over the VoiceBank-DEMAND test split, using the author's own
STFT, feature build and reconstruction path from
[`eco8-neaixt`](https://github.com/LarocheC/eco8-neaixt).

```bash
python inversion.py   --models <LiSenNet> --data <VBD> --eco8 <eco8-neaixt>
python fingerprint.py --models <LiSenNet> --data <VBD> --eco8 <eco8-neaixt> --nisqa <NISQA>
```

## The anomaly

Measured over 206 clips, int8 minus fp32, paired 95% CI:

| variant | ΔPESQ | clips worse | Δ pooled SNR |
|---|---|---|---|
| `conv-hardened` | −0.018 ± 0.006 | 68.0% | **+0.288 ± 0.124 dB** |
| `gru` | −0.042 ± 0.010 | 83.0% | −0.271 ± 0.144 dB |
| `conv` | −0.097 ± 0.011 | 92.7% | **+0.044 ± 0.170 dB** |

The measured ΔPESQ reproduces the published −0.015 / −0.076 / −0.115, so this is
not a broken pipeline. At 6-bit weights the split is wider still: PESQ
2.910 → 2.848 while pooled SNR goes 10.15 → 10.61.

Two instruments, opposite answers, same model.

## Four mechanisms, three refuted

| hypothesis | test | outcome |
|---|---|---|
| int8 smooths the mask | gain-field roughness | **refuted** — int8 is 10–13% *rougher* |
| systematic gain bias | sweep uniform gain; rescale int8 to fp32's mean | **refuted** — PESQ invariant to uniform gain (±0.000); debiasing keeps the whole effect |
| monotone distortion of the gain transfer curve | 24-entry LUT fitted on held-out clips | **refuted** — recovers −0.004 / +0.004 PESQ, inside the CI |
| zero-mean jitter on the compressed gain | synthetic jitter sweep | **explains PESQ, not SNR** |

The third is worth dwelling on. The int8 error *does* have a systematic
S-shape — it lifts gains in 0.3–0.6 by up to +0.014 and lowers gains above 0.7 —
so the mask's contrast is compressed toward its middle. But correcting exactly
that conditional mean, with a LUT fitted on disjoint clips, recovers **no** PESQ
while pushing SNR *further* up. The perceptual damage is not in the mean.

It is in the jitter. Adding zero-mean Gaussian noise of std ≈ 0.02 to the fp32
compressed gain reproduces the PESQ loss quantitatively (−0.012 to −0.018
against int8's −0.009 to −0.019 on the same clips), and the measured int8 gain
error is `mean +0.0017, std 0.037` — essentially zero-mean.

**So the perceptual cost of int8 here is mask jitter, not any systematic
distortion.** That is actionable: a QAT objective for this model class should be
penalising gain-field jitter, not weight MSE or output MSE.

The SNR half is still unexplained. Independent jitter *lowers* SNR (−0.16 dB at
std 0.02), so whatever raises it comes from the signal-correlated part of the
int8 error. Stated as an open question rather than a fourth story.

## Do artefacts have perceptual fingerprints?

If the damage is jitter rather than smoothing, a *dimensional* rater should say
so. Using NISQA v2 (the same rater that built the corpus targets in
[`llm-sqa`](https://github.com/LarocheC/llm-sqa)), on perturbations of one
enhancer where only the gain field changes:

| condition | ΔMOS | noisiness | discontinuity | coloration | loudness | disc:noise |
|---|---|---|---|---|---|---|
| jitter (strong) | −0.431 | −0.320 | −0.303 | −0.243 | −0.168 | **0.95** |
| smoothing (strong) | −0.538 | −0.832 | −0.088 | −0.149 | −0.279 | **0.11** |
| uniform gain ×0.75 | −0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |

**At audible severity the mechanisms separate cleanly**: jitter damages
discontinuity almost as much as noisiness, over-smoothing barely touches it — a
~9× difference on that axis. The uniform-gain control moves *nothing*, which
both validates the setup and explains why the gain-bias hypothesis failed: after
level normalisation a uniform gain is perceptually null.

**At int8 severity it does not.** The whole int8 effect is 0.02–0.03 MOS, and
the per-dimension differences do not reach significance. The fingerprint idea is
sound; NISQA's resolution is the binding constraint.

That is the handoff point to a sharper instrument — see
[SQA_PROTOCOL.md](SQA_PROTOCOL.md) for running this against a calibrated
descriptive audio LLM, with the positive control that decides in a day whether
it has the dynamic range.

## Why this sits in a repository about a 2015 NMF paper

It is the same thread. The diagnostics that made it possible
(`spnmf.diagnostics.per_part_snr`) were built to ask what a pooled quality score
hides, after a structured decomposition turned out to be doing nothing. The
answer, twice over, has been that the measurement instrument was the interesting
part.

## Reproducing

`inversion.py` needs `torch`, `onnxruntime`, `pesq`, `soundfile`, `pyarrow`;
`fingerprint.py` additionally needs a NISQA clone (weights ship in-repo).
Both take `--models`, `--data`, `--eco8` paths; see each file's docstring.

The RMS normalisation in `eco8-neaixt/common/dataset.py`
(`sqrt(len / Σx²)`, applied to clean and noisy alike) is load-bearing — without
it the enhancer loses ~1.3 PESQ and every number above is meaningless.
