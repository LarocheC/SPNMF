# What frame-skipping actually costs a streaming enhancer

Dynamic-compute work for audio usually asks *which frames can I skip without
hurting accuracy*. On a **stateful** streaming graph that turns out to be the
wrong question: most of the damage is not the held mask, it is the recurrence
losing its place.

Measured on `claroche1/LiSenNet` `conv-hardened`, the int8 **streaming** graph
(17 FIFO state tensors), VoiceBank-DEMAND, 30 clips, paired 95% CI:

| arm | ΔPESQ |
|---|---|
| periodic skip 20% | −0.108 ± 0.031 |
| **content-adaptive skip 20%** | **−0.014 ± 0.009** |
| adaptive 20%, state kept fresh | −0.001 ± 0.003 |
| periodic skip 30% | −0.182 ± 0.036 |
| **content-adaptive skip 30%** | **−0.029 ± 0.014** |
| adaptive 30%, state kept fresh | −0.006 ± 0.007 |

Two results:

1. **Content-adaptive skipping beats periodic by 6–8×** at matched skip rate.
   The routing statistic is spectral flux of the noisy magnitude — already
   computed in the front end, no learned gate.
2. **79–96% of the residual damage is state staleness.** The "fresh-state" arms
   hold exactly the same masks on exactly the same frames, but keep running the
   network so the FIFO state stays current; they cost almost nothing. What
   skipping really costs is the recurrence's implicit time axis.

That reframes the design target. If most of the cost is state divergence rather
than mask error, the useful question is how to keep state coherent cheaply —
and it predicts the advantage should **collapse** on the stateless windowed
graph, where there is no state to lose. That is the experiment to run next, and
it is a two-line change: swap `g_best_streaming_int8_static.onnx` for
`g_best_windowed_int8_static.onnx`.

## Caveats

Absolute PESQ depends heavily on the clip subset — on one unlucky stride the
enhancer scored *below* the noisy input (2.067 vs 2.128), because that subset
was high-SNR. Deltas are paired and stable; absolute values are not. Verify any
absolute number against the published full-split figures before quoting it.

The routing here is oracle-free but also un-costed: HOLD frames still pay
STFT, feature build, ISTFT and overlap-add. No energy claim is made, and none
should be until it is measured in joules on real silicon.

```bash
python skip_routing.py --models <LiSenNet> --data <VBD> --eco8 <eco8-neaixt>
```
