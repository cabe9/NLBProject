# STNDT-lite Depth-4 Diagnostic Summary

Local public-test scores are against the frozen public NLB target HDF5, not a
live EvalAI leaderboard.

| Recipe | Validation signal | Local public-test co-bps | Notes |
|---|---:|---:|---|
| 4-layer constant, `mask_prob=0.5` | sanity mean `0.358319` | `0.374211` | First validated depth-4 promotion. |
| 4-layer constant, `mask_prob=0.6` | sanity mean `0.365366` | `0.379537` | Validation and public-test both improved. |
| Screen H objective neighborhood | anchor `0.364878`; best non-anchor `0.364161` | n/a | Nearby objective tweaks did not beat the anchor. |

Interpretation: train/val selection remained directionally useful from
`mask_prob=0.5` to `mask_prob=0.6`, but the local objective neighborhood around
`mask_prob=0.6` appears saturated. The next reasonable validation axis is
architecture or regularization, not another near-repeat of Screen H.
