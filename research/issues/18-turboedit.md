---
key: TURBOEDIT
title: "[X13] TurboEdit: does edit-friendly inversion carry the watermark through? (regeneration-thesis test)"
labels: experiment, spotlight, gpu-host
milestone: "M2: main results (Nov 7)"
---
## Hypothesis
In edit-friendly DDPM inversion (TurboEdit, arXiv:2408.00735) the per-step target `x_s = sqrt(a_s) x0 + sqrt(1-a_s) eps` is a closed-form noising of the **watermarked input**; the edit is only the correction `mu(x^_t,c) - mu(x_t,c) + w (mu(x^_t,c^) - mu(x^_t,c))` added on top (Eq. 10). SDEdit instead re-samples from noise. Prediction: **at matched edit magnitude, TurboEdit preserves watermarks far better than SDEdit**, and its regeneration-EWO is low. If true, this is the cleanest single demonstration that watermark loss is governed by carrier regeneration, not by visible change.

## Implemented
`TurboEdit` in `src/wmedit/editors.py` (SDXL-Turbo, 4 steps from t=599, Delta=200, w=1.5, last-step clip 15.5), from the paper's equations; the official repo has no licence. GPU test: `test_turboedit_reconstructs_when_prompts_match`.

## Tasks
- [ ] Run the GPU test; check reconstruction PSNR with c^ = c (EF property) and visual quality vs. Fig. 4 of the paper
- [ ] Add to `configs/kaggle.yaml` (done) and rerun the Kaggle kernel; then stage 2 (#15)
- [ ] Sweep w in {1.0, 1.5, 2.0, 3.0} to trace detection vs. edit magnitude; compare to the SDEdit strength sweep in the probes
- [ ] EWO (#1) and universality on the TurboEdit rows: does regen-EWO put TurboEdit and SDEdit on one curve?
- [ ] Paper: Sec. 1 motivation paragraph + one panel of the teaser

## Risk
SDXL-Turbo at 512x512 needs ~7 GB fp16 (fits a T4). The DDPM posterior mean for skipped steps is my reading of the paper (Appendix lacks the exact scheduler); note it in the supplement.
