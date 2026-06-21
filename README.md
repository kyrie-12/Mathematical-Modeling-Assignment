# Mathematical Modeling Assignment

This project studies degradation-aware enhancement for DRIVE retinal images.

## Current pipeline

1. Generate normal, noisy, blurred, low-contrast, and mixed inputs.
2. Extract interpretable image-quality features.
3. Train a five-class random-forest baseline.
4. Train three random-forest component classifiers for noise, blur, and low contrast.
5. Estimate noise severity with a random-forest regressor.
6. Select NLM strength from a training-only downstream Dice grid search.
7. Route each predicted component combination to a matching enhancement chain.
8. Segment vessels with a fixed CLAHE, multi-scale Frangi, Otsu, and morphology pipeline.
9. Compare no enhancement, fixed enhancement, and adaptive enhancement.

The component code is ordered as `noise-blur-low_contrast`:

| Code | Predicted degradation | Enhancement strategy |
|---|---|---|
| `000` | normal | identity |
| `100` | noise | severity-aware NLM or identity fallback |
| `010` | blur | unsharp mask, mild CLAHE |
| `001` | low contrast | gamma, CLAHE |
| `110` | noise and blur | severity-aware NLM, unsharp mask |
| `101` | noise and low contrast | severity-aware NLM, weak CLAHE |
| `011` | blur and low contrast | unsharp mask, gamma, mild CLAHE |
| `111` | all three | severity-aware NLM, weak CLAHE |

The original five-class prediction is retained in the model bundle and result tables as a baseline.

## Reproduce stages 5-8

```powershell
.\.venv\Scripts\python.exe src\05_train_degradation_classifier.py
.\.venv\Scripts\python.exe src\06_tune_noise_enhancement.py
.\.venv\Scripts\python.exe src\06_adaptive_enhancement.py --overwrite
.\.venv\Scripts\python.exe src\07_vessel_segmentation.py --overwrite
.\.venv\Scripts\python.exe src\08_evaluate_results.py
.\.venv\Scripts\python.exe src\09_module_ablation.py --overwrite
.\.venv\Scripts\python.exe src\10_noise_robustness.py
```

The training-only NLM search selects identity for sigma 5, `h=5` for sigma 15, and `h=12`
for sigma 30. Current test Dice scores are `0.5349` without enhancement, `0.3670` with fixed
enhancement, and `0.6059` with component-aware, noise-severity-aware adaptive enhancement.

## Stage 9 module ablation

Stage 9 keeps the segmentation evaluator fixed and compares the full adaptive pipeline with:

- fixed enhancement for every image (remove degradation recognition),
- CLAHE only for noisy images (remove denoising),
- unsharp masking only for low-contrast images (remove contrast enhancement), and
- fixed NLM only for blurred images (remove sharpening).

The script writes paired per-sample metrics, a summary table, the exact protocol, and a Dice
comparison figure under `results/stage9_ablation/`.

## Stage 10 noise-severity robustness

Stage 10 groups the 60 test noise samples by Gaussian noise sigma (`5`, `15`, and `30`). It
compares all three treatment groups using Dice, IoU, precision, recall, HD/HD95, PSNR, and
SSIM. Paired bootstrap confidence intervals and Wilcoxon tests use the 20 DRIVE test image
IDs as independent units. Outputs are written under `results/stage10_noise_robustness/`.
