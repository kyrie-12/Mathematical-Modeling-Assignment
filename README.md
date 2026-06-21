# Degradation-Aware Retinal Image Enhancement for Vessel Segmentation

This repository contains the code for a mathematical modeling course project on
low-quality retinal image degradation recognition, adaptive image enhancement,
and vessel segmentation evaluation on the DRIVE dataset.

The project studies a practical question:

> Can we first identify the degradation type of a low-quality medical image and
> then choose a matching enhancement strategy that improves downstream vessel
> segmentation?

The final pipeline combines interpretable image-quality features, random forest
models, traditional image enhancement, and a fixed vessel segmentation evaluator.

## Project Highlights

- Builds synthetic low-quality retinal images with noise, blur, low contrast,
  and mixed degradation.
- Extracts interpretable degradation features from image statistics, gradients,
  texture, residuals, and frequency information.
- Trains random forest models for degradation recognition and noise severity
  estimation.
- Uses adaptive enhancement routes based on predicted degradation components.
- Evaluates enhancement quality using both image-quality metrics and medical
  segmentation metrics.
- Includes module ablation and robustness grouping experiments.

## Dataset

The project uses the public DRIVE retinal vessel segmentation dataset.

Expected local structure:

```text
data/raw/DRIVE/
  training/
    images/
    1st_manual/
    mask/
  test/
    images/
    1st_manual/
    mask/
```

The dataset is not included in this repository. Put the downloaded DRIVE files
under `data/raw/DRIVE/` before running the pipeline.

## Method Overview

The complete workflow is:

```text
DRIVE images
  -> synthetic degradation generation
  -> degradation feature extraction
  -> random forest degradation recognition
  -> adaptive enhancement
  -> fixed vessel segmentation
  -> image-quality and segmentation evaluation
  -> ablation and robustness analysis
```

The adaptive enhancement component code is ordered as:

```text
noise-blur-low_contrast
```

| Code | Predicted degradation components | Enhancement strategy |
|---|---|---|
| `000` | normal | identity |
| `100` | noise | severity-aware NLM or identity fallback |
| `010` | blur | unsharp mask + mild CLAHE |
| `001` | low contrast | Gamma correction + CLAHE |
| `110` | noise + blur | severity-aware NLM + unsharp mask |
| `101` | noise + low contrast | severity-aware NLM + weak CLAHE |
| `011` | blur + low contrast | unsharp mask + Gamma correction + mild CLAHE |
| `111` | noise + blur + low contrast | severity-aware NLM + weak CLAHE |

The original five-class classifier is retained as a baseline, while the final
adaptive route uses multi-label component prediction and noise severity
estimation.

## Environment

Python 3.11 is recommended.

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Main dependencies:

- NumPy
- OpenCV
- scikit-image
- scikit-learn
- SciPy
- Matplotlib

## Repository Structure

```text
config/
  stage6_enhancement.json          # selected enhancement parameters

src/
  04_generate_degradations.py       # generate synthetic degraded images
  05_extract_features.py            # extract degradation-recognition features
  05_train_degradation_classifier.py # train RF/SVM and component models
  06_tune_noise_enhancement.py      # tune NLM strength using training Dice
  06_adaptive_enhancement.py        # generate fixed and adaptive enhanced images
  07_vessel_segmentation.py         # fixed vessel segmentation evaluator
  08_evaluate_results.py            # compute metrics and summaries
  09_module_ablation.py             # module ablation experiments
  10_noise_robustness.py            # robustness grouping experiments

data/                               # local data and generated images, ignored by git
results/                            # experiment outputs, ignored by git
```

## Reproduce the Full Pipeline

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe src\04_generate_degradations.py
.\.venv\Scripts\python.exe src\05_extract_features.py
.\.venv\Scripts\python.exe src\05_train_degradation_classifier.py
.\.venv\Scripts\python.exe src\06_tune_noise_enhancement.py
.\.venv\Scripts\python.exe src\06_adaptive_enhancement.py --overwrite
.\.venv\Scripts\python.exe src\07_vessel_segmentation.py --overwrite
.\.venv\Scripts\python.exe src\08_evaluate_results.py
.\.venv\Scripts\python.exe src\09_module_ablation.py --overwrite
.\.venv\Scripts\python.exe src\10_noise_robustness.py
```

The training-only NLM search selected:

| Noise sigma | Selected NLM strength |
|---:|---:|
| 5 | 0, identity fallback |
| 15 | 5 |
| 30 | 12 |

## Main Results

The final test set contains 280 degraded image variants from the 20 DRIVE test
images. The three main treatment groups are no enhancement, fixed enhancement,
and adaptive enhancement.

| Group | Dice | IoU | SSIM | HD95 |
|---|---:|---:|---:|---:|
| No enhancement | 0.5349 | 0.3805 | 0.7838 | 17.1968 |
| Fixed enhancement | 0.3670 | 0.2284 | 0.7623 | 22.3810 |
| Adaptive enhancement | 0.6059 | 0.4504 | 0.8565 | 17.4064 |

The adaptive strategy improves average Dice and IoU compared with both no
enhancement and fixed enhancement. It also improves SSIM, which indicates better
structure preservation. However, HD95 is close to the no-enhancement result,
showing that boundary errors are not fully solved by the current enhancement
pipeline.

## Stage 9: Module Ablation

Stage 9 evaluates whether each module contributes to the final adaptive result.

| Ablation setting | Target subset | Full Dice | Ablated Dice | Dice contribution |
|---|---|---:|---:|---:|
| Remove degradation recognition | all | 0.6059 | 0.3670 | +0.2388 |
| Remove denoising | noise | 0.4859 | 0.4556 | +0.0302 |
| Remove contrast enhancement | low contrast | 0.7121 | 0.5991 | +0.1129 |
| Remove sharpening | blur | 0.7143 | 0.3138 | +0.4005 |

The ablation study shows that degradation recognition and blur-oriented
sharpening contribute strongly. Denoising improves Dice on noisy images, but it
can also trade recall, precision, and boundary behavior, so it should not be
interpreted as universally beneficial.

## Stage 10: Robustness Grouping

The robustness experiment follows a paper-style grouping strategy. Noise samples
are grouped by Gaussian noise sigma. Mixed degradation samples are grouped by
component count:

- two-component mixed degradation is treated as medium mixed degradation;
- three-component mixed degradation is treated as strong mixed degradation.

| Group | No enhancement Dice | Fixed enhancement Dice | Adaptive enhancement Dice | Adaptive - Fixed |
|---|---:|---:|---:|---:|
| Medium mixed degradation | 0.4350 | 0.3253 | 0.5678 | +0.2425 |
| Strong mixed degradation | 0.3512 | 0.2835 | 0.3399 | +0.0565 |
| Noise sigma 5 | 0.6367 | 0.4578 | 0.6435 | +0.1857 |
| Noise sigma 15 | 0.3808 | 0.4966 | 0.4818 | -0.0148 |
| Noise sigma 30 | 0.3022 | 0.3739 | 0.3323 | -0.0417 |

The robustness results show that adaptive enhancement is especially useful for
mixed degradation, where a single fixed enhancement chain is less well matched
to the input. For pure noise degradation, adaptive enhancement performs best
under light noise, is close to fixed enhancement under medium noise, and falls
behind fixed enhancement under strong noise. This indicates that the current
adaptive strategy is safer for degradation-type matching and structure
preservation, but it is not guaranteed to achieve the best Dice under every
noise intensity.

## Generated Outputs

Important outputs are written to:

```text
results/stage8_evaluation/
results/stage9_ablation/
results/stage10_noise_robustness/
```

Useful files include:

- `results/stage8_evaluation/tables/metrics_per_sample.csv`
- `results/stage8_evaluation/tables/test_overall_metrics.json`
- `results/stage9_ablation/tables/module_ablation_summary.csv`
- `results/stage10_noise_robustness/tables/paper_style_robustness_summary.csv`
- `results/stage10_noise_robustness/figures/paper_style_robustness.png`

These outputs are generated locally and are ignored by git.

## Conclusion

This project supports three main conclusions:

1. Interpretable degradation features combined with random forest models can
   identify synthetic retinal image degradation reliably enough to drive
   enhancement selection.
2. Adaptive enhancement improves average downstream vessel segmentation compared
   with a fixed enhancement pipeline.
3. The benefit of adaptive enhancement is condition-dependent: it is strong for
   mixed degradation and light noise, but strong pure noise still requires better
   denoising or segmentation-aware enhancement design.
