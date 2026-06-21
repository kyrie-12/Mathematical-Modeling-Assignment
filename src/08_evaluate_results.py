import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_COLUMNS = [
    "mse",
    "psnr",
    "ssim",
    "entropy",
    "contrast_std",
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "pixel_accuracy",
    "hausdorff_distance",
    "hausdorff_95",
]
GROUP_LABELS = {
    "no_enhancement": "No enhancement",
    "fixed_enhancement": "Fixed enhancement",
    "adaptive_enhancement": "Adaptive enhancement",
}
DEGRADATION_ORDER = ["normal", "noise", "blur", "low_contrast", "mixed", "all"]


def read_color(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def read_mask(path: str, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def quality_channel(image: np.ndarray, channel: str) -> np.ndarray:
    if channel == "green":
        return image[:, :, 1]
    if channel == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image quality channel: {channel}")


def image_quality_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    fov_mask: np.ndarray,
    config: dict,
) -> dict[str, float]:
    reference_channel = quality_channel(reference, config["image_quality_channel"])
    candidate_channel = quality_channel(candidate, config["image_quality_channel"])
    if reference_channel.shape != candidate_channel.shape:
        raise ValueError("Reference and candidate image shapes differ")

    reference_values = reference_channel[fov_mask].astype(np.float64)
    candidate_values = candidate_channel[fov_mask].astype(np.float64)
    mse = float(np.mean(np.square(reference_values - candidate_values)))
    if mse <= 1e-12:
        psnr = float(config["psnr_cap_db"])
    else:
        psnr = float(10.0 * np.log10(float(config["data_range"]) ** 2 / mse))

    _, ssim_map = structural_similarity(
        reference_channel,
        candidate_channel,
        data_range=float(config["data_range"]),
        full=True,
    )
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": float(np.mean(ssim_map[fov_mask])),
        "entropy": float(shannon_entropy(candidate_channel[fov_mask].astype(np.uint8))),
        "contrast_std": float(np.std(candidate_values) / float(config["data_range"])),
    }


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)


def hausdorff_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    prediction_boundary = mask_boundary(prediction)
    ground_truth_boundary = mask_boundary(ground_truth)
    if not np.any(prediction_boundary) and not np.any(ground_truth_boundary):
        return 0.0, 0.0
    if not np.any(prediction_boundary) or not np.any(ground_truth_boundary):
        diagonal = float(np.hypot(*prediction.shape))
        return diagonal, diagonal

    distance_to_ground_truth = distance_transform_edt(~ground_truth_boundary)
    distance_to_prediction = distance_transform_edt(~prediction_boundary)
    distances = np.concatenate(
        [
            distance_to_ground_truth[prediction_boundary],
            distance_to_prediction[ground_truth_boundary],
        ]
    )
    return float(np.max(distances)), float(np.percentile(distances, 95))


def segmentation_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    fov_mask: np.ndarray,
) -> dict[str, float]:
    prediction = prediction & fov_mask
    ground_truth = ground_truth & fov_mask
    tp = int(np.logical_and(prediction, ground_truth).sum())
    fp = int(np.logical_and(prediction, ~ground_truth & fov_mask).sum())
    fn = int(np.logical_and(~prediction & fov_mask, ground_truth).sum())
    tn = int(np.logical_and(~prediction & fov_mask, ~ground_truth & fov_mask).sum())
    epsilon = 1e-12
    dice = 2.0 * tp / (2.0 * tp + fp + fn + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    hausdorff, hausdorff_95 = hausdorff_metrics(prediction, ground_truth)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "pixel_accuracy": float(accuracy),
        "hausdorff_distance": hausdorff,
        "hausdorff_95": hausdorff_95,
    }


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["split"], row["degradation_type"], row["group"])].append(row)
        buckets[(row["split"], "all", row["group"])].append(row)

    summary_rows = []
    for (split, degradation_type, group), values in sorted(buckets.items()):
        summary = {
            "split": split,
            "degradation_type": degradation_type,
            "group": group,
            "samples": len(values),
        }
        for metric in METRIC_COLUMNS:
            metric_values = np.asarray([float(row[metric]) for row in values], dtype=np.float64)
            summary[f"{metric}_mean"] = float(metric_values.mean())
            summary[f"{metric}_std"] = float(metric_values.std(ddof=1)) if len(metric_values) > 1 else 0.0
        summary_rows.append(summary)
    return summary_rows


def test_improvements(summary_rows: list[dict]) -> list[dict]:
    lookup = {
        (row["degradation_type"], row["group"]): row
        for row in summary_rows
        if row["split"] == "test"
    }
    higher_is_better = ["psnr", "ssim", "dice", "iou", "precision", "recall"]
    lower_is_better = ["mse", "hausdorff_distance", "hausdorff_95"]
    output = []
    for degradation_type in DEGRADATION_ORDER:
        adaptive = lookup[(degradation_type, "adaptive_enhancement")]
        no_enhancement = lookup[(degradation_type, "no_enhancement")]
        fixed = lookup[(degradation_type, "fixed_enhancement")]
        row = {"degradation_type": degradation_type}
        for metric in higher_is_better:
            row[f"adaptive_vs_no_{metric}"] = float(
                adaptive[f"{metric}_mean"] - no_enhancement[f"{metric}_mean"]
            )
            row[f"adaptive_vs_fixed_{metric}"] = float(
                adaptive[f"{metric}_mean"] - fixed[f"{metric}_mean"]
            )
        for metric in lower_is_better:
            row[f"adaptive_vs_no_{metric}_reduction"] = float(
                no_enhancement[f"{metric}_mean"] - adaptive[f"{metric}_mean"]
            )
            row[f"adaptive_vs_fixed_{metric}_reduction"] = float(
                fixed[f"{metric}_mean"] - adaptive[f"{metric}_mean"]
            )
        output.append(row)
    return output


def summary_lookup(summary_rows: list[dict], split: str = "test") -> dict:
    return {
        (row["degradation_type"], row["group"]): row
        for row in summary_rows
        if row["split"] == split
    }


def grouped_bar_chart(
    path: Path,
    summary_rows: list[dict],
    metrics: list[str],
    titles: list[str],
) -> None:
    lookup = summary_lookup(summary_rows)
    degradations = DEGRADATION_ORDER[:-1]
    groups = ["no_enhancement", "fixed_enhancement", "adaptive_enhancement"]
    colors = ["#7A7A7A", "#C7813A", "#397367"]
    figure, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5))
    axes = np.atleast_1d(axes)
    x = np.arange(len(degradations))
    width = 0.24
    for axis, metric, title in zip(axes, metrics, titles):
        for index, (group, color) in enumerate(zip(groups, colors)):
            values = [lookup[(degradation, group)][f"{metric}_mean"] for degradation in degradations]
            axis.bar(x + (index - 1) * width, values, width, label=GROUP_LABELS[group], color=color)
        axis.set_xticks(x, degradations, rotation=20)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image quality and DRIVE vessel segmentation results.")
    parser.add_argument("--segmentation-manifest", default="data/metadata/segmentation_manifest.csv")
    parser.add_argument("--degradation-manifest", default="data/metadata/degradation_manifest.csv")
    parser.add_argument("--config", default="config/stage8_evaluation.json")
    parser.add_argument("--results-root", default="results/stage8_evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segmentation_rows = load_csv(Path(args.segmentation_manifest).resolve())
    degradation_rows = load_csv(Path(args.degradation_manifest).resolve())
    with Path(args.config).resolve().open(encoding="utf-8") as file:
        config = json.load(file)
    results_root = Path(args.results_root).resolve()
    table_dir = results_root / "tables"
    figure_dir = results_root / "figures"

    original_by_degraded = {
        Path(row["degraded_image_path"]).resolve().as_posix(): row["original_image_path"]
        for row in degradation_rows
    }
    group_columns = {
        "no_enhancement": ("degraded_image_path", "no_enhancement_mask_path"),
        "fixed_enhancement": ("fixed_image_path", "fixed_enhancement_mask_path"),
        "adaptive_enhancement": ("adaptive_image_path", "adaptive_enhancement_mask_path"),
    }

    metric_rows = []
    for index, row in enumerate(segmentation_rows, start=1):
        degraded_key = Path(row["degraded_image_path"]).resolve().as_posix()
        reference_path = original_by_degraded.get(degraded_key)
        if not reference_path:
            raise KeyError(f"No original image for: {row['degraded_image_path']}")
        reference = read_color(reference_path)
        fov_mask = read_mask(row["fov_mask_path"], reference.shape[:2])
        ground_truth = read_mask(row["manual_mask_path"], reference.shape[:2])

        for group in config["groups"]:
            image_column, mask_column = group_columns[group]
            candidate = read_color(row[image_column])
            prediction = read_mask(row[mask_column], reference.shape[:2])
            quality = image_quality_metrics(reference, candidate, fov_mask, config)
            segmentation = segmentation_metrics(prediction, ground_truth, fov_mask)
            metric_rows.append(
                {
                    "image_id": row["image_id"],
                    "split": row["split"],
                    "degradation_type": row["degradation_type"],
                    "predicted_degradation": row["predicted_degradation"],
                    "group": group,
                    "reference_image_path": reference_path,
                    "evaluated_image_path": row[image_column],
                    "prediction_mask_path": row[mask_column],
                    **quality,
                    **segmentation,
                }
            )
        if index % 25 == 0 or index == len(segmentation_rows):
            print(f"Evaluated {index}/{len(segmentation_rows)} samples")

    summary_rows = summarize(metric_rows)
    improvement_rows = test_improvements(summary_rows)
    write_csv(table_dir / "metrics_per_sample.csv", metric_rows)
    write_csv(table_dir / "metrics_summary.csv", summary_rows)
    write_csv(table_dir / "test_adaptive_improvements.csv", improvement_rows)

    lookup = summary_lookup(summary_rows)
    overall = {
        "test_samples": sum(row["split"] == "test" for row in segmentation_rows),
        "groups": {},
        "config": config,
    }
    for group in config["groups"]:
        row = lookup[("all", group)]
        overall["groups"][group] = {
            metric: float(row[f"{metric}_mean"]) for metric in METRIC_COLUMNS
        }
    with (table_dir / "test_overall_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(overall, file, ensure_ascii=False, indent=2)

    grouped_bar_chart(
        figure_dir / "segmentation_overlap_by_degradation.png",
        summary_rows,
        ["dice", "iou"],
        ["Test Dice by Degradation", "Test IoU by Degradation"],
    )
    grouped_bar_chart(
        figure_dir / "hausdorff_by_degradation.png",
        summary_rows,
        ["hausdorff_distance", "hausdorff_95"],
        ["Test Hausdorff Distance", "Test HD95"],
    )
    grouped_bar_chart(
        figure_dir / "image_quality_by_degradation.png",
        summary_rows,
        ["psnr", "ssim"],
        ["Test PSNR by Degradation", "Test SSIM by Degradation"],
    )

    print(f"Per-sample metrics: {table_dir / 'metrics_per_sample.csv'}")
    print(f"Summary: {table_dir / 'metrics_summary.csv'}")
    for group, metrics in overall["groups"].items():
        print(
            f"{group}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
            f"HD={metrics['hausdorff_distance']:.2f}, PSNR={metrics['psnr']:.2f}, "
            f"SSIM={metrics['ssim']:.4f}"
        )


if __name__ == "__main__":
    main()
