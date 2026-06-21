import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEGMENTATION_METRICS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "pixel_accuracy",
    "hausdorff_distance",
    "hausdorff_95",
]
EXPERIMENT_ORDER = [
    "remove_recognition",
    "remove_denoising",
    "remove_contrast_enhancement",
    "remove_sharpening",
]
EXPERIMENT_LABELS = {
    "remove_recognition": "Remove recognition",
    "remove_denoising": "Remove denoising",
    "remove_contrast_enhancement": "Remove contrast",
    "remove_sharpening": "Remove sharpening",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_image(path: Path, image: np.ndarray, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Could not write image: {path}")


def sample_key_from_metric(row: dict) -> tuple[str, str, str]:
    name = Path(row["evaluated_image_path"]).stem
    if "_adaptive_components-" in name:
        name = name.split("_adaptive_components-", 1)[0]
    elif name.endswith("_fixed"):
        name = name[: -len("_fixed")]
    return row["image_id"], row["degradation_type"], name


def metric_values(row: dict) -> dict[str, float]:
    return {metric: float(row[metric]) for metric in SEGMENTATION_METRICS}


def paired_row(
    experiment: str,
    target: str,
    image_id: str,
    sample_key: str,
    full: dict[str, float],
    ablated: dict[str, float],
) -> dict:
    row = {
        "experiment": experiment,
        "target_degradation": target,
        "image_id": image_id,
        "sample_key": sample_key,
    }
    for metric in SEGMENTATION_METRICS:
        row[f"full_{metric}"] = full[metric]
        row[f"ablated_{metric}"] = ablated[metric]
    row["dice_contribution"] = full["dice"] - ablated["dice"]
    row["iou_contribution"] = full["iou"] - ablated["iou"]
    row["hd_reduction"] = ablated["hausdorff_distance"] - full["hausdorff_distance"]
    row["hd95_reduction"] = ablated["hausdorff_95"] - full["hausdorff_95"]
    return row


def apply_ablation(
    experiment: str,
    image: np.ndarray,
    fov_mask: np.ndarray,
    enhancement_config: dict,
    stage6,
) -> np.ndarray:
    if experiment == "remove_denoising":
        ablated = stage6.apply_clahe(
            image,
            enhancement_config["clahe"]["standard_clip_limit"],
            enhancement_config["clahe"]["tile_grid_size"],
        )
    elif experiment == "remove_contrast_enhancement":
        ablated = stage6.apply_unsharp_mask(image, enhancement_config["unsharp_mask"])
    elif experiment == "remove_sharpening":
        ablated = stage6.apply_nlm(image, enhancement_config["nlm"])
    else:
        raise ValueError(f"Unsupported computed ablation: {experiment}")
    return stage6.apply_inside_fov(image, ablated, fov_mask)


def summarize(per_sample_rows: list[dict]) -> list[dict]:
    summary_rows = []
    for experiment in EXPERIMENT_ORDER:
        values = [row for row in per_sample_rows if row["experiment"] == experiment]
        if not values:
            raise ValueError(f"No values for experiment: {experiment}")
        summary = {
            "experiment": experiment,
            "target_degradation": values[0]["target_degradation"],
            "samples": len(values),
        }
        for metric in SEGMENTATION_METRICS:
            summary[f"full_{metric}"] = float(
                np.mean([float(row[f"full_{metric}"]) for row in values])
            )
            summary[f"ablated_{metric}"] = float(
                np.mean([float(row[f"ablated_{metric}"]) for row in values])
            )
        summary["dice_contribution"] = summary["full_dice"] - summary["ablated_dice"]
        summary["iou_contribution"] = summary["full_iou"] - summary["ablated_iou"]
        summary["hd_reduction"] = (
            summary["ablated_hausdorff_distance"] - summary["full_hausdorff_distance"]
        )
        summary["hd95_reduction"] = summary["ablated_hausdorff_95"] - summary["full_hausdorff_95"]
        summary_rows.append(summary)
    return summary_rows


def save_summary_figure(path: Path, summary_rows: list[dict]) -> None:
    labels = [EXPERIMENT_LABELS[row["experiment"]] for row in summary_rows]
    full = [row["full_dice"] for row in summary_rows]
    ablated = [row["ablated_dice"] for row in summary_rows]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(positions - width / 2, full, width, label="Full adaptive", color="#287271")
    axis.bar(positions + width / 2, ablated, width, label="Ablated", color="#D9822B")
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_ylabel("Mean Dice")
    axis.set_ylim(0.0, 0.8)
    axis.set_title("Stage 9 Module Ablation")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage 9 module ablation experiments.")
    parser.add_argument("--adaptive-manifest", default="data/metadata/adaptive_enhancement_manifest.csv")
    parser.add_argument("--segmentation-manifest", default="data/metadata/segmentation_manifest.csv")
    parser.add_argument("--stage8-metrics", default="results/stage8_evaluation/tables/metrics_per_sample.csv")
    parser.add_argument("--enhancement-config", default="config/stage6_enhancement.json")
    parser.add_argument("--segmentation-config", default="config/stage7_segmentation.json")
    parser.add_argument("--output-root", default="data/processed/ablation")
    parser.add_argument("--results-root", default="results/stage9_ablation")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage6 = load_module("stage6_enhancement", PROJECT_ROOT / "src" / "06_adaptive_enhancement.py")
    stage7 = load_module("stage7_segmentation", PROJECT_ROOT / "src" / "07_vessel_segmentation.py")
    stage8 = load_module("stage8_evaluation", PROJECT_ROOT / "src" / "08_evaluate_results.py")

    adaptive_rows = load_csv(Path(args.adaptive_manifest).resolve())
    segmentation_rows = load_csv(Path(args.segmentation_manifest).resolve())
    stage8_rows = load_csv(Path(args.stage8_metrics).resolve())
    segmentation_by_path = {row["degraded_image_path"]: row for row in segmentation_rows}
    with Path(args.enhancement_config).resolve().open(encoding="utf-8") as file:
        enhancement_config = json.load(file)
    with Path(args.segmentation_config).resolve().open(encoding="utf-8") as file:
        segmentation_config = json.load(file)["segmentation"]

    output_root = Path(args.output_root).resolve()
    results_root = Path(args.results_root).resolve()
    per_sample_rows = []

    existing = {}
    for row in stage8_rows:
        if row["split"] != "test" or row["group"] not in {
            "adaptive_enhancement",
            "fixed_enhancement",
        }:
            continue
        existing[(sample_key_from_metric(row), row["group"])] = metric_values(row)
    paired_keys = sorted({key for key, group in existing if group == "adaptive_enhancement"})
    for image_id, degradation_type, sample_key in paired_keys:
        full = existing[((image_id, degradation_type, sample_key), "adaptive_enhancement")]
        ablated = existing[((image_id, degradation_type, sample_key), "fixed_enhancement")]
        per_sample_rows.append(
            paired_row(
                "remove_recognition",
                "all",
                image_id,
                sample_key,
                full,
                ablated,
            )
        )

    computed_experiments = {
        "remove_denoising": "noise",
        "remove_contrast_enhancement": "low_contrast",
        "remove_sharpening": "blur",
    }
    selected_rows = [
        row
        for row in adaptive_rows
        if row["split"] == "test"
        and any(row["degradation_type"] == target for target in computed_experiments.values())
    ]
    processed = 0
    for experiment, target in computed_experiments.items():
        for row in [item for item in selected_rows if item["degradation_type"] == target]:
            degraded = stage6.read_color(row["degraded_image_path"])
            fov_mask = stage6.read_fov_mask(row["fov_mask_path"], degraded.shape[:2])
            ground_truth = stage8.read_mask(row["manual_mask_path"], degraded.shape[:2])
            segmentation_row = segmentation_by_path[row["degraded_image_path"]]
            full_mask = stage8.read_mask(
                segmentation_row["adaptive_enhancement_mask_path"],
                degraded.shape[:2],
            )
            full_metrics = stage8.segmentation_metrics(full_mask, ground_truth, fov_mask)

            ablated_image = apply_ablation(
                experiment,
                degraded,
                fov_mask,
                enhancement_config,
                stage6,
            )
            ablated_mask, _ = stage7.segment_vessels(
                ablated_image,
                fov_mask,
                segmentation_config,
            )
            source_stem = Path(row["degraded_image_path"]).stem
            image_path = output_root / experiment / target / f"{source_stem}_ablated.png"
            mask_path = output_root / experiment / target / f"{source_stem}_ablated_mask.png"
            save_image(image_path, ablated_image, args.overwrite)
            save_image(mask_path, ablated_mask, args.overwrite)
            ablated_metrics = stage8.segmentation_metrics(
                ablated_mask > 0,
                ground_truth,
                fov_mask,
            )
            per_sample_rows.append(
                paired_row(
                    experiment,
                    target,
                    row["image_id"],
                    source_stem,
                    full_metrics,
                    ablated_metrics,
                )
            )
            processed += 1
            if processed % 20 == 0:
                print(f"Computed {processed}/180 module-ablation samples")

    summary_rows = summarize(per_sample_rows)
    table_dir = results_root / "tables"
    write_csv(table_dir / "module_ablation_per_sample.csv", per_sample_rows)
    write_csv(table_dir / "module_ablation_summary.csv", summary_rows)
    with (table_dir / "module_ablation_protocol.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selection_split": "test",
                "full_reference": "current adaptive enhancement",
                "experiments": {
                    "remove_recognition": "all samples use the fixed enhancement baseline",
                    "remove_denoising": "noise samples use CLAHE only",
                    "remove_contrast_enhancement": "low-contrast samples use unsharp masking only",
                    "remove_sharpening": "blur samples use fixed NLM only",
                },
                "segmentation_config": segmentation_config,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    save_summary_figure(results_root / "figures" / "module_ablation_dice.png", summary_rows)

    for row in summary_rows:
        print(
            f"{row['experiment']}: full Dice={row['full_dice']:.4f}, "
            f"ablated Dice={row['ablated_dice']:.4f}, "
            f"contribution={row['dice_contribution']:.4f}"
        )
    print(f"Outputs: {results_root}")


if __name__ == "__main__":
    main()
