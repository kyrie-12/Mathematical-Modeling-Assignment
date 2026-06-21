import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dice_score(prediction: np.ndarray, ground_truth: np.ndarray, fov_mask: np.ndarray) -> float:
    predicted = (prediction > 0) & fov_mask
    truth = (ground_truth > 0) & fov_mask
    true_positive = np.count_nonzero(predicted & truth)
    false_positive = np.count_nonzero(predicted & ~truth & fov_mask)
    false_negative = np.count_nonzero(~predicted & truth)
    return float(2.0 * true_positive / max(2 * true_positive + false_positive + false_negative, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune NLM strength on training-only noisy DRIVE images."
    )
    parser.add_argument("--features", default="data/metadata/degradation_features.csv")
    parser.add_argument("--enhancement-config", default="config/stage6_enhancement.json")
    parser.add_argument("--segmentation-config", default="config/stage7_segmentation.json")
    parser.add_argument("--results-root", default="results/stage6_noise_tuning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage6 = load_module("stage6_enhancement", PROJECT_ROOT / "src" / "06_adaptive_enhancement.py")
    stage7 = load_module("stage7_segmentation", PROJECT_ROOT / "src" / "07_vessel_segmentation.py")

    with Path(args.features).resolve().open(encoding="utf-8", newline="") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["split"] == "train" and row["degradation_type"] == "noise"
        ]
    if not rows:
        raise ValueError("No training-only noise rows found")
    with Path(args.enhancement_config).resolve().open(encoding="utf-8") as file:
        enhancement_config = json.load(file)
    with Path(args.segmentation_config).resolve().open(encoding="utf-8") as file:
        segmentation_config = json.load(file)["segmentation"]

    candidates = [float(value) for value in enhancement_config["noise_severity"]["candidate_h"]]
    per_sample = []
    grouped_scores = defaultdict(list)
    for row_index, row in enumerate(rows, start=1):
        image = stage6.read_color(row["degraded_image_path"])
        fov_mask = stage6.read_fov_mask(row["fov_mask_path"], image.shape[:2])
        ground_truth = cv2.imread(row["manual_mask_path"], cv2.IMREAD_GRAYSCALE)
        if ground_truth is None:
            raise ValueError(f"Could not read manual mask: {row['manual_mask_path']}")
        sigma = int(round(float(row["noise_sigma"])))
        for h in candidates:
            if h <= 0:
                enhanced = image.copy()
                strategy = "identity"
            else:
                nlm_config = {**enhancement_config["nlm"], "h": h, "h_color": h}
                enhanced = stage6.apply_nlm(image, nlm_config)
                enhanced = stage6.apply_inside_fov(image, enhanced, fov_mask)
                strategy = f"nlm_h{h:g}"
            prediction, _ = stage7.segment_vessels(enhanced, fov_mask, segmentation_config)
            dice = dice_score(prediction, ground_truth, fov_mask)
            grouped_scores[(sigma, h)].append(dice)
            per_sample.append(
                {
                    "image_id": row["image_id"],
                    "noise_sigma": sigma,
                    "h": h,
                    "strategy": strategy,
                    "dice": dice,
                }
            )
        if row_index % 10 == 0 or row_index == len(rows):
            print(f"Tuned {row_index}/{len(rows)} training noise images")

    summary_rows = []
    selected_h = {}
    for sigma in sorted({key[0] for key in grouped_scores}):
        sigma_rows = []
        for h in candidates:
            values = grouped_scores[(sigma, h)]
            item = {
                "noise_sigma": sigma,
                "h": h,
                "strategy": "identity" if h <= 0 else f"nlm_h{h:g}",
                "samples": len(values),
                "dice_mean": float(np.mean(values)),
                "dice_std": float(np.std(values, ddof=1)),
            }
            sigma_rows.append(item)
            summary_rows.append(item)
        best = max(sigma_rows, key=lambda item: (item["dice_mean"], -item["h"]))
        selected_h[str(sigma)] = float(best["h"])

    results_root = Path(args.results_root).resolve()
    table_dir = results_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / "noise_h_grid_per_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(per_sample[0].keys()))
        writer.writeheader()
        writer.writerows(per_sample)
    with (table_dir / "noise_h_grid_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    result = {
        "selection_split": "train",
        "selection_degradation": "noise",
        "selection_metric": "mean_dice",
        "candidate_h": candidates,
        "selected_h": selected_h,
    }
    with (table_dir / "noise_h_selection.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(f"Selected h by noise sigma: {selected_h}")
    print(f"Outputs: {results_root}")


if __name__ == "__main__":
    main()
