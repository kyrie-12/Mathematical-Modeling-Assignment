import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_ORDER = ["normal", "noise", "blur", "low_contrast", "mixed"]
STRATEGY_NAMES = {
    "normal": "identity",
    "noise": "non_local_means",
    "blur": "unsharp_mask_then_mild_clahe",
    "low_contrast": "gamma_then_clahe",
    "mixed": "non_local_means_then_clahe",
}


def read_color(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def read_fov_mask(path: str, shape: tuple[int, int]) -> np.ndarray:
    if not path:
        return np.ones(shape, dtype=bool)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read FOV mask: {path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def apply_inside_fov(original: np.ndarray, enhanced: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
    output = original.copy()
    output[fov_mask] = enhanced[fov_mask]
    return output


def apply_nlm(image: np.ndarray, config: dict) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=float(config["h"]),
        hColor=float(config["h_color"]),
        templateWindowSize=int(config["template_window_size"]),
        searchWindowSize=int(config["search_window_size"]),
    )


def apply_unsharp_mask(image: np.ndarray, config: dict) -> np.ndarray:
    sigma = float(config["sigma"])
    amount = float(config["amount"])
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def apply_clahe(image: np.ndarray, clip_limit: float, tile_grid_size: list[int]) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])),
    )
    enhanced_lightness = clahe.apply(lightness)
    enhanced_lab = cv2.merge((enhanced_lightness, channel_a, channel_b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    values = np.arange(256, dtype=np.float32) / 255.0
    lookup = np.clip(np.power(values, gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image, lookup)


def enhance_by_prediction(
    image: np.ndarray,
    predicted_class: str,
    config: dict,
) -> np.ndarray:
    if predicted_class == "normal":
        return image.copy()
    if predicted_class == "noise":
        return apply_nlm(image, config["nlm"])
    if predicted_class == "blur":
        enhanced = apply_unsharp_mask(image, config["unsharp_mask"])
        return apply_clahe(
            enhanced,
            config["clahe"]["mild_clip_limit"],
            config["clahe"]["tile_grid_size"],
        )
    if predicted_class == "low_contrast":
        enhanced = apply_gamma(image, config["gamma"]["value"])
        return apply_clahe(
            enhanced,
            config["clahe"]["standard_clip_limit"],
            config["clahe"]["tile_grid_size"],
        )
    if predicted_class == "mixed":
        enhanced = apply_nlm(image, config["nlm"])
        return apply_clahe(
            enhanced,
            config["clahe"]["standard_clip_limit"],
            config["clahe"]["tile_grid_size"],
        )
    raise ValueError(f"Unsupported predicted class: {predicted_class}")


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Feature table is empty: {path}")
    return rows


def feature_matrix(rows: list[dict], columns: list[str]) -> np.ndarray:
    matrix = np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature table contains NaN or infinite values")
    return matrix


def save_preview(path: Path, preview_rows: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    classes = [label for label in CLASS_ORDER if label in preview_rows]
    figure, axes = plt.subplots(len(classes), 2, figsize=(9, 3 * len(classes)))
    axes = np.atleast_2d(axes)
    for row_index, label in enumerate(classes):
        before, after = preview_rows[label]
        axes[row_index, 0].imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
        axes[row_index, 0].set_title(f"Predicted {label}: before")
        axes[row_index, 1].imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
        axes[row_index, 1].set_title(f"Strategy: {STRATEGY_NAMES[label]}")
        axes[row_index, 0].axis("off")
        axes[row_index, 1].axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply RF-driven adaptive enhancement to DRIVE images.")
    parser.add_argument("--features", default="data/metadata/degradation_features.csv")
    parser.add_argument(
        "--model",
        default="results/stage5_random_forest/models/random_forest_degradation_classifier.joblib",
    )
    parser.add_argument("--config", default="config/stage6_enhancement.json")
    parser.add_argument("--output-root", default="data/processed/enhanced/adaptive")
    parser.add_argument("--manifest", default="data/metadata/adaptive_enhancement_manifest.csv")
    parser.add_argument("--results-root", default="results/stage6_adaptive_enhancement")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = Path(args.features).resolve()
    model_path = Path(args.model).resolve()
    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    results_root = Path(args.results_root).resolve()

    rows = load_rows(feature_path)
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    features = feature_matrix(rows, feature_columns)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    probability_index = {label: index for index, label in enumerate(model.classes_)}

    manifest_rows = []
    preview_rows = {}
    for index, (row, predicted_class, probability) in enumerate(
        zip(rows, predictions, probabilities), start=1
    ):
        source_path = Path(row["degraded_image_path"])
        image = read_color(str(source_path))
        fov_mask = read_fov_mask(row["fov_mask_path"], image.shape[:2])
        enhanced = enhance_by_prediction(image, predicted_class, config)
        enhanced = apply_inside_fov(image, enhanced, fov_mask)

        output_path = (
            output_root
            / row["split"]
            / row["degradation_type"]
            / f"{source_path.stem}_adaptive_pred-{predicted_class}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not output_path.exists():
            if not cv2.imwrite(str(output_path), enhanced):
                raise ValueError(f"Could not write image: {output_path}")

        confidence = float(np.max(probability))
        output_row = {
            **row,
            "predicted_degradation": predicted_class,
            "prediction_confidence": confidence,
            "enhancement_strategy": STRATEGY_NAMES[predicted_class],
            "enhanced_image_path": output_path.as_posix(),
        }
        for label in CLASS_ORDER:
            output_row[f"prob_{label}"] = float(probability[probability_index[label]])
        manifest_rows.append(output_row)

        if row["split"] == "test" and predicted_class not in preview_rows:
            preview_rows[predicted_class] = (image, enhanced)
        if index % 50 == 0 or index == len(rows):
            print(f"Enhanced {index}/{len(rows)} images")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    true_labels = np.asarray([row["degradation_type"] for row in rows])
    summary = {
        "images": len(rows),
        "routing_accuracy_all_splits": float(np.mean(predictions == true_labels)),
        "predicted_class_counts": dict(Counter(predictions)),
        "strategy_counts": dict(Counter(STRATEGY_NAMES[label] for label in predictions)),
        "config": config,
    }
    table_dir = results_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / "adaptive_enhancement_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    save_preview(results_root / "figures" / "adaptive_enhancement_samples.png", preview_rows)

    print(f"Manifest: {manifest_path}")
    print(f"Predicted classes: {summary['predicted_class_counts']}")
    print(f"Outputs: {output_root}")


if __name__ == "__main__":
    main()
