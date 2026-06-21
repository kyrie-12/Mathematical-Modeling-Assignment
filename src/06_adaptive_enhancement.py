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
COMPONENT_COLUMNS = ["has_noise", "has_blur", "has_low_contrast"]
STRATEGY_NAMES = {
    "000": "identity",
    "100": "non_local_means",
    "010": "unsharp_mask_then_mild_clahe",
    "001": "gamma_then_clahe",
    "110": "non_local_means_then_unsharp_mask",
    "101": "non_local_means_then_weak_clahe",
    "011": "unsharp_mask_then_gamma_then_mild_clahe",
    "111": "non_local_means_then_weak_clahe",
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


def select_nlm_h(predicted_noise_sigma: float, config: dict) -> float:
    selected = config["noise_severity"]["selected_h"]
    nearest_sigma = min(selected, key=lambda value: abs(float(value) - predicted_noise_sigma))
    return float(selected[nearest_sigma])


def apply_severity_nlm(
    image: np.ndarray,
    predicted_noise_sigma: float,
    config: dict,
) -> tuple[np.ndarray, float]:
    h = select_nlm_h(predicted_noise_sigma, config)
    if h <= 0:
        return image.copy(), h
    nlm_config = {**config["nlm"], "h": h, "h_color": h}
    return apply_nlm(image, nlm_config), h


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


def component_code(components: np.ndarray | list[int] | tuple[int, ...]) -> str:
    return "".join(str(int(value)) for value in components)


def components_to_class(components: np.ndarray | list[int] | tuple[int, ...]) -> str:
    active = [name for name, value in zip(("noise", "blur", "low_contrast"), components) if value]
    if not active:
        return "normal"
    if len(active) == 1:
        return active[0]
    return "mixed"


def enhance_by_components(
    image: np.ndarray,
    components: np.ndarray | list[int] | tuple[int, ...],
    config: dict,
    predicted_noise_sigma: float,
) -> tuple[np.ndarray, float]:
    code = component_code(components)
    if code == "000":
        return image.copy(), 0.0
    if code == "100":
        return apply_severity_nlm(image, predicted_noise_sigma, config)
    if code == "010":
        enhanced = apply_unsharp_mask(image, config["unsharp_mask"])
        return (
            apply_clahe(
                enhanced,
                config["clahe"]["mild_clip_limit"],
                config["clahe"]["tile_grid_size"],
            ),
            0.0,
        )
    if code == "001":
        enhanced = apply_gamma(image, config["gamma"]["value"])
        return (
            apply_clahe(
                enhanced,
                config["clahe"]["standard_clip_limit"],
                config["clahe"]["tile_grid_size"],
            ),
            0.0,
        )
    if code == "110":
        enhanced, h = apply_severity_nlm(image, predicted_noise_sigma, config)
        return apply_unsharp_mask(enhanced, config["unsharp_mask"]), h
    if code in {"101", "111"}:
        enhanced, h = apply_severity_nlm(image, predicted_noise_sigma, config)
        return (
            apply_clahe(
                enhanced,
                config["clahe"]["weak_clip_limit"],
                config["clahe"]["tile_grid_size"],
            ),
            h,
        )
    if code == "011":
        enhanced = apply_unsharp_mask(image, config["unsharp_mask"])
        enhanced = apply_gamma(enhanced, config["gamma"]["value"])
        return (
            apply_clahe(
                enhanced,
                config["clahe"]["mild_clip_limit"],
                config["clahe"]["tile_grid_size"],
            ),
            0.0,
        )
    raise ValueError(f"Unsupported component code: {code}")


def strategy_name(code: str, selected_h: float) -> str:
    base = STRATEGY_NAMES[code]
    if code[0] == "0":
        return base
    if selected_h <= 0:
        return base.replace("non_local_means", "identity_noise_fallback")
    return base.replace("non_local_means", f"non_local_means_h{selected_h:g}")


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


def save_preview(
    path: Path,
    preview_rows: dict[str, tuple[np.ndarray, np.ndarray, str]],
) -> None:
    codes = [code for code in STRATEGY_NAMES if code in preview_rows]
    figure, axes = plt.subplots(len(codes), 2, figsize=(9, 3 * len(codes)))
    axes = np.atleast_2d(axes)
    for row_index, code in enumerate(codes):
        before, after, applied_strategy = preview_rows[code]
        axes[row_index, 0].imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
        axes[row_index, 0].set_title(f"Predicted components {code}: before")
        axes[row_index, 1].imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
        strategy_title = applied_strategy.replace("_then_", "\n")
        axes[row_index, 1].set_title(f"Strategy:\n{strategy_title}", fontsize=9)
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
    component_models = model_bundle.get("component_models")
    component_columns = model_bundle.get("component_columns")
    component_thresholds = model_bundle.get("component_thresholds", {})
    if not component_models or component_columns != COMPONENT_COLUMNS:
        raise ValueError("Model bundle does not contain the expected multi-label component models")
    noise_severity_model = model_bundle.get("noise_severity_model")
    if noise_severity_model is None:
        raise ValueError("Model bundle does not contain a noise severity regressor")
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    features = feature_matrix(rows, feature_columns)
    legacy_predictions = model.predict(features)
    legacy_probabilities = model.predict_proba(features)
    probability_index = {label: index for index, label in enumerate(model.classes_)}
    component_probabilities = np.zeros((len(rows), len(COMPONENT_COLUMNS)), dtype=np.float64)
    for component_index, component in enumerate(COMPONENT_COLUMNS):
        component_model = component_models[component]
        positive_index = int(np.flatnonzero(component_model.classes_ == 1)[0])
        component_probabilities[:, component_index] = component_model.predict_proba(features)[
            :, positive_index
        ]
    thresholds = np.asarray(
        [float(component_thresholds.get(component, 0.5)) for component in COMPONENT_COLUMNS]
    )
    component_predictions = (component_probabilities >= thresholds).astype(np.uint8)
    predicted_noise_sigmas = noise_severity_model.predict(features)

    manifest_rows = []
    preview_rows = {}
    applied_strategies = []
    for index, (
        row,
        legacy_prediction,
        legacy_probability,
        predicted_components,
        component_probability,
        predicted_noise_sigma,
    ) in enumerate(
        zip(
            rows,
            legacy_predictions,
            legacy_probabilities,
            component_predictions,
            component_probabilities,
            predicted_noise_sigmas,
        ),
        start=1,
    ):
        source_path = Path(row["degraded_image_path"])
        image = read_color(str(source_path))
        fov_mask = read_fov_mask(row["fov_mask_path"], image.shape[:2])
        code = component_code(predicted_components)
        predicted_class = components_to_class(predicted_components)
        enhanced, selected_h = enhance_by_components(
            image,
            predicted_components,
            config,
            float(predicted_noise_sigma),
        )
        enhanced = apply_inside_fov(image, enhanced, fov_mask)
        applied_strategy = strategy_name(code, selected_h)
        applied_strategies.append(applied_strategy)

        output_path = (
            output_root
            / row["split"]
            / row["degradation_type"]
            / f"{source_path.stem}_adaptive_components-{code}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not output_path.exists():
            if not cv2.imwrite(str(output_path), enhanced):
                raise ValueError(f"Could not write image: {output_path}")

        confidence = float(np.mean(np.maximum(component_probability, 1.0 - component_probability)))
        output_row = {
            **row,
            "predicted_degradation": predicted_class,
            "legacy_predicted_degradation": legacy_prediction,
            "predicted_component_code": code,
            "predicted_noise_sigma": float(predicted_noise_sigma),
            "selected_nlm_h": selected_h,
            "prediction_confidence": confidence,
            "enhancement_strategy": applied_strategy,
            "enhanced_image_path": output_path.as_posix(),
        }
        for label in CLASS_ORDER:
            output_row[f"prob_{label}"] = float(
                legacy_probability[probability_index[label]]
            )
        for component_index, component in enumerate(COMPONENT_COLUMNS):
            output_row[f"pred_{component}"] = int(predicted_components[component_index])
            output_row[f"prob_{component}"] = float(component_probability[component_index])
        manifest_rows.append(output_row)

        if row["split"] == "test" and code not in preview_rows:
            preview_rows[code] = (image, enhanced, applied_strategy)
        if index % 50 == 0 or index == len(rows):
            print(f"Enhanced {index}/{len(rows)} images")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    true_labels = np.asarray([row["degradation_type"] for row in rows])
    true_components = np.asarray(
        [
            [
                int(bool(row["noise_sigma"])),
                int(bool(row["blur_kernel"])),
                int(bool(row["contrast_alpha"])),
            ]
            for row in rows
        ],
        dtype=np.uint8,
    )
    summary = {
        "images": len(rows),
        "legacy_routing_accuracy_all_splits": float(
            np.mean(legacy_predictions == true_labels)
        ),
        "multilabel_subset_accuracy_all_splits": float(
            np.mean(np.all(component_predictions == true_components, axis=1))
        ),
        "predicted_component_counts": dict(
            Counter(component_code(values) for values in component_predictions)
        ),
        "strategy_counts": dict(
            Counter(applied_strategies)
        ),
        "config": config,
    }
    table_dir = results_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / "adaptive_enhancement_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    save_preview(results_root / "figures" / "adaptive_enhancement_samples.png", preview_rows)

    print(f"Manifest: {manifest_path}")
    print(f"Predicted components: {summary['predicted_component_counts']}")
    print(f"Outputs: {output_root}")


if __name__ == "__main__":
    main()
