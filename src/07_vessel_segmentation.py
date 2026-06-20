import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage import img_as_float32, img_as_ubyte
from skimage.restoration import denoise_nl_means, estimate_sigma

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUPS = ("no_enhancement", "fixed_enhancement", "adaptive_enhancement")


def read_color(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def read_fov_mask(path: str, shape: tuple[int, int]) -> np.ndarray:
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


def apply_nlm(image: np.ndarray, config: dict, fov_mask: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_float = img_as_float32(image_rgb)
    y_indices, x_indices = np.where(fov_mask)
    y0, y1 = y_indices.min(), y_indices.max() + 1
    x0, x1 = x_indices.min(), x_indices.max() + 1
    sigma_patch = image_float[y0:y1, x0:x1].copy()
    patch_mask = fov_mask[y0:y1, x0:x1]
    median_color = np.median(sigma_patch[patch_mask], axis=0)
    sigma_patch[~patch_mask] = median_color
    sigma = float(np.mean(estimate_sigma(sigma_patch, channel_axis=-1)))
    denoised = denoise_nl_means(
        image_float,
        h=float(config["h_factor"]) * sigma,
        sigma=sigma,
        fast_mode=bool(config["fast_mode"]),
        patch_size=int(config["patch_size"]),
        patch_distance=int(config["patch_distance"]),
        channel_axis=-1,
    )
    return cv2.cvtColor(img_as_ubyte(denoised), cv2.COLOR_RGB2BGR)


def apply_clahe(image: np.ndarray, config: dict) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    tile_size = config["tile_grid_size"]
    clahe = cv2.createCLAHE(
        clipLimit=float(config["standard_clip_limit"]),
        tileGridSize=(int(tile_size[0]), int(tile_size[1])),
    )
    enhanced_lightness = clahe.apply(lightness)
    return cv2.cvtColor(
        cv2.merge((enhanced_lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )


def apply_unsharp_mask(image: np.ndarray, config: dict) -> np.ndarray:
    sigma = float(config["sigma"])
    amount = float(config["amount"])
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def fixed_enhancement(
    image: np.ndarray,
    fov_mask: np.ndarray,
    enhancement_config: dict,
) -> np.ndarray:
    enhanced = apply_nlm(image, enhancement_config["nlm"], fov_mask)
    enhanced = apply_clahe(enhanced, enhancement_config["clahe"])
    enhanced = apply_unsharp_mask(enhanced, enhancement_config["unsharp_mask"])
    return apply_inside_fov(image, enhanced, fov_mask)


def remove_small_components(binary: np.ndarray, minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == label] = 255
    return cleaned


def segment_vessels(
    image: np.ndarray,
    fov_mask: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, float]:
    green = image[:, :, 1]
    green_float = green.astype(np.float32)
    background = cv2.GaussianBlur(
        green_float,
        (0, 0),
        sigmaX=float(config["background_sigma"]),
        sigmaY=float(config["background_sigma"]),
    )
    vessel_response = np.maximum(background - green_float, 0.0)
    roi_values = vessel_response[fov_mask]
    low, high = np.percentile(roi_values, (1, 99))
    if high <= low:
        normalized = np.zeros_like(green, dtype=np.uint8)
    else:
        normalized = np.clip((vessel_response - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    normalized[~fov_mask] = 0
    tile_size = config["normalization_tile_grid_size"]
    normalizer = cv2.createCLAHE(
        clipLimit=float(config["normalization_clahe_clip_limit"]),
        tileGridSize=(int(tile_size[0]), int(tile_size[1])),
    )
    normalized = normalizer.apply(normalized)
    normalized[~fov_mask] = 0

    otsu_input = normalized[fov_mask].reshape(-1, 1)
    otsu_threshold, _ = cv2.threshold(
        otsu_input,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    final_threshold = float(otsu_threshold) / float(config["threshold_multiplier"])
    binary = np.zeros_like(normalized, dtype=np.uint8)
    binary[(normalized >= final_threshold) & fov_mask] = 255

    open_size = int(config["opening_kernel_size"])
    close_size = int(config["closing_kernel_size"])
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    binary = remove_small_components(binary, int(config["minimum_component_area"]))
    binary[~fov_mask] = 0
    return binary, final_threshold


def save_image(path: Path, image: np.ndarray, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Could not write image: {path}")


def load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def save_preview(path: Path, examples: dict[str, dict[str, np.ndarray]]) -> None:
    labels = [label for label in ("normal", "noise", "blur", "low_contrast", "mixed") if label in examples]
    figure, axes = plt.subplots(len(labels), 4, figsize=(13, 3 * len(labels)))
    axes = np.atleast_2d(axes)
    for row_index, label in enumerate(labels):
        item = examples[label]
        axes[row_index, 0].imshow(cv2.cvtColor(item["input"], cv2.COLOR_BGR2RGB))
        axes[row_index, 0].set_title(f"{label}: degraded")
        axes[row_index, 1].imshow(item["no_enhancement"], cmap="gray")
        axes[row_index, 1].set_title("No enhancement mask")
        axes[row_index, 2].imshow(item["fixed_enhancement"], cmap="gray")
        axes[row_index, 2].set_title("Fixed enhancement mask")
        axes[row_index, 3].imshow(item["adaptive_enhancement"], cmap="gray")
        axes[row_index, 3].set_title("Adaptive enhancement mask")
        for axis in axes[row_index]:
            axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed DRIVE vessel segmentation protocol.")
    parser.add_argument("--manifest", default="data/metadata/adaptive_enhancement_manifest.csv")
    parser.add_argument("--enhancement-config", default="config/stage6_enhancement.json")
    parser.add_argument("--segmentation-config", default="config/stage7_segmentation.json")
    parser.add_argument("--fixed-root", default="data/processed/enhanced/fixed")
    parser.add_argument("--mask-root", default="data/processed/segmentation")
    parser.add_argument("--output-manifest", default="data/metadata/segmentation_manifest.csv")
    parser.add_argument("--results-root", default="results/stage7_segmentation")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest(Path(args.manifest).resolve())
    with Path(args.enhancement_config).resolve().open(encoding="utf-8") as file:
        enhancement_config = json.load(file)
    with Path(args.segmentation_config).resolve().open(encoding="utf-8") as file:
        stage7_config = json.load(file)
    segmentation_config = stage7_config["segmentation"]
    fixed_root = Path(args.fixed_root).resolve()
    mask_root = Path(args.mask_root).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    results_root = Path(args.results_root).resolve()

    output_rows = []
    previews = {}
    vessel_ratios = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        degraded = read_color(row["degraded_image_path"])
        adaptive = read_color(row["enhanced_image_path"])
        fov_mask = read_fov_mask(row["fov_mask_path"], degraded.shape[:2])
        fixed = fixed_enhancement(degraded, fov_mask, enhancement_config)

        source_stem = Path(row["degraded_image_path"]).stem
        fixed_path = fixed_root / row["split"] / row["degradation_type"] / f"{source_stem}_fixed.png"
        save_image(fixed_path, fixed, args.overwrite)

        images = {
            "no_enhancement": degraded,
            "fixed_enhancement": fixed,
            "adaptive_enhancement": adaptive,
        }
        masks = {}
        thresholds = {}
        mask_paths = {}
        for group, image in images.items():
            mask, threshold = segment_vessels(image, fov_mask, segmentation_config)
            mask_path = mask_root / group / row["split"] / row["degradation_type"] / f"{source_stem}_{group}_mask.png"
            save_image(mask_path, mask, args.overwrite)
            masks[group] = mask
            thresholds[group] = threshold
            mask_paths[group] = mask_path
            vessel_ratios[group].append(float(np.mean(mask[fov_mask] > 0)))

        output_rows.append(
            {
                "image_id": row["image_id"],
                "split": row["split"],
                "degradation_type": row["degradation_type"],
                "predicted_degradation": row["predicted_degradation"],
                "manual_mask_path": row["manual_mask_path"],
                "fov_mask_path": row["fov_mask_path"],
                "degraded_image_path": row["degraded_image_path"],
                "fixed_image_path": fixed_path.as_posix(),
                "adaptive_image_path": row["enhanced_image_path"],
                "no_enhancement_mask_path": mask_paths["no_enhancement"].as_posix(),
                "fixed_enhancement_mask_path": mask_paths["fixed_enhancement"].as_posix(),
                "adaptive_enhancement_mask_path": mask_paths["adaptive_enhancement"].as_posix(),
                "no_enhancement_threshold": thresholds["no_enhancement"],
                "fixed_enhancement_threshold": thresholds["fixed_enhancement"],
                "adaptive_enhancement_threshold": thresholds["adaptive_enhancement"],
            }
        )

        if row["split"] == "test" and row["degradation_type"] not in previews:
            previews[row["degradation_type"]] = {"input": degraded, **masks}
        if index % 25 == 0 or index == len(rows):
            print(f"Segmented {index}/{len(rows)} samples")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "samples": len(output_rows),
        "groups": list(GROUPS),
        "mean_vessel_pixel_ratio": {
            group: float(np.mean(values)) for group, values in vessel_ratios.items()
        },
        "segmentation_config": segmentation_config,
        "fixed_enhancement_order": stage7_config["fixed_enhancement_order"],
    }
    table_dir = results_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / "segmentation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    save_preview(results_root / "figures" / "segmentation_samples.png", previews)

    print(f"Segmentation manifest: {output_manifest}")
    print(f"Mask root: {mask_root}")


if __name__ == "__main__":
    main()
