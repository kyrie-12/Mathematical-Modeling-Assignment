import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import shannon_entropy


FEATURE_COLUMNS = [
    "gray_mean",
    "gray_std",
    "green_mean",
    "green_std",
    "entropy",
    "intensity_range_90",
    "laplacian_variance",
    "edge_density",
    "high_frequency_energy",
    "noise_mad_estimate",
    "gradient_mean",
    "gradient_std",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
]


def read_image(path: str, flag: int) -> np.ndarray:
    image = cv2.imread(path, flag)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def analysis_mask(mask_path: str, shape: tuple[int, int]) -> np.ndarray:
    if not mask_path:
        return np.ones(shape, dtype=bool)
    mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 0).astype(np.uint8)
    eroded = cv2.erode(binary, np.ones((5, 5), dtype=np.uint8), iterations=1)
    return (eroded if eroded.any() else binary).astype(bool)


def extract_features(image_path: str, fov_mask_path: str) -> dict[str, float]:
    image = read_image(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    green = image[:, :, 1]
    roi_mask = analysis_mask(fov_mask_path, gray.shape)
    gray_roi = gray[roi_mask].astype(np.float32)
    green_roi = green[roi_mask].astype(np.float32)

    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0).astype(np.float32)
    high_frequency = gray.astype(np.float32) - blurred
    high_frequency_roi = high_frequency[roi_mask]
    centered_high_frequency = high_frequency_roi - np.median(high_frequency_roi)

    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)[roi_mask]

    p05, p95 = np.percentile(gray_roi, (5, 95))
    return {
        "gray_mean": float(gray_roi.mean()),
        "gray_std": float(gray_roi.std()),
        "green_mean": float(green_roi.mean()),
        "green_std": float(green_roi.std()),
        "entropy": float(shannon_entropy(gray_roi.astype(np.uint8))),
        "intensity_range_90": float(p95 - p05),
        "laplacian_variance": float(laplacian[roi_mask].var()),
        "edge_density": float(np.mean(edges[roi_mask] > 0)),
        "high_frequency_energy": float(np.mean(np.square(high_frequency_roi))),
        "noise_mad_estimate": float(np.median(np.abs(centered_high_frequency)) / 0.6745),
        "gradient_mean": float(gradient.mean()),
        "gradient_std": float(gradient.std()),
        "dark_pixel_ratio": float(np.mean(gray_roi < 50)),
        "bright_pixel_ratio": float(np.mean(gray_roi > 200)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract interpretable degradation features from DRIVE images.")
    parser.add_argument("--manifest", default="data/metadata/degradation_manifest.csv")
    parser.add_argument("--output", default="data/metadata/degradation_features.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    with manifest_path.open(encoding="utf-8", newline="") as file:
        manifest_rows = list(csv.DictReader(file))

    if not manifest_rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    output_rows = []
    for index, row in enumerate(manifest_rows, start=1):
        features = extract_features(row["degraded_image_path"], row["fov_mask_path"])
        output_rows.append({**row, **features})
        if index % 50 == 0 or index == len(manifest_rows):
            print(f"Extracted {index}/{len(manifest_rows)} images")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(manifest_rows[0].keys()) + FEATURE_COLUMNS
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Feature table: {output_path}")
    print(f"Rows: {len(output_rows)}, features: {len(FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()
