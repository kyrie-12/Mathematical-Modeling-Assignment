import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import shannon_entropy


FEATURE_COLUMNS = [
    "gray_mean",
    "gray_variance",
    "gray_skewness",
    "gray_kurtosis",
    "entropy",
    "intensity_range_90",
    "local_contrast_std",
    "laplacian_variance",
    "gradient_mean",
    "edge_density",
    "high_frequency_energy_ratio",
    "local_residual_variance",
    "smoothing_difference_mean",
    "smoothing_difference_std",
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
    roi_mask = analysis_mask(fov_mask_path, gray.shape)
    gray_roi = gray[roi_mask].astype(np.float32)
    gray_mean = float(gray_roi.mean())
    centered = gray_roi - gray_mean
    gray_std = float(gray_roi.std())
    safe_std = max(gray_std, 1e-6)
    skewness = float(np.mean(np.power(centered / safe_std, 3)))
    kurtosis = float(np.mean(np.power(centered / safe_std, 4)) - 3.0)

    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    gray_float = gray.astype(np.float32)
    blurred = cv2.GaussianBlur(gray_float, (5, 5), sigmaX=0)
    residual = gray_float - blurred
    residual_roi = residual[roi_mask]

    local_mean = cv2.GaussianBlur(gray_float, (0, 0), sigmaX=3.0)
    local_square_mean = cv2.GaussianBlur(np.square(gray_float), (0, 0), sigmaX=3.0)
    local_std = np.sqrt(np.maximum(local_square_mean - np.square(local_mean), 0.0))

    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)[roi_mask]

    p05, p95 = np.percentile(gray_roi, (5, 95))
    y_indices, x_indices = np.where(roi_mask)
    y0, y1 = y_indices.min(), y_indices.max() + 1
    x0, x1 = x_indices.min(), x_indices.max() + 1
    fft_patch = gray_float[y0:y1, x0:x1].copy()
    patch_mask = roi_mask[y0:y1, x0:x1]
    fft_patch[~patch_mask] = gray_mean
    window = np.outer(np.hanning(fft_patch.shape[0]), np.hanning(fft_patch.shape[1]))
    spectrum = np.square(np.abs(np.fft.fftshift(np.fft.fft2((fft_patch - gray_mean) * window))))
    center_y, center_x = spectrum.shape[0] // 2, spectrum.shape[1] // 2
    half_y, half_x = max(1, spectrum.shape[0] // 8), max(1, spectrum.shape[1] // 8)
    low_frequency_energy = spectrum[
        center_y - half_y : center_y + half_y,
        center_x - half_x : center_x + half_x,
    ].sum()
    total_frequency_energy = max(float(spectrum.sum()), 1e-12)
    high_frequency_ratio = float((total_frequency_energy - low_frequency_energy) / total_frequency_energy)

    return {
        "gray_mean": gray_mean,
        "gray_variance": float(gray_roi.var()),
        "gray_skewness": skewness,
        "gray_kurtosis": kurtosis,
        "entropy": float(shannon_entropy(gray_roi.astype(np.uint8))),
        "intensity_range_90": float(p95 - p05),
        "local_contrast_std": float(local_std[roi_mask].std()),
        "laplacian_variance": float(laplacian[roi_mask].var()),
        "gradient_mean": float(gradient.mean()),
        "edge_density": float(np.mean(edges[roi_mask] > 0)),
        "high_frequency_energy_ratio": high_frequency_ratio,
        "local_residual_variance": float(residual_roi.var()),
        "smoothing_difference_mean": float(np.mean(np.abs(residual_roi))),
        "smoothing_difference_std": float(residual_roi.std()),
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
