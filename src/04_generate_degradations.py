import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from skimage import exposure, img_as_ubyte
from skimage.util import random_noise


NOISE_LEVELS = (15, 30)
BLUR_SIGMAS = (1.2, 2.0)
CONTRAST_LEVELS = (0.45, 0.65)
MIXED_CONFIGS = (
    ("medium", 15, 1.2, 0.55),
    ("strong", 30, 2.0, 0.45),
)


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def save_png(path: Path, image: np.ndarray, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Could not write image: {path}")


def add_gaussian_noise(image: np.ndarray, sigma: int, rng: np.random.Generator) -> np.ndarray:
    noisy = random_noise(
        image,
        mode="gaussian",
        mean=0.0,
        var=(sigma / 255.0) ** 2,
        rng=rng,
        clip=True,
    )
    return img_as_ubyte(noisy)


def add_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def reduce_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    reduced = np.empty_like(image)
    for channel in range(image.shape[2]):
        values = image[:, :, channel]
        mean = float(values.mean())
        output_low = mean * (1.0 - alpha)
        output_high = mean + alpha * (255.0 - mean)
        reduced[:, :, channel] = exposure.rescale_intensity(
            values,
            in_range=(0, 255),
            out_range=(output_low, output_high),
        ).astype(np.uint8)
    return reduced


def find_companion(drive_root: Path, split: str, image_path: Path, kind: str) -> str:
    image_id = image_path.stem.split("_")[0]
    if kind == "manual":
        directory = drive_root / split / "1st_manual"
        stems = (f"{image_id}_manual1",)
    else:
        directory = drive_root / split / "mask"
        split_name = "training" if split == "training" else "test"
        stems = (f"{image_id}_{split_name}_mask",)

    for stem in stems:
        for suffix in (".gif", ".png", ".tif", ".tiff"):
            candidate = directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate.as_posix()
    return ""


def manifest_row(
    drive_root: Path,
    split: str,
    source: Path,
    output: Path,
    degradation_type: str,
    noise_sigma="",
    blur_sigma="",
    contrast_alpha="",
    mixed_mode="",
) -> dict:
    return {
        "image_id": source.stem,
        "split": "train" if split == "training" else "test",
        "original_image_path": source.as_posix(),
        "degraded_image_path": output.as_posix(),
        "manual_mask_path": find_companion(drive_root, split, source, "manual"),
        "fov_mask_path": find_companion(drive_root, split, source, "fov"),
        "degradation_type": degradation_type,
        "noise_sigma": noise_sigma,
        "blur_sigma": blur_sigma,
        "contrast_alpha": contrast_alpha,
        "mixed_mode": mixed_mode,
    }


def generate_one(
    drive_root: Path,
    output_root: Path,
    split: str,
    source: Path,
    rng: np.random.Generator,
    overwrite: bool,
) -> list[dict]:
    split_name = "train" if split == "training" else "test"
    image = load_rgb(source)
    rows = []

    output = output_root / split_name / "normal" / f"{source.stem}_normal.png"
    save_png(output, image, overwrite)
    rows.append(manifest_row(drive_root, split, source, output, "normal"))

    for sigma in NOISE_LEVELS:
        degraded = add_gaussian_noise(image, sigma, rng)
        output = output_root / split_name / "noise" / f"{source.stem}_noise_sigma{sigma}.png"
        save_png(output, degraded, overwrite)
        rows.append(manifest_row(drive_root, split, source, output, "noise", noise_sigma=sigma))

    for sigma in BLUR_SIGMAS:
        degraded = add_gaussian_blur(image, sigma)
        sigma_name = str(sigma).replace(".", "")
        output = output_root / split_name / "blur" / f"{source.stem}_blur_sigma{sigma_name}.png"
        save_png(output, degraded, overwrite)
        rows.append(manifest_row(drive_root, split, source, output, "blur", blur_sigma=sigma))

    for alpha in CONTRAST_LEVELS:
        degraded = reduce_contrast(image, alpha)
        alpha_name = str(alpha).replace(".", "")
        output = output_root / split_name / "low_contrast" / f"{source.stem}_lowcontrast_alpha{alpha_name}.png"
        save_png(output, degraded, overwrite)
        rows.append(manifest_row(drive_root, split, source, output, "low_contrast", contrast_alpha=alpha))

    for mode, noise_sigma, blur_sigma, alpha in MIXED_CONFIGS:
        degraded = image.copy()
        degraded = reduce_contrast(degraded, alpha)
        degraded = add_gaussian_blur(degraded, blur_sigma)
        degraded = add_gaussian_noise(degraded, noise_sigma, rng)
        output = output_root / split_name / "mixed" / f"{source.stem}_mixed_{mode}.png"
        save_png(output, degraded, overwrite)
        rows.append(
            manifest_row(
                drive_root,
                split,
                source,
                output,
                "mixed",
                noise_sigma=noise_sigma,
                blur_sigma=blur_sigma,
                contrast_alpha=alpha,
                mixed_mode=mode,
            )
        )

    return rows


def discover_images(drive_root: Path, split: str) -> list[Path]:
    image_dir = drive_root / split / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    images = []
    for pattern in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
        images.extend(image_dir.glob(pattern))
    return sorted(images)


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate labeled degradations for the DRIVE dataset.")
    parser.add_argument("--drive-root", default="data/raw/DRIVE")
    parser.add_argument("--output-root", default="data/processed/degraded")
    parser.add_argument("--manifest", default="data/metadata/degradation_manifest.csv")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drive_root = Path(args.drive_root).resolve()
    output_root = Path(args.output_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    rng = np.random.default_rng(args.seed)
    rows = []

    for split in ("training", "test"):
        for source in discover_images(drive_root, split):
            rows.extend(generate_one(drive_root, output_root, split, source, rng, args.overwrite))

    write_manifest(manifest_path, rows)
    counts = Counter((row["split"], row["degradation_type"]) for row in rows)
    print(f"Generated manifest rows: {len(rows)}")
    for key in sorted(counts):
        print(f"{key[0]:5s} {key[1]:13s}: {counts[key]}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
