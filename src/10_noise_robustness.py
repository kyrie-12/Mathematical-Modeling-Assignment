import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUP_ORDER = ["no_enhancement", "fixed_enhancement", "adaptive_enhancement"]
GROUP_LABELS = {
    "no_enhancement": "No enhancement",
    "fixed_enhancement": "Fixed enhancement",
    "adaptive_enhancement": "Adaptive enhancement",
}
GROUP_COLORS = {
    "no_enhancement": "#4C78A8",
    "fixed_enhancement": "#E07A2D",
    "adaptive_enhancement": "#2A7F62",
}
METRICS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "hausdorff_distance",
    "hausdorff_95",
    "psnr",
    "ssim",
]


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


def noise_sigma(row: dict) -> int:
    match = re.search(r"sigma(\d+)", row["evaluated_image_path"])
    if match is None:
        raise ValueError(f"Could not parse noise sigma: {row['evaluated_image_path']}")
    return int(match.group(1))


def robustness_group(row: dict) -> tuple[str, str, int]:
    path = row["evaluated_image_path"]
    if row["degradation_type"] == "noise":
        sigma = noise_sigma(row)
        return "noise", f"sigma_{sigma}", sigma
    if row["degradation_type"] == "mixed":
        name = Path(path).stem
        components = sum(token in name for token in ["noise", "blur", "contrast"])
        if components >= 3:
            return "mixed", "strong_three_component", 2
        return "mixed", "medium_two_component", 1
    raise ValueError(f"Unsupported degradation type: {row['degradation_type']}")


def summarize(rows: list[dict], selected_h: dict[str, float]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["noise_sigma"], row["group"])].append(row)
    summary_rows = []
    for sigma in sorted({key[0] for key in buckets}):
        for group in GROUP_ORDER:
            values = buckets[(sigma, group)]
            summary = {
                "noise_sigma": sigma,
                "group": group,
                "samples": len(values),
                "adaptive_h": selected_h.get(str(sigma), "") if group == "adaptive_enhancement" else "",
            }
            for metric in METRICS:
                metric_values = np.asarray([float(row[metric]) for row in values])
                summary[f"{metric}_mean"] = float(np.mean(metric_values))
                summary[f"{metric}_std"] = float(np.std(metric_values, ddof=1))
            summary_rows.append(summary)
    return summary_rows


def summarize_paper_style(rows: list[dict], selected_h: dict[str, float]) -> list[dict]:
    buckets = defaultdict(list)
    orders = {}
    for row in rows:
        case, severity, order = robustness_group(row)
        row["robustness_case"] = case
        row["severity_label"] = severity
        row["severity_order"] = order
        buckets[(case, severity, row["group"])].append(row)
        orders[(case, severity)] = order

    summary_rows = []
    for case, severity in sorted(orders, key=lambda key: (key[0], orders[key])):
        for group in GROUP_ORDER:
            values = buckets[(case, severity, group)]
            summary = {
                "robustness_case": case,
                "severity_label": severity,
                "severity_order": orders[(case, severity)],
                "group": group,
                "samples": len(values),
                "adaptive_h": "",
            }
            if case == "noise" and group == "adaptive_enhancement":
                sigma = severity.replace("sigma_", "")
                summary["adaptive_h"] = selected_h.get(sigma, "")
            for metric in METRICS:
                metric_values = np.asarray([float(row[metric]) for row in values])
                summary[f"{metric}_mean"] = float(np.mean(metric_values))
                summary[f"{metric}_std"] = float(np.std(metric_values, ddof=1))
            summary_rows.append(summary)
    return summary_rows


def paired_comparisons(
    rows: list[dict],
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["noise_sigma"], row["image_id"])][row["group"]] = row
    rng = np.random.default_rng(seed)
    output_rows = []
    for sigma in sorted({key[0] for key in grouped}):
        image_ids = sorted(image_id for group_sigma, image_id in grouped if group_sigma == sigma)
        for baseline in ["no_enhancement", "fixed_enhancement"]:
            comparison = {
                "noise_sigma": sigma,
                "baseline": baseline,
                "samples": len(image_ids),
            }
            for metric in METRICS:
                adaptive_values = np.asarray(
                    [float(grouped[(sigma, image_id)]["adaptive_enhancement"][metric]) for image_id in image_ids]
                )
                baseline_values = np.asarray(
                    [float(grouped[(sigma, image_id)][baseline][metric]) for image_id in image_ids]
                )
                differences = adaptive_values - baseline_values
                bootstrap_indices = rng.integers(
                    0,
                    len(differences),
                    size=(bootstrap_samples, len(differences)),
                )
                bootstrap_means = np.mean(differences[bootstrap_indices], axis=1)
                lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
                try:
                    p_value = float(wilcoxon(differences).pvalue)
                except ValueError:
                    p_value = 1.0
                comparison[f"adaptive_{metric}"] = float(np.mean(adaptive_values))
                comparison[f"baseline_{metric}"] = float(np.mean(baseline_values))
                comparison[f"delta_{metric}"] = float(np.mean(differences))
                comparison[f"delta_{metric}_ci_low"] = float(lower)
                comparison[f"delta_{metric}_ci_high"] = float(upper)
                comparison[f"delta_{metric}_wilcoxon_p"] = p_value
            output_rows.append(comparison)
    return output_rows


def paper_style_paired_comparisons(
    rows: list[dict],
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    grouped = defaultdict(dict)
    orders = {}
    for row in rows:
        case, severity, order = robustness_group(row)
        match_key = row["evaluated_image_path"]
        if row["group"] != "no_enhancement":
            match_key = re.sub(r"/enhanced/(fixed|adaptive)/", "/degraded/", match_key)
            match_key = re.sub(r"_(fixed|adaptive_components-[0-9]+)\.png$", ".png", match_key)
        grouped[(case, severity, row["image_id"], Path(match_key).stem)][row["group"]] = row
        orders[(case, severity)] = order

    rng = np.random.default_rng(seed)
    output_rows = []
    for case, severity in sorted(orders, key=lambda key: (key[0], orders[key])):
        sample_keys = sorted(
            key for key in grouped if key[0] == case and key[1] == severity
        )
        complete_keys = [
            key for key in sample_keys if all(group in grouped[key] for group in GROUP_ORDER)
        ]
        for baseline in ["no_enhancement", "fixed_enhancement"]:
            comparison = {
                "robustness_case": case,
                "severity_label": severity,
                "severity_order": orders[(case, severity)],
                "baseline": baseline,
                "samples": len(complete_keys),
            }
            for metric in METRICS:
                adaptive_values = np.asarray(
                    [float(grouped[key]["adaptive_enhancement"][metric]) for key in complete_keys]
                )
                baseline_values = np.asarray(
                    [float(grouped[key][baseline][metric]) for key in complete_keys]
                )
                differences = adaptive_values - baseline_values
                bootstrap_indices = rng.integers(
                    0,
                    len(differences),
                    size=(bootstrap_samples, len(differences)),
                )
                bootstrap_means = np.mean(differences[bootstrap_indices], axis=1)
                lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
                try:
                    p_value = float(wilcoxon(differences).pvalue)
                except ValueError:
                    p_value = 1.0
                comparison[f"adaptive_{metric}"] = float(np.mean(adaptive_values))
                comparison[f"baseline_{metric}"] = float(np.mean(baseline_values))
                comparison[f"delta_{metric}"] = float(np.mean(differences))
                comparison[f"delta_{metric}_ci_low"] = float(lower)
                comparison[f"delta_{metric}_ci_high"] = float(upper)
                comparison[f"delta_{metric}_wilcoxon_p"] = p_value
            output_rows.append(comparison)
    return output_rows


def save_figure(path: Path, summary_rows: list[dict]) -> None:
    sigmas = sorted({int(row["noise_sigma"]) for row in summary_rows})
    panels = [
        ("dice", "Dice", True),
        ("iou", "IoU", True),
        ("hausdorff_95", "HD95 / pixel", False),
        ("ssim", "SSIM", True),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (metric, title, higher_is_better) in zip(axes.flat, panels):
        for group in GROUP_ORDER:
            selected = sorted(
                [row for row in summary_rows if row["group"] == group],
                key=lambda row: int(row["noise_sigma"]),
            )
            means = [float(row[f"{metric}_mean"]) for row in selected]
            stds = [float(row[f"{metric}_std"]) for row in selected]
            axis.errorbar(
                sigmas,
                means,
                yerr=stds,
                marker="o",
                capsize=3,
                linewidth=2,
                label=GROUP_LABELS[group],
                color=GROUP_COLORS[group],
            )
        direction = "higher is better" if higher_is_better else "lower is better"
        axis.set_title(f"{title} ({direction})")
        axis.set_xlabel("Gaussian noise sigma")
        axis.set_xticks(sigmas)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Stage 10 Noise-Severity Robustness", fontsize=15)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_paper_style_figure(path: Path, summary_rows: list[dict]) -> None:
    panels = [("noise", "Noise severity"), ("mixed", "Mixed degradation severity")]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    width = 0.24
    for axis, (case, title) in zip(axes, panels):
        selected = [row for row in summary_rows if row["robustness_case"] == case]
        severities = sorted(
            {(row["severity_label"], int(row["severity_order"])) for row in selected},
            key=lambda item: item[1],
        )
        x = np.arange(len(severities))
        for offset, group in zip([-width, 0, width], GROUP_ORDER):
            group_rows = {
                row["severity_label"]: row for row in selected if row["group"] == group
            }
            means = [float(group_rows[label]["dice_mean"]) for label, _ in severities]
            stds = [float(group_rows[label]["dice_std"]) for label, _ in severities]
            axis.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=3,
                label=GROUP_LABELS[group],
                color=GROUP_COLORS[group],
            )
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels([label.replace("_", "\n") for label, _ in severities])
        axis.set_ylabel("Dice")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.suptitle("Paper-style Robustness Grouping", fontsize=15)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def comparison_statement(row: dict) -> str:
    delta = float(row["delta_dice"])
    lower = float(row["delta_dice_ci_low"])
    upper = float(row["delta_dice_ci_high"])
    baseline = GROUP_LABELS[row["baseline"]].lower()
    if lower > 0:
        relation = "significantly higher than"
    elif upper < 0:
        relation = "significantly lower than"
    else:
        relation = "not significantly different from"
    return (
        f"adaptive Dice is {relation} {baseline} "
        f"(delta={delta:.4f}, 95% CI=[{lower:.4f}, {upper:.4f}])"
    )


def save_conclusion(
    path: Path,
    summary_rows: list[dict],
    comparison_rows: list[dict],
) -> None:
    by_key = {(int(row["noise_sigma"]), row["group"]): row for row in summary_rows}
    comparisons = {
        (int(row["noise_sigma"]), row["baseline"]): row for row in comparison_rows
    }
    lines = []
    for sigma in sorted({key[0] for key in by_key}):
        adaptive = by_key[(sigma, "adaptive_enhancement")]
        no_enhancement = by_key[(sigma, "no_enhancement")]
        fixed = by_key[(sigma, "fixed_enhancement")]
        scores = {
            "no enhancement": float(no_enhancement["dice_mean"]),
            "fixed enhancement": float(fixed["dice_mean"]),
            "adaptive enhancement": float(adaptive["dice_mean"]),
        }
        best = max(scores, key=scores.get)
        lines.append(
            f"sigma={sigma}: adaptive Dice={scores['adaptive enhancement']:.4f}, "
            f"no enhancement={scores['no enhancement']:.4f}, "
            f"fixed enhancement={scores['fixed enhancement']:.4f}; best={best}. "
            f"{comparison_statement(comparisons[(sigma, 'no_enhancement')])}; "
            f"{comparison_statement(comparisons[(sigma, 'fixed_enhancement')])}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_paper_style_conclusion(
    path: Path,
    summary_rows: list[dict],
    comparison_rows: list[dict],
) -> None:
    by_key = {
        (row["robustness_case"], row["severity_label"], row["group"]): row
        for row in summary_rows
    }
    comparisons = {
        (row["robustness_case"], row["severity_label"], row["baseline"]): row
        for row in comparison_rows
    }
    ordered = sorted(
        {
            (row["robustness_case"], row["severity_label"], int(row["severity_order"]))
            for row in summary_rows
        },
        key=lambda item: (item[0], item[2]),
    )
    lines = [
        "This paper-style robustness check groups noise by sigma and mixed degradation by component count.",
        "For the current dataset, two-component mixed samples are treated as medium mixed degradation, and three-component mixed samples are treated as strong mixed degradation.",
        "",
    ]
    for case, severity, _ in ordered:
        no_enhancement = by_key[(case, severity, "no_enhancement")]
        fixed = by_key[(case, severity, "fixed_enhancement")]
        adaptive = by_key[(case, severity, "adaptive_enhancement")]
        adaptive_vs_fixed = comparisons[(case, severity, "fixed_enhancement")]
        lines.append(
            f"{case}/{severity}: no Dice={float(no_enhancement['dice_mean']):.4f}, "
            f"fixed Dice={float(fixed['dice_mean']):.4f}, "
            f"adaptive Dice={float(adaptive['dice_mean']):.4f}, "
            f"adaptive-fixed delta={float(adaptive_vs_fixed['delta_dice']):.4f}, "
            f"95% CI=[{float(adaptive_vs_fixed['delta_dice_ci_low']):.4f}, "
            f"{float(adaptive_vs_fixed['delta_dice_ci_high']):.4f}]."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage 10 noise-severity robustness analysis.")
    parser.add_argument("--metrics", default="results/stage8_evaluation/tables/metrics_per_sample.csv")
    parser.add_argument("--enhancement-config", default="config/stage6_enhancement.json")
    parser.add_argument("--results-root", default="results/stage10_noise_robustness")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        row
        for row in load_csv(Path(args.metrics).resolve())
        if row["split"] == "test" and row["degradation_type"] == "noise"
    ]
    if not rows:
        raise ValueError("No test noise metrics found")
    for row in rows:
        row["noise_sigma"] = noise_sigma(row)
    with Path(args.enhancement_config).resolve().open(encoding="utf-8") as file:
        selected_h = json.load(file)["noise_severity"]["selected_h"]

    summary_rows = summarize(rows, selected_h)
    comparison_rows = paired_comparisons(rows, args.bootstrap_samples, args.seed)
    results_root = Path(args.results_root).resolve()
    table_dir = results_root / "tables"
    write_csv(table_dir / "noise_severity_summary.csv", summary_rows)
    write_csv(table_dir / "noise_severity_paired_comparisons.csv", comparison_rows)

    paper_style_rows = [
        row
        for row in load_csv(Path(args.metrics).resolve())
        if row["split"] == "test" and row["degradation_type"] in {"noise", "mixed"}
    ]
    paper_style_summary_rows = summarize_paper_style(paper_style_rows, selected_h)
    paper_style_comparison_rows = paper_style_paired_comparisons(
        paper_style_rows,
        args.bootstrap_samples,
        args.seed,
    )
    write_csv(table_dir / "paper_style_robustness_summary.csv", paper_style_summary_rows)
    write_csv(table_dir / "paper_style_robustness_paired_comparisons.csv", paper_style_comparison_rows)

    with (table_dir / "noise_robustness_protocol.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "split": "test",
                "independent_unit": "DRIVE image_id",
                "noise_sigmas": sorted({row["noise_sigma"] for row in rows}),
                "paper_style_groups": {
                    "noise": "grouped by Gaussian noise sigma",
                    "mixed": "two degradation components are treated as medium mixed; three degradation components are treated as strong mixed",
                },
                "adaptive_h": selected_h,
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    save_figure(results_root / "figures" / "noise_severity_robustness.png", summary_rows)
    save_paper_style_figure(
        results_root / "figures" / "paper_style_robustness.png",
        paper_style_summary_rows,
    )
    save_conclusion(
        table_dir / "noise_severity_conclusion.txt",
        summary_rows,
        comparison_rows,
    )
    save_paper_style_conclusion(
        table_dir / "paper_style_robustness_conclusion.txt",
        paper_style_summary_rows,
        paper_style_comparison_rows,
    )

    by_key = {(int(row["noise_sigma"]), row["group"]): row for row in summary_rows}
    for sigma in sorted({key[0] for key in by_key}):
        print(
            f"sigma={sigma}: no={float(by_key[(sigma, 'no_enhancement')]['dice_mean']):.4f}, "
            f"fixed={float(by_key[(sigma, 'fixed_enhancement')]['dice_mean']):.4f}, "
            f"adaptive={float(by_key[(sigma, 'adaptive_enhancement')]['dice_mean']):.4f}"
        )
    print("Paper-style robustness:")
    for row in paper_style_summary_rows:
        if row["group"] == "adaptive_enhancement":
            fixed = next(
                other
                for other in paper_style_summary_rows
                if other["robustness_case"] == row["robustness_case"]
                and other["severity_label"] == row["severity_label"]
                and other["group"] == "fixed_enhancement"
            )
            print(
                f"{row['robustness_case']}/{row['severity_label']}: "
                f"fixed={float(fixed['dice_mean']):.4f}, "
                f"adaptive={float(row['dice_mean']):.4f}"
            )
    print(f"Outputs: {results_root}")


if __name__ == "__main__":
    main()
