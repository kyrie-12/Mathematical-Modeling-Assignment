import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


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
CLASS_ORDER = ["normal", "noise", "blur", "low_contrast", "mixed"]


def load_feature_table(path: Path):
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Feature table is empty: {path}")

    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if not train_rows or not test_rows:
        raise ValueError("Feature table must contain both train and test rows")

    def to_arrays(selected_rows):
        features = np.asarray(
            [[float(row[column]) for column in FEATURE_COLUMNS] for row in selected_rows],
            dtype=np.float64,
        )
        labels = np.asarray([row["degradation_type"] for row in selected_rows])
        if not np.isfinite(features).all():
            raise ValueError("Feature table contains NaN or infinite values")
        return features, labels

    x_train, y_train = to_arrays(train_rows)
    x_test, y_test = to_arrays(test_rows)
    return train_rows, test_rows, x_train, y_train, x_test, y_test


def distribution(labels) -> dict[str, int]:
    counts = Counter(labels)
    return {label: int(counts.get(label, 0)) for label in CLASS_ORDER}


def save_confusion_matrix(path: Path, y_true, y_pred, normalize=None) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER, normalize=normalize)
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_ORDER)
    figure, axis = plt.subplots(figsize=(8, 7))
    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        values_format=".2f" if normalize else "d",
    )
    axis.set_title("Random Forest Confusion Matrix" + (" (Normalized)" if normalize else ""))
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_feature_importance(table_path: Path, figure_path: Path, model) -> None:
    pairs = sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda item: item[1], reverse=True)
    with table_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["feature", "importance"])
        writer.writerows((name, float(value)) for name, value in pairs)

    names = [name for name, _ in reversed(pairs)]
    values = [value for _, value in reversed(pairs)]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(names, values, color="#397367")
    axis.set_xlabel("Importance")
    axis.set_title("Random Forest Feature Importance")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def write_classification_report(path: Path, report: dict) -> None:
    rows = []
    for label in CLASS_ORDER:
        values = report[label]
        rows.append(
            {
                "class": label,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1_score": values["f1-score"],
                "support": int(values["support"]),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Random Forest degradation classifier.")
    parser.add_argument("--features", default="data/metadata/degradation_features.csv")
    parser.add_argument("--output-root", default="results/stage5_random_forest")
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = Path(args.features).resolve()
    output_root = Path(args.output_root).resolve()
    model_dir = output_root / "models"
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    for directory in (model_dir, table_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_rows, test_rows, x_train, y_train, x_test, y_test = load_feature_table(feature_path)
    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": "RandomForestClassifier",
        "parameters": model.get_params(),
        "features": FEATURE_COLUMNS,
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "train_distribution": distribution(y_train),
        "test_distribution": distribution(y_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }

    model_bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "class_order": CLASS_ORDER,
        "seed": args.seed,
    }
    joblib.dump(model_bundle, model_dir / "random_forest_degradation_classifier.joblib")

    with (table_dir / "random_forest_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    write_classification_report(table_dir / "random_forest_classification_report.csv", report)

    probability_index = {label: index for index, label in enumerate(model.classes_)}
    prediction_fields = [
        "image_id",
        "degraded_image_path",
        "true_label",
        "predicted_label",
        *[f"prob_{label}" for label in CLASS_ORDER],
    ]
    with (table_dir / "random_forest_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=prediction_fields)
        writer.writeheader()
        for row, predicted, probability in zip(test_rows, predictions, probabilities):
            output_row = {
                "image_id": row["image_id"],
                "degraded_image_path": row["degraded_image_path"],
                "true_label": row["degradation_type"],
                "predicted_label": predicted,
            }
            for label in CLASS_ORDER:
                output_row[f"prob_{label}"] = float(probability[probability_index[label]])
            writer.writerow(output_row)

    save_confusion_matrix(figure_dir / "random_forest_confusion_matrix.png", y_test, predictions)
    save_confusion_matrix(
        figure_dir / "random_forest_confusion_matrix_normalized.png",
        y_test,
        predictions,
        normalize="true",
    )
    save_feature_importance(
        table_dir / "random_forest_feature_importance.csv",
        figure_dir / "random_forest_feature_importance.png",
        model,
    )

    print(f"Training samples: {len(train_rows)}")
    print(f"Test samples: {len(test_rows)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Outputs: {output_root}")


if __name__ == "__main__":
    main()
