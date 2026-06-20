import argparse
import csv
import json
import os
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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
CLASS_ORDER = ["normal", "noise", "blur", "low_contrast", "mixed"]


def load_feature_table(path: Path):
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Feature table is empty: {path}")

    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]

    def arrays(selected_rows):
        x = np.asarray([[float(row[name]) for name in FEATURE_COLUMNS] for row in selected_rows], dtype=float)
        y = np.asarray([row["degradation_type"] for row in selected_rows])
        return x, y

    return rows, train_rows, test_rows, *arrays(train_rows), *arrays(test_rows)


def evaluate_model(name, model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=CLASS_ORDER,
            output_dict=True,
            zero_division=0,
        ),
    }
    return model, predictions, metrics


def save_confusion_matrix(path: Path, y_true, predictions, title: str) -> None:
    matrix = confusion_matrix(y_true, predictions, labels=CLASS_ORDER)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_ORDER)
    figure, axis = plt.subplots(figsize=(8, 7))
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_feature_importance(path: Path, model: RandomForestClassifier) -> None:
    order = np.argsort(model.feature_importances_)
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(np.asarray(FEATURE_COLUMNS)[order], model.feature_importances_[order], color="#397367")
    axis.set_xlabel("Importance")
    axis.set_title("Random Forest Feature Importance")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF and SVM degradation classifiers.")
    parser.add_argument("--features", default="data/metadata/degradation_features.csv")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = Path(args.features).resolve()
    results_root = Path(args.results_root).resolve()
    model_dir = results_root / "models"
    table_dir = results_root / "tables"
    figure_dir = results_root / "figures"
    for directory in (model_dir, table_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows, train_rows, test_rows, x_train, y_train, x_test, y_test = load_feature_table(feature_path)
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=1,
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=args.seed,
                    ),
                ),
            ]
        ),
    }

    all_metrics = {
        "dataset": {
            "train_samples": len(train_rows),
            "test_samples": len(test_rows),
            "feature_count": len(FEATURE_COLUMNS),
            "features": FEATURE_COLUMNS,
        },
        "models": {},
    }
    prediction_columns = ["image_id", "degraded_image_path", "true_label"]
    prediction_rows = [
        {
            "image_id": row["image_id"],
            "degraded_image_path": row["degraded_image_path"],
            "true_label": row["degradation_type"],
        }
        for row in test_rows
    ]
    fitted_models = {}

    for name, model in models.items():
        fitted, predictions, metrics = evaluate_model(name, model, x_train, y_train, x_test, y_test)
        fitted_models[name] = fitted
        all_metrics["models"][name] = metrics
        joblib.dump(fitted, model_dir / f"{name}.joblib")
        save_confusion_matrix(
            figure_dir / f"confusion_matrix_{name}.png",
            y_test,
            predictions,
            f"{name.replace('_', ' ').title()} Confusion Matrix",
        )
        prediction_column = f"prediction_{name}"
        prediction_columns.append(prediction_column)
        for row, prediction in zip(prediction_rows, predictions):
            row[prediction_column] = prediction
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    save_feature_importance(figure_dir / "random_forest_feature_importance.png", fitted_models["random_forest"])
    best_name = max(all_metrics["models"], key=lambda name: all_metrics["models"][name]["macro_f1"])
    joblib.dump(fitted_models[best_name], model_dir / "best_degradation_classifier.joblib")
    all_metrics["best_model"] = best_name

    with (table_dir / "degradation_classifier_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(all_metrics, file, ensure_ascii=False, indent=2)
    with (table_dir / "degradation_classifier_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=prediction_columns)
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"Best model: {best_name}")
    print(f"Results: {results_root}")


if __name__ == "__main__":
    main()
