from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluate binary probabilities at one classification threshold."""

    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "precision_class_1": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall_class_1": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1_class_1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }


def find_best_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    min_precision: float | None = None,
) -> dict[str, float]:
    """Find the threshold that maximizes F1, optionally under precision constraint."""

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    rows = []

    for threshold, threshold_precision, threshold_recall in zip(
        thresholds,
        precision[:-1],
        recall[:-1],
    ):
        if threshold_precision + threshold_recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2
                * threshold_precision
                * threshold_recall
                / (threshold_precision + threshold_recall)
            )

        rows.append(
            {
                "threshold": float(threshold),
                "precision_class_1": float(threshold_precision),
                "recall_class_1": float(threshold_recall),
                "f1_class_1": float(f1),
            }
        )

    threshold_df = pd.DataFrame(rows)

    if min_precision is not None:
        constrained = threshold_df[
            threshold_df["precision_class_1"] >= min_precision
        ].copy()

        if not constrained.empty:
            best_row = constrained.sort_values(
                ["recall_class_1", "f1_class_1"],
                ascending=False,
            ).iloc[0]

            return {
                "threshold": float(best_row["threshold"]),
                "selection_rule": f"max recall with precision >= {min_precision}",
                "validation_precision_class_1": float(best_row["precision_class_1"]),
                "validation_recall_class_1": float(best_row["recall_class_1"]),
                "validation_f1_class_1": float(best_row["f1_class_1"]),
            }

    best_row = threshold_df.sort_values(
        "f1_class_1",
        ascending=False,
    ).iloc[0]

    return {
        "threshold": float(best_row["threshold"]),
        "selection_rule": "max f1",
        "validation_precision_class_1": float(best_row["precision_class_1"]),
        "validation_recall_class_1": float(best_row["recall_class_1"]),
        "validation_f1_class_1": float(best_row["f1_class_1"]),
    }


def get_positive_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return class-1 probabilities from a fitted binary classifier."""

    probabilities = model.predict_proba(X)

    if probabilities.shape[1] != 2:
        raise ValueError("Expected binary classifier probabilities.")

    return probabilities[:, 1]
