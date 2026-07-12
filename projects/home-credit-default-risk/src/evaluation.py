from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    model: Any,
    X: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate a fitted binary classification model.

    Parameters
    ----------
    model:
        A fitted classifier or Pipeline supporting predict_proba().
    X:
        Feature data used for evaluation.
    y_true:
        Actual binary target values.
    threshold:
        Probability threshold used to convert probabilities into class labels.

    Returns
    -------
    dict
        Dictionary containing probabilities, predictions and evaluation metrics.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    if not hasattr(model, "predict_proba"):
        raise TypeError("The provided model must support predict_proba().")

    probabilities = model.predict_proba(X)

    if probabilities.shape[1] != 2:
        raise ValueError(
            "evaluate_classifier currently supports binary classification only."
        )

    positive_probabilities = probabilities[:, 1]

    predictions = (positive_probabilities >= threshold).astype(int)

    matrix = confusion_matrix(y_true, predictions)

    metrics = {
        "threshold": threshold,
        "roc_auc": roc_auc_score(
            y_true,
            positive_probabilities,
        ),
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
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
        "confusion_matrix": matrix,
        "classification_report": classification_report(
            y_true,
            predictions,
            zero_division=0,
        ),
        "predictions": predictions,
        "probabilities": positive_probabilities,
    }

    return metrics


# Cross validation model comparison function
def compare_model_cv(
    name: str,
    model_pipeline: Any,
    X,
    y,
    cv,
    scoring: str = "roc_auc",
) -> dict[str, float | str]:
    """
    Evaluate one model Pipeline using cross-validation.

    Returns the model name, mean score, standard deviation,
    minimum score and maximum score.
    """

    scores = cross_val_score(
        estimator=model_pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    return {
        "model": name,
        "mean_cv_roc_auc": scores.mean(),
        "std_cv_roc_auc": scores.std(),
        "min_cv_roc_auc": scores.min(),
        "max_cv_roc_auc": scores.max(),
    }
