"""Metrics, model selection and external evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pipeline.config import DEFAULT_CONFIG, PipelineConfig
from pipeline.contracts import (
    DataBundle,
    ExternalEvaluation,
    ModelArtifact,
    PreprocessingBundle,
    SelectionResult,
)


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_true, prediction, zero_division=0),
        "F0.5": fbeta_score(
            y_true,
            prediction,
            beta=0.5,
            zero_division=0,
        ),
        "Balanced Accuracy": balanced_accuracy_score(y_true, prediction),
        "ROC-AUC": roc_auc_score(y_true, probability),
        "PR-AUC": average_precision_score(y_true, probability),
    }


def build_validation_results(
    artifacts: dict[str, ModelArtifact],
    y_validation: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Create one comparable validation row per model artifact."""

    records: list[dict[str, object]] = []
    for model_name, artifact in artifacts.items():
        if artifact.validation_probability is None:
            raise ValueError(f"{model_name} has no validation probabilities.")
        records.append(
            {
                "Model": model_name,
                "Algorithm": artifact.algorithm,
                "Tuned": artifact.tuned,
                "Feature set": artifact.feature_set,
                "Imbalance strategy": artifact.imbalance_strategy,
                "Oversampling": artifact.oversampled,
                "Training rows": artifact.training_rows,
                "Best iteration": artifact.best_iteration,
                "Training seconds": artifact.training_seconds,
                "Search seconds": artifact.tuning_seconds,
                "CV PR-AUC mean": artifact.cv_pr_auc_mean,
                "CV PR-AUC std": artifact.cv_pr_auc_std,
                **classification_metrics(
                    y_validation,
                    artifact.validation_probability,
                    threshold=threshold,
                ),
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values(
            ["Tuned", "CV PR-AUC mean", "PR-AUC"],
            ascending=[False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def select_tuned_model(
    validation_results: pd.DataFrame,
    artifacts: dict[str, ModelArtifact],
    y_validation: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> SelectionResult:
    """Select by CV PR-AUC equivalence, then operating-point validation F1."""

    ranking = (
        validation_results.loc[validation_results["Tuned"]]
        .sort_values(
            ["CV PR-AUC mean", "F1", "PR-AUC"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    if ranking.empty:
        raise ValueError("No tuned models are available for selection.")

    best_cv_pr_auc = float(ranking.iloc[0]["CV PR-AUC mean"])
    shortlist = ranking.loc[
        ranking["CV PR-AUC mean"]
        >= best_cv_pr_auc - config.cv_tie_tolerance
    ].copy()
    selected_model_name = (
        shortlist.sort_values(
            ["F1", "PR-AUC", "CV PR-AUC mean"],
            ascending=False,
        )
        .iloc[0]["Model"]
    )
    artifact = artifacts[str(selected_model_name)]
    if artifact.validation_probability is None:
        raise ValueError("Selected artifact has no validation probabilities.")

    precision_values, recall_values, threshold_values = precision_recall_curve(
        y_validation,
        artifact.validation_probability,
    )
    precision_at_threshold = precision_values[:-1]
    recall_at_threshold = recall_values[:-1]
    beta = 0.5
    f05_values = (
        (1 + beta**2) * precision_at_threshold * recall_at_threshold
        / (
            beta**2 * precision_at_threshold
            + recall_at_threshold
            + 1e-12
        )
    )
    best_threshold_index = int(np.nanargmax(f05_values))
    validation_f05_threshold = float(threshold_values[best_threshold_index])
    operating_threshold = config.operating_threshold

    threshold_comparison = pd.DataFrame(
        [
            {
                "Decision rule": "Default threshold",
                **classification_metrics(
                    y_validation,
                    artifact.validation_probability,
                    threshold=0.50,
                ),
            },
            {
                "Decision rule": "Validation-best F0.5 reference",
                **classification_metrics(
                    y_validation,
                    artifact.validation_probability,
                    threshold=validation_f05_threshold,
                ),
            },
            {
                "Decision rule": "Business operating threshold",
                **classification_metrics(
                    y_validation,
                    artifact.validation_probability,
                    threshold=operating_threshold,
                ),
            },
        ]
    )
    return SelectionResult(
        selected_model_name=str(selected_model_name),
        artifact=artifact,
        ranking=ranking,
        shortlist=shortlist,
        operating_threshold=operating_threshold,
        validation_f05_threshold=validation_f05_threshold,
        threshold_comparison=threshold_comparison,
        precision_curve=precision_at_threshold,
        recall_curve=recall_at_threshold,
        threshold_curve=threshold_values,
        f05_curve=f05_values,
    )


def evaluate_external_test(
    selection: SelectionResult,
    data: DataBundle,
    preprocessing: PreprocessingBundle,
) -> ExternalEvaluation:
    """Evaluate the frozen selected model on the untouched external split."""

    feature_set = selection.artifact.feature_set
    probability = selection.artifact.predict_probability(
        preprocessing.matrices[feature_set]["test"]
    )
    prediction = (
        probability >= selection.operating_threshold
    ).astype(int)
    confusion = confusion_matrix(
        data.y_test,
        prediction,
        labels=[0, 1],
    )
    results = pd.DataFrame(
        [
            {
                "Model": selection.selected_model_name,
                "Algorithm": selection.artifact.algorithm,
                "Feature set": feature_set,
                "Rows": len(data.y_test),
                "Actual churn rate": data.y_test.mean(),
                **classification_metrics(
                    data.y_test,
                    probability,
                    threshold=selection.operating_threshold,
                ),
            }
        ]
    )
    return ExternalEvaluation(
        results=results,
        probability=probability,
        prediction=prediction,
        confusion=confusion,
    )

