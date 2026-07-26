"""All report plots used by the orchestration notebook."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from pipeline.contracts import (
    ExternalEvaluation,
    FeatureImportanceResult,
    ModelArtifact,
    SelectionResult,
)


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")


def plot_validation_comparison(
    validation_results: pd.DataFrame,
    artifacts: dict[str, ModelArtifact],
    y_validation: pd.Series,
    threshold: float,
) -> Figure:
    """Plot metrics, ROC curves and PR curves for the three tuned models."""

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
    ]
    tuned_results = validation_results.loc[
        validation_results["Tuned"]
    ].copy()
    tuned_artifacts = {
        name: artifact
        for name, artifact in artifacts.items()
        if artifact.tuned
    }
    validation_long = tuned_results.melt(
        id_vars="Model",
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Score",
    )

    fig, axes = plt.subplots(1, 3, figsize=(25, 7))
    sns.barplot(
        data=validation_long,
        x="Metric",
        y="Score",
        hue="Model",
        ax=axes[0],
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Validation metrics at threshold {threshold:.2f}")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(title="", fontsize=8)

    for model_name, artifact in tuned_artifacts.items():
        if artifact.validation_probability is None:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_validation,
            artifact.validation_probability,
        )
        roc_auc = roc_auc_score(
            y_validation,
            artifact.validation_probability,
        )
        axes[1].plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{model_name} ({roc_auc:.3f})",
        )
    axes[1].plot([0, 1], [0, 1], "--", color="#888888")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("Validation ROC curves")
    axes[1].legend(title="ROC-AUC", fontsize=8)

    for model_name, artifact in tuned_artifacts.items():
        if artifact.validation_probability is None:
            continue
        precision_values, recall_values, _ = precision_recall_curve(
            y_validation,
            artifact.validation_probability,
        )
        pr_auc = average_precision_score(
            y_validation,
            artifact.validation_probability,
        )
        axes[2].plot(
            recall_values,
            precision_values,
            linewidth=2,
            label=f"{model_name} ({pr_auc:.3f})",
        )
    axes[2].axhline(y_validation.mean(), linestyle="--", color="#888888")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_title("Validation precision-recall curves")
    axes[2].legend(title="PR-AUC", fontsize=8)

    fig.suptitle(
        "Tuned model comparison - external test excluded from tuning",
        fontsize=16,
    )
    fig.tight_layout()
    return fig


def plot_threshold_selection(
    selection: SelectionResult,
    y_validation: pd.Series,
) -> Figure:
    """Plot threshold trade-offs and validation confusion matrix."""

    if selection.artifact.validation_probability is None:
        raise ValueError("Selected artifact has no validation probabilities.")
    prediction = (
        selection.artifact.validation_probability
        >= selection.operating_threshold
    ).astype(int)
    validation_confusion = confusion_matrix(
        y_validation,
        prediction,
        labels=[0, 1],
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(
        selection.threshold_curve,
        selection.precision_curve,
        label="Precision",
    )
    axes[0].plot(
        selection.threshold_curve,
        selection.recall_curve,
        label="Recall",
    )
    axes[0].plot(
        selection.threshold_curve,
        selection.f05_curve,
        label="F0.5",
    )
    axes[0].axvline(
        selection.validation_f05_threshold,
        color="#777777",
        linestyle=":",
        linewidth=2,
        label=(
            "F0.5 reference = "
            f"{selection.validation_f05_threshold:.3f}"
        ),
    )
    axes[0].axvline(
        selection.operating_threshold,
        color="#C44E52",
        linestyle="--",
        linewidth=2,
        label=f"Operating = {selection.operating_threshold:.3f}",
    )
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("Decision threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title(
        f"Validation operating threshold - {selection.selected_model_name}"
    )
    axes[0].legend()

    sns.heatmap(
        validation_confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=axes[1],
    )
    axes[1].set_title(
        "Validation confusion matrix @ threshold "
        f"{selection.operating_threshold:.2f}"
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    fig.tight_layout()
    return fig


def plot_external_evaluation(
    external: ExternalEvaluation,
    y_test: pd.Series,
    selection: SelectionResult,
) -> Figure:
    """Plot external confusion matrix, ROC and precision-recall curves."""

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        external.probability,
    )
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test,
        external.probability,
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    sns.heatmap(
        external.confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=axes[0],
    )
    axes[0].set_title("External test confusion matrix")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    axes[1].plot(false_positive_rate, true_positive_rate, linewidth=2)
    axes[1].plot([0, 1], [0, 1], "--", color="#888888")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title(
        "External ROC curve - AUC "
        f"{roc_auc_score(y_test, external.probability):.3f}"
    )

    axes[2].plot(recall_curve, precision_curve, linewidth=2)
    axes[2].axhline(y_test.mean(), linestyle="--", color="#888888")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_title(
        "External precision-recall curve - AP "
        f"{average_precision_score(y_test, external.probability):.3f}"
    )

    fig.suptitle(
        f"External synthetic test - {selection.selected_model_name} "
        f"@ threshold {selection.operating_threshold:.3f}",
        fontsize=15,
    )
    fig.tight_layout()
    return fig


def plot_probability_separation(
    selection: SelectionResult,
    y_validation: pd.Series,
    external: ExternalEvaluation,
    y_test: pd.Series,
) -> tuple[Figure, pd.DataFrame]:
    """Plot class-conditional predicted-score distributions."""

    if selection.artifact.validation_probability is None:
        raise ValueError("Selected artifact has no validation probabilities.")
    probability_inputs = [
        (
            "Validation",
            y_validation,
            selection.artifact.validation_probability,
        ),
        ("External test", y_test, external.probability),
    ]
    summary_records: list[dict[str, object]] = []
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 6),
        sharex=True,
        sharey=True,
    )
    probability_bins = np.linspace(0, 1, 31)
    class_styles = {
        0: ("Actual retained (0)", "#4C72B0"),
        1: ("Actual churn (1)", "#C44E52"),
    }

    for axis, (split_name, target, probability) in zip(
        axes,
        probability_inputs,
    ):
        target_array = np.asarray(target)
        for actual_class, (class_label, class_color) in class_styles.items():
            class_probability = probability[target_array == actual_class]
            axis.hist(
                class_probability,
                bins=probability_bins,
                density=True,
                histtype="stepfilled",
                alpha=0.32,
                linewidth=1.5,
                edgecolor=class_color,
                color=class_color,
                label=class_label,
            )
            summary_records.append(
                {
                    "Split": split_name,
                    "Actual class": class_label,
                    "Customers": len(class_probability),
                    "Mean probability": class_probability.mean(),
                    "Median probability": np.median(class_probability),
                    "Q25 probability": np.quantile(
                        class_probability,
                        0.25,
                    ),
                    "Q75 probability": np.quantile(
                        class_probability,
                        0.75,
                    ),
                    "Share above threshold": (
                        class_probability >= selection.operating_threshold
                    ).mean(),
                }
            )

        axis.axvspan(
            selection.operating_threshold,
            1.0,
            color="#C44E52",
            alpha=0.05,
            label="Predicted-churn region",
        )
        axis.axvline(
            selection.operating_threshold,
            color="#222222",
            linestyle="--",
            linewidth=2,
            label=f"Threshold = {selection.operating_threshold:.3f}",
        )
        axis.set_xlim(0, 1)
        axis.set_xlabel("Predicted probability P(Exited = 1)")
        axis.set_ylabel("Class-normalized density")
        axis.set_title(f"{split_name} probability separation")
        axis.legend(loc="upper center", fontsize=9)

    fig.suptitle(
        "Predicted-score separation by actual class - "
        f"{selection.selected_model_name}",
        fontsize=16,
    )
    fig.tight_layout()
    return fig, pd.DataFrame(summary_records)


def plot_feature_importance(
    importance: FeatureImportanceResult,
    selection: SelectionResult,
    top_n: int = 20,
) -> Figure:
    """Plot normalized source-feature importance."""

    top_importance = (
        importance.source.head(top_n)
        .sort_values("Importance share", ascending=True)
    )
    fig, axis = plt.subplots(figsize=(11, 8))
    bars = axis.barh(
        top_importance["Source feature"],
        top_importance["Importance share"],
        color="#4C72B0",
    )
    axis.bar_label(
        bars,
        labels=[
            f"{value:.1%}"
            for value in top_importance["Importance share"]
        ],
        padding=3,
    )
    axis.set_xlabel("Share of normalized model importance")
    axis.set_ylabel("")
    axis.set_title(
        "Top source-feature importance - "
        f"{selection.selected_model_name} "
        f"({selection.artifact.algorithm})"
    )
    maximum = float(top_importance["Importance share"].max())
    axis.set_xlim(0, maximum * 1.18 if maximum > 0 else 1)
    fig.tight_layout()
    return fig

