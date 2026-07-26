"""Small data contracts passed between pipeline steps."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DataBundle:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    split_summary: pd.DataFrame
    external_overlap: int
    model_columns: list[str]


@dataclass(slots=True)
class FeatureBundle:
    engineer: Any
    frames: dict[str, dict[str, pd.DataFrame]]
    engineered_feature_names: list[str]
    rule_feature_names: list[str]
    catalog: pd.DataFrame


@dataclass(slots=True)
class PreprocessingBundle:
    preprocessors: dict[str, Any]
    matrices: dict[str, dict[str, Any]]
    inventory: pd.DataFrame


@dataclass(slots=True)
class SearchSpec:
    name: str
    model_family: str
    algorithm: str
    estimator: Any
    parameters: dict[str, list[Any]]


@dataclass(slots=True)
class ModelArtifact:
    name: str
    model: Any
    model_family: str
    algorithm: str
    feature_set: str
    imbalance_strategy: str
    scale_pos_weight: float
    oversampled: bool
    training_rows: int
    training_seconds: float
    best_iteration: int
    tuned: bool = False
    cv_pr_auc_mean: float = np.nan
    cv_pr_auc_std: float = np.nan
    tuning_seconds: float = np.nan
    best_parameters: dict[str, Any] = field(default_factory=dict)
    validation_probability: np.ndarray | None = None

    def predict_probability(self, matrix: Any) -> np.ndarray:
        """Return P(churn=1) while respecting each boosting API."""

        if self.model_family == "lightgbm":
            return self.model.predict_proba(
                matrix,
                num_iteration=self.best_iteration,
            )[:, 1]
        if self.model_family == "xgboost":
            return self.model.predict_proba(
                matrix,
                iteration_range=(0, self.best_iteration),
            )[:, 1]
        return self.model.predict_proba(matrix)[:, 1]


@dataclass(slots=True)
class TuningResult:
    artifacts: dict[str, ModelArtifact]
    summary: pd.DataFrame


@dataclass(slots=True)
class SelectionResult:
    selected_model_name: str
    artifact: ModelArtifact
    ranking: pd.DataFrame
    shortlist: pd.DataFrame
    operating_threshold: float
    validation_f05_threshold: float
    threshold_comparison: pd.DataFrame
    precision_curve: np.ndarray
    recall_curve: np.ndarray
    threshold_curve: np.ndarray
    f05_curve: np.ndarray


@dataclass(slots=True)
class ExternalEvaluation:
    results: pd.DataFrame
    probability: np.ndarray
    prediction: np.ndarray
    confusion: np.ndarray


@dataclass(slots=True)
class FeatureImportanceResult:
    processed: pd.DataFrame
    source: pd.DataFrame
    rule_features: pd.DataFrame

