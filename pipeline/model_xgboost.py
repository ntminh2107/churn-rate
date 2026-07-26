"""XGBoost model definitions, baseline and tuning space."""

import time

from xgboost import XGBClassifier

from pipeline.config import PipelineConfig
from pipeline.contracts import DataBundle, ModelArtifact, PreprocessingBundle, SearchSpec


FINAL_TUNED_PARAMETERS = {
    "subsample": 0.85,
    "scale_pos_weight": 1.50,
    "reg_lambda": 2.00,
    "reg_alpha": 0.00,
    "n_estimators": 500,
    "min_child_weight": 3,
    "max_depth": 5,
    "learning_rate": 0.015,
    "gamma": 0.30,
    "colsample_bytree": 0.70,
}


def make_search_spec(
    train_scale_pos_weight: float,
    config: PipelineConfig,
) -> SearchSpec:
    """Return the fold-safe randomized-search definition for XGBoost."""

    return SearchSpec(
        name="Tuned XGBoost",
        model_family="xgboost",
        algorithm="XGBoost",
        estimator=XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=config.random_state,
            n_jobs=1,
            verbosity=0,
        ),
        parameters={
            "model__n_estimators": [200, 350, 500, 700, 900],
            "model__learning_rate": [0.015, 0.025, 0.05, 0.08],
            "model__max_depth": [3, 4, 5, 6],
            "model__min_child_weight": [1, 3, 5, 8],
            "model__subsample": [0.70, 0.85, 1.00],
            "model__colsample_bytree": [0.70, 0.85, 1.00],
            "model__gamma": [0.0, 0.1, 0.3, 0.5],
            "model__reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "model__scale_pos_weight": [
                1.0,
                1.5,
                2.0,
                train_scale_pos_weight,
            ],
        },
    )


def build_tuned_model(
    parameters: dict[str, object],
    config: PipelineConfig,
    n_jobs: int = -1,
) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=config.random_state,
        n_jobs=n_jobs,
        verbosity=0,
        **parameters,
    )


def train_baseline(
    data: DataBundle,
    preprocessing: PreprocessingBundle,
    config: PipelineConfig,
) -> ModelArtifact:
    """Train the engineered XGBoost early-stopping baseline."""

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=2_000,
        learning_rate=0.025,
        max_depth=4,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=1.00,
        eval_metric="auc",
        early_stopping_rounds=100,
        tree_method="hist",
        random_state=config.random_state,
        n_jobs=-1,
        verbosity=0,
    )
    started_at = time.perf_counter()
    model.fit(
        preprocessing.matrices["engineered"]["train"],
        data.y_train,
        eval_set=[
            (
                preprocessing.matrices["engineered"]["validation"],
                data.y_validation,
            )
        ],
        verbose=False,
    )
    artifact = ModelArtifact(
        name="Engineered XGBoost",
        model=model,
        model_family="xgboost",
        algorithm="XGBoost",
        feature_set="engineered",
        imbalance_strategy="None",
        scale_pos_weight=1.0,
        oversampled=False,
        training_rows=len(data.y_train),
        training_seconds=time.perf_counter() - started_at,
        best_iteration=int(model.best_iteration) + 1,
    )
    artifact.validation_probability = artifact.predict_probability(
        preprocessing.matrices["engineered"]["validation"]
    )
    return artifact

