"""Leakage-safe hyperparameter tuning across the three model families."""

import time
from collections.abc import Callable

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from pipeline.config import DEFAULT_CONFIG, PipelineConfig
from pipeline.contracts import (
    DataBundle,
    ModelArtifact,
    PreprocessingBundle,
    SearchSpec,
    TuningResult,
)
from pipeline.features import ChurnFeatureEngineer
from pipeline.model_lightgbm import (
    build_tuned_model as build_tuned_lightgbm,
)
from pipeline.model_lightgbm import make_search_spec as make_lightgbm_search_spec
from pipeline.model_random_forest import (
    build_tuned_model as build_tuned_random_forest,
)
from pipeline.model_random_forest import (
    make_search_spec as make_random_forest_search_spec,
)
from pipeline.model_xgboost import (
    build_tuned_model as build_tuned_xgboost,
)
from pipeline.model_xgboost import make_search_spec as make_xgboost_search_spec


FINAL_BUILDERS: dict[str, Callable[..., object]] = {
    "lightgbm": build_tuned_lightgbm,
    "xgboost": build_tuned_xgboost,
    "random_forest": build_tuned_random_forest,
}


def make_tuning_specs(
    data: DataBundle,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> list[SearchSpec]:
    negative_count = int((data.y_train == 0).sum())
    positive_count = int((data.y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count
    return [
        make_lightgbm_search_spec(scale_pos_weight, config),
        make_xgboost_search_spec(scale_pos_weight, config),
        make_random_forest_search_spec(config),
    ]


def _imbalance_description(
    model_family: str,
    parameters: dict[str, object],
) -> tuple[str, float]:
    if model_family == "random_forest":
        class_weight = parameters.get("class_weight")
        description = f"class_weight={class_weight}" if class_weight else "None"
        return description, 1.0

    scale_pos_weight = float(parameters.get("scale_pos_weight", 1.0))
    description = (
        f"scale_pos_weight={scale_pos_weight:.4f}"
        if scale_pos_weight != 1.0
        else "None"
    )
    return description, scale_pos_weight


def tune_models(
    data: DataBundle,
    preprocessing: PreprocessingBundle,
    config: PipelineConfig = DEFAULT_CONFIG,
    *,
    search_n_jobs: int = -1,
    final_model_n_jobs: int = -1,
    verbose: int = 1,
) -> TuningResult:
    """Tune all model families using train-only stratified CV.

    `ChurnFeatureEngineer` and preprocessing are included inside the search
    pipeline, so both are fitted independently inside every fold.
    """

    cv_strategy = StratifiedKFold(
        n_splits=config.cv_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    artifacts: dict[str, ModelArtifact] = {}
    records: list[dict[str, object]] = []

    for spec in make_tuning_specs(data, config):
        fold_safe_pipeline = Pipeline(
            [
                ("features", ChurnFeatureEngineer(age_bins=config.age_bins)),
                (
                    "preprocessor",
                    clone(preprocessing.preprocessors["engineered"]),
                ),
                ("model", spec.estimator),
            ]
        )
        search = RandomizedSearchCV(
            estimator=fold_safe_pipeline,
            param_distributions=spec.parameters,
            n_iter=config.tuning_iterations,
            scoring="average_precision",
            n_jobs=search_n_jobs,
            cv=cv_strategy,
            refit=True,
            random_state=config.random_state,
            verbose=verbose,
            return_train_score=False,
            error_score="raise",
        )

        tuning_started_at = time.perf_counter()
        search.fit(data.X_train, data.y_train)
        tuning_seconds = time.perf_counter() - tuning_started_at

        best_parameters = {
            key.removeprefix("model__"): value
            for key, value in search.best_params_.items()
        }
        best_index = int(search.best_index_)
        cv_pr_auc_std = float(search.cv_results_["std_test_score"][best_index])

        final_model = FINAL_BUILDERS[spec.model_family](
            best_parameters,
            config,
            n_jobs=final_model_n_jobs,
        )
        final_fit_started_at = time.perf_counter()
        final_model.fit(
            preprocessing.matrices["engineered"]["train"],
            data.y_train,
        )
        final_fit_seconds = time.perf_counter() - final_fit_started_at
        imbalance_strategy, scale_pos_weight = _imbalance_description(
            spec.model_family,
            best_parameters,
        )

        artifact = ModelArtifact(
            name=spec.name,
            model=final_model,
            model_family=spec.model_family,
            algorithm=spec.algorithm,
            feature_set="engineered",
            imbalance_strategy=imbalance_strategy,
            scale_pos_weight=scale_pos_weight,
            oversampled=False,
            training_rows=len(data.y_train),
            training_seconds=final_fit_seconds,
            best_iteration=int(best_parameters["n_estimators"]),
            tuned=True,
            cv_pr_auc_mean=float(search.best_score_),
            cv_pr_auc_std=cv_pr_auc_std,
            tuning_seconds=tuning_seconds,
            best_parameters=best_parameters,
        )
        artifact.validation_probability = artifact.predict_probability(
            preprocessing.matrices["engineered"]["validation"]
        )
        artifacts[artifact.name] = artifact
        records.append(
            {
                "Model": artifact.name,
                "CV PR-AUC mean": artifact.cv_pr_auc_mean,
                "CV PR-AUC std": artifact.cv_pr_auc_std,
                "Search seconds": artifact.tuning_seconds,
                "Best parameters": artifact.best_parameters,
            }
        )

    summary = (
        pd.DataFrame(records)
        .sort_values("CV PR-AUC mean", ascending=False)
        .reset_index(drop=True)
    )
    return TuningResult(artifacts=artifacts, summary=summary)

