"""LightGBM model definitions, baselines and tuning space."""

import time

import lightgbm as lgb
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier

from pipeline.config import PipelineConfig
from pipeline.contracts import DataBundle, ModelArtifact, PreprocessingBundle, SearchSpec


FINAL_TUNED_PARAMETERS = {
    "subsample_freq": 1,
    "subsample": 0.70,
    "scale_pos_weight": 1.50,
    "reg_lambda": 0.50,
    "reg_alpha": 0.50,
    "num_leaves": 15,
    "n_estimators": 600,
    "min_child_samples": 10,
    "max_depth": 10,
    "learning_rate": 0.025,
    "colsample_bytree": 0.70,
}


def make_search_spec(
    train_scale_pos_weight: float,
    config: PipelineConfig,
) -> SearchSpec:
    """Return the fold-safe randomized-search definition for LightGBM."""

    return SearchSpec(
        name="Tuned LightGBM",
        model_family="lightgbm",
        algorithm="LightGBM",
        estimator=LGBMClassifier(
            objective="binary",
            random_state=config.random_state,
            n_jobs=1,
            verbosity=-1,
        ),
        parameters={
            "model__n_estimators": [150, 250, 400, 600, 800],
            "model__learning_rate": [0.015, 0.025, 0.05, 0.08],
            "model__num_leaves": [15, 31, 47, 63],
            "model__max_depth": [-1, 4, 6, 8, 10],
            "model__min_child_samples": [10, 20, 30, 50, 80],
            "model__subsample": [0.70, 0.85, 1.00],
            "model__subsample_freq": [1],
            "model__colsample_bytree": [0.70, 0.85, 1.00],
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
) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        random_state=config.random_state,
        n_jobs=n_jobs,
        verbosity=-1,
        **parameters,
    )


def train_baseline_variants(
    data: DataBundle,
    preprocessing: PreprocessingBundle,
    config: PipelineConfig,
) -> dict[str, ModelArtifact]:
    """Train raw/engineered/weighted/oversampled LightGBM baselines."""

    negative_count = int((data.y_train == 0).sum())
    positive_count = int((data.y_train == 1).sum())
    train_scale_pos_weight = negative_count / positive_count
    common_parameters = {
        "objective": "binary",
        "n_estimators": 2_000,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.10,
        "reg_lambda": 1.00,
        "random_state": config.random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }
    variants = [
        {
            "name": "Raw LGBM",
            "feature_set": "raw",
            "scale_pos_weight": 1.0,
            "imbalance_strategy": "None",
            "oversample": False,
        },
        {
            "name": "Engineered LGBM",
            "feature_set": "engineered",
            "scale_pos_weight": 1.0,
            "imbalance_strategy": "None",
            "oversample": False,
        },
        {
            "name": "Engineered + weight",
            "feature_set": "engineered",
            "scale_pos_weight": train_scale_pos_weight,
            "imbalance_strategy": "scale_pos_weight",
            "oversample": False,
        },
        {
            "name": "Engineered + oversampling",
            "feature_set": "engineered",
            "scale_pos_weight": 1.0,
            "imbalance_strategy": "RandomOverSampler",
            "oversample": True,
        },
    ]

    artifacts: dict[str, ModelArtifact] = {}
    for variant in variants:
        feature_set = str(variant["feature_set"])
        train_matrix = preprocessing.matrices[feature_set]["train"]
        train_target = data.y_train
        if variant["oversample"]:
            oversampler = RandomOverSampler(random_state=config.random_state)
            train_matrix, train_target = oversampler.fit_resample(
                train_matrix,
                train_target,
            )

        model = LGBMClassifier(
            **common_parameters,
            scale_pos_weight=float(variant["scale_pos_weight"]),
        )
        started_at = time.perf_counter()
        model.fit(
            train_matrix,
            train_target,
            eval_set=[
                (
                    preprocessing.matrices[feature_set]["validation"],
                    data.y_validation,
                )
            ],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=100,
                    first_metric_only=True,
                    verbose=False,
                ),
                lgb.log_evaluation(period=0),
            ],
        )
        artifact = ModelArtifact(
            name=str(variant["name"]),
            model=model,
            model_family="lightgbm",
            algorithm="LightGBM",
            feature_set=feature_set,
            imbalance_strategy=str(variant["imbalance_strategy"]),
            scale_pos_weight=float(variant["scale_pos_weight"]),
            oversampled=bool(variant["oversample"]),
            training_rows=len(train_target),
            training_seconds=time.perf_counter() - started_at,
            best_iteration=int(model.best_iteration_),
        )
        artifact.validation_probability = artifact.predict_probability(
            preprocessing.matrices[feature_set]["validation"]
        )
        artifacts[artifact.name] = artifact
    return artifacts

