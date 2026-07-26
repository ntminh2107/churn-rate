"""Random Forest model definitions, baseline and tuning space."""

import time

from sklearn.ensemble import RandomForestClassifier

from pipeline.config import PipelineConfig
from pipeline.contracts import DataBundle, ModelArtifact, PreprocessingBundle, SearchSpec


FINAL_TUNED_PARAMETERS = {
    "n_estimators": 800,
    "min_samples_split": 10,
    "min_samples_leaf": 2,
    "max_features": "log2",
    "max_depth": 8,
    "class_weight": {0: 1, 1: 2},
    "bootstrap": True,
}


def make_search_spec(
    config: PipelineConfig,
) -> SearchSpec:
    """Return the fold-safe randomized-search definition for Random Forest."""

    return SearchSpec(
        name="Tuned Random Forest",
        model_family="random_forest",
        algorithm="Random Forest",
        estimator=RandomForestClassifier(
            random_state=config.random_state,
            n_jobs=1,
        ),
        parameters={
            "model__n_estimators": [300, 500, 800, 1_000],
            "model__max_depth": [None, 8, 12, 16, 20],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8, 12],
            "model__max_features": ["sqrt", "log2", 0.50, 0.75],
            "model__class_weight": [
                None,
                "balanced",
                "balanced_subsample",
                {0: 1, 1: 2},
                {0: 1, 1: 3},
            ],
            "model__bootstrap": [True],
        },
    )


def build_tuned_model(
    parameters: dict[str, object],
    config: PipelineConfig,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=config.random_state,
        n_jobs=n_jobs,
        **parameters,
    )


def train_baseline(
    data: DataBundle,
    preprocessing: PreprocessingBundle,
    config: PipelineConfig,
) -> ModelArtifact:
    """Train the engineered Random Forest baseline."""

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=config.random_state,
        n_jobs=-1,
    )
    started_at = time.perf_counter()
    model.fit(
        preprocessing.matrices["engineered"]["train"],
        data.y_train,
    )
    artifact = ModelArtifact(
        name="Engineered Random Forest",
        model=model,
        model_family="random_forest",
        algorithm="Random Forest",
        feature_set="engineered",
        imbalance_strategy="None",
        scale_pos_weight=1.0,
        oversampled=False,
        training_rows=len(data.y_train),
        training_seconds=time.perf_counter() - started_at,
        best_iteration=model.n_estimators,
    )
    artifact.validation_probability = artifact.predict_probability(
        preprocessing.matrices["engineered"]["validation"]
    )
    return artifact

