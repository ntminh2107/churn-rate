"""Normalize model-specific importance back to source features."""

import numpy as np
import pandas as pd

from pipeline.contracts import (
    FeatureBundle,
    FeatureImportanceResult,
    ModelArtifact,
    PreprocessingBundle,
)


def extract_processed_importance(
    artifact: ModelArtifact,
    feature_names: list[str],
) -> np.ndarray:
    model = artifact.model
    if artifact.model_family == "lightgbm":
        values = model.booster_.feature_importance(importance_type="gain")
    elif artifact.model_family == "xgboost":
        score_map = model.get_booster().get_score(importance_type="total_gain")
        values = np.zeros(len(feature_names), dtype=float)
        name_to_index = {
            name: index for index, name in enumerate(feature_names)
        }
        for key, value in score_map.items():
            if key.startswith("f") and key[1:].isdigit():
                feature_index = int(key[1:])
            else:
                feature_index = name_to_index.get(key)
            if feature_index is not None and feature_index < len(values):
                values[feature_index] = value
    elif artifact.model_family == "random_forest":
        values = model.feature_importances_
    else:
        raise ValueError(
            f"Unsupported model family: {artifact.model_family}"
        )

    values = np.asarray(values, dtype=float)
    if len(values) != len(feature_names):
        raise AssertionError("Importance vector does not match transformed schema.")
    return values


def calculate_source_importance(
    artifact: ModelArtifact,
    features: FeatureBundle,
    preprocessing: PreprocessingBundle,
) -> FeatureImportanceResult:
    """Aggregate one-hot importances back to original/engineered columns."""

    feature_set = artifact.feature_set
    preprocessor = preprocessing.preprocessors[feature_set]
    transformed_names = preprocessor.get_feature_names_out().tolist()
    importance_values = extract_processed_importance(
        artifact,
        transformed_names,
    )
    categorical_columns = (
        features.frames[feature_set]["train"]
        .select_dtypes(exclude=np.number)
        .columns.tolist()
    )

    def to_source_feature(transformed_name: str) -> str:
        transformer_name, remainder = transformed_name.split("__", 1)
        if transformer_name == "num":
            return remainder
        for source_feature in sorted(
            categorical_columns,
            key=len,
            reverse=True,
        ):
            if (
                remainder == source_feature
                or remainder.startswith(source_feature + "_")
            ):
                return source_feature
        return remainder

    processed = pd.DataFrame(
        {
            "Processed feature": transformed_names,
            "Source feature": [
                to_source_feature(name) for name in transformed_names
            ],
            "Importance": importance_values,
        }
    )
    source = (
        processed.groupby("Source feature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_total = source["Importance"].sum()
    source["Importance share"] = (
        source["Importance"] / importance_total
        if importance_total > 0
        else 0.0
    )
    rule_features = (
        source.loc[
            source["Source feature"].isin(features.rule_feature_names)
        ]
        .sort_values("Importance share", ascending=False)
        .reset_index(drop=True)
    )
    return FeatureImportanceResult(
        processed=processed,
        source=source,
        rule_features=rule_features,
    )
