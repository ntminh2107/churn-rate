"""Preprocessing fitted only on training data."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pipeline.config import DEFAULT_CONFIG, PipelineConfig
from pipeline.contracts import FeatureBundle, PreprocessingBundle


def build_preprocessor(
    frame: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build numeric/categorical preprocessing for a known input schema."""

    numeric_columns = frame.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = frame.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    return preprocessor, numeric_columns, categorical_columns


def fit_preprocessors(
    features: FeatureBundle,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> PreprocessingBundle:
    """Fit one preprocessor per feature set and transform all splits."""

    preprocessors: dict[str, ColumnTransformer] = {}
    matrices: dict[str, dict[str, Any]] = {}
    inventory: list[dict[str, object]] = []

    for feature_set, frames in features.frames.items():
        preprocessor, numeric_columns, categorical_columns = build_preprocessor(
            frames["train"]
        )
        matrices[feature_set] = {
            "train": preprocessor.fit_transform(frames["train"]),
            "validation": preprocessor.transform(frames["validation"]),
            "test": preprocessor.transform(frames["test"]),
        }
        preprocessors[feature_set] = preprocessor
        transformed_names = preprocessor.get_feature_names_out().tolist()
        forbidden = [*config.excluded_columns, config.target]
        if any(
            excluded in transformed_name
            for excluded in forbidden
            for transformed_name in transformed_names
        ):
            raise AssertionError("Excluded identifiers/target leaked into matrix.")
        inventory.append(
            {
                "Feature set": feature_set,
                "Input columns": frames["train"].shape[1],
                "Numeric columns": len(numeric_columns),
                "Categorical columns": len(categorical_columns),
                "Processed columns": len(transformed_names),
                "Train matrix shape": str(matrices[feature_set]["train"].shape),
            }
        )

    return PreprocessingBundle(
        preprocessors=preprocessors,
        matrices=matrices,
        inventory=pd.DataFrame(inventory),
    )
