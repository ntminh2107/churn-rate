"""Data loading, schema validation and leakage-safe splitting."""

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.config import DEFAULT_CONFIG, PipelineConfig
from pipeline.contracts import DataBundle


def _validate_target(frame: pd.DataFrame, target: str, source_name: str) -> None:
    if target not in frame.columns:
        raise ValueError(f"Missing target column {target!r} in {source_name}.")


def load_and_split_data(
    config: PipelineConfig = DEFAULT_CONFIG,
) -> DataBundle:
    """Load source/external data and create a stratified train/validation split."""

    source_df = pd.read_csv(config.train_data_path)
    external_df = pd.read_csv(config.external_test_path)
    _validate_target(source_df, config.target, config.train_data_path.name)
    _validate_target(external_df, config.target, config.external_test_path.name)

    excluded = [*config.excluded_columns, config.target]
    X = source_df.drop(columns=excluded, errors="ignore").copy()
    y = source_df[config.target].astype(int).copy()
    external_X = external_df.drop(columns=excluded, errors="ignore").copy()
    external_y = external_df[config.target].astype(int).copy()

    leaked_columns = set(excluded).intersection(X.columns)
    if leaked_columns:
        raise AssertionError(f"Excluded columns leaked into features: {leaked_columns}")
    if set(external_X.columns) != set(X.columns):
        missing = sorted(set(X.columns) - set(external_X.columns))
        extra = sorted(set(external_X.columns) - set(X.columns))
        raise ValueError(
            "External schema differs from source schema. "
            f"Missing={missing}, extra={extra}"
        )
    external_X = external_X[X.columns]

    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=config.validation_size,
        stratify=y,
        random_state=config.random_state,
    )
    if not set(X_train.index).isdisjoint(X_validation.index):
        raise AssertionError("Train and validation indices overlap.")

    source_keys = X.astype(str).agg("\x1f".join, axis=1)
    external_keys = external_X.astype(str).agg("\x1f".join, axis=1)
    external_overlap = int(external_keys.isin(set(source_keys)).sum())

    split_summary = pd.DataFrame(
        {
            "Split": ["Train", "Validation", "External test"],
            "Source": [
                config.train_data_path.name,
                config.train_data_path.name,
                config.external_test_path.name,
            ],
            "Rows": [len(X_train), len(X_validation), len(external_X)],
            "Churn rate": [
                y_train.mean(),
                y_validation.mean(),
                external_y.mean(),
            ],
            "Target use": [
                "Model fitting and cross-validation",
                "Operating comparison and model tie-break",
                "Final diagnostic evaluation only",
            ],
        }
    )

    return DataBundle(
        X_train=X_train,
        X_validation=X_validation,
        X_test=external_X,
        y_train=y_train,
        y_validation=y_validation,
        y_test=external_y,
        split_summary=split_summary,
        external_overlap=external_overlap,
        model_columns=X.columns.tolist(),
    )
