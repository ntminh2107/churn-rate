"""Central configuration for the churn pipeline."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Immutable configuration shared by notebook and command-line runs."""

    train_data_path: Path = PROJECT_ROOT / "archive" / "Customer-Churn-Records.csv"
    external_test_path: Path = (
        PROJECT_ROOT / "archive" / "Synthetic-Customer-Churn-Test.csv"
    )
    target: str = "Exited"
    excluded_columns: tuple[str, ...] = (
        "RowNumber",
        "CustomerId",
        "Surname",
        "Complain",
    )
    dropped_engineered_features: tuple[str, ...] = (
        "VeryLowCreditInactive",
        "LowCreditShortTenureHasCard",
        "ShortTenureUnder5",
        "VeryLowCreditScore",
        "LowCreditScore425",
        "InactiveWithMultipleProducts",
        "ZeroBalance",
        "CardButInactive",
    )
    validation_size: float = 0.15
    random_state: int = 42
    age_bins: int = 5
    operating_threshold: float = 0.40
    cv_tie_tolerance: float = 0.002
    tuning_iterations: int = 18
    cv_splits: int = 5

    def __post_init__(self) -> None:
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")
        if not 0 < self.operating_threshold < 1:
            raise ValueError("operating_threshold must be between 0 and 1.")
        if self.age_bins < 2:
            raise ValueError("age_bins must be at least 2.")
        if self.tuning_iterations < 1:
            raise ValueError("tuning_iterations must be positive.")
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be at least 2.")
        if len(set(self.dropped_engineered_features)) != len(
            self.dropped_engineered_features
        ):
            raise ValueError("dropped_engineered_features contains duplicates.")


DEFAULT_CONFIG = PipelineConfig()
