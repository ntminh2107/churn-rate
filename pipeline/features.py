"""Train-fitted feature engineering and rule auditing."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pipeline.config import DEFAULT_CONFIG, PipelineConfig
from pipeline.contracts import DataBundle, FeatureBundle


REQUIRED_FEATURE_COLUMNS = {
    "Age",
    "Balance",
    "CreditScore",
    "EstimatedSalary",
    "Geography",
    "HasCrCard",
    "IsActiveMember",
    "NumOfProducts",
    "Point Earned",
    "Tenure",
}

RULE_FEATURE_NAMES = [
    "IsInactiveMember",
    "VeryLowCreditScore",
    "LowCreditScore425",
    "ShortTenureUnder5",
    "VeryLowCreditInactive",
    "LowCreditShortTenureHasCard",
    "InactiveWithOneProduct",
    "InactiveWithThreePlusProducts",
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain and interaction features without learning from validation/test."""

    def __init__(self, age_bins: int = 5):
        self.age_bins = age_bins

    @staticmethod
    def _validate_columns(frame: pd.DataFrame) -> None:
        missing = sorted(REQUIRED_FEATURE_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"Missing columns required for feature engineering: {missing}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "ChurnFeatureEngineer":
        self._validate_columns(X)
        age_quantiles = X["Age"].quantile(
            np.linspace(0, 1, self.age_bins + 1)
        ).to_numpy(dtype=float)
        self.age_edges_ = np.unique(age_quantiles)
        if len(self.age_edges_) < 2:
            raise ValueError("Age must contain at least two distinct values.")
        self.age_edges_[0] = -np.inf
        self.age_edges_[-1] = np.inf
        self.age_labels_ = [
            f"age_q{index + 1}" for index in range(len(self.age_edges_) - 1)
        ]
        self.older_age_threshold_ = float(X["Age"].quantile(0.75))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            ["age_edges_", "age_labels_", "older_age_threshold_"],
        )
        self._validate_columns(X)
        result = X.copy()

        result["ProductGroup"] = np.select(
            [
                result["NumOfProducts"] == 1,
                result["NumOfProducts"] == 2,
                result["NumOfProducts"] >= 3,
            ],
            ["1_product", "2_products", "3_plus_products"],
            default="other",
        )
        result["HasTwoProducts"] = (result["NumOfProducts"] == 2).astype(int)
        result["HasThreePlusProducts"] = (
            result["NumOfProducts"] >= 3
        ).astype(int)

        inactive = result["IsActiveMember"] == 0
        very_low_credit = result["CreditScore"] <= 400
        low_credit_425 = result["CreditScore"] <= 425
        short_tenure = result["Tenure"] < 5
        has_credit_card = result["HasCrCard"] == 1

        result["IsInactiveMember"] = inactive.astype(int)
        result["VeryLowCreditScore"] = very_low_credit.astype(int)
        result["LowCreditScore425"] = low_credit_425.astype(int)
        result["ShortTenureUnder5"] = short_tenure.astype(int)
        result["VeryLowCreditInactive"] = (
            very_low_credit & inactive
        ).astype(int)
        result["LowCreditShortTenureHasCard"] = (
            low_credit_425 & short_tenure & has_credit_card
        ).astype(int)
        result["InactiveWithOneProduct"] = (
            inactive & (result["NumOfProducts"] == 1)
        ).astype(int)
        result["InactiveWithThreePlusProducts"] = (
            inactive & (result["NumOfProducts"] >= 3)
        ).astype(int)

        result["ZeroBalance"] = (result["Balance"] == 0).astype(int)
        result["LogBalance"] = np.log1p(result["Balance"].clip(lower=0))
        result["BalanceToSalary"] = (
            result["Balance"] / (result["EstimatedSalary"].abs() + 1.0)
        )
        result["BalancePerProduct"] = (
            result["Balance"] / result["NumOfProducts"].clip(lower=1)
        )

        result["AgeGroup"] = pd.cut(
            result["Age"],
            bins=self.age_edges_,
            labels=self.age_labels_,
            include_lowest=True,
        ).astype("object")
        result["IsOlderCustomer"] = (
            result["Age"] >= self.older_age_threshold_
        ).astype(int)

        result["InactiveWithBalance"] = (
            inactive & (result["Balance"] > 0)
        ).astype(int)
        result["InactiveWithMultipleProducts"] = (
            inactive & (result["NumOfProducts"] >= 2)
        ).astype(int)
        result["CardButInactive"] = (
            inactive & (result["HasCrCard"] == 1)
        ).astype(int)

        activity_label = np.where(
            result["IsActiveMember"] == 1,
            "active",
            "inactive",
        )
        result["GeographyActivity"] = (
            result["Geography"].astype(str) + "_" + activity_label
        )
        result["GeographyAgeGroup"] = (
            result["Geography"].astype(str)
            + "_"
            + result["AgeGroup"].astype(str)
        )
        result["PointsPerTenure"] = (
            result["Point Earned"] / (result["Tenure"] + 1.0)
        )

        return result


def audit_risk_rules(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Audit user-provided risk hypotheses on train only."""

    rule_masks = {
        "CreditScore <= 400 and inactive": (
            (X_train["CreditScore"] <= 400)
            & (X_train["IsActiveMember"] == 0)
        ),
        "Inactive member": X_train["IsActiveMember"] == 0,
        "NumOfProducts >= 3": X_train["NumOfProducts"] >= 3,
        "CreditScore <= 425, Tenure < 5 and HasCrCard": (
            (X_train["CreditScore"] <= 425)
            & (X_train["Tenure"] < 5)
            & (X_train["HasCrCard"] == 1)
        ),
    }
    baseline_churn_rate = y_train.mean()
    records: list[dict[str, object]] = []
    for rule_name, rule_mask in rule_masks.items():
        rule_target = y_train.loc[rule_mask]
        support = len(rule_target)
        exited = int(rule_target.sum())
        non_exited = support - exited
        exit_rate = exited / support if support else np.nan
        records.append(
            {
                "Rule": rule_name,
                "Support": support,
                "Exited": exited,
                "Non-exited": non_exited,
                "Exit rate": exit_rate,
                "Lift vs train baseline": (
                    exit_rate / baseline_churn_rate if support else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def make_feature_catalog() -> pd.DataFrame:
    """Return documentation for all engineered features."""

    rows = [
        ("ProductGroup", "Categorical", "1 product / 2 products / 3+ products"),
        ("HasTwoProducts", "Binary", "NumOfProducts == 2"),
        ("HasThreePlusProducts", "Binary", "NumOfProducts >= 3"),
        ("IsInactiveMember", "Binary", "IsActiveMember == 0"),
        ("VeryLowCreditScore", "Binary", "CreditScore <= 400"),
        ("LowCreditScore425", "Binary", "CreditScore <= 425"),
        ("ShortTenureUnder5", "Binary", "Tenure < 5"),
        (
            "VeryLowCreditInactive",
            "Binary interaction",
            "CreditScore <= 400 and inactive",
        ),
        (
            "LowCreditShortTenureHasCard",
            "Binary interaction",
            "CreditScore <= 425 and Tenure < 5 and HasCrCard == 1",
        ),
        (
            "InactiveWithOneProduct",
            "Binary interaction",
            "Inactive and NumOfProducts == 1",
        ),
        (
            "InactiveWithThreePlusProducts",
            "Binary interaction",
            "Inactive and NumOfProducts >= 3",
        ),
        ("ZeroBalance", "Binary", "Balance == 0"),
        ("LogBalance", "Numeric", "log1p(Balance)"),
        (
            "BalanceToSalary",
            "Numeric",
            "Balance / (abs(EstimatedSalary) + 1)",
        ),
        (
            "BalancePerProduct",
            "Numeric",
            "Balance / max(NumOfProducts, 1)",
        ),
        ("AgeGroup", "Categorical", "Age quantile bins fitted on train only"),
        ("IsOlderCustomer", "Binary", "Age >= train 75th percentile"),
        ("InactiveWithBalance", "Binary", "Inactive and Balance > 0"),
        (
            "InactiveWithMultipleProducts",
            "Binary",
            "Inactive and NumOfProducts >= 2",
        ),
        ("CardButInactive", "Binary", "Has card but inactive"),
        (
            "GeographyActivity",
            "Categorical interaction",
            "Geography x activity state",
        ),
        (
            "GeographyAgeGroup",
            "Categorical interaction",
            "Geography x AgeGroup",
        ),
        ("PointsPerTenure", "Numeric", "Point Earned / (Tenure + 1)"),
    ]
    return pd.DataFrame(rows, columns=["Feature", "Type", "Definition"])


def build_feature_bundle(
    data: DataBundle,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> FeatureBundle:
    """Fit feature engineering on train and transform validation/test."""

    engineer = ChurnFeatureEngineer(age_bins=config.age_bins)
    train_engineered = engineer.fit_transform(data.X_train, data.y_train)
    validation_engineered = engineer.transform(data.X_validation)
    test_engineered = engineer.transform(data.X_test)

    engineered_feature_names = [
        column
        for column in train_engineered.columns
        if column not in data.X_train.columns
    ]
    excluded = {*config.excluded_columns, config.target}
    if excluded.intersection(train_engineered.columns):
        raise AssertionError("Excluded identifiers/target leaked into engineered data.")
    if not set(RULE_FEATURE_NAMES).issubset(train_engineered.columns):
        raise AssertionError("One or more rule features were not created.")
    if not (
        train_engineered.columns.tolist()
        == validation_engineered.columns.tolist()
        == test_engineered.columns.tolist()
    ):
        raise AssertionError("Engineered schemas differ across splits.")

    frames = {
        "raw": {
            "train": data.X_train,
            "validation": data.X_validation,
            "test": data.X_test,
        },
        "engineered": {
            "train": train_engineered,
            "validation": validation_engineered,
            "test": test_engineered,
        },
    }
    return FeatureBundle(
        engineer=engineer,
        frames=frames,
        engineered_feature_names=engineered_feature_names,
        rule_feature_names=RULE_FEATURE_NAMES.copy(),
        catalog=make_feature_catalog(),
    )

