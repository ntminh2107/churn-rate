"""Fast demo-time training and prediction with frozen tuned parameters."""

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

from pipeline.config import DEFAULT_CONFIG, PipelineConfig
from pipeline.contracts import (
    DataBundle,
    FeatureBundle,
    ModelArtifact,
    PreprocessingBundle,
)
from pipeline.data import load_and_split_data
from pipeline.features import build_feature_bundle
from pipeline.model_lightgbm import (
    FINAL_TUNED_PARAMETERS as LIGHTGBM_PARAMETERS,
)
from pipeline.model_lightgbm import build_tuned_model as build_lightgbm
from pipeline.model_random_forest import (
    FINAL_TUNED_PARAMETERS as RANDOM_FOREST_PARAMETERS,
)
from pipeline.model_random_forest import (
    build_tuned_model as build_random_forest,
)
from pipeline.model_xgboost import (
    FINAL_TUNED_PARAMETERS as XGBOOST_PARAMETERS,
)
from pipeline.model_xgboost import build_tuned_model as build_xgboost
from pipeline.preprocessing import fit_preprocessors


@dataclass(slots=True)
class DemoBundle:
    """Everything the Streamlit app needs for prediction and EDA."""

    config: PipelineConfig
    data: DataBundle
    features: FeatureBundle
    preprocessing: PreprocessingBundle
    artifacts: dict[str, ModelArtifact]
    eda_frame: pd.DataFrame


FROZEN_MODEL_SPECS = {
    "Tuned XGBoost": {
        "family": "xgboost",
        "algorithm": "XGBoost",
        "parameters": XGBOOST_PARAMETERS,
        "builder": build_xgboost,
        "cv_pr_auc": 0.700688,
    },
    "Tuned LightGBM": {
        "family": "lightgbm",
        "algorithm": "LightGBM",
        "parameters": LIGHTGBM_PARAMETERS,
        "builder": build_lightgbm,
        "cv_pr_auc": 0.700792,
    },
    "Tuned Random Forest": {
        "family": "random_forest",
        "algorithm": "Random Forest",
        "parameters": RANDOM_FOREST_PARAMETERS,
        "builder": build_random_forest,
        "cv_pr_auc": 0.688571,
    },
}


def train_demo_models(
    config: PipelineConfig = DEFAULT_CONFIG,
) -> DemoBundle:
    """Fit the three frozen tuned models without repeating hyperparameter search."""

    data = load_and_split_data(config)
    features = build_feature_bundle(data, config)
    preprocessing = fit_preprocessors(features, config)
    train_matrix = preprocessing.matrices["engineered"]["train"]
    validation_matrix = preprocessing.matrices["engineered"]["validation"]
    artifacts: dict[str, ModelArtifact] = {}

    for model_name, spec in FROZEN_MODEL_SPECS.items():
        parameters = dict(spec["parameters"])
        model = spec["builder"](parameters, config, n_jobs=-1)
        started_at = time.perf_counter()
        model.fit(train_matrix, data.y_train)
        training_seconds = time.perf_counter() - started_at

        if spec["family"] == "random_forest":
            imbalance_strategy = (
                f"class_weight={parameters.get('class_weight')}"
            )
            scale_pos_weight = 1.0
        else:
            scale_pos_weight = float(
                parameters.get("scale_pos_weight", 1.0)
            )
            imbalance_strategy = (
                f"scale_pos_weight={scale_pos_weight:.4f}"
            )

        artifact = ModelArtifact(
            name=model_name,
            model=model,
            model_family=str(spec["family"]),
            algorithm=str(spec["algorithm"]),
            feature_set="engineered",
            imbalance_strategy=imbalance_strategy,
            scale_pos_weight=scale_pos_weight,
            oversampled=False,
            training_rows=len(data.y_train),
            training_seconds=training_seconds,
            best_iteration=int(parameters["n_estimators"]),
            tuned=True,
            cv_pr_auc_mean=float(spec["cv_pr_auc"]),
            best_parameters=parameters,
        )
        artifact.validation_probability = artifact.predict_probability(
            validation_matrix
        )
        artifacts[model_name] = artifact

    source_features = pd.concat(
        [data.X_train, data.X_validation],
        axis=0,
    ).sort_index()
    source_target = pd.concat(
        [data.y_train, data.y_validation],
        axis=0,
    ).sort_index()
    eda_frame = source_features.copy()
    eda_frame[config.target] = source_target

    return DemoBundle(
        config=config,
        data=data,
        features=features,
        preprocessing=preprocessing,
        artifacts=artifacts,
        eda_frame=eda_frame,
    )


def generate_random_customer(
    bundle: DemoBundle,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic customer by sampling each train marginal."""

    rng = np.random.default_rng(seed)
    reference = bundle.features.frames["raw"]["train"]
    record: dict[str, object] = {}
    for column in reference.columns:
        non_missing = reference[column].dropna()
        if non_missing.empty:
            record[column] = np.nan
            continue
        sampled = non_missing.iloc[int(rng.integers(0, len(non_missing)))]
        if pd.api.types.is_integer_dtype(reference[column].dtype):
            sampled = int(sampled)
        elif pd.api.types.is_float_dtype(reference[column].dtype):
            sampled = float(sampled)
        record[column] = sampled
    return pd.DataFrame([record], columns=reference.columns)


def normalize_customer(
    customer: pd.DataFrame,
    bundle: DemoBundle,
) -> pd.DataFrame:
    """Validate and order one or more raw customer rows for inference."""

    expected_columns = bundle.data.model_columns
    missing = sorted(set(expected_columns) - set(customer.columns))
    extra = sorted(set(customer.columns) - set(expected_columns))
    if missing or extra:
        raise ValueError(
            f"Customer schema mismatch. Missing={missing}, extra={extra}"
        )
    return customer[expected_columns].copy()


def predict_customer(
    bundle: DemoBundle,
    customer: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Score a raw customer with all frozen models."""

    threshold = (
        bundle.config.operating_threshold
        if threshold is None
        else float(threshold)
    )
    customer = normalize_customer(customer, bundle)
    engineered = bundle.features.engineer.transform(customer)
    matrix = bundle.preprocessing.preprocessors["engineered"].transform(
        engineered
    )

    records: list[dict[str, object]] = []
    for model_name, artifact in bundle.artifacts.items():
        probability = float(artifact.predict_probability(matrix)[0])
        if probability >= 0.70:
            risk_level = "Rất cao"
        elif probability >= threshold:
            risk_level = "Cao"
        elif probability >= max(0.25, threshold * 0.65):
            risk_level = "Trung bình"
        else:
            risk_level = "Thấp"
        records.append(
            {
                "Model": model_name,
                "Algorithm": artifact.algorithm,
                "Churn probability": probability,
                "Threshold": threshold,
                "Prediction": (
                    "Churn" if probability >= threshold else "Retained"
                ),
                "Risk level": risk_level,
                "CV PR-AUC": artifact.cv_pr_auc_mean,
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values("Churn probability", ascending=False)
        .reset_index(drop=True)
    )


def customer_risk_signals(customer: pd.DataFrame) -> list[str]:
    """Return human-readable rule signals for the demo explanation."""

    row = customer.iloc[0]
    signals: list[str] = []
    if int(row["IsActiveMember"]) == 0:
        signals.append("Khách hàng đang không hoạt động")
    if int(row["CreditScore"]) <= 400:
        signals.append("Credit score rất thấp (≤ 400)")
    elif int(row["CreditScore"]) <= 425:
        signals.append("Credit score thấp (≤ 425)")
    if int(row["NumOfProducts"]) >= 3:
        signals.append("Đang sử dụng từ 3 sản phẩm trở lên")
    if (
        int(row["CreditScore"]) <= 425
        and int(row["Tenure"]) < 5
        and int(row["HasCrCard"]) == 1
    ):
        signals.append("Low credit + tenure ngắn + có credit card")
    if int(row["IsActiveMember"]) == 0 and float(row["Balance"]) > 0:
        signals.append("Không hoạt động nhưng vẫn còn số dư")
    if int(row["NumOfProducts"]) == 2:
        signals.append("Hai sản phẩm — tín hiệu giữ chân tích cực trong train")
    return signals
