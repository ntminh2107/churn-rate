"""Baseline training orchestration."""

from pipeline.config import DEFAULT_CONFIG, PipelineConfig
from pipeline.contracts import DataBundle, ModelArtifact, PreprocessingBundle
from pipeline.model_lightgbm import train_baseline_variants
from pipeline.model_random_forest import train_baseline as train_rf_baseline
from pipeline.model_xgboost import train_baseline as train_xgb_baseline


def train_baseline_models(
    data: DataBundle,
    preprocessing: PreprocessingBundle,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> dict[str, ModelArtifact]:
    """Train all six baseline/ablation variants used by the report."""

    artifacts = train_baseline_variants(data, preprocessing, config)
    xgb_artifact = train_xgb_baseline(data, preprocessing, config)
    rf_artifact = train_rf_baseline(data, preprocessing, config)
    artifacts[xgb_artifact.name] = xgb_artifact
    artifacts[rf_artifact.name] = rf_artifact
    return artifacts
