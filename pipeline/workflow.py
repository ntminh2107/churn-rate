"""Optional end-to-end orchestration for non-notebook callers."""

from dataclasses import dataclass

import pandas as pd

from pipeline.config import DEFAULT_CONFIG, PipelineConfig
from pipeline.contracts import (
    DataBundle,
    ExternalEvaluation,
    FeatureBundle,
    FeatureImportanceResult,
    ModelArtifact,
    PreprocessingBundle,
    SelectionResult,
    TuningResult,
)
from pipeline.data import load_and_split_data
from pipeline.evaluation import (
    build_validation_results,
    evaluate_external_test,
    select_tuned_model,
)
from pipeline.features import build_feature_bundle
from pipeline.importance import calculate_source_importance
from pipeline.preprocessing import fit_preprocessors
from pipeline.training import train_baseline_models
from pipeline.tuning import tune_models


@dataclass(slots=True)
class PipelineRun:
    data: DataBundle
    features: FeatureBundle
    preprocessing: PreprocessingBundle
    baseline_artifacts: dict[str, ModelArtifact]
    tuning: TuningResult
    artifacts: dict[str, ModelArtifact]
    validation_results: pd.DataFrame
    selection: SelectionResult
    external: ExternalEvaluation
    importance: FeatureImportanceResult


def run_training_workflow(
    config: PipelineConfig = DEFAULT_CONFIG,
    *,
    include_baselines: bool = True,
    search_n_jobs: int = -1,
    final_model_n_jobs: int = -1,
    verbose: int = 1,
) -> PipelineRun:
    """Run every non-visual training/evaluation step."""

    data = load_and_split_data(config)
    features = build_feature_bundle(data, config)
    preprocessing = fit_preprocessors(features, config)
    baseline_artifacts = (
        train_baseline_models(data, preprocessing, config)
        if include_baselines
        else {}
    )
    tuning = tune_models(
        data,
        preprocessing,
        config,
        search_n_jobs=search_n_jobs,
        final_model_n_jobs=final_model_n_jobs,
        verbose=verbose,
    )
    artifacts = {**baseline_artifacts, **tuning.artifacts}
    validation_results = build_validation_results(
        artifacts,
        data.y_validation,
        threshold=config.operating_threshold,
    )
    selection = select_tuned_model(
        validation_results,
        artifacts,
        data.y_validation,
        config,
    )
    external = evaluate_external_test(selection, data, preprocessing)
    importance = calculate_source_importance(
        selection.artifact,
        features,
        preprocessing,
    )
    return PipelineRun(
        data=data,
        features=features,
        preprocessing=preprocessing,
        baseline_artifacts=baseline_artifacts,
        tuning=tuning,
        artifacts=artifacts,
        validation_results=validation_results,
        selection=selection,
        external=external,
        importance=importance,
    )

