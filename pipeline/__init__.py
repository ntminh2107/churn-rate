"""Reusable customer-churn training pipeline.

The notebook is intentionally kept as an orchestration and reporting layer.
All trainable transformations and model logic live in this package.
"""

from pipeline.config import DEFAULT_CONFIG, PipelineConfig

__all__ = ["DEFAULT_CONFIG", "PipelineConfig"]

