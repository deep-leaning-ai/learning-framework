"""
Core module - Contains core business logic and port interfaces
"""

__version__ = "0.1.0"

from core.models import (
    ModelConfig,
    TrainingConfig,
    DatasetInfo,
    ExperimentResult,
    TrainingHistory,
    ModelType,
    OptimizerType,
)

from core.contracts import (
    ModelPort,
    DataPort,
    MetricsPort,
    TrackingPort,
    ConfigPort,
)

__all__ = [
    "__version__",
    "ModelConfig",
    "TrainingConfig",
    "DatasetInfo",
    "ExperimentResult",
    "TrainingHistory",
    "ModelType",
    "OptimizerType",
    "ModelPort",
    "DataPort",
    "MetricsPort",
    "TrackingPort",
    "ConfigPort",
]
