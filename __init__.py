"""
Learning Framework - Educational AI learning framework with unified interface

This package provides a unified interface for training and evaluating
machine learning models across different frameworks (TensorFlow, PyTorch, etc.).
"""

__version__ = "0.1.0"

from core.models import (
    ModelConfig,
    TrainingConfig,
    DatasetInfo,
    ExperimentResult,
    ModelType,
    OptimizerType,
)
from core.contracts import ModelPort, DataPort, MetricsPort

from services.experiment_service import ExperimentService

from config.schema import ExperimentConfig, FrameworkType, TaskType
from config.loader import YAMLConfigLoader

from utils.exceptions import (
    AIFrameworkException,
    ConfigException,
    ModelException,
    DataException,
)
from utils.metrics import MetricsCalculator

__all__ = [
    "__version__",
    "ModelConfig",
    "TrainingConfig",
    "DatasetInfo",
    "ExperimentResult",
    "ModelType",
    "OptimizerType",
    "ModelPort",
    "DataPort",
    "MetricsPort",
    "ExperimentService",
    "ExperimentConfig",
    "FrameworkType",
    "TaskType",
    "YAMLConfigLoader",
    "AIFrameworkException",
    "ConfigException",
    "ModelException",
    "DataException",
    "MetricsCalculator",
]
