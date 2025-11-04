"""
Framework-specific adapters

This module provides adapters for different ML frameworks.
Each adapter implements the ModelPort protocol.

Available adapters:
    - KerasModelAdapter: TensorFlow/Keras models
    - PyTorchModelAdapter: PyTorch models
    - (Future) HuggingFaceModelAdapter: Transformers models

Usage:
    from learning_framework.adapters.frameworks import KerasModelAdapter
    from learning_framework.adapters.frameworks import PyTorchModelAdapter

    keras_adapter = KerasModelAdapter()
    pytorch_adapter = PyTorchModelAdapter()
    model = keras_adapter.build(model_config)

Note: Import adapters directly to avoid loading heavy framework dependencies
"""

from adapters.frameworks.keras_adapter import KerasModelAdapter
from adapters.frameworks.pytorch_adapter import PyTorchModelAdapter

__all__ = [
    "KerasModelAdapter",
    "PyTorchModelAdapter",
]
