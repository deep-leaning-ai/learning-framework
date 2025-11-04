"""
Adapters module - Framework-specific implementations

This module provides adapters for different ML frameworks and data loaders.

Submodules:
    - frameworks: ML framework adapters (Keras, PyTorch, etc.)
    - loaders: Data loading adapters (File, Database, etc.)
    - mock_adapter: Mock implementations for testing

Note: Framework adapters should be imported directly to avoid loading heavy dependencies
    Example: from learning_framework.adapters.frameworks import KerasModelAdapter
    Example: from learning_framework.adapters.loaders import FileDataLoader
"""

from .mock_adapter import MockModelAdapter, MockDataAdapter

__all__ = [
    "MockModelAdapter",
    "MockDataAdapter",
]
