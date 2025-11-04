"""
Data loaders module

This module provides data loading adapters that implement the DataPort protocol.

Available loaders:
    - FileDataLoader: Load data from CSV, NumPy files
    - (Future) DatabaseDataLoader: Load from SQL databases
    - (Future) APIDataLoader: Load from REST APIs

Usage:
    from learning_framework.adapters.loaders import FileDataLoader

    loader = FileDataLoader()
    X, y = loader.load("data.csv", target_column="label")
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split(X, y)
"""

from adapters.loaders.file_data_loader import FileDataLoader

__all__ = [
    "FileDataLoader",
]
