"""
Metrics calculation system.

This module provides implementations for various evaluation metrics
for classification and regression tasks, implementing the MetricsPort interface.
"""

import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class MetricsCalculator:
    """
    메트릭 계산기

    MetricsPort 프로토콜을 구현하며, scikit-learn을 사용하여
    분류 및 회귀 메트릭을 계산합니다.
    """

    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        정확도 계산

        Args:
            y_true: 실제 값
            y_pred: 예측 값

        Returns:
            정확도 (0-1)

        Raises:
            ValueError: 입력 배열의 크기가 다르거나 비어있는 경우
        """
        y_true = self._validate_and_convert(y_true, y_pred, "classification")
        y_pred = self._validate_and_convert(y_pred, y_true, "classification")

        return float(accuracy_score(y_true, y_pred))

    def calculate_precision_recall_f1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Precision, Recall, F1 계산

        Args:
            y_true: 실제 값
            y_pred: 예측 값
            average: 평균 방법 ('micro', 'macro', 'weighted', 'binary')

        Returns:
            {'precision': float, 'recall': float, 'f1': float} 딕셔너리

        Raises:
            ValueError: 입력 배열의 크기가 다른 경우
        """
        y_true = self._validate_and_convert(y_true, y_pred, "classification")
        y_pred = self._validate_and_convert(y_pred, y_true, "classification")

        # zero_division=0으로 설정하여 경고 방지
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Confusion Matrix 계산

        Args:
            y_true: 실제 값
            y_pred: 예측 값

        Returns:
            Confusion Matrix (2D numpy array)

        Raises:
            ValueError: 입력 배열의 크기가 다른 경우
        """
        y_true = self._validate_and_convert(y_true, y_pred, "classification")
        y_pred = self._validate_and_convert(y_pred, y_true, "classification")

        return confusion_matrix(y_true, y_pred)

    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        회귀 메트릭 계산 (MSE, RMSE, MAE, R2)

        Args:
            y_true: 실제 값
            y_pred: 예측 값

        Returns:
            회귀 메트릭 딕셔너리

        Raises:
            ValueError: 입력 배열의 크기가 다른 경우
        """
        y_true = self._validate_and_convert(y_true, y_pred, "regression")
        y_pred = self._validate_and_convert(y_pred, y_true, "regression")

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }

    def _validate_and_convert(
        self,
        y: np.ndarray,
        y_other: np.ndarray,
        task_type: str
    ) -> np.ndarray:
        """
        입력 검증 및 변환

        Args:
            y: 검증할 배열
            y_other: 비교할 배열
            task_type: 태스크 타입 ('classification' 또는 'regression')

        Returns:
            검증 및 변환된 numpy 배열

        Raises:
            ValueError: 입력이 비어있거나 크기가 맞지 않는 경우
        """
        # numpy 배열로 변환
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # 2차원 배열을 1차원으로 변환
        if y.ndim > 1:
            y = y.ravel()

        # 빈 배열 체크
        if len(y) == 0:
            raise ValueError("Input arrays cannot be empty")

        # 크기 일치 체크
        if isinstance(y_other, np.ndarray):
            y_other_flat = y_other.ravel() if y_other.ndim > 1 else y_other
            if len(y) != len(y_other_flat):
                raise ValueError(
                    f"Input arrays must have the same size. "
                    f"Got {len(y)} and {len(y_other_flat)}"
                )

        return y
