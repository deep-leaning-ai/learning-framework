"""
TDD Tests for metrics calculation system.
All tests follow Given-When-Then pattern.
"""

import pytest
import numpy as np
from typing import Dict

from utils.metrics import MetricsCalculator


class TestMetricsCalculatorAccuracy:
    """정확도 계산 테스트"""

    def test_calculate_perfect_accuracy(self):
        # Given: 모두 정확하게 예측한 경우
        calculator = MetricsCalculator()
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        # When: 정확도 계산
        accuracy = calculator.calculate_accuracy(y_true, y_pred)

        # Then: 정확도가 1.0
        assert accuracy == 1.0

    def test_calculate_zero_accuracy(self):
        # Given: 모두 틀리게 예측한 경우
        calculator = MetricsCalculator()
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])

        # When: 정확도 계산
        accuracy = calculator.calculate_accuracy(y_true, y_pred)

        # Then: 정확도가 0.0
        assert accuracy == 0.0

    def test_calculate_partial_accuracy(self):
        # Given: 일부만 정확하게 예측한 경우
        calculator = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])

        # When: 정확도 계산
        accuracy = calculator.calculate_accuracy(y_true, y_pred)

        # Then: 정확도가 0.5
        assert accuracy == 0.5

    def test_accuracy_with_mismatched_shapes_raises_error(self):
        # Given: 크기가 다른 배열
        calculator = MetricsCalculator()
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1])

        # When/Then: ValueError 발생
        with pytest.raises(ValueError) as exc_info:
            calculator.calculate_accuracy(y_true, y_pred)

        assert "shape" in str(exc_info.value).lower() or "size" in str(exc_info.value).lower()


class TestMetricsCalculatorPrecisionRecallF1:
    """Precision, Recall, F1 계산 테스트"""

    def test_calculate_binary_classification_metrics(self):
        # Given: 이진 분류 데이터
        calculator = MetricsCalculator()
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0])

        # When: 메트릭 계산
        metrics = calculator.calculate_precision_recall_f1(y_true, y_pred)

        # Then: precision, recall, f1이 모두 포함됨
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_calculate_multiclass_metrics_weighted(self):
        # Given: 다중 클래스 분류 데이터
        calculator = MetricsCalculator()
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])

        # When: weighted average로 메트릭 계산
        metrics = calculator.calculate_precision_recall_f1(y_true, y_pred, average="weighted")

        # Then: 메트릭이 올바르게 계산됨
        assert isinstance(metrics, dict)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_calculate_metrics_macro_average(self):
        # Given: 다중 클래스 데이터
        calculator = MetricsCalculator()
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        # When: macro average로 계산
        metrics = calculator.calculate_precision_recall_f1(y_true, y_pred, average="macro")

        # Then: 완벽한 예측이므로 모든 메트릭이 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_metrics_with_mismatched_shapes_raises_error(self):
        # Given: 크기가 다른 배열
        calculator = MetricsCalculator()
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1])

        # When/Then: ValueError 발생
        with pytest.raises(ValueError):
            calculator.calculate_precision_recall_f1(y_true, y_pred)


class TestMetricsCalculatorConfusionMatrix:
    """Confusion Matrix 계산 테스트"""

    def test_calculate_binary_confusion_matrix(self):
        # Given: 이진 분류 데이터
        calculator = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])

        # When: Confusion Matrix 계산
        cm = calculator.calculate_confusion_matrix(y_true, y_pred)

        # Then: 2x2 행렬이 반환됨
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)

    def test_calculate_multiclass_confusion_matrix(self):
        # Given: 3-클래스 분류 데이터
        calculator = MetricsCalculator()
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 1, 2])

        # When: Confusion Matrix 계산
        cm = calculator.calculate_confusion_matrix(y_true, y_pred)

        # Then: 3x3 행렬이 반환됨
        assert cm.shape == (3, 3)
        assert cm.sum() == len(y_true)

    def test_perfect_prediction_confusion_matrix(self):
        # Given: 완벽한 예측
        calculator = MetricsCalculator()
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        # When: Confusion Matrix 계산
        cm = calculator.calculate_confusion_matrix(y_true, y_pred)

        # Then: 대각선만 값이 있음
        assert np.trace(cm) == len(y_true)
        assert cm.sum() == len(y_true)

    def test_confusion_matrix_with_mismatched_shapes_raises_error(self):
        # Given: 크기가 다른 배열
        calculator = MetricsCalculator()
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1])

        # When/Then: ValueError 발생
        with pytest.raises(ValueError):
            calculator.calculate_confusion_matrix(y_true, y_pred)


class TestMetricsCalculatorRegressionMetrics:
    """회귀 메트릭 계산 테스트"""

    def test_calculate_regression_metrics_perfect_prediction(self):
        # Given: 완벽한 예측
        calculator = MetricsCalculator()
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # When: 회귀 메트릭 계산
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)

        # Then: 모든 에러 메트릭이 0, R2는 1.0
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0

    def test_calculate_regression_metrics_with_errors(self):
        # Given: 오차가 있는 예측
        calculator = MetricsCalculator()
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        # When: 회귀 메트릭 계산
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)

        # Then: 메트릭이 올바르게 계산됨
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["mse"] > 0
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert 0.0 <= metrics["r2"] <= 1.0

    def test_calculate_mse_manually_verified(self):
        # Given: 간단한 예측 데이터
        calculator = MetricsCalculator()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])

        # When: MSE 계산
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)

        # Then: MSE = ((0.5)^2 + (0.5)^2 + (0.5)^2) / 3 = 0.25
        assert abs(metrics["mse"] - 0.25) < 1e-10

    def test_calculate_rmse_is_sqrt_of_mse(self):
        # Given: 예측 데이터
        calculator = MetricsCalculator()
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.2, 2.1, 2.9, 4.3])

        # When: 회귀 메트릭 계산
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)

        # Then: RMSE = sqrt(MSE)
        assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 1e-10

    def test_regression_metrics_with_mismatched_shapes_raises_error(self):
        # Given: 크기가 다른 배열
        calculator = MetricsCalculator()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])

        # When/Then: ValueError 발생
        with pytest.raises(ValueError):
            calculator.calculate_regression_metrics(y_true, y_pred)


class TestMetricsCalculatorEdgeCases:
    """엣지 케이스 테스트"""

    def test_single_sample_accuracy(self):
        # Given: 단일 샘플
        calculator = MetricsCalculator()
        y_true = np.array([1])
        y_pred = np.array([1])

        # When: 정확도 계산
        accuracy = calculator.calculate_accuracy(y_true, y_pred)

        # Then: 정확도 1.0
        assert accuracy == 1.0

    def test_empty_arrays_raise_error(self):
        # Given: 빈 배열
        calculator = MetricsCalculator()
        y_true = np.array([])
        y_pred = np.array([])

        # When/Then: ValueError 발생
        with pytest.raises(ValueError):
            calculator.calculate_accuracy(y_true, y_pred)

    def test_all_same_predictions_in_binary(self):
        # Given: 모두 같은 값으로 예측
        calculator = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])

        # When: 메트릭 계산
        metrics = calculator.calculate_precision_recall_f1(y_true, y_pred, average="binary")

        # Then: 메트릭이 계산됨 (일부는 0일 수 있음)
        assert isinstance(metrics, dict)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_regression_with_negative_values(self):
        # Given: 음수 값을 포함한 회귀 데이터
        calculator = MetricsCalculator()
        y_true = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y_pred = np.array([-1.8, -1.1, 0.1, 0.9, 2.1])

        # When: 회귀 메트릭 계산
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)

        # Then: 메트릭이 올바르게 계산됨
        assert metrics["mse"] > 0
        assert metrics["r2"] > 0

    def test_1d_array_handling(self):
        # Given: 1차원 배열
        calculator = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])

        # When: 메트릭 계산
        accuracy = calculator.calculate_accuracy(y_true, y_pred)
        cm = calculator.calculate_confusion_matrix(y_true, y_pred)

        # Then: 정상 작동
        assert isinstance(accuracy, float)
        assert cm.ndim == 2


class TestMetricsCalculatorValidation:
    """입력 검증 테스트"""

    def test_non_numpy_array_inputs_converted(self):
        # Given: 리스트 입력
        calculator = MetricsCalculator()
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]

        # When: 정확도 계산
        accuracy = calculator.calculate_accuracy(y_true, y_pred)

        # Then: 정상 작동 (numpy로 자동 변환)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_2d_arrays_flattened_for_classification(self):
        # Given: 2차원 배열
        calculator = MetricsCalculator()
        y_true = np.array([[0], [1], [0], [1]])
        y_pred = np.array([[0], [1], [1], [1]])

        # When: 정확도 계산
        accuracy = calculator.calculate_accuracy(y_true, y_pred)

        # Then: 정상 작동
        assert isinstance(accuracy, float)
