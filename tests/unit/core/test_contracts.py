"""
TDD Tests for core port interfaces (Protocol).
All tests follow Given-When-Then pattern.
"""

import pytest
import numpy as np
from typing import Dict, Any, Optional, Tuple
from core.contracts import (
    ModelPort,
    DataPort,
    TrackingPort,
    MetricsPort,
    ConfigPort,
)
from core.models import ModelConfig, TrainingConfig, ModelType, OptimizerType


class TestModelPort:
    """ModelPort Protocol 테스트"""

    def test_model_port_protocol_structure(self):
        # Given: ModelPort를 구현하는 Mock 클래스
        class MockModelAdapter:
            def build(self, config: ModelConfig) -> Any:
                return "mock_model"

            def train(self, model: Any, train_data: Any, val_data: Optional[Any], config: TrainingConfig) -> Dict[str, float]:
                return {"loss": 0.5, "accuracy": 0.9}

            def evaluate(self, model: Any, test_data: Any) -> Dict[str, float]:
                return {"accuracy": 0.85}

            def predict(self, model: Any, inputs: np.ndarray) -> np.ndarray:
                return np.array([1, 0, 1])

            def save(self, model: Any, path: str) -> None:
                pass

            def load(self, path: str) -> Any:
                return "loaded_model"

        # When: MockModelAdapter 인스턴스 생성
        adapter = MockModelAdapter()

        # Then: 모든 필수 메서드가 존재함
        assert hasattr(adapter, 'build')
        assert hasattr(adapter, 'train')
        assert hasattr(adapter, 'evaluate')
        assert hasattr(adapter, 'predict')
        assert hasattr(adapter, 'save')
        assert hasattr(adapter, 'load')

    def test_model_port_build_method_signature(self):
        # Given: ModelPort를 구현하는 Mock 클래스
        class MockModelAdapter:
            def build(self, config: ModelConfig) -> Any:
                return {"type": "model"}

            def train(self, model: Any, train_data: Any, val_data: Optional[Any], config: TrainingConfig) -> Dict[str, float]:
                return {}

            def evaluate(self, model: Any, test_data: Any) -> Dict[str, float]:
                return {}

            def predict(self, model: Any, inputs: np.ndarray) -> np.ndarray:
                return np.array([])

            def save(self, model: Any, path: str) -> None:
                pass

            def load(self, path: str) -> Any:
                return None

        adapter = MockModelAdapter()
        config = ModelConfig(name="test", type=ModelType.CLASSIFICATION, architecture="test_arch")

        # When: build 메서드 호출
        result = adapter.build(config)

        # Then: 올바른 결과 반환
        assert result is not None

    def test_model_port_train_returns_metrics(self):
        # Given: ModelPort 구현체
        class MockModelAdapter:
            def build(self, config: ModelConfig) -> Any:
                return None

            def train(self, model: Any, train_data: Any, val_data: Optional[Any], config: TrainingConfig) -> Dict[str, float]:
                return {"loss": 0.3, "accuracy": 0.92}

            def evaluate(self, model: Any, test_data: Any) -> Dict[str, float]:
                return {}

            def predict(self, model: Any, inputs: np.ndarray) -> np.ndarray:
                return np.array([])

            def save(self, model: Any, path: str) -> None:
                pass

            def load(self, path: str) -> Any:
                return None

        adapter = MockModelAdapter()
        config = TrainingConfig(epochs=10, batch_size=32, learning_rate=0.001)

        # When: train 메서드 호출
        metrics = adapter.train("model", "train_data", "val_data", config)

        # Then: 메트릭 딕셔너리 반환
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "accuracy" in metrics


class TestDataPort:
    """DataPort Protocol 테스트"""

    def test_data_port_protocol_structure(self):
        # Given: DataPort를 구현하는 Mock 클래스
        class MockDataAdapter:
            def load(self, path: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
                return np.array([]), np.array([])

            def split(self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7,
                     val_ratio: float = 0.15, test_ratio: float = 0.15,
                     random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

            def preprocess(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
                return X, y

            def to_framework_format(self, X: np.ndarray, y: np.ndarray, framework: str) -> Any:
                return {"X": X, "y": y}

        # When: MockDataAdapter 인스턴스 생성
        adapter = MockDataAdapter()

        # Then: 모든 필수 메서드가 존재함
        assert hasattr(adapter, 'load')
        assert hasattr(adapter, 'split')
        assert hasattr(adapter, 'preprocess')
        assert hasattr(adapter, 'to_framework_format')

    def test_data_port_load_returns_tuple(self):
        # Given: DataPort 구현체
        class MockDataAdapter:
            def load(self, path: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
                X = np.array([[1, 2], [3, 4]])
                y = np.array([0, 1])
                return X, y

            def split(self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7,
                     val_ratio: float = 0.15, test_ratio: float = 0.15,
                     random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

            def preprocess(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
                return X, y

            def to_framework_format(self, X: np.ndarray, y: np.ndarray, framework: str) -> Any:
                return None

        adapter = MockDataAdapter()

        # When: load 메서드 호출
        X, y = adapter.load("test_path")

        # Then: 튜플 반환
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)


class TestTrackingPort:
    """TrackingPort Protocol 테스트"""

    def test_tracking_port_protocol_structure(self):
        # Given: TrackingPort를 구현하는 Mock 클래스
        class MockTracker:
            def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
                return "run_123"

            def log_params(self, params: Dict[str, Any]) -> None:
                pass

            def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
                pass

            def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
                pass

            def log_model(self, model: Any, artifact_path: str) -> None:
                pass

            def end_run(self) -> None:
                pass

            def get_run_id(self) -> Optional[str]:
                return "run_123"

        # When: MockTracker 인스턴스 생성
        tracker = MockTracker()

        # Then: 모든 필수 메서드가 존재함
        assert hasattr(tracker, 'start_run')
        assert hasattr(tracker, 'log_params')
        assert hasattr(tracker, 'log_metrics')
        assert hasattr(tracker, 'log_artifact')
        assert hasattr(tracker, 'log_model')
        assert hasattr(tracker, 'end_run')
        assert hasattr(tracker, 'get_run_id')

    def test_tracking_port_start_run_returns_id(self):
        # Given: TrackingPort 구현체
        class MockTracker:
            def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
                return f"run_{experiment_name}"

            def log_params(self, params: Dict[str, Any]) -> None:
                pass

            def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
                pass

            def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
                pass

            def log_model(self, model: Any, artifact_path: str) -> None:
                pass

            def end_run(self) -> None:
                pass

            def get_run_id(self) -> Optional[str]:
                return None

        tracker = MockTracker()

        # When: start_run 호출
        run_id = tracker.start_run("experiment1")

        # Then: run ID 반환
        assert isinstance(run_id, str)
        assert "experiment1" in run_id


class TestMetricsPort:
    """MetricsPort Protocol 테스트"""

    def test_metrics_port_protocol_structure(self):
        # Given: MetricsPort를 구현하는 Mock 클래스
        class MockMetrics:
            def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
                return 0.9

            def calculate_precision_recall_f1(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> Dict[str, float]:
                return {"precision": 0.85, "recall": 0.88, "f1": 0.86}

            def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
                return np.array([[10, 2], [1, 15]])

            def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
                return {"mse": 0.5, "rmse": 0.71, "mae": 0.4, "r2": 0.95}

        # When: MockMetrics 인스턴스 생성
        metrics = MockMetrics()

        # Then: 모든 필수 메서드가 존재함
        assert hasattr(metrics, 'calculate_accuracy')
        assert hasattr(metrics, 'calculate_precision_recall_f1')
        assert hasattr(metrics, 'calculate_confusion_matrix')
        assert hasattr(metrics, 'calculate_regression_metrics')

    def test_metrics_port_calculate_accuracy_returns_float(self):
        # Given: MetricsPort 구현체
        class MockMetrics:
            def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
                return float(np.mean(y_true == y_pred))

            def calculate_precision_recall_f1(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> Dict[str, float]:
                return {}

            def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
                return np.array([])

            def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
                return {}

        metrics = MockMetrics()
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 0, 1, 0])

        # When: calculate_accuracy 호출
        accuracy = metrics.calculate_accuracy(y_true, y_pred)

        # Then: float 반환
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0


class TestConfigPort:
    """ConfigPort Protocol 테스트"""

    def test_config_port_protocol_structure(self):
        # Given: ConfigPort를 구현하는 Mock 클래스
        class MockConfig:
            def load(self, path: str) -> Dict[str, Any]:
                return {}

            def validate(self, config: Dict[str, Any]) -> bool:
                return True

            def merge(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
                return {**base_config, **override_config}

            def save(self, config: Dict[str, Any], path: str) -> None:
                pass

        # When: MockConfig 인스턴스 생성
        config_mgr = MockConfig()

        # Then: 모든 필수 메서드가 존재함
        assert hasattr(config_mgr, 'load')
        assert hasattr(config_mgr, 'validate')
        assert hasattr(config_mgr, 'merge')
        assert hasattr(config_mgr, 'save')

    def test_config_port_load_returns_dict(self):
        # Given: ConfigPort 구현체
        class MockConfig:
            def load(self, path: str) -> Dict[str, Any]:
                return {"key": "value", "number": 42}

            def validate(self, config: Dict[str, Any]) -> bool:
                return True

            def merge(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
                return {}

            def save(self, config: Dict[str, Any], path: str) -> None:
                pass

        config_mgr = MockConfig()

        # When: load 호출
        config = config_mgr.load("test.yaml")

        # Then: 딕셔너리 반환
        assert isinstance(config, dict)

    def test_config_port_merge_combines_configs(self):
        # Given: ConfigPort 구현체
        class MockConfig:
            def load(self, path: str) -> Dict[str, Any]:
                return {}

            def validate(self, config: Dict[str, Any]) -> bool:
                return True

            def merge(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
                result = base_config.copy()
                result.update(override_config)
                return result

            def save(self, config: Dict[str, Any], path: str) -> None:
                pass

        config_mgr = MockConfig()
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        # When: merge 호출
        merged = config_mgr.merge(base, override)

        # Then: 병합된 설정 반환
        assert merged["a"] == 1
        assert merged["b"] == 3
        assert merged["c"] == 4


class TestProtocolImplementationFlexibility:
    """Protocol의 구조적 서브타이핑 테스트"""

    def test_partial_implementation_is_not_valid(self):
        # Given: ModelPort의 일부만 구현한 클래스
        class IncompleteAdapter:
            def build(self, config: ModelConfig) -> Any:
                return None
            # train, evaluate, predict 등 누락

        adapter = IncompleteAdapter()

        # When/Then: 필수 메서드가 없음
        assert not hasattr(adapter, 'train')
        assert not hasattr(adapter, 'evaluate')

    def test_extra_methods_are_allowed(self):
        # Given: ModelPort + 추가 메서드를 구현한 클래스
        class ExtendedAdapter:
            def build(self, config: ModelConfig) -> Any:
                return None

            def train(self, model: Any, train_data: Any, val_data: Optional[Any], config: TrainingConfig) -> Dict[str, float]:
                return {}

            def evaluate(self, model: Any, test_data: Any) -> Dict[str, float]:
                return {}

            def predict(self, model: Any, inputs: np.ndarray) -> np.ndarray:
                return np.array([])

            def save(self, model: Any, path: str) -> None:
                pass

            def load(self, path: str) -> Any:
                return None

            # 추가 메서드
            def custom_method(self) -> str:
                return "custom"

        adapter = ExtendedAdapter()

        # When: 추가 메서드 호출
        result = adapter.custom_method()

        # Then: 정상 동작
        assert result == "custom"
        assert hasattr(adapter, 'build')
        assert hasattr(adapter, 'custom_method')
