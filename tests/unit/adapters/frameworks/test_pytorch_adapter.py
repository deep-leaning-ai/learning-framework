"""
TDD Tests for PyTorch adapter.
All tests follow Given-When-Then pattern.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, Any

from adapters.frameworks.pytorch_adapter import PyTorchModelAdapter
from core.models import ModelConfig, TrainingConfig, ModelType, OptimizerType
from core.exceptions import ModelException, DataException


class TestPyTorchAdapterInitialization:
    """PyTorch 어댑터 초기화 테스트"""

    def test_create_pytorch_adapter(self):
        # Given: PyTorch 어댑터
        # When: PyTorchModelAdapter 생성
        adapter = PyTorchModelAdapter()

        # Then: 어댑터가 올바르게 생성됨
        assert adapter is not None
        assert hasattr(adapter, 'device')


class TestPyTorchAdapterModelBuilding:
    """모델 빌드 테스트"""

    def test_build_sequential_classification_model(self):
        # Given: 분류용 모델 설정
        adapter = PyTorchModelAdapter()
        config = ModelConfig(
            name="test_classifier",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (20,),
                "layers": [
                    {"type": "dense", "units": 64, "activation": "relu"},
                    {"type": "dense", "units": 32, "activation": "relu"},
                    {"type": "dense", "units": 10, "activation": "softmax"}
                ]
            }
        )

        # When: 모델 빌드
        model = adapter.build(config)

        # Then: PyTorch 모델이 생성됨
        assert model is not None
        assert hasattr(model, 'forward')

    def test_build_sequential_regression_model(self):
        # Given: 회귀용 모델 설정
        adapter = PyTorchModelAdapter()
        config = ModelConfig(
            name="test_regressor",
            type=ModelType.REGRESSION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (10,),
                "layers": [
                    {"type": "dense", "units": 64, "activation": "relu"},
                    {"type": "dense", "units": 32, "activation": "relu"},
                    {"type": "dense", "units": 1}
                ]
            }
        )

        # When: 모델 빌드
        model = adapter.build(config)

        # Then: 회귀 모델이 생성됨
        assert model is not None
        assert hasattr(model, 'forward')

    def test_build_model_with_dropout(self):
        # Given: Dropout 레이어를 포함한 설정
        adapter = PyTorchModelAdapter()
        config = ModelConfig(
            name="test_dropout",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (20,),
                "layers": [
                    {"type": "dense", "units": 64, "activation": "relu"},
                    {"type": "dropout", "rate": 0.5},
                    {"type": "dense", "units": 10, "activation": "softmax"}
                ]
            }
        )

        # When: 모델 빌드
        model = adapter.build(config)

        # Then: Dropout 레이어가 포함된 모델 생성
        assert model is not None


class TestPyTorchAdapterModelTraining:
    """모델 학습 테스트"""

    def test_train_classification_model(self):
        # Given: 분류 모델과 학습 데이터
        adapter = PyTorchModelAdapter()
        model_config = ModelConfig(
            name="train_classifier",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (10,),
                "layers": [
                    {"type": "dense", "units": 64, "activation": "relu"},
                    {"type": "dense", "units": 3}
                ]
            }
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(
            epochs=5,
            batch_size=32,
            learning_rate=0.001,
            optimizer=OptimizerType.ADAM
        )

        # 학습 데이터 생성
        X_train = np.random.rand(100, 10).astype(np.float32)
        y_train = np.random.randint(0, 3, size=(100,))

        # When: 모델 학습
        metrics = adapter.train(
            model=model,
            train_data=(X_train, y_train),
            val_data=None,
            config=training_config
        )

        # Then: 학습 메트릭 반환
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_train_with_validation_data(self):
        # Given: 검증 데이터를 포함한 학습
        adapter = PyTorchModelAdapter()
        model_config = ModelConfig(
            name="train_with_val",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (5,),
                "layers": [
                    {"type": "dense", "units": 32, "activation": "relu"},
                    {"type": "dense", "units": 2}
                ]
            }
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(
            epochs=3,
            batch_size=16,
            learning_rate=0.01
        )

        X_train = np.random.rand(80, 5).astype(np.float32)
        y_train = np.random.randint(0, 2, size=(80,))
        X_val = np.random.rand(20, 5).astype(np.float32)
        y_val = np.random.randint(0, 2, size=(20,))

        # When: 검증 데이터와 함께 학습
        metrics = adapter.train(
            model=model,
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            config=training_config
        )

        # Then: 검증 메트릭도 포함
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics or "loss" in metrics

    def test_train_regression_model(self):
        # Given: 회귀 모델
        adapter = PyTorchModelAdapter()
        model_config = ModelConfig(
            name="train_regressor",
            type=ModelType.REGRESSION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (8,),
                "layers": [
                    {"type": "dense", "units": 32, "activation": "relu"},
                    {"type": "dense", "units": 1}
                ]
            }
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(
            epochs=5,
            batch_size=32,
            learning_rate=0.01
        )

        X_train = np.random.rand(100, 8).astype(np.float32)
        y_train = np.random.rand(100).astype(np.float32)

        # When: 회귀 모델 학습
        metrics = adapter.train(
            model=model,
            train_data=(X_train, y_train),
            val_data=None,
            config=training_config
        )

        # Then: 회귀 메트릭 반환
        assert isinstance(metrics, dict)
        assert "loss" in metrics


class TestPyTorchAdapterModelEvaluation:
    """모델 평가 테스트"""

    def test_evaluate_classification_model(self):
        # Given: 학습된 분류 모델
        adapter = PyTorchModelAdapter()
        model_config = ModelConfig(
            name="eval_classifier",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (5,),
                "layers": [
                    {"type": "dense", "units": 32, "activation": "relu"},
                    {"type": "dense", "units": 2}
                ]
            }
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(epochs=2, batch_size=16, learning_rate=0.01)
        X_train = np.random.rand(50, 5).astype(np.float32)
        y_train = np.random.randint(0, 2, size=(50,))
        adapter.train(model, (X_train, y_train), None, training_config)

        # 테스트 데이터
        X_test = np.random.rand(20, 5).astype(np.float32)
        y_test = np.random.randint(0, 2, size=(20,))

        # When: 모델 평가
        metrics = adapter.evaluate(model, (X_test, y_test))

        # Then: 평가 메트릭 반환
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_evaluate_regression_model(self):
        # Given: 학습된 회귀 모델
        adapter = PyTorchModelAdapter()
        model_config = ModelConfig(
            name="eval_regressor",
            type=ModelType.REGRESSION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (4,),
                "layers": [
                    {"type": "dense", "units": 16, "activation": "relu"},
                    {"type": "dense", "units": 1}
                ]
            }
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(epochs=2, batch_size=16, learning_rate=0.01)
        X_train = np.random.rand(50, 4).astype(np.float32)
        y_train = np.random.rand(50).astype(np.float32)
        adapter.train(model, (X_train, y_train), None, training_config)

        # 테스트 데이터
        X_test = np.random.rand(20, 4).astype(np.float32)
        y_test = np.random.rand(20).astype(np.float32)

        # When: 회귀 모델 평가
        metrics = adapter.evaluate(model, (X_test, y_test))

        # Then: 회귀 메트릭 반환
        assert isinstance(metrics, dict)
        assert "loss" in metrics


class TestPyTorchAdapterModelPrediction:
    """모델 예측 테스트"""

    def test_predict_classification(self):
        # Given: 학습된 분류 모델
        adapter = PyTorchModelAdapter()
        model_config = ModelConfig(
            name="pred_classifier",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (6,),
                "layers": [
                    {"type": "dense", "units": 16, "activation": "relu"},
                    {"type": "dense", "units": 3}
                ]
            }
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(epochs=2, batch_size=16, learning_rate=0.01)
        X_train = np.random.rand(50, 6).astype(np.float32)
        y_train = np.random.randint(0, 3, size=(50,))
        adapter.train(model, (X_train, y_train), None, training_config)

        # When: 예측 수행
        X_test = np.random.rand(10, 6).astype(np.float32)
        predictions = adapter.predict(model, X_test)

        # Then: 예측 결과 반환
        assert predictions is not None
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)

    def test_predict_regression(self):
        # Given: 학습된 회귀 모델
        adapter = PyTorchModelAdapter()
        model_config = ModelConfig(
            name="pred_regressor",
            type=ModelType.REGRESSION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (4,),
                "layers": [
                    {"type": "dense", "units": 16, "activation": "relu"},
                    {"type": "dense", "units": 1}
                ]
            }
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(epochs=2, batch_size=16, learning_rate=0.01)
        X_train = np.random.rand(50, 4).astype(np.float32)
        y_train = np.random.rand(50).astype(np.float32)
        adapter.train(model, (X_train, y_train), None, training_config)

        # When: 회귀 예측
        X_test = np.random.rand(10, 4).astype(np.float32)
        predictions = adapter.predict(model, X_test)

        # Then: 연속형 예측 결과 반환
        assert predictions is not None
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)


class TestPyTorchAdapterModelSaveLoad:
    """모델 저장/로드 테스트"""

    def test_save_and_load_model(self, tmp_path):
        # Given: 학습된 모델
        adapter = PyTorchModelAdapter()
        model_config = ModelConfig(
            name="save_load_model",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (5,),
                "layers": [
                    {"type": "dense", "units": 16, "activation": "relu"},
                    {"type": "dense", "units": 2}
                ]
            }
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(epochs=2, batch_size=16, learning_rate=0.01)
        X_train = np.random.rand(50, 5).astype(np.float32)
        y_train = np.random.randint(0, 2, size=(50,))
        adapter.train(model, (X_train, y_train), None, training_config)

        # 원본 예측
        X_test = np.random.rand(10, 5).astype(np.float32)
        original_predictions = adapter.predict(model, X_test)

        # When: 모델 저장 및 로드
        model_path = tmp_path / "test_model.pth"
        adapter.save(model, str(model_path))

        # 로드를 위한 동일한 구조의 모델 생성
        model_structure = adapter.build(model_config)
        loaded_model = adapter.load(str(model_path), model_structure=model_structure)

        # Then: 로드된 모델이 동일한 예측 수행
        loaded_predictions = adapter.predict(loaded_model, X_test)
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions, decimal=5)

    def test_save_creates_directory_if_not_exists(self, tmp_path):
        # Given: 존재하지 않는 디렉토리 경로
        adapter = PyTorchModelAdapter()
        model_config = ModelConfig(
            name="dir_test",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (3,),
                "layers": [
                    {"type": "dense", "units": 8, "activation": "relu"},
                    {"type": "dense", "units": 2}
                ]
            }
        )
        model = adapter.build(model_config)

        nested_path = tmp_path / "nested" / "dir" / "model.pth"

        # When: 모델 저장
        adapter.save(model, str(nested_path))

        # Then: 디렉토리가 생성되고 모델이 저장됨
        assert nested_path.exists()


class TestPyTorchAdapterErrorHandling:
    """에러 처리 테스트"""

    def test_build_with_invalid_architecture(self):
        # Given: 잘못된 아키텍처
        adapter = PyTorchModelAdapter()
        config = ModelConfig(
            name="invalid_arch",
            type=ModelType.CLASSIFICATION,
            architecture="unsupported_architecture",
            hyperparameters={
                "input_shape": (10,),
                "layers": [
                    {"type": "dense", "units": 32}
                ]
            }
        )

        # When/Then: 에러 발생
        with pytest.raises(ModelException):
            adapter.build(config)

    def test_train_with_invalid_data_format(self):
        # Given: 잘못된 데이터 형식
        adapter = PyTorchModelAdapter()
        model_config = ModelConfig(
            name="invalid_data",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (5,),
                "layers": [
                    {"type": "dense", "units": 16, "activation": "relu"},
                    {"type": "dense", "units": 2}
                ]
            }
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(epochs=2, batch_size=16, learning_rate=0.01)

        # 잘못된 데이터 형식 (튜플이 아님)
        X_train = np.random.rand(50, 5).astype(np.float32)

        # When/Then: 에러 발생
        with pytest.raises(DataException):
            adapter.train(model, X_train, None, training_config)

    def test_load_nonexistent_model(self):
        # Given: 존재하지 않는 모델 경로
        adapter = PyTorchModelAdapter()

        # When/Then: 에러 발생
        with pytest.raises(ModelException):
            adapter.load("nonexistent_model.pth")


class TestPyTorchAdapterOptimizers:
    """옵티마이저 테스트"""

    def test_use_sgd_optimizer(self):
        # Given: SGD 옵티마이저 설정
        adapter = PyTorchModelAdapter()
        model_config = ModelConfig(
            name="sgd_model",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (5,),
                "layers": [
                    {"type": "dense", "units": 16, "activation": "relu"},
                    {"type": "dense", "units": 2}
                ]
            }
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(
            epochs=2,
            batch_size=16,
            learning_rate=0.01,
            optimizer=OptimizerType.SGD
        )

        X_train = np.random.rand(50, 5).astype(np.float32)
        y_train = np.random.randint(0, 2, size=(50,))

        # When: SGD로 학습
        metrics = adapter.train(model, (X_train, y_train), None, training_config)

        # Then: 학습 성공
        assert isinstance(metrics, dict)
        assert "loss" in metrics

    def test_use_rmsprop_optimizer(self):
        # Given: RMSprop 옵티마이저 설정
        adapter = PyTorchModelAdapter()
        model_config = ModelConfig(
            name="rmsprop_model",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (5,),
                "layers": [
                    {"type": "dense", "units": 16, "activation": "relu"},
                    {"type": "dense", "units": 2}
                ]
            }
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(
            epochs=2,
            batch_size=16,
            learning_rate=0.01,
            optimizer=OptimizerType.RMSPROP
        )

        X_train = np.random.rand(50, 5).astype(np.float32)
        y_train = np.random.randint(0, 2, size=(50,))

        # When: RMSprop로 학습
        metrics = adapter.train(model, (X_train, y_train), None, training_config)

        # Then: 학습 성공
        assert isinstance(metrics, dict)
        assert "loss" in metrics

    def test_use_adamw_optimizer(self):
        # Given: AdamW 옵티마이저 설정
        adapter = PyTorchModelAdapter()
        model_config = ModelConfig(
            name="adamw_model",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (5,),
                "layers": [
                    {"type": "dense", "units": 16, "activation": "relu"},
                    {"type": "dense", "units": 2}
                ]
            }
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(
            epochs=2,
            batch_size=16,
            learning_rate=0.01,
            optimizer=OptimizerType.ADAMW
        )

        X_train = np.random.rand(50, 5).astype(np.float32)
        y_train = np.random.randint(0, 2, size=(50,))

        # When: AdamW로 학습
        metrics = adapter.train(model, (X_train, y_train), None, training_config)

        # Then: 학습 성공
        assert isinstance(metrics, dict)
        assert "loss" in metrics
