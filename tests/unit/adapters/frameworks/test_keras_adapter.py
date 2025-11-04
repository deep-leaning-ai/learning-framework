"""
TDD Tests for Keras adapter.
All tests follow Given-When-Then pattern.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, Any

from learning_framework.adapters.frameworks.keras_adapter import KerasModelAdapter
from learning_framework.core.models import ModelConfig, TrainingConfig, ModelType, OptimizerType
from learning_framework.core.exceptions import ModelException, DataException


class TestKerasAdapterInitialization:
    """Keras 어댑터 초기화 테스트"""

    def test_create_keras_adapter(self):
        # Given: Keras 어댑터
        # When: KerasModelAdapter 생성
        adapter = KerasModelAdapter()

        # Then: 어댑터가 올바르게 생성됨
        assert adapter is not None


class TestKerasAdapterModelBuilding:
    """모델 빌드 테스트"""

    def test_build_sequential_classification_model(self):
        # Given: 분류용 모델 설정
        adapter = KerasModelAdapter()
        config = ModelConfig(
            name="test_classifier",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dense", "units": 10, "activation": "softmax"}
            ],
            input_shape=(20,)
        )

        # When: 모델 빌드
        model = adapter.build(config)

        # Then: Keras 모델이 생성됨
        assert model is not None
        assert hasattr(model, 'layers')
        assert len(model.layers) == 3

    def test_build_sequential_regression_model(self):
        # Given: 회귀용 모델 설정
        adapter = KerasModelAdapter()
        config = ModelConfig(
            name="test_regressor",
            type=ModelType.REGRESSION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dense", "units": 1}
            ],
            input_shape=(10,)
        )

        # When: 모델 빌드
        model = adapter.build(config)

        # Then: 회귀 모델이 생성됨
        assert model is not None
        assert hasattr(model, 'layers')

    def test_build_model_with_dropout(self):
        # Given: Dropout 레이어를 포함한 설정
        adapter = KerasModelAdapter()
        config = ModelConfig(
            name="test_dropout",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dropout", "rate": 0.5},
                {"type": "dense", "units": 10, "activation": "softmax"}
            ],
            input_shape=(20,)
        )

        # When: 모델 빌드
        model = adapter.build(config)

        # Then: Dropout 레이어가 포함된 모델 생성
        assert model is not None
        assert len(model.layers) == 3

    def test_build_model_with_batch_normalization(self):
        # Given: BatchNormalization 레이어를 포함한 설정
        adapter = KerasModelAdapter()
        config = ModelConfig(
            name="test_batchnorm",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "batch_normalization"},
                {"type": "dense", "units": 10, "activation": "softmax"}
            ],
            input_shape=(20,)
        )

        # When: 모델 빌드
        model = adapter.build(config)

        # Then: BatchNormalization 레이어가 포함된 모델 생성
        assert model is not None


class TestKerasAdapterModelTraining:
    """모델 학습 테스트"""

    def test_train_classification_model(self):
        # Given: 분류 모델과 학습 데이터
        adapter = KerasModelAdapter()
        model_config = ModelConfig(
            name="train_classifier",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dense", "units": 3, "activation": "softmax"}
            ],
            input_shape=(10,)
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(
            epochs=5,
            batch_size=32,
            learning_rate=0.001,
            optimizer=OptimizerType.ADAM
        )

        # 학습 데이터 생성
        X_train = np.random.rand(100, 10)
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
        adapter = KerasModelAdapter()
        model_config = ModelConfig(
            name="train_with_val",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dense", "units": 2, "activation": "softmax"}
            ],
            input_shape=(5,)
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(
            epochs=3,
            batch_size=16,
            learning_rate=0.01
        )

        X_train = np.random.rand(80, 5)
        y_train = np.random.randint(0, 2, size=(80,))
        X_val = np.random.rand(20, 5)
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
        adapter = KerasModelAdapter()
        model_config = ModelConfig(
            name="train_regressor",
            type=ModelType.REGRESSION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dense", "units": 1}
            ],
            input_shape=(8,)
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(
            epochs=5,
            batch_size=32,
            learning_rate=0.01
        )

        X_train = np.random.rand(100, 8)
        y_train = np.random.rand(100, 1)

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


class TestKerasAdapterModelEvaluation:
    """모델 평가 테스트"""

    def test_evaluate_classification_model(self):
        # Given: 학습된 분류 모델
        adapter = KerasModelAdapter()
        model_config = ModelConfig(
            name="eval_classifier",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dense", "units": 2, "activation": "softmax"}
            ],
            input_shape=(5,)
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(epochs=2, batch_size=16, learning_rate=0.01)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, size=(50,))
        adapter.train(model, (X_train, y_train), None, training_config)

        # 테스트 데이터
        X_test = np.random.rand(20, 5)
        y_test = np.random.randint(0, 2, size=(20,))

        # When: 모델 평가
        metrics = adapter.evaluate(model, (X_test, y_test))

        # Then: 평가 메트릭 반환
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_evaluate_regression_model(self):
        # Given: 학습된 회귀 모델
        adapter = KerasModelAdapter()
        model_config = ModelConfig(
            name="eval_regressor",
            type=ModelType.REGRESSION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 16, "activation": "relu"},
                {"type": "dense", "units": 1}
            ],
            input_shape=(4,)
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(epochs=2, batch_size=16, learning_rate=0.01)
        X_train = np.random.rand(50, 4)
        y_train = np.random.rand(50, 1)
        adapter.train(model, (X_train, y_train), None, training_config)

        # 테스트 데이터
        X_test = np.random.rand(20, 4)
        y_test = np.random.rand(20, 1)

        # When: 회귀 모델 평가
        metrics = adapter.evaluate(model, (X_test, y_test))

        # Then: 회귀 메트릭 반환
        assert isinstance(metrics, dict)
        assert "loss" in metrics


class TestKerasAdapterModelPrediction:
    """모델 예측 테스트"""

    def test_predict_classification(self):
        # Given: 학습된 분류 모델
        adapter = KerasModelAdapter()
        model_config = ModelConfig(
            name="pred_classifier",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 16, "activation": "relu"},
                {"type": "dense", "units": 3, "activation": "softmax"}
            ],
            input_shape=(6,)
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(epochs=2, batch_size=16, learning_rate=0.01)
        X_train = np.random.rand(50, 6)
        y_train = np.random.randint(0, 3, size=(50,))
        adapter.train(model, (X_train, y_train), None, training_config)

        # When: 예측 수행
        X_test = np.random.rand(10, 6)
        predictions = adapter.predict(model, X_test)

        # Then: 예측 결과 반환
        assert predictions is not None
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)

    def test_predict_regression(self):
        # Given: 학습된 회귀 모델
        adapter = KerasModelAdapter()
        model_config = ModelConfig(
            name="pred_regressor",
            type=ModelType.REGRESSION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 16, "activation": "relu"},
                {"type": "dense", "units": 1}
            ],
            input_shape=(4,)
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(epochs=2, batch_size=16, learning_rate=0.01)
        X_train = np.random.rand(50, 4)
        y_train = np.random.rand(50, 1)
        adapter.train(model, (X_train, y_train), None, training_config)

        # When: 회귀 예측
        X_test = np.random.rand(10, 4)
        predictions = adapter.predict(model, X_test)

        # Then: 연속형 예측 결과 반환
        assert predictions is not None
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)


class TestKerasAdapterModelSaveLoad:
    """모델 저장/로드 테스트"""

    def test_save_and_load_model(self, tmp_path):
        # Given: 학습된 모델
        adapter = KerasModelAdapter()
        model_config = ModelConfig(
            name="save_load_model",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 16, "activation": "relu"},
                {"type": "dense", "units": 2, "activation": "softmax"}
            ],
            input_shape=(5,)
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(epochs=2, batch_size=16, learning_rate=0.01)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, size=(50,))
        adapter.train(model, (X_train, y_train), None, training_config)

        # 원본 예측
        X_test = np.random.rand(10, 5)
        original_predictions = adapter.predict(model, X_test)

        # When: 모델 저장 및 로드
        model_path = tmp_path / "test_model.keras"
        adapter.save(model, str(model_path))
        loaded_model = adapter.load(str(model_path))

        # Then: 로드된 모델이 동일한 예측 수행
        loaded_predictions = adapter.predict(loaded_model, X_test)
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions, decimal=5)

    def test_save_creates_directory_if_not_exists(self, tmp_path):
        # Given: 존재하지 않는 디렉토리 경로
        adapter = KerasModelAdapter()
        model_config = ModelConfig(
            name="dir_test",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 8, "activation": "relu"},
                {"type": "dense", "units": 2, "activation": "softmax"}
            ],
            input_shape=(3,)
        )
        model = adapter.build(model_config)

        nested_path = tmp_path / "nested" / "dir" / "model.keras"

        # When: 모델 저장
        adapter.save(model, str(nested_path))

        # Then: 디렉토리가 생성되고 모델이 저장됨
        assert nested_path.exists()


class TestKerasAdapterErrorHandling:
    """에러 처리 테스트"""

    def test_build_with_invalid_layer_type(self):
        # Given: 잘못된 레이어 타입
        adapter = KerasModelAdapter()
        config = ModelConfig(
            name="invalid_layer",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            layers=[
                {"type": "invalid_layer_type", "units": 32}
            ],
            input_shape=(10,)
        )

        # When/Then: 에러 발생
        with pytest.raises(ModelException):
            adapter.build(config)

    def test_train_with_mismatched_data_shape(self):
        # Given: 입력 형태가 맞지 않는 데이터
        adapter = KerasModelAdapter()
        model_config = ModelConfig(
            name="shape_mismatch",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 16, "activation": "relu"},
                {"type": "dense", "units": 2, "activation": "softmax"}
            ],
            input_shape=(5,)
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(epochs=2, batch_size=16, learning_rate=0.01)

        # 잘못된 형태의 데이터
        X_train = np.random.rand(50, 10)  # input_shape=(5,)인데 10차원 데이터
        y_train = np.random.randint(0, 2, size=(50,))

        # When/Then: 에러 발생
        with pytest.raises((ModelException, ValueError)):
            adapter.train(model, (X_train, y_train), None, training_config)

    def test_load_nonexistent_model(self):
        # Given: 존재하지 않는 모델 경로
        adapter = KerasModelAdapter()

        # When/Then: 에러 발생
        with pytest.raises(ModelException):
            adapter.load("nonexistent_model.keras")


class TestKerasAdapterOptimizers:
    """옵티마이저 테스트"""

    def test_use_sgd_optimizer(self):
        # Given: SGD 옵티마이저 설정
        adapter = KerasModelAdapter()
        model_config = ModelConfig(
            name="sgd_model",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 16, "activation": "relu"},
                {"type": "dense", "units": 2, "activation": "softmax"}
            ],
            input_shape=(5,)
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(
            epochs=2,
            batch_size=16,
            learning_rate=0.01,
            optimizer=OptimizerType.SGD
        )

        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, size=(50,))

        # When: SGD로 학습
        metrics = adapter.train(model, (X_train, y_train), None, training_config)

        # Then: 학습 성공
        assert isinstance(metrics, dict)
        assert "loss" in metrics

    def test_use_rmsprop_optimizer(self):
        # Given: RMSprop 옵티마이저 설정
        adapter = KerasModelAdapter()
        model_config = ModelConfig(
            name="rmsprop_model",
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            layers=[
                {"type": "dense", "units": 16, "activation": "relu"},
                {"type": "dense", "units": 2, "activation": "softmax"}
            ],
            input_shape=(5,)
        )
        model = adapter.build(model_config)

        training_config = TrainingConfig(
            epochs=2,
            batch_size=16,
            learning_rate=0.01,
            optimizer=OptimizerType.RMSPROP
        )

        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, size=(50,))

        # When: RMSprop로 학습
        metrics = adapter.train(model, (X_train, y_train), None, training_config)

        # Then: 학습 성공
        assert isinstance(metrics, dict)
        assert "loss" in metrics
