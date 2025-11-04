"""
Keras model adapter implementation.

Implements ModelPort for Keras/TensorFlow models.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers
except ImportError:
    raise ImportError(
        "TensorFlow/Keras is required for KerasModelAdapter. "
        "Install with: pip install tensorflow"
    )

from learning_framework.core.contracts import ModelPort
from learning_framework.core.models import ModelConfig, TrainingConfig, ModelType, OptimizerType
from learning_framework.core.exceptions import ModelException, DataException


class KerasModelAdapter:
    """
    Keras 모델 어댑터

    Keras/TensorFlow를 사용한 딥러닝 모델 구축, 학습, 평가를 담당합니다.
    """

    def build(self, config: ModelConfig) -> Any:
        """
        Keras 모델 구축

        Args:
            config: 모델 설정

        Returns:
            구축된 Keras 모델

        Raises:
            ModelException: 모델 구축 실패 시
        """
        try:
            if config.architecture.lower() == "sequential":
                return self._build_sequential_model(config)
            else:
                raise ModelException(
                    f"Unsupported architecture: {config.architecture}",
                    details={"architecture": config.architecture}
                )
        except Exception as e:
            if isinstance(e, ModelException):
                raise
            raise ModelException(
                f"Failed to build model: {str(e)}",
                details={"config": config.name}
            )

    def _build_sequential_model(self, config: ModelConfig) -> keras.Model:
        """
        Sequential 모델 구축

        Args:
            config: 모델 설정

        Returns:
            Sequential 모델
        """
        model = models.Sequential()

        # 첫 번째 레이어에 input_shape 추가
        first_layer = True

        for layer_config in config.layers:
            layer = self._create_layer(layer_config, config.input_shape if first_layer else None)
            model.add(layer)

            # Dense나 Conv 레이어가 추가되면 input_shape는 더 이상 필요 없음
            if layer_config["type"] in ["dense", "conv2d", "conv1d"]:
                first_layer = False

        return model

    def _create_layer(self, layer_config: Dict[str, Any], input_shape: Optional[Tuple] = None) -> layers.Layer:
        """
        레이어 생성

        Args:
            layer_config: 레이어 설정
            input_shape: 입력 형태 (첫 레이어용)

        Returns:
            Keras 레이어

        Raises:
            ModelException: 지원하지 않는 레이어 타입
        """
        layer_type = layer_config["type"].lower()

        try:
            if layer_type == "dense":
                units = layer_config["units"]
                activation = layer_config.get("activation", None)
                if input_shape:
                    return layers.Dense(units, activation=activation, input_shape=input_shape)
                else:
                    return layers.Dense(units, activation=activation)

            elif layer_type == "dropout":
                rate = layer_config["rate"]
                return layers.Dropout(rate)

            elif layer_type == "batch_normalization":
                return layers.BatchNormalization()

            elif layer_type == "conv2d":
                filters = layer_config["filters"]
                kernel_size = layer_config["kernel_size"]
                activation = layer_config.get("activation", None)
                if input_shape:
                    return layers.Conv2D(filters, kernel_size, activation=activation, input_shape=input_shape)
                else:
                    return layers.Conv2D(filters, kernel_size, activation=activation)

            elif layer_type == "maxpooling2d":
                pool_size = layer_config.get("pool_size", (2, 2))
                return layers.MaxPooling2D(pool_size)

            elif layer_type == "flatten":
                return layers.Flatten()

            else:
                raise ModelException(
                    f"Unsupported layer type: {layer_type}",
                    details={"layer_type": layer_type}
                )
        except KeyError as e:
            raise ModelException(
                f"Missing required parameter for {layer_type} layer: {str(e)}",
                details={"layer_type": layer_type}
            )

    def train(
        self,
        model: Any,
        train_data: Any,
        val_data: Optional[Any],
        config: TrainingConfig
    ) -> Dict[str, float]:
        """
        모델 학습

        Args:
            model: Keras 모델
            train_data: 학습 데이터 (X, y) 튜플
            val_data: 검증 데이터 (X, y) 튜플 (선택)
            config: 학습 설정

        Returns:
            학습 메트릭 딕셔너리
        """
        try:
            # 데이터 추출
            if isinstance(train_data, tuple) and len(train_data) == 2:
                X_train, y_train = train_data
            else:
                raise DataException(
                    "train_data must be a tuple of (X, y)",
                    details={"type": type(train_data)}
                )

            # 검증 데이터 처리
            validation_data = None
            if val_data is not None:
                if isinstance(val_data, tuple) and len(val_data) == 2:
                    validation_data = val_data
                else:
                    raise DataException(
                        "val_data must be a tuple of (X, y)",
                        details={"type": type(val_data)}
                    )

            # 옵티마이저 생성
            optimizer = self._create_optimizer(config)

            # 모델 컴파일 (아직 컴파일되지 않은 경우)
            if not model.optimizer:
                # 출력 레이어를 기반으로 손실 함수와 메트릭 결정
                output_shape = model.layers[-1].output_shape

                if len(output_shape) > 1 and output_shape[-1] > 1:
                    # 다중 클래스 분류
                    loss = "sparse_categorical_crossentropy"
                    metrics = ["accuracy"]
                else:
                    # 회귀 또는 이진 분류
                    # y 데이터 타입으로 판단
                    if y_train.dtype in [np.float32, np.float64]:
                        loss = "mse"
                        metrics = ["mae"]
                    else:
                        loss = "binary_crossentropy"
                        metrics = ["accuracy"]

                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            else:
                # 이미 컴파일된 모델의 옵티마이저 업데이트
                model.optimizer = optimizer

            # 학습
            history = model.fit(
                X_train,
                y_train,
                epochs=config.epochs,
                batch_size=config.batch_size,
                validation_data=validation_data,
                verbose=0  # 출력 억제
            )

            # 마지막 에포크의 메트릭 반환
            result_metrics = {}
            for key, values in history.history.items():
                result_metrics[key] = float(values[-1])

            return result_metrics

        except Exception as e:
            if isinstance(e, (ModelException, DataException)):
                raise
            raise ModelException(
                f"Training failed: {str(e)}",
                details={"error": str(e)}
            )

    def _create_optimizer(self, config: TrainingConfig) -> optimizers.Optimizer:
        """
        옵티마이저 생성

        Args:
            config: 학습 설정

        Returns:
            Keras 옵티마이저
        """
        lr = config.learning_rate

        if config.optimizer == OptimizerType.ADAM:
            return optimizers.Adam(learning_rate=lr)
        elif config.optimizer == OptimizerType.SGD:
            return optimizers.SGD(learning_rate=lr)
        elif config.optimizer == OptimizerType.RMSPROP:
            return optimizers.RMSprop(learning_rate=lr)
        else:
            # 기본값은 Adam
            return optimizers.Adam(learning_rate=lr)

    def evaluate(self, model: Any, test_data: Any) -> Dict[str, float]:
        """
        모델 평가

        Args:
            model: Keras 모델
            test_data: 테스트 데이터 (X, y) 튜플

        Returns:
            평가 메트릭 딕셔너리
        """
        try:
            if isinstance(test_data, tuple) and len(test_data) == 2:
                X_test, y_test = test_data
            else:
                raise DataException(
                    "test_data must be a tuple of (X, y)",
                    details={"type": type(test_data)}
                )

            # 평가
            results = model.evaluate(X_test, y_test, verbose=0)

            # 결과를 딕셔너리로 변환
            metrics = {}
            if isinstance(results, list):
                # metrics_names와 results를 매핑
                for name, value in zip(model.metrics_names, results):
                    metrics[name] = float(value)
            else:
                metrics["loss"] = float(results)

            return metrics

        except Exception as e:
            if isinstance(e, DataException):
                raise
            raise ModelException(
                f"Evaluation failed: {str(e)}",
                details={"error": str(e)}
            )

    def predict(self, model: Any, inputs: np.ndarray) -> np.ndarray:
        """
        예측 수행

        Args:
            model: Keras 모델
            inputs: 입력 데이터

        Returns:
            예측 결과
        """
        try:
            predictions = model.predict(inputs, verbose=0)

            # 분류 모델의 경우 클래스 인덱스 반환
            if predictions.shape[-1] > 1:
                # 다중 클래스: argmax
                return np.argmax(predictions, axis=-1)
            else:
                # 회귀 또는 이진 분류: 그대로 반환
                return predictions.flatten()

        except Exception as e:
            raise ModelException(
                f"Prediction failed: {str(e)}",
                details={"error": str(e)}
            )

    def save(self, model: Any, path: str) -> None:
        """
        모델 저장

        Args:
            model: Keras 모델
            path: 저장 경로
        """
        try:
            # 디렉토리 생성
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Keras 3.0+ 형식으로 저장
            model.save(str(save_path))

        except Exception as e:
            raise ModelException(
                f"Failed to save model: {str(e)}",
                details={"path": path}
            )

    def load(self, path: str) -> Any:
        """
        모델 로드

        Args:
            path: 모델 파일 경로

        Returns:
            로드된 Keras 모델
        """
        try:
            if not Path(path).exists():
                raise ModelException(
                    f"Model file not found: {path}",
                    details={"path": path}
                )

            model = keras.models.load_model(path)
            return model

        except Exception as e:
            if isinstance(e, ModelException):
                raise
            raise ModelException(
                f"Failed to load model: {str(e)}",
                details={"path": path}
            )
