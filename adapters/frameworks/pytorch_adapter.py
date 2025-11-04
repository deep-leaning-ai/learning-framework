"""
PyTorch model adapter implementation.

Implements ModelPort for PyTorch models.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    raise ImportError(
        "PyTorch is required for PyTorchModelAdapter. "
        "Install with: pip install torch"
    )

from core.contracts import ModelPort
from core.models import ModelConfig, TrainingConfig, ModelType, OptimizerType
from core.exceptions import ModelException, DataException


class PyTorchModelAdapter:
    """
    PyTorch 모델 어댑터

    PyTorch를 사용한 딥러닝 모델 구축, 학습, 평가를 담당합니다.
    """

    def __init__(self):
        """Initialize PyTorch adapter with device detection"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self, config: ModelConfig) -> Any:
        """
        PyTorch 모델 구축

        Args:
            config: 모델 설정

        Returns:
            구축된 PyTorch 모델

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

    def _build_sequential_model(self, config: ModelConfig) -> nn.Module:
        """
        Sequential 모델 구축

        Args:
            config: 모델 설정

        Returns:
            Sequential 모델
        """
        layers = []
        input_dim = config.hyperparameters.get("input_shape", (0,))[0] if config.hyperparameters.get("input_shape") else None

        for i, layer_config in enumerate(config.hyperparameters.get("layers", [])):
            layer = self._create_layer(layer_config, input_dim if i == 0 else None)

            if layer is not None:
                layers.append(layer)

            # Update input_dim for next layer if current layer is Linear
            if layer_config["type"] == "dense" and input_dim is not None:
                input_dim = layer_config["units"]

        model = nn.Sequential(*layers)
        model.to(self.device)
        return model

    def _create_layer(self, layer_config: Dict[str, Any], input_dim: Optional[int] = None) -> Optional[nn.Module]:
        """
        레이어 생성

        Args:
            layer_config: 레이어 설정
            input_dim: 입력 차원 (첫 레이어용)

        Returns:
            PyTorch 레이어

        Raises:
            ModelException: 지원하지 않는 레이어 타입
        """
        layer_type = layer_config["type"].lower()

        try:
            if layer_type == "dense":
                units = layer_config["units"]
                if input_dim is None:
                    raise ModelException(
                        "input_dim is required for the first Dense layer",
                        details={"layer_type": layer_type}
                    )

                layer = nn.Linear(input_dim, units)

                # activation이 있으면 Sequential로 묶어서 반환
                activation = layer_config.get("activation", None)
                if activation:
                    activation_layer = self._get_activation(activation)
                    return nn.Sequential(layer, activation_layer)
                return layer

            elif layer_type == "dropout":
                rate = layer_config["rate"]
                return nn.Dropout(p=rate)

            elif layer_type == "batch_normalization":
                # BatchNorm은 input feature 수를 알아야 하므로 동적으로 처리 필요
                # 여기서는 기본값으로 처리
                return nn.BatchNorm1d(num_features=1)  # 실제로는 이전 레이어의 output dim 필요

            elif layer_type == "conv2d":
                in_channels = layer_config.get("in_channels", 1)
                filters = layer_config["filters"]
                kernel_size = layer_config["kernel_size"]
                activation = layer_config.get("activation", None)

                layer = nn.Conv2d(in_channels, filters, kernel_size)

                if activation:
                    activation_layer = self._get_activation(activation)
                    return nn.Sequential(layer, activation_layer)
                return layer

            elif layer_type == "maxpooling2d":
                pool_size = layer_config.get("pool_size", (2, 2))
                if isinstance(pool_size, tuple):
                    pool_size = pool_size[0]
                return nn.MaxPool2d(kernel_size=pool_size)

            elif layer_type == "flatten":
                return nn.Flatten()

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

    def _get_activation(self, activation: str) -> nn.Module:
        """
        활성화 함수 반환

        Args:
            activation: 활성화 함수 이름

        Returns:
            PyTorch 활성화 함수 모듈
        """
        activation = activation.lower()

        if activation == "relu":
            return nn.ReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "softmax":
            return nn.Softmax(dim=1)
        elif activation == "tanh":
            return nn.Tanh()
        else:
            raise ModelException(
                f"Unsupported activation: {activation}",
                details={"activation": activation}
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
            model: PyTorch 모델
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

            # NumPy to Tensor
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.LongTensor(y_train).to(self.device) if y_train.dtype in [np.int32, np.int64] else torch.FloatTensor(y_train).to(self.device)

            # DataLoader 생성
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

            # 검증 데이터 처리
            val_loader = None
            if val_data is not None:
                if isinstance(val_data, tuple) and len(val_data) == 2:
                    X_val, y_val = val_data
                    X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                    y_val_tensor = torch.LongTensor(y_val).to(self.device) if y_val.dtype in [np.int32, np.int64] else torch.FloatTensor(y_val).to(self.device)
                    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
                else:
                    raise DataException(
                        "val_data must be a tuple of (X, y)",
                        details={"type": type(val_data)}
                    )

            # 옵티마이저 생성
            optimizer = self._create_optimizer(model, config)

            # 손실 함수 선택
            criterion = self._get_loss_function(y_train)

            # 학습 루프
            model.train()
            epoch_metrics = {}

            for epoch in range(config.epochs):
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()

                    outputs = model(batch_X)

                    # 손실 계산
                    if isinstance(criterion, nn.CrossEntropyLoss):
                        loss = criterion(outputs, batch_y)
                        _, predicted = torch.max(outputs.data, 1)
                        train_correct += (predicted == batch_y).sum().item()
                        train_total += batch_y.size(0)
                    elif isinstance(criterion, nn.MSELoss):
                        if batch_y.dim() == 1:
                            batch_y = batch_y.unsqueeze(1)
                        loss = criterion(outputs, batch_y)
                    else:
                        loss = criterion(outputs.squeeze(), batch_y)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                # 에폭 평균 손실
                avg_train_loss = train_loss / len(train_loader)
                epoch_metrics["loss"] = avg_train_loss

                # 분류 정확도
                if train_total > 0:
                    epoch_metrics["accuracy"] = train_correct / train_total

                # 검증 평가
                if val_loader is not None:
                    val_metrics = self._evaluate_loader(model, val_loader, criterion)
                    epoch_metrics["val_loss"] = val_metrics["loss"]
                    if "accuracy" in val_metrics:
                        epoch_metrics["val_accuracy"] = val_metrics["accuracy"]

            return epoch_metrics

        except Exception as e:
            if isinstance(e, (ModelException, DataException)):
                raise
            raise ModelException(
                f"Training failed: {str(e)}",
                details={"error": str(e)}
            )

    def _create_optimizer(self, model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
        """
        옵티마이저 생성

        Args:
            model: PyTorch 모델
            config: 학습 설정

        Returns:
            PyTorch 옵티마이저
        """
        lr = config.learning_rate

        if config.optimizer == OptimizerType.ADAM:
            return optim.Adam(model.parameters(), lr=lr)
        elif config.optimizer == OptimizerType.SGD:
            return optim.SGD(model.parameters(), lr=lr)
        elif config.optimizer == OptimizerType.RMSPROP:
            return optim.RMSprop(model.parameters(), lr=lr)
        elif config.optimizer == OptimizerType.ADAMW:
            return optim.AdamW(model.parameters(), lr=lr)
        else:
            # 기본값은 Adam
            return optim.Adam(model.parameters(), lr=lr)

    def _get_loss_function(self, y_data: np.ndarray) -> nn.Module:
        """
        손실 함수 선택

        Args:
            y_data: 타겟 데이터

        Returns:
            손실 함수
        """
        # 정수형이면 분류 문제
        if y_data.dtype in [np.int32, np.int64]:
            # 다중 클래스 분류
            num_classes = len(np.unique(y_data))
            if num_classes > 2:
                return nn.CrossEntropyLoss()
            else:
                # 이진 분류
                return nn.BCEWithLogitsLoss()
        else:
            # 회귀 문제
            return nn.MSELoss()

    def _evaluate_loader(self, model: nn.Module, data_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """
        DataLoader로 모델 평가

        Args:
            model: PyTorch 모델
            data_loader: 데이터 로더
            criterion: 손실 함수

        Returns:
            평가 메트릭
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                outputs = model(batch_X)

                # 손실 계산
                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss = criterion(outputs, batch_y)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)
                elif isinstance(criterion, nn.MSELoss):
                    if batch_y.dim() == 1:
                        batch_y = batch_y.unsqueeze(1)
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)

                total_loss += loss.item()

        metrics = {"loss": total_loss / len(data_loader)}

        if total > 0:
            metrics["accuracy"] = correct / total

        model.train()
        return metrics

    def evaluate(self, model: Any, test_data: Any) -> Dict[str, float]:
        """
        모델 평가

        Args:
            model: PyTorch 모델
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

            # NumPy to Tensor
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_test_tensor = torch.LongTensor(y_test).to(self.device) if y_test.dtype in [np.int32, np.int64] else torch.FloatTensor(y_test).to(self.device)

            # DataLoader 생성
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # 손실 함수
            criterion = self._get_loss_function(y_test)

            # 평가
            metrics = self._evaluate_loader(model, test_loader, criterion)

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
            model: PyTorch 모델
            inputs: 입력 데이터

        Returns:
            예측 결과
        """
        try:
            model.eval()

            # NumPy to Tensor
            inputs_tensor = torch.FloatTensor(inputs).to(self.device)

            with torch.no_grad():
                outputs = model(inputs_tensor)

                # 분류 모델인지 확인
                if outputs.shape[-1] > 1:
                    # 다중 클래스: argmax
                    _, predictions = torch.max(outputs, 1)
                    return predictions.cpu().numpy()
                else:
                    # 회귀 또는 이진 분류
                    return outputs.squeeze().cpu().numpy()

        except Exception as e:
            raise ModelException(
                f"Prediction failed: {str(e)}",
                details={"error": str(e)}
            )

    def save(self, model: Any, path: str) -> None:
        """
        모델 저장

        Args:
            model: PyTorch 모델
            path: 저장 경로
        """
        try:
            # 디렉토리 생성
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 모델 저장 (state_dict 방식)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
            }, str(save_path))

        except Exception as e:
            raise ModelException(
                f"Failed to save model: {str(e)}",
                details={"path": path}
            )

    def load(self, path: str, model_structure: Optional[nn.Module] = None) -> Any:
        """
        모델 로드

        Args:
            path: 모델 파일 경로
            model_structure: 모델 구조 (필요 시)

        Returns:
            로드된 PyTorch 모델
        """
        try:
            if not Path(path).exists():
                raise ModelException(
                    f"Model file not found: {path}",
                    details={"path": path}
                )

            checkpoint = torch.load(path, map_location=self.device)

            if model_structure is None:
                raise ModelException(
                    "model_structure is required for PyTorch model loading",
                    details={"path": path}
                )

            model_structure.load_state_dict(checkpoint['model_state_dict'])
            model_structure.to(self.device)

            return model_structure

        except Exception as e:
            if isinstance(e, ModelException):
                raise
            raise ModelException(
                f"Failed to load model: {str(e)}",
                details={"path": path}
            )
