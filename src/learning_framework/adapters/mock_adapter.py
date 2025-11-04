"""
Mock adapters for testing and development.

This module provides mock implementations of ModelPort and DataPort
for testing purposes without requiring actual ML frameworks.
"""

import pickle
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from learning_framework.core.models import ModelConfig, TrainingConfig
from learning_framework.core.contracts import ModelPort, DataPort
from learning_framework.utils.exceptions import ModelException, DataException


class MockModelAdapter:
    """
    Mock 모델 어댑터

    ModelPort 프로토콜을 구현하며, 실제 ML 프레임워크 없이
    테스트와 개발에 사용할 수 있는 가짜 모델을 제공합니다.
    """

    def build(self, config: ModelConfig) -> Dict[str, Any]:
        """
        Mock 모델 구축

        Args:
            config: 모델 설정

        Returns:
            Mock 모델 (딕셔너리)
        """
        model = {
            "type": config.name,
            "config": config,
            "architecture": config.architecture,
            "trained": False,
            "weights": None
        }
        return model

    def train(
        self,
        model: Any,
        train_data: Any,
        val_data: Optional[Any],
        config: TrainingConfig
    ) -> Dict[str, float]:
        """
        Mock 학습 수행

        Args:
            model: 학습할 모델
            train_data: 학습 데이터
            val_data: 검증 데이터 (선택)
            config: 학습 설정

        Returns:
            Mock 메트릭 딕셔너리
        """
        # 모델을 "학습됨" 상태로 표시
        if isinstance(model, dict):
            model["trained"] = True
            model["weights"] = np.random.rand(10)  # 가짜 가중치

        # Mock 메트릭 생성
        base_loss = 0.5
        base_accuracy = 0.8

        # 에폭 수에 따라 성능이 향상되는 것처럼 시뮬레이션
        improvement_factor = min(config.epochs / 100, 0.3)

        metrics = {
            "loss": base_loss * (1 - improvement_factor),
            "accuracy": min(base_accuracy + improvement_factor, 0.99)
        }

        # 검증 데이터가 있으면 검증 메트릭도 추가
        if val_data is not None:
            metrics["val_loss"] = metrics["loss"] * 1.1
            metrics["val_accuracy"] = metrics["accuracy"] * 0.95

        return metrics

    def evaluate(self, model: Any, test_data: Any) -> Dict[str, float]:
        """
        Mock 평가 수행

        Args:
            model: 평가할 모델
            test_data: 테스트 데이터

        Returns:
            Mock 평가 메트릭
        """
        # Mock 평가 메트릭 생성
        metrics = {
            "accuracy": 0.85,
            "loss": 0.35,
            "precision": 0.83,
            "recall": 0.87,
            "f1": 0.85
        }
        return metrics

    def predict(self, model: Any, inputs: np.ndarray) -> np.ndarray:
        """
        Mock 예측 수행

        Args:
            model: 예측에 사용할 모델
            inputs: 입력 데이터

        Returns:
            Mock 예측 결과
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)

        # 입력 크기만큼 랜덤 예측 생성
        n_samples = len(inputs)

        # 분류 문제라고 가정하고 0 또는 1 예측
        predictions = np.random.randint(0, 2, size=n_samples)

        return predictions

    def save(self, model: Any, path: str) -> None:
        """
        Mock 모델 저장

        Args:
            model: 저장할 모델
            path: 저장 경로

        Raises:
            ModelException: 저장 실패 시
        """
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            raise ModelException(
                f"Failed to save mock model: {str(e)}",
                context={"path": path}
            ) from e

    def load(self, path: str) -> Any:
        """
        Mock 모델 로드

        Args:
            path: 모델 파일 경로

        Returns:
            로드된 Mock 모델

        Raises:
            ModelException: 로드 실패 시
        """
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError as e:
            raise ModelException(
                f"Model file not found: {path}",
                context={"path": path},
                recovery_hint="Check if the file path is correct"
            ) from e
        except Exception as e:
            raise ModelException(
                f"Failed to load mock model: {str(e)}",
                context={"path": path}
            ) from e


class MockDataAdapter:
    """
    Mock 데이터 어댑터

    DataPort 프로토콜을 구현하며, 실제 데이터 없이
    테스트와 개발에 사용할 수 있는 가짜 데이터를 생성합니다.
    """

    def load(self, path: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mock 데이터 로드

        Args:
            path: 데이터 파일 경로 (사용되지 않음)
            **kwargs: 추가 옵션
                - n_samples: 샘플 수 (기본값: 100)
                - n_features: 피처 수 (기본값: 10)

        Returns:
            (X, y) 튜플 - Mock 피처와 타겟
        """
        n_samples = kwargs.get('n_samples', 100)
        n_features = kwargs.get('n_features', 10)

        # 랜덤 데이터 생성
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, size=n_samples)

        return X, y

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        데이터 분할

        Args:
            X: 피처 데이터
            y: 타겟 데이터
            train_ratio: 학습 데이터 비율
            val_ratio: 검증 데이터 비율
            test_ratio: 테스트 데이터 비율
            random_state: 난수 시드

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test) 튜플
        """
        if random_state is not None:
            np.random.seed(random_state)

        n_samples = len(X)

        # 인덱스 섞기
        indices = np.random.permutation(n_samples)

        # 분할 지점 계산
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        # 인덱스 분할
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        # 데이터 분할
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        데이터 전처리 (Mock)

        Args:
            X: 피처 데이터
            y: 타겟 데이터 (선택)

        Returns:
            (전처리된 X, 전처리된 y) 튜플
        """
        # Mock 전처리: 표준화 시뮬레이션
        X_processed = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # y는 그대로 반환
        y_processed = y if y is not None else None

        return X_processed, y_processed

    def to_framework_format(
        self,
        X: np.ndarray,
        y: np.ndarray,
        framework: str
    ) -> Any:
        """
        프레임워크별 데이터 포맷으로 변환 (Mock)

        Args:
            X: 피처 데이터
            y: 타겟 데이터
            framework: 프레임워크 이름

        Returns:
            Mock 프레임워크 데이터 객체
        """
        # 프레임워크별 Mock 데이터 생성
        if framework.lower() == "sklearn":
            # sklearn은 numpy 그대로 사용
            return {"X": X, "y": y, "framework": "sklearn"}

        elif framework.lower() in ["tensorflow", "keras"]:
            # TensorFlow/Keras Dataset 시뮬레이션
            return {
                "X": X,
                "y": y,
                "framework": "tensorflow",
                "batch_size": 32,
                "shuffle": True
            }

        elif framework.lower() == "pytorch":
            # PyTorch DataLoader 시뮬레이션
            return {
                "X": X,
                "y": y,
                "framework": "pytorch",
                "batch_size": 32,
                "shuffle": True,
                "num_workers": 0
            }

        else:
            # 기본 포맷
            return {"X": X, "y": y, "framework": framework}
