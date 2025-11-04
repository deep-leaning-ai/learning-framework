"""
Core port interfaces using Protocol.

These protocols define the contracts that adapters must implement
to integrate with the framework. They enable dependency inversion
and testability through duck typing.
"""

from typing import Protocol, Dict, Any, Optional, Tuple, List
import numpy as np
from learning_framework.core.models import ModelConfig, TrainingConfig, ExperimentResult


class ModelPort(Protocol):
    """
    모델 관련 포트 인터페이스

    AI 프레임워크 어댑터들이 구현해야 하는 모델 관련 메서드를 정의합니다.
    """

    def build(self, config: ModelConfig) -> Any:
        """
        모델 구축

        Args:
            config: 모델 설정

        Returns:
            구축된 모델 객체 (프레임워크별로 다름)
        """
        ...

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
            model: 학습할 모델
            train_data: 학습 데이터
            val_data: 검증 데이터 (선택)
            config: 학습 설정

        Returns:
            학습 결과 메트릭 딕셔너리
        """
        ...

    def evaluate(self, model: Any, test_data: Any) -> Dict[str, float]:
        """
        모델 평가

        Args:
            model: 평가할 모델
            test_data: 테스트 데이터

        Returns:
            평가 메트릭 딕셔너리
        """
        ...

    def predict(self, model: Any, inputs: np.ndarray) -> np.ndarray:
        """
        예측 수행

        Args:
            model: 예측에 사용할 모델
            inputs: 입력 데이터 (numpy array)

        Returns:
            예측 결과 (numpy array로 정규화됨)
        """
        ...

    def save(self, model: Any, path: str) -> None:
        """
        모델 저장

        Args:
            model: 저장할 모델
            path: 저장 경로
        """
        ...

    def load(self, path: str) -> Any:
        """
        모델 로드

        Args:
            path: 모델 파일 경로

        Returns:
            로드된 모델 객체
        """
        ...


class DataPort(Protocol):
    """
    데이터 관련 포트 인터페이스

    데이터 로딩, 전처리, 분할을 담당하는 어댑터들이 구현해야 하는 메서드를 정의합니다.
    """

    def load(self, path: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터 로드

        Args:
            path: 데이터 파일 경로
            **kwargs: 추가 로드 옵션

        Returns:
            (X, y) 튜플 - 피처와 타겟
        """
        ...

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
        ...

    def preprocess(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        데이터 전처리

        Args:
            X: 피처 데이터
            y: 타겟 데이터 (선택)

        Returns:
            (전처리된 X, 전처리된 y) 튜플
        """
        ...

    def to_framework_format(self, X: np.ndarray, y: np.ndarray, framework: str) -> Any:
        """
        프레임워크별 데이터 포맷으로 변환

        Args:
            X: 피처 데이터
            y: 타겟 데이터
            framework: 프레임워크 이름 ('tensorflow', 'pytorch', 'sklearn' 등)

        Returns:
            프레임워크별 데이터 객체 (DataLoader, Dataset 등)
        """
        ...


class TrackingPort(Protocol):
    """
    실험 추적 포트 인터페이스

    MLflow, W&B 등의 실험 추적 도구와 통합하기 위한 메서드를 정의합니다.
    """

    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
        """
        실험 실행 시작

        Args:
            experiment_name: 실험 이름
            run_name: 실행 이름 (선택)

        Returns:
            실행 ID
        """
        ...

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        파라미터 로깅

        Args:
            params: 하이퍼파라미터 딕셔너리
        """
        ...

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        메트릭 로깅

        Args:
            metrics: 메트릭 딕셔너리
            step: 스텝/에폭 번호 (선택)
        """
        ...

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        아티팩트 저장

        Args:
            local_path: 로컬 파일 경로
            artifact_path: 저장될 아티팩트 경로 (선택)
        """
        ...

    def log_model(self, model: Any, artifact_path: str) -> None:
        """
        모델 로깅

        Args:
            model: 저장할 모델
            artifact_path: 모델 저장 경로
        """
        ...

    def end_run(self) -> None:
        """실험 실행 종료"""
        ...

    def get_run_id(self) -> Optional[str]:
        """
        현재 실행 ID 조회

        Returns:
            실행 ID 또는 None
        """
        ...


class MetricsPort(Protocol):
    """
    메트릭 계산 포트 인터페이스

    다양한 평가 메트릭을 계산하는 메서드를 정의합니다.
    """

    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        정확도 계산

        Args:
            y_true: 실제 값
            y_pred: 예측 값

        Returns:
            정확도 (0-1)
        """
        ...

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
            average: 평균 방법 ('micro', 'macro', 'weighted')

        Returns:
            {'precision': float, 'recall': float, 'f1': float} 딕셔너리
        """
        ...

    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Confusion Matrix 계산

        Args:
            y_true: 실제 값
            y_pred: 예측 값

        Returns:
            Confusion Matrix (2D numpy array)
        """
        ...

    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        회귀 메트릭 계산 (MSE, RMSE, MAE, R2)

        Args:
            y_true: 실제 값
            y_pred: 예측 값

        Returns:
            회귀 메트릭 딕셔너리
        """
        ...


class ConfigPort(Protocol):
    """
    설정 관리 포트 인터페이스

    설정 파일 로드 및 검증을 담당하는 메서드를 정의합니다.
    """

    def load(self, path: str) -> Dict[str, Any]:
        """
        설정 파일 로드

        Args:
            path: 설정 파일 경로

        Returns:
            설정 딕셔너리
        """
        ...

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        설정 검증

        Args:
            config: 설정 딕셔너리

        Returns:
            검증 성공 여부
        """
        ...

    def merge(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        설정 병합

        Args:
            base_config: 기본 설정
            override_config: 오버라이드 설정

        Returns:
            병합된 설정 딕셔너리
        """
        ...

    def save(self, config: Dict[str, Any], path: str) -> None:
        """
        설정 파일 저장

        Args:
            config: 설정 딕셔너리
            path: 저장 경로
        """
        ...
