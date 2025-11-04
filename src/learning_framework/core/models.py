"""
Core data models using dataclasses.

These models represent the core domain entities and value objects
without any framework-specific dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class ModelType(Enum):
    """모델 타입 열거형"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"


class OptimizerType(Enum):
    """옵티마이저 타입 열거형"""
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"


@dataclass(frozen=True)
class ModelConfig:
    """
    모델 설정 Value Object

    Args:
        name: 모델 이름
        type: 모델 타입 (classification, regression 등)
        architecture: 아키텍처 이름 (resnet50, bert 등)
        hyperparameters: 하이퍼파라미터 딕셔너리
    """
    name: str
    type: ModelType
    architecture: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Model name cannot be empty")
        if not self.architecture or not self.architecture.strip():
            raise ValueError("Architecture must be specified")


@dataclass(frozen=True)
class TrainingConfig:
    """
    학습 설정 Value Object

    Args:
        epochs: 학습 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        optimizer: 옵티마이저 타입
        early_stopping: 조기 종료 여부
        patience: 조기 종료 patience
        validation_split: 검증 데이터 비율
    """
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: OptimizerType = OptimizerType.ADAM
    early_stopping: bool = True
    patience: int = 10
    validation_split: float = 0.1

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValueError("Learning rate must be in range (0, 1]")
        if self.patience < 0:
            raise ValueError("Patience must be non-negative")
        if not 0 <= self.validation_split < 1:
            raise ValueError("Validation split must be in range [0, 1)")


@dataclass(frozen=True)
class DatasetInfo:
    """
    데이터셋 정보 Value Object

    Args:
        name: 데이터셋 이름
        train_size: 학습 데이터 크기
        val_size: 검증 데이터 크기
        test_size: 테스트 데이터 크기
        num_features: 피처 개수
        num_classes: 클래스 개수 (분류 문제의 경우)
    """
    name: str
    train_size: int
    val_size: int
    test_size: int
    num_features: int
    num_classes: Optional[int] = None

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Dataset name cannot be empty")
        if self.train_size < 0:
            raise ValueError("Train size must be non-negative")
        if self.val_size < 0:
            raise ValueError("Validation size must be non-negative")
        if self.test_size < 0:
            raise ValueError("Test size must be non-negative")
        if self.num_features <= 0:
            raise ValueError("Number of features must be positive")
        if self.num_classes is not None and self.num_classes <= 0:
            raise ValueError("Number of classes must be positive if specified")

    @property
    def total_size(self) -> int:
        """전체 데이터셋 크기"""
        return self.train_size + self.val_size + self.test_size


@dataclass
class ExperimentResult:
    """
    실험 결과 엔티티

    Args:
        experiment_id: 실험 고유 ID
        model_config: 모델 설정
        training_config: 학습 설정
        dataset_info: 데이터셋 정보
        metrics: 메트릭 딕셔너리
        created_at: 생성 시간
        duration_seconds: 실행 시간 (초)
    """
    experiment_id: str
    model_config: ModelConfig
    training_config: TrainingConfig
    dataset_info: DatasetInfo
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.experiment_id or not self.experiment_id.strip():
            raise ValueError("Experiment ID cannot be empty")
        if self.duration_seconds < 0:
            raise ValueError("Duration must be non-negative")

    def add_metric(self, name: str, value: float) -> None:
        """
        메트릭 추가

        Args:
            name: 메트릭 이름
            value: 메트릭 값
        """
        if not name or not name.strip():
            raise ValueError("Metric name cannot be empty")
        self.metrics[name] = value

    def get_metric(self, name: str) -> Optional[float]:
        """
        메트릭 조회

        Args:
            name: 메트릭 이름

        Returns:
            메트릭 값 또는 None
        """
        return self.metrics.get(name)


@dataclass
class TrainingHistory:
    """
    학습 히스토리 엔티티

    Args:
        epoch_metrics: 에폭별 메트릭 리스트
        best_epoch: 최고 성능 에폭
        best_metric_value: 최고 메트릭 값
    """
    epoch_metrics: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_metric_value: float = 0.0

    def add_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        에폭 메트릭 추가

        Args:
            epoch: 에폭 번호
            metrics: 메트릭 딕셔너리
        """
        if epoch < 0:
            raise ValueError("Epoch must be non-negative")
        if not metrics:
            raise ValueError("Metrics cannot be empty")

        self.epoch_metrics.append({
            'epoch': epoch,
            **metrics
        })

    def get_epoch_metric(self, epoch: int, metric_name: str) -> Optional[float]:
        """
        특정 에폭의 메트릭 조회

        Args:
            epoch: 에폭 번호
            metric_name: 메트릭 이름

        Returns:
            메트릭 값 또는 None
        """
        for epoch_data in self.epoch_metrics:
            if epoch_data.get('epoch') == epoch:
                return epoch_data.get(metric_name)
        return None
