"""
Configuration schemas using Pydantic.

These schemas define and validate configuration from YAML/JSON files.
They provide automatic validation, type checking, and default values.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class FrameworkType(str, Enum):
    """프레임워크 타입"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    KERAS = "keras"
    HUGGINGFACE = "huggingface"


class TaskType(str, Enum):
    """태스크 타입"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"


class OptimizerType(str, Enum):
    """옵티마이저 타입"""
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"


class SchedulerType(str, Enum):
    """스케줄러 타입"""
    STEP = "step"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class ExperimentConfig(BaseModel):
    """
    실험 기본 설정 스키마

    Attributes:
        name: 실험 이름
        description: 실험 설명
        version: 실험 버전
        tags: 태그 리스트
        seed: 랜덤 시드
    """
    name: str = Field(..., min_length=1, description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    version: str = Field(default="1.0.0", description="Experiment version")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """이름 검증"""
        if not v or not v.strip():
            raise ValueError("Experiment name cannot be empty or whitespace")
        return v.strip()


class ModelConfigSchema(BaseModel):
    """
    모델 설정 스키마

    Attributes:
        name: 모델 이름
        framework: 프레임워크 타입
        task_type: 태스크 타입
        architecture: 아키텍처 이름
        hyperparameters: 하이퍼파라미터 딕셔너리
        pretrained: 사전학습 모델 사용 여부
        checkpoint_path: 체크포인트 경로
    """
    name: str = Field(..., min_length=1, description="Model name")
    framework: FrameworkType = Field(..., description="Framework type")
    task_type: TaskType = Field(..., description="Task type")
    architecture: str = Field(..., min_length=1, description="Model architecture")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model hyperparameters"
    )
    pretrained: bool = Field(default=False, description="Use pretrained weights")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint")

    @field_validator('name', 'architecture')
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """비어있지 않은지 검증"""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace")
        return v.strip()


class TrainingConfigSchema(BaseModel):
    """
    학습 설정 스키마

    Attributes:
        epochs: 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        optimizer: 옵티마이저 타입
        scheduler: 스케줄러 타입
        early_stopping: 조기 종료 여부
        patience: 조기 종료 patience
        validation_split: 검증 데이터 비율
        save_best_only: 최고 모델만 저장
        checkpoint_dir: 체크포인트 저장 디렉토리
    """
    epochs: int = Field(..., gt=0, description="Number of training epochs")
    batch_size: int = Field(..., gt=0, description="Batch size")
    learning_rate: float = Field(..., gt=0.0, le=1.0, description="Learning rate")
    optimizer: OptimizerType = Field(
        default=OptimizerType.ADAM,
        description="Optimizer type"
    )
    scheduler: Optional[SchedulerType] = Field(None, description="Learning rate scheduler")
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(default=10, ge=0, description="Early stopping patience")
    validation_split: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="Validation data split ratio"
    )
    save_best_only: bool = Field(default=True, description="Save only best model")
    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory to save checkpoints"
    )

    @field_validator('learning_rate')
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """학습률 범위 검증"""
        if v <= 0.0 or v > 1.0:
            raise ValueError("Learning rate must be in range (0, 1]")
        return v


class DataConfigSchema(BaseModel):
    """
    데이터 설정 스키마

    Attributes:
        name: 데이터셋 이름
        path: 데이터 경로
        train_path: 학습 데이터 경로
        val_path: 검증 데이터 경로
        test_path: 테스트 데이터 경로
        format: 데이터 포맷
        shuffle: 셔플 여부
        num_workers: 데이터 로딩 워커 수
        preprocessing: 전처리 설정
    """
    name: str = Field(..., min_length=1, description="Dataset name")
    path: Optional[str] = Field(None, description="Main data path")
    train_path: Optional[str] = Field(None, description="Training data path")
    val_path: Optional[str] = Field(None, description="Validation data path")
    test_path: Optional[str] = Field(None, description="Test data path")
    format: str = Field(default="csv", description="Data format (csv, json, parquet)")
    shuffle: bool = Field(default=True, description="Shuffle data")
    num_workers: int = Field(default=4, ge=0, description="Number of data loading workers")
    preprocessing: Dict[str, Any] = Field(
        default_factory=dict,
        description="Preprocessing configurations"
    )

    @model_validator(mode='after')
    def validate_paths(self) -> 'DataConfigSchema':
        """데이터 경로 검증"""
        if not self.path and not (self.train_path and self.val_path):
            raise ValueError(
                "Either 'path' or both 'train_path' and 'val_path' must be provided"
            )
        return self


class LoggingConfigSchema(BaseModel):
    """
    로깅 설정 스키마

    Attributes:
        level: 로그 레벨
        log_dir: 로그 저장 디렉토리
        console: 콘솔 로깅 여부
        file: 파일 로깅 여부
        tensorboard: TensorBoard 사용 여부
        mlflow: MLflow 사용 여부
        wandb: W&B 사용 여부
    """
    level: str = Field(
        default="INFO",
        description="Logging level",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    log_dir: str = Field(default="logs", description="Log directory")
    console: bool = Field(default=True, description="Enable console logging")
    file: bool = Field(default=True, description="Enable file logging")
    tensorboard: bool = Field(default=False, description="Enable TensorBoard")
    mlflow: bool = Field(default=False, description="Enable MLflow tracking")
    wandb: bool = Field(default=False, description="Enable Weights & Biases tracking")


class TrackingConfigSchema(BaseModel):
    """
    실험 추적 설정 스키마

    Attributes:
        backend: 추적 백엔드 (mlflow, wandb, tensorboard)
        experiment_name: 실험 이름
        run_name: 실행 이름
        log_interval: 로깅 주기 (steps)
        save_artifacts: 아티팩트 저장 여부
    """
    backend: str = Field(
        default="mlflow",
        description="Tracking backend",
        pattern="^(mlflow|wandb|tensorboard|none)$"
    )
    experiment_name: Optional[str] = Field(None, description="Experiment name")
    run_name: Optional[str] = Field(None, description="Run name")
    log_interval: int = Field(default=10, gt=0, description="Logging interval in steps")
    save_artifacts: bool = Field(default=True, description="Save artifacts")


class FullConfigSchema(BaseModel):
    """
    전체 설정 스키마 (모든 설정을 포함)

    Attributes:
        experiment: 실험 설정
        model: 모델 설정
        training: 학습 설정
        data: 데이터 설정
        logging: 로깅 설정
        tracking: 추적 설정
    """
    experiment: ExperimentConfig
    model: ModelConfigSchema
    training: TrainingConfigSchema
    data: DataConfigSchema
    logging: LoggingConfigSchema = Field(default_factory=LoggingConfigSchema)
    tracking: TrackingConfigSchema = Field(default_factory=TrackingConfigSchema)

    class Config:
        """Pydantic 설정"""
        validate_assignment = True
        extra = "forbid"  # 추가 필드 금지

    @model_validator(mode='after')
    def validate_config_consistency(self) -> 'FullConfigSchema':
        """설정 일관성 검증"""
        # 실험 이름과 추적 실험 이름 동기화
        if not self.tracking.experiment_name:
            self.tracking.experiment_name = self.experiment.name

        return self
