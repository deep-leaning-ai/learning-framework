"""
TDD Tests for configuration schemas (Pydantic).
All tests follow Given-When-Then pattern.
"""

import pytest
from pydantic import ValidationError
from learning_framework.config.schema import (
    ExperimentConfig,
    ModelConfigSchema,
    TrainingConfigSchema,
    DataConfigSchema,
    LoggingConfigSchema,
    TrackingConfigSchema,
    FullConfigSchema,
    FrameworkType,
    TaskType,
    OptimizerType,
    SchedulerType,
)


class TestExperimentConfig:
    """ExperimentConfig 스키마 테스트"""

    def test_valid_experiment_config(self):
        # Given: 유효한 실험 설정 데이터
        data = {
            "name": "test_experiment",
            "description": "Test description",
            "version": "1.0.0",
            "tags": ["test", "classification"],
            "seed": 42
        }

        # When: ExperimentConfig 생성
        config = ExperimentConfig(**data)

        # Then: 올바르게 생성됨
        assert config.name == "test_experiment"
        assert config.description == "Test description"
        assert config.version == "1.0.0"
        assert config.tags == ["test", "classification"]
        assert config.seed == 42

    def test_default_values(self):
        # Given: 최소한의 필수 필드만 제공
        data = {"name": "minimal_experiment"}

        # When: ExperimentConfig 생성
        config = ExperimentConfig(**data)

        # Then: 기본값이 적용됨
        assert config.name == "minimal_experiment"
        assert config.description is None
        assert config.version == "1.0.0"
        assert config.tags == []
        assert config.seed == 42

    def test_empty_name_raises_error(self):
        # Given: 빈 이름
        data = {"name": ""}

        # When/Then: ValidationError 발생
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(**data)

        assert "name" in str(exc_info.value).lower()

    def test_whitespace_name_raises_error(self):
        # Given: 공백만 있는 이름
        data = {"name": "   "}

        # When/Then: ValidationError 발생
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(**data)

        assert "whitespace" in str(exc_info.value).lower()

    def test_negative_seed_raises_error(self):
        # Given: 음수 seed
        data = {"name": "test", "seed": -1}

        # When/Then: ValidationError 발생
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(**data)

        assert "seed" in str(exc_info.value).lower()


class TestModelConfigSchema:
    """ModelConfigSchema 테스트"""

    def test_valid_model_config(self):
        # Given: 유효한 모델 설정 데이터
        data = {
            "name": "resnet50",
            "framework": "pytorch",
            "task_type": "classification",
            "architecture": "ResNet",
            "hyperparameters": {"layers": 50},
            "pretrained": True
        }

        # When: ModelConfigSchema 생성
        config = ModelConfigSchema(**data)

        # Then: 올바르게 생성됨
        assert config.name == "resnet50"
        assert config.framework == FrameworkType.PYTORCH
        assert config.task_type == TaskType.CLASSIFICATION
        assert config.architecture == "ResNet"
        assert config.hyperparameters == {"layers": 50}
        assert config.pretrained is True

    def test_default_hyperparameters(self):
        # Given: hyperparameters 없는 설정
        data = {
            "name": "simple_model",
            "framework": "tensorflow",
            "task_type": "regression",
            "architecture": "MLP"
        }

        # When: ModelConfigSchema 생성
        config = ModelConfigSchema(**data)

        # Then: 빈 딕셔너리가 기본값
        assert config.hyperparameters == {}
        assert config.pretrained is False

    def test_invalid_framework_raises_error(self):
        # Given: 잘못된 프레임워크
        data = {
            "name": "test",
            "framework": "invalid_framework",
            "task_type": "classification",
            "architecture": "test"
        }

        # When/Then: ValidationError 발생
        with pytest.raises(ValidationError) as exc_info:
            ModelConfigSchema(**data)

        assert "framework" in str(exc_info.value).lower()

    def test_empty_architecture_raises_error(self):
        # Given: 빈 아키텍처
        data = {
            "name": "test",
            "framework": "pytorch",
            "task_type": "classification",
            "architecture": ""
        }

        # When/Then: ValidationError 발생
        with pytest.raises(ValidationError) as exc_info:
            ModelConfigSchema(**data)

        error_str = str(exc_info.value).lower()
        assert "architecture" in error_str and ("empty" in error_str or "character" in error_str)


class TestTrainingConfigSchema:
    """TrainingConfigSchema 테스트"""

    def test_valid_training_config(self):
        # Given: 유효한 학습 설정 데이터
        data = {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "scheduler": "cosine",
            "early_stopping": True,
            "patience": 10
        }

        # When: TrainingConfigSchema 생성
        config = TrainingConfigSchema(**data)

        # Then: 올바르게 생성됨
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.optimizer == OptimizerType.ADAM
        assert config.scheduler == SchedulerType.COSINE
        assert config.early_stopping is True
        assert config.patience == 10

    def test_default_values(self):
        # Given: 필수 필드만 제공
        data = {
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01
        }

        # When: TrainingConfigSchema 생성
        config = TrainingConfigSchema(**data)

        # Then: 기본값이 적용됨
        assert config.optimizer == OptimizerType.ADAM
        assert config.scheduler is None
        assert config.early_stopping is True
        assert config.patience == 10
        assert config.validation_split == 0.1

    def test_zero_epochs_raises_error(self):
        # Given: 0 epochs
        data = {
            "epochs": 0,
            "batch_size": 32,
            "learning_rate": 0.001
        }

        # When/Then: ValidationError 발생
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfigSchema(**data)

        assert "epochs" in str(exc_info.value).lower()

    def test_negative_batch_size_raises_error(self):
        # Given: 음수 batch_size
        data = {
            "epochs": 10,
            "batch_size": -1,
            "learning_rate": 0.001
        }

        # When/Then: ValidationError 발생
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfigSchema(**data)

        assert "batch_size" in str(exc_info.value).lower()

    def test_learning_rate_out_of_range_raises_error(self):
        # Given: 범위를 벗어난 learning_rate
        data = {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 1.5
        }

        # When/Then: ValidationError 발생
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfigSchema(**data)

        assert "learning_rate" in str(exc_info.value).lower()

    def test_negative_patience_raises_error(self):
        # Given: 음수 patience
        data = {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "patience": -1
        }

        # When/Then: ValidationError 발생
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfigSchema(**data)

        assert "patience" in str(exc_info.value).lower()


class TestDataConfigSchema:
    """DataConfigSchema 테스트"""

    def test_valid_data_config_with_single_path(self):
        # Given: 단일 경로를 가진 데이터 설정
        data = {
            "name": "mnist",
            "path": "data/mnist.csv"
        }

        # When: DataConfigSchema 생성
        config = DataConfigSchema(**data)

        # Then: 올바르게 생성됨
        assert config.name == "mnist"
        assert config.path == "data/mnist.csv"
        assert config.format == "csv"

    def test_valid_data_config_with_split_paths(self):
        # Given: 분할된 경로를 가진 데이터 설정
        data = {
            "name": "cifar10",
            "train_path": "data/train.csv",
            "val_path": "data/val.csv",
            "test_path": "data/test.csv"
        }

        # When: DataConfigSchema 생성
        config = DataConfigSchema(**data)

        # Then: 올바르게 생성됨
        assert config.name == "cifar10"
        assert config.train_path == "data/train.csv"
        assert config.val_path == "data/val.csv"
        assert config.test_path == "data/test.csv"

    def test_no_paths_raises_error(self):
        # Given: 경로가 없는 데이터 설정
        data = {"name": "test"}

        # When/Then: ValidationError 발생
        with pytest.raises(ValidationError) as exc_info:
            DataConfigSchema(**data)

        assert "path" in str(exc_info.value).lower()

    def test_only_train_path_raises_error(self):
        # Given: train_path만 있는 설정
        data = {
            "name": "test",
            "train_path": "data/train.csv"
        }

        # When/Then: ValidationError 발생 (val_path 필요)
        with pytest.raises(ValidationError) as exc_info:
            DataConfigSchema(**data)

        assert "path" in str(exc_info.value).lower()


class TestFullConfigSchema:
    """FullConfigSchema 통합 테스트"""

    def test_valid_full_config(self):
        # Given: 완전한 설정 데이터
        data = {
            "experiment": {
                "name": "full_test",
                "seed": 123
            },
            "model": {
                "name": "resnet",
                "framework": "pytorch",
                "task_type": "classification",
                "architecture": "ResNet50"
            },
            "training": {
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "data": {
                "name": "dataset",
                "path": "data/test.csv"
            }
        }

        # When: FullConfigSchema 생성
        config = FullConfigSchema(**data)

        # Then: 모든 설정이 올바르게 생성됨
        assert config.experiment.name == "full_test"
        assert config.model.name == "resnet"
        assert config.training.epochs == 50
        assert config.data.name == "dataset"
        assert config.logging is not None
        assert config.tracking is not None

    def test_experiment_name_syncs_to_tracking(self):
        # Given: tracking.experiment_name이 없는 설정
        data = {
            "experiment": {"name": "sync_test"},
            "model": {
                "name": "model",
                "framework": "pytorch",
                "task_type": "classification",
                "architecture": "test"
            },
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "data": {
                "name": "data",
                "path": "data.csv"
            }
        }

        # When: FullConfigSchema 생성
        config = FullConfigSchema(**data)

        # Then: experiment 이름이 tracking으로 동기화됨
        assert config.tracking.experiment_name == "sync_test"

    def test_extra_fields_forbidden(self):
        # Given: 정의되지 않은 추가 필드를 포함한 설정
        data = {
            "experiment": {"name": "test"},
            "model": {
                "name": "model",
                "framework": "pytorch",
                "task_type": "classification",
                "architecture": "test"
            },
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "data": {
                "name": "data",
                "path": "data.csv"
            },
            "unknown_field": "should_fail"
        }

        # When/Then: ValidationError 발생
        with pytest.raises(ValidationError) as exc_info:
            FullConfigSchema(**data)

        assert "extra" in str(exc_info.value).lower() or "unknown" in str(exc_info.value).lower()

    def test_missing_required_section_raises_error(self):
        # Given: 필수 섹션이 없는 설정
        data = {
            "experiment": {"name": "test"},
            "model": {
                "name": "model",
                "framework": "pytorch",
                "task_type": "classification",
                "architecture": "test"
            }
            # training과 data 섹션 누락
        }

        # When/Then: ValidationError 발생
        with pytest.raises(ValidationError) as exc_info:
            FullConfigSchema(**data)

        error_str = str(exc_info.value).lower()
        assert "training" in error_str or "data" in error_str


class TestLoggingConfigSchema:
    """LoggingConfigSchema 테스트"""

    def test_valid_logging_config(self):
        # Given: 유효한 로깅 설정
        data = {
            "level": "DEBUG",
            "log_dir": "custom_logs",
            "console": False,
            "mlflow": True
        }

        # When: LoggingConfigSchema 생성
        config = LoggingConfigSchema(**data)

        # Then: 올바르게 생성됨
        assert config.level == "DEBUG"
        assert config.log_dir == "custom_logs"
        assert config.console is False
        assert config.mlflow is True

    def test_invalid_log_level_raises_error(self):
        # Given: 잘못된 로그 레벨
        data = {"level": "INVALID"}

        # When/Then: ValidationError 발생
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfigSchema(**data)

        assert "level" in str(exc_info.value).lower()


class TestTrackingConfigSchema:
    """TrackingConfigSchema 테스트"""

    def test_valid_tracking_config(self):
        # Given: 유효한 추적 설정
        data = {
            "backend": "wandb",
            "experiment_name": "test_exp",
            "run_name": "run_001",
            "log_interval": 50
        }

        # When: TrackingConfigSchema 생성
        config = TrackingConfigSchema(**data)

        # Then: 올바르게 생성됨
        assert config.backend == "wandb"
        assert config.experiment_name == "test_exp"
        assert config.run_name == "run_001"
        assert config.log_interval == 50

    def test_invalid_backend_raises_error(self):
        # Given: 잘못된 백엔드
        data = {"backend": "invalid_backend"}

        # When/Then: ValidationError 발생
        with pytest.raises(ValidationError) as exc_info:
            TrackingConfigSchema(**data)

        assert "backend" in str(exc_info.value).lower()
