"""
TDD Tests for core data models.
All tests follow Given-When-Then pattern.
"""

import pytest
from datetime import datetime
from core.models import (
    ModelConfig,
    TrainingConfig,
    DatasetInfo,
    ExperimentResult,
    TrainingHistory,
    ModelType,
    OptimizerType,
)


class TestModelConfig:
    """ModelConfig 데이터 클래스 테스트"""

    def test_valid_model_config_creation(self):
        # Given: 유효한 모델 설정 데이터
        name = "test_model"
        model_type = ModelType.CLASSIFICATION
        architecture = "resnet50"
        hyperparameters = {"layers": 50, "dropout": 0.2}

        # When: ModelConfig 생성
        config = ModelConfig(
            name=name,
            type=model_type,
            architecture=architecture,
            hyperparameters=hyperparameters
        )

        # Then: 설정이 올바르게 생성됨
        assert config.name == name
        assert config.type == model_type
        assert config.architecture == architecture
        assert config.hyperparameters == hyperparameters

    def test_empty_name_raises_error(self):
        # Given: 빈 이름
        # When: ModelConfig 생성 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="name cannot be empty"):
            ModelConfig(
                name="",
                type=ModelType.CLASSIFICATION,
                architecture="resnet"
            )

    def test_whitespace_name_raises_error(self):
        # Given: 공백만 있는 이름
        # When: ModelConfig 생성 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="name cannot be empty"):
            ModelConfig(
                name="   ",
                type=ModelType.CLASSIFICATION,
                architecture="resnet"
            )

    def test_empty_architecture_raises_error(self):
        # Given: 빈 아키텍처
        # When: ModelConfig 생성 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="Architecture must be specified"):
            ModelConfig(
                name="test",
                type=ModelType.CLASSIFICATION,
                architecture=""
            )

    def test_model_config_is_immutable(self):
        # Given: 생성된 ModelConfig
        config = ModelConfig(
            name="test",
            type=ModelType.CLASSIFICATION,
            architecture="resnet"
        )

        # When/Then: 속성 변경 시도하면 에러 발생
        with pytest.raises(Exception):  # dataclass frozen error
            config.name = "new_name"


class TestTrainingConfig:
    """TrainingConfig 데이터 클래스 테스트"""

    def test_valid_training_config_creation(self):
        # Given: 유효한 학습 설정 데이터
        epochs = 100
        batch_size = 32
        learning_rate = 0.001

        # When: TrainingConfig 생성
        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # Then: 설정이 올바르게 생성됨
        assert config.epochs == epochs
        assert config.batch_size == batch_size
        assert config.learning_rate == learning_rate
        assert config.optimizer == OptimizerType.ADAM  # default
        assert config.early_stopping is True  # default

    def test_negative_epochs_raises_error(self):
        # Given: 음수 epochs
        # When: TrainingConfig 생성 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="Epochs must be positive"):
            TrainingConfig(
                epochs=-1,
                batch_size=32,
                learning_rate=0.001
            )

    def test_zero_epochs_raises_error(self):
        # Given: 0 epochs
        # When: TrainingConfig 생성 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="Epochs must be positive"):
            TrainingConfig(
                epochs=0,
                batch_size=32,
                learning_rate=0.001
            )

    def test_negative_batch_size_raises_error(self):
        # Given: 음수 batch_size
        # When: TrainingConfig 생성 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="Batch size must be positive"):
            TrainingConfig(
                epochs=10,
                batch_size=-1,
                learning_rate=0.001
            )

    def test_invalid_learning_rate_too_small_raises_error(self):
        # Given: 0 이하의 learning_rate
        # When: TrainingConfig 생성 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="Learning rate must be in range"):
            TrainingConfig(
                epochs=10,
                batch_size=32,
                learning_rate=0.0
            )

    def test_invalid_learning_rate_too_large_raises_error(self):
        # Given: 1 초과의 learning_rate
        # When: TrainingConfig 생성 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="Learning rate must be in range"):
            TrainingConfig(
                epochs=10,
                batch_size=32,
                learning_rate=1.5
            )

    def test_negative_patience_raises_error(self):
        # Given: 음수 patience
        # When: TrainingConfig 생성 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="Patience must be non-negative"):
            TrainingConfig(
                epochs=10,
                batch_size=32,
                learning_rate=0.001,
                patience=-1
            )

    def test_invalid_validation_split_raises_error(self):
        # Given: 범위를 벗어난 validation_split
        # When: TrainingConfig 생성 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="Validation split must be in range"):
            TrainingConfig(
                epochs=10,
                batch_size=32,
                learning_rate=0.001,
                validation_split=1.5
            )


class TestDatasetInfo:
    """DatasetInfo 데이터 클래스 테스트"""

    def test_valid_dataset_info_creation(self):
        # Given: 유효한 데이터셋 정보
        name = "mnist"
        train_size = 50000
        val_size = 10000
        test_size = 10000
        num_features = 784
        num_classes = 10

        # When: DatasetInfo 생성
        info = DatasetInfo(
            name=name,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            num_features=num_features,
            num_classes=num_classes
        )

        # Then: 정보가 올바르게 생성됨
        assert info.name == name
        assert info.train_size == train_size
        assert info.val_size == val_size
        assert info.test_size == test_size
        assert info.num_features == num_features
        assert info.num_classes == num_classes

    def test_total_size_property(self):
        # Given: 데이터셋 정보
        info = DatasetInfo(
            name="test",
            train_size=100,
            val_size=20,
            test_size=30,
            num_features=10
        )

        # When: total_size 조회
        total = info.total_size

        # Then: 올바른 합계 반환
        assert total == 150

    def test_empty_name_raises_error(self):
        # Given: 빈 이름
        # When: DatasetInfo 생성 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="name cannot be empty"):
            DatasetInfo(
                name="",
                train_size=100,
                val_size=20,
                test_size=30,
                num_features=10
            )

    def test_negative_train_size_raises_error(self):
        # Given: 음수 train_size
        # When: DatasetInfo 생성 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="Train size must be non-negative"):
            DatasetInfo(
                name="test",
                train_size=-1,
                val_size=20,
                test_size=30,
                num_features=10
            )

    def test_negative_num_features_raises_error(self):
        # Given: 음수 또는 0인 num_features
        # When: DatasetInfo 생성 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="Number of features must be positive"):
            DatasetInfo(
                name="test",
                train_size=100,
                val_size=20,
                test_size=30,
                num_features=0
            )

    def test_invalid_num_classes_raises_error(self):
        # Given: 0 이하의 num_classes
        # When: DatasetInfo 생성 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="Number of classes must be positive"):
            DatasetInfo(
                name="test",
                train_size=100,
                val_size=20,
                test_size=30,
                num_features=10,
                num_classes=0
            )


class TestExperimentResult:
    """ExperimentResult 엔티티 테스트"""

    def test_valid_experiment_result_creation(self):
        # Given: 유효한 실험 결과 데이터
        experiment_id = "exp_001"
        model_config = ModelConfig(
            name="test_model",
            type=ModelType.CLASSIFICATION,
            architecture="resnet"
        )
        training_config = TrainingConfig(
            epochs=10,
            batch_size=32,
            learning_rate=0.001
        )
        dataset_info = DatasetInfo(
            name="test_data",
            train_size=100,
            val_size=20,
            test_size=30,
            num_features=10
        )

        # When: ExperimentResult 생성
        result = ExperimentResult(
            experiment_id=experiment_id,
            model_config=model_config,
            training_config=training_config,
            dataset_info=dataset_info
        )

        # Then: 결과가 올바르게 생성됨
        assert result.experiment_id == experiment_id
        assert result.model_config == model_config
        assert result.training_config == training_config
        assert result.dataset_info == dataset_info
        assert isinstance(result.created_at, datetime)

    def test_add_metric(self):
        # Given: ExperimentResult 인스턴스
        result = ExperimentResult(
            experiment_id="exp_001",
            model_config=ModelConfig(
                name="test",
                type=ModelType.CLASSIFICATION,
                architecture="resnet"
            ),
            training_config=TrainingConfig(
                epochs=10,
                batch_size=32,
                learning_rate=0.001
            ),
            dataset_info=DatasetInfo(
                name="test",
                train_size=100,
                val_size=20,
                test_size=30,
                num_features=10
            )
        )

        # When: 메트릭 추가
        result.add_metric("accuracy", 0.95)
        result.add_metric("loss", 0.15)

        # Then: 메트릭이 올바르게 저장됨
        assert result.metrics["accuracy"] == 0.95
        assert result.metrics["loss"] == 0.15

    def test_get_metric(self):
        # Given: 메트릭이 추가된 ExperimentResult
        result = ExperimentResult(
            experiment_id="exp_001",
            model_config=ModelConfig(
                name="test",
                type=ModelType.CLASSIFICATION,
                architecture="resnet"
            ),
            training_config=TrainingConfig(
                epochs=10,
                batch_size=32,
                learning_rate=0.001
            ),
            dataset_info=DatasetInfo(
                name="test",
                train_size=100,
                val_size=20,
                test_size=30,
                num_features=10
            )
        )
        result.add_metric("accuracy", 0.95)

        # When: 메트릭 조회
        accuracy = result.get_metric("accuracy")
        nonexistent = result.get_metric("nonexistent")

        # Then: 올바른 값 반환
        assert accuracy == 0.95
        assert nonexistent is None

    def test_empty_metric_name_raises_error(self):
        # Given: ExperimentResult 인스턴스
        result = ExperimentResult(
            experiment_id="exp_001",
            model_config=ModelConfig(
                name="test",
                type=ModelType.CLASSIFICATION,
                architecture="resnet"
            ),
            training_config=TrainingConfig(
                epochs=10,
                batch_size=32,
                learning_rate=0.001
            ),
            dataset_info=DatasetInfo(
                name="test",
                train_size=100,
                val_size=20,
                test_size=30,
                num_features=10
            )
        )

        # When: 빈 메트릭 이름으로 추가 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="Metric name cannot be empty"):
            result.add_metric("", 0.95)


class TestTrainingHistory:
    """TrainingHistory 엔티티 테스트"""

    def test_add_epoch(self):
        # Given: TrainingHistory 인스턴스
        history = TrainingHistory()

        # When: 에폭 메트릭 추가
        history.add_epoch(0, {"loss": 0.5, "accuracy": 0.8})
        history.add_epoch(1, {"loss": 0.3, "accuracy": 0.9})

        # Then: 히스토리에 올바르게 저장됨
        assert len(history.epoch_metrics) == 2
        assert history.epoch_metrics[0]["epoch"] == 0
        assert history.epoch_metrics[0]["loss"] == 0.5
        assert history.epoch_metrics[1]["accuracy"] == 0.9

    def test_get_epoch_metric(self):
        # Given: 에폭 메트릭이 추가된 TrainingHistory
        history = TrainingHistory()
        history.add_epoch(0, {"loss": 0.5, "accuracy": 0.8})
        history.add_epoch(1, {"loss": 0.3, "accuracy": 0.9})

        # When: 특정 에폭의 메트릭 조회
        loss_epoch_0 = history.get_epoch_metric(0, "loss")
        accuracy_epoch_1 = history.get_epoch_metric(1, "accuracy")
        nonexistent = history.get_epoch_metric(2, "loss")

        # Then: 올바른 값 반환
        assert loss_epoch_0 == 0.5
        assert accuracy_epoch_1 == 0.9
        assert nonexistent is None

    def test_negative_epoch_raises_error(self):
        # Given: TrainingHistory 인스턴스
        history = TrainingHistory()

        # When: 음수 에폭으로 추가 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="Epoch must be non-negative"):
            history.add_epoch(-1, {"loss": 0.5})

    def test_empty_metrics_raises_error(self):
        # Given: TrainingHistory 인스턴스
        history = TrainingHistory()

        # When: 빈 메트릭으로 추가 시도
        # Then: ValueError 발생
        with pytest.raises(ValueError, match="Metrics cannot be empty"):
            history.add_epoch(0, {})
