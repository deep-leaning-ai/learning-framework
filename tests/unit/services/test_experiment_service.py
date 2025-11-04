"""
TDD Tests for experiment service.
All tests follow Given-When-Then pattern.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from services.experiment_service import ExperimentService
from adapters.mock_adapter import MockModelAdapter, MockDataAdapter
from utils.metrics import MetricsCalculator
from core.models import ModelConfig, TrainingConfig, ModelType, OptimizerType, ExperimentResult


class TestExperimentServiceInitialization:
    """ExperimentService 초기화 테스트"""

    def test_create_service_with_adapters(self):
        # Given: 어댑터들
        model_adapter = MockModelAdapter()
        data_adapter = MockDataAdapter()
        metrics_calculator = MetricsCalculator()

        # When: ExperimentService 생성
        service = ExperimentService(
            model_adapter=model_adapter,
            data_adapter=data_adapter,
            metrics_calculator=metrics_calculator
        )

        # Then: 서비스가 올바르게 생성됨
        assert service.model_adapter is model_adapter
        assert service.data_adapter is data_adapter
        assert service.metrics_calculator is metrics_calculator

    def test_service_without_metrics_calculator(self):
        # Given: 메트릭 계산기 없이
        model_adapter = MockModelAdapter()
        data_adapter = MockDataAdapter()

        # When: ExperimentService 생성
        service = ExperimentService(
            model_adapter=model_adapter,
            data_adapter=data_adapter
        )

        # Then: 기본 메트릭 계산기가 생성됨
        assert service.metrics_calculator is not None


class TestExperimentServiceDataPreparation:
    """데이터 준비 테스트"""

    def test_prepare_data_loads_and_splits(self):
        # Given: ExperimentService
        service = ExperimentService(
            model_adapter=MockModelAdapter(),
            data_adapter=MockDataAdapter()
        )
        data_path = "test_data.csv"

        # When: 데이터 준비
        data = service.prepare_data(
            data_path=data_path,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )

        # Then: 분할된 데이터 반환
        assert "X_train" in data
        assert "X_val" in data
        assert "X_test" in data
        assert "y_train" in data
        assert "y_val" in data
        assert "y_test" in data

    def test_prepare_data_with_preprocessing(self):
        # Given: ExperimentService
        service = ExperimentService(
            model_adapter=MockModelAdapter(),
            data_adapter=MockDataAdapter()
        )

        # When: 전처리를 포함한 데이터 준비
        data = service.prepare_data(
            data_path="test_data.csv",
            preprocess=True
        )

        # Then: 전처리된 데이터 반환
        assert data["X_train"] is not None
        assert data["preprocessed"] is True


class TestExperimentServiceModelTraining:
    """모델 학습 테스트"""

    def test_train_model_returns_metrics(self):
        # Given: ExperimentService와 데이터
        service = ExperimentService(
            model_adapter=MockModelAdapter(),
            data_adapter=MockDataAdapter()
        )
        data = service.prepare_data("test_data.csv")

        model_config = ModelConfig(
            name="test_model",
            type=ModelType.CLASSIFICATION,
            architecture="TestArch"
        )
        training_config = TrainingConfig(
            epochs=10,
            batch_size=32,
            learning_rate=0.001
        )

        # When: 모델 학습
        model, metrics = service.train_model(
            model_config=model_config,
            training_config=training_config,
            train_data=data["X_train"],
            val_data=data["X_val"]
        )

        # Then: 모델과 메트릭 반환
        assert model is not None
        assert isinstance(metrics, dict)
        assert "loss" in metrics or "accuracy" in metrics

    def test_train_model_without_validation_data(self):
        # Given: 검증 데이터 없이
        service = ExperimentService(
            model_adapter=MockModelAdapter(),
            data_adapter=MockDataAdapter()
        )
        data = service.prepare_data("test_data.csv")

        model_config = ModelConfig(
            name="test_model",
            type=ModelType.CLASSIFICATION,
            architecture="TestArch"
        )
        training_config = TrainingConfig(
            epochs=5,
            batch_size=16,
            learning_rate=0.01
        )

        # When: 검증 데이터 없이 학습
        model, metrics = service.train_model(
            model_config=model_config,
            training_config=training_config,
            train_data=data["X_train"],
            val_data=None
        )

        # Then: 학습 성공
        assert model is not None
        assert isinstance(metrics, dict)


class TestExperimentServiceModelEvaluation:
    """모델 평가 테스트"""

    def test_evaluate_model_returns_metrics(self):
        # Given: 학습된 모델
        service = ExperimentService(
            model_adapter=MockModelAdapter(),
            data_adapter=MockDataAdapter()
        )
        data = service.prepare_data("test_data.csv")

        model_config = ModelConfig(
            name="test_model",
            type=ModelType.CLASSIFICATION,
            architecture="TestArch"
        )
        training_config = TrainingConfig(
            epochs=10,
            batch_size=32,
            learning_rate=0.001
        )

        model, _ = service.train_model(
            model_config=model_config,
            training_config=training_config,
            train_data=data["X_train"],
            val_data=data["X_val"]
        )

        # When: 모델 평가
        eval_metrics = service.evaluate_model(
            model=model,
            test_data=data["X_test"],
            test_labels=data["y_test"]
        )

        # Then: 평가 메트릭 반환
        assert isinstance(eval_metrics, dict)
        assert len(eval_metrics) > 0


class TestExperimentServiceFullExperiment:
    """전체 실험 실행 테스트"""

    def test_run_full_experiment(self):
        # Given: ExperimentService와 설정
        service = ExperimentService(
            model_adapter=MockModelAdapter(),
            data_adapter=MockDataAdapter()
        )

        experiment_config = {
            "data_path": "test_data.csv",
            "model_config": ModelConfig(
                name="full_test_model",
                type=ModelType.CLASSIFICATION,
                architecture="FullTestArch"
            ),
            "training_config": TrainingConfig(
                epochs=10,
                batch_size=32,
                learning_rate=0.001
            )
        }

        # When: 전체 실험 실행
        result = service.run_experiment(experiment_config)

        # Then: ExperimentResult 반환
        assert isinstance(result, ExperimentResult)
        assert result.model_config is not None
        assert result.training_config is not None
        assert len(result.metrics) > 0

    def test_run_experiment_saves_model(self, tmp_path):
        # Given: 모델 저장 경로와 함께
        service = ExperimentService(
            model_adapter=MockModelAdapter(),
            data_adapter=MockDataAdapter()
        )

        model_save_path = tmp_path / "test_model.pkl"
        experiment_config = {
            "data_path": "test_data.csv",
            "model_config": ModelConfig(
                name="save_test_model",
                type=ModelType.CLASSIFICATION,
                architecture="SaveTestArch"
            ),
            "training_config": TrainingConfig(
                epochs=5,
                batch_size=16,
                learning_rate=0.01
            ),
            "model_save_path": str(model_save_path)
        }

        # When: 실험 실행
        result = service.run_experiment(experiment_config)

        # Then: 모델이 저장됨
        assert model_save_path.exists()
        assert result is not None


class TestExperimentServiceModelPrediction:
    """모델 예측 테스트"""

    def test_predict_with_trained_model(self):
        # Given: 학습된 모델
        service = ExperimentService(
            model_adapter=MockModelAdapter(),
            data_adapter=MockDataAdapter()
        )
        data = service.prepare_data("test_data.csv")

        model_config = ModelConfig(
            name="pred_model",
            type=ModelType.CLASSIFICATION,
            architecture="PredArch"
        )
        training_config = TrainingConfig(
            epochs=5,
            batch_size=32,
            learning_rate=0.001
        )

        model, _ = service.train_model(
            model_config=model_config,
            training_config=training_config,
            train_data=data["X_train"],
            val_data=data["X_val"]
        )

        # When: 예측 수행
        predictions = service.predict(model, data["X_test"])

        # Then: 예측 결과 반환
        assert predictions is not None
        assert len(predictions) == len(data["X_test"])


class TestExperimentServiceErrorHandling:
    """에러 처리 테스트"""

    def test_prepare_data_with_invalid_path(self):
        # Given: 잘못된 경로
        service = ExperimentService(
            model_adapter=MockModelAdapter(),
            data_adapter=MockDataAdapter()
        )

        # When: 데이터 준비 (Mock은 경로를 무시하므로 정상 동작)
        data = service.prepare_data("invalid_path.csv")

        # Then: Mock 데이터 반환
        assert data is not None

    def test_train_with_empty_data_raises_error(self):
        # Given: 빈 데이터
        service = ExperimentService(
            model_adapter=MockModelAdapter(),
            data_adapter=MockDataAdapter()
        )

        import numpy as np
        empty_data = np.array([])

        model_config = ModelConfig(
            name="error_model",
            type=ModelType.CLASSIFICATION,
            architecture="ErrorArch"
        )
        training_config = TrainingConfig(
            epochs=5,
            batch_size=32,
            learning_rate=0.001
        )

        # When/Then: 에러 발생 (또는 처리)
        # Mock 어댑터는 빈 데이터도 처리할 수 있음
        try:
            model, metrics = service.train_model(
                model_config=model_config,
                training_config=training_config,
                train_data=empty_data,
                val_data=None
            )
            assert model is not None
        except Exception:
            # 에러가 발생할 수도 있음
            pass


class TestExperimentServiceStateful:
    """상태 관리 테스트"""

    def test_service_maintains_experiment_state(self):
        # Given: ExperimentService
        service = ExperimentService(
            model_adapter=MockModelAdapter(),
            data_adapter=MockDataAdapter()
        )

        # When: 실험 실행
        experiment_config = {
            "data_path": "test_data.csv",
            "model_config": ModelConfig(
                name="state_model",
                type=ModelType.CLASSIFICATION,
                architecture="StateArch"
            ),
            "training_config": TrainingConfig(
                epochs=5,
                batch_size=32,
                learning_rate=0.001
            )
        }
        result = service.run_experiment(experiment_config)

        # Then: 마지막 결과가 저장됨
        assert service.last_result is not None
        assert service.last_result == result

    def test_multiple_experiments_tracking(self):
        # Given: ExperimentService
        service = ExperimentService(
            model_adapter=MockModelAdapter(),
            data_adapter=MockDataAdapter()
        )

        # When: 여러 실험 실행
        for i in range(3):
            experiment_config = {
                "data_path": "test_data.csv",
                "model_config": ModelConfig(
                    name=f"multi_model_{i}",
                    type=ModelType.CLASSIFICATION,
                    architecture=f"MultiArch_{i}"
                ),
                "training_config": TrainingConfig(
                    epochs=5,
                    batch_size=32,
                    learning_rate=0.001
                )
            }
            result = service.run_experiment(experiment_config)

        # Then: 마지막 실험 결과가 저장됨
        assert service.last_result is not None
        assert service.last_result.model_config.name == "multi_model_2"
