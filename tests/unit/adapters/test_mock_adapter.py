"""
TDD Tests for mock adapters (ModelPort and DataPort implementations).
All tests follow Given-When-Then pattern.
"""

import pytest
import numpy as np
from typing import Dict, Any

from learning_framework.adapters.mock_adapter import MockModelAdapter, MockDataAdapter
from learning_framework.core.models import ModelConfig, TrainingConfig, ModelType, OptimizerType


class TestMockModelAdapter:
    """MockModelAdapter 테스트"""

    def test_build_returns_mock_model(self):
        # Given: MockModelAdapter와 ModelConfig
        adapter = MockModelAdapter()
        config = ModelConfig(
            name="mock_model",
            type=ModelType.CLASSIFICATION,
            architecture="MockArch"
        )

        # When: 모델 구축
        model = adapter.build(config)

        # Then: 딕셔너리 형태의 모델 반환
        assert isinstance(model, dict)
        assert "type" in model
        assert model["type"] == "mock_model"

    def test_build_stores_config_in_model(self):
        # Given: 설정이 포함된 ModelConfig
        adapter = MockModelAdapter()
        config = ModelConfig(
            name="test_model",
            type=ModelType.CLASSIFICATION,
            architecture="TestArch",
            hyperparameters={"layers": 3}
        )

        # When: 모델 구축
        model = adapter.build(config)

        # Then: 설정이 모델에 저장됨
        assert model["config"] == config
        assert model["config"].hyperparameters["layers"] == 3

    def test_train_returns_metrics(self):
        # Given: 모델, 데이터, 학습 설정
        adapter = MockModelAdapter()
        model = {"type": "mock_model"}
        train_data = np.array([[1, 2], [3, 4]])
        val_data = np.array([[5, 6]])
        config = TrainingConfig(epochs=10, batch_size=32, learning_rate=0.001)

        # When: 학습 수행
        metrics = adapter.train(model, train_data, val_data, config)

        # Then: 메트릭 딕셔너리 반환
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_train_without_validation_data(self):
        # Given: 검증 데이터 없이 학습
        adapter = MockModelAdapter()
        model = {"type": "mock_model"}
        train_data = np.array([[1, 2], [3, 4]])
        config = TrainingConfig(epochs=5, batch_size=16, learning_rate=0.01)

        # When: 학습 수행
        metrics = adapter.train(model, train_data, None, config)

        # Then: 메트릭 반환
        assert isinstance(metrics, dict)
        assert "loss" in metrics

    def test_evaluate_returns_metrics(self):
        # Given: 모델과 테스트 데이터
        adapter = MockModelAdapter()
        model = {"type": "mock_model"}
        test_data = np.array([[1, 2], [3, 4]])

        # When: 평가 수행
        metrics = adapter.evaluate(model, test_data)

        # Then: 평가 메트릭 반환
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics or "loss" in metrics

    def test_predict_returns_predictions(self):
        # Given: 모델과 입력 데이터
        adapter = MockModelAdapter()
        model = {"type": "mock_model"}
        inputs = np.array([[1, 2], [3, 4], [5, 6]])

        # When: 예측 수행
        predictions = adapter.predict(model, inputs)

        # Then: 입력과 같은 크기의 예측 반환
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(inputs)

    def test_save_creates_model_file(self, tmp_path):
        # Given: 모델과 저장 경로
        adapter = MockModelAdapter()
        model = {"type": "mock_model", "weights": [1, 2, 3]}
        save_path = tmp_path / "model.pkl"

        # When: 모델 저장
        adapter.save(model, str(save_path))

        # Then: 파일이 생성됨
        assert save_path.exists()

    def test_load_returns_model(self, tmp_path):
        # Given: 저장된 모델
        adapter = MockModelAdapter()
        model = {"type": "mock_model", "weights": [1, 2, 3]}
        model_path = tmp_path / "model.pkl"
        adapter.save(model, str(model_path))

        # When: 모델 로드
        loaded_model = adapter.load(str(model_path))

        # Then: 원본 모델과 동일한 내용
        assert loaded_model["type"] == model["type"]
        assert loaded_model["weights"] == model["weights"]

    def test_load_nonexistent_file_raises_error(self):
        # Given: 존재하지 않는 파일
        adapter = MockModelAdapter()
        nonexistent_path = "/nonexistent/model.pkl"

        # When/Then: 에러 발생
        with pytest.raises(Exception):
            adapter.load(nonexistent_path)


class TestMockDataAdapter:
    """MockDataAdapter 테스트"""

    def test_load_returns_data_tuple(self):
        # Given: MockDataAdapter와 경로
        adapter = MockDataAdapter()
        path = "mock_data.csv"

        # When: 데이터 로드
        X, y = adapter.load(path)

        # Then: numpy 배열 튜플 반환
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y)
        assert X.ndim == 2  # 2차원 피처 배열

    def test_load_with_kwargs(self):
        # Given: 추가 옵션과 함께 로드
        adapter = MockDataAdapter()
        path = "mock_data.csv"

        # When: 옵션과 함께 로드
        X, y = adapter.load(path, n_samples=50, n_features=5)

        # Then: 지정된 크기의 데이터 생성
        assert X.shape[0] == 50
        assert X.shape[1] == 5
        assert len(y) == 50

    def test_split_returns_six_arrays(self):
        # Given: 데이터
        adapter = MockDataAdapter()
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        # When: 데이터 분할
        X_train, X_val, X_test, y_train, y_val, y_test = adapter.split(
            X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        # Then: 6개의 배열 반환
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_val, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert len(X_train) + len(X_val) + len(X_test) == len(X)

    def test_split_with_custom_ratios(self):
        # Given: 데이터와 커스텀 비율
        adapter = MockDataAdapter()
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        # When: 커스텀 비율로 분할
        X_train, X_val, X_test, y_train, y_val, y_test = adapter.split(
            X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )

        # Then: 비율에 맞게 분할됨
        assert len(X_train) == 60
        assert len(X_val) == 20
        assert len(X_test) == 20

    def test_split_with_random_state(self):
        # Given: 동일한 random_state
        adapter = MockDataAdapter()
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        # When: 같은 random_state로 두 번 분할
        result1 = adapter.split(X, y, random_state=42)
        result2 = adapter.split(X, y, random_state=42)

        # Then: 동일한 결과
        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[3], result2[3])

    def test_preprocess_returns_processed_data(self):
        # Given: 전처리할 데이터
        adapter = MockDataAdapter()
        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([0, 1])

        # When: 전처리 수행
        X_processed, y_processed = adapter.preprocess(X, y)

        # Then: 처리된 데이터 반환
        assert isinstance(X_processed, np.ndarray)
        assert isinstance(y_processed, np.ndarray)
        assert X_processed.shape == X.shape
        assert len(y_processed) == len(y)

    def test_preprocess_without_labels(self):
        # Given: 레이블 없는 데이터
        adapter = MockDataAdapter()
        X = np.array([[1, 2, 3], [4, 5, 6]])

        # When: 전처리 수행
        X_processed, y_processed = adapter.preprocess(X, None)

        # Then: X만 처리되고 y는 None
        assert isinstance(X_processed, np.ndarray)
        assert y_processed is None

    def test_to_framework_format_returns_converted_data(self):
        # Given: 데이터와 프레임워크 이름
        adapter = MockDataAdapter()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        # When: 프레임워크 포맷으로 변환
        data = adapter.to_framework_format(X, y, "pytorch")

        # Then: 변환된 데이터 반환
        assert data is not None
        assert "X" in data
        assert "y" in data

    def test_to_framework_format_tensorflow(self):
        # Given: 데이터
        adapter = MockDataAdapter()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        # When: TensorFlow 포맷으로 변환
        data = adapter.to_framework_format(X, y, "tensorflow")

        # Then: 변환된 데이터 반환
        assert data is not None
        assert "framework" in data
        assert data["framework"] == "tensorflow"

    def test_to_framework_format_sklearn(self):
        # Given: 데이터
        adapter = MockDataAdapter()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        # When: sklearn 포맷으로 변환
        data = adapter.to_framework_format(X, y, "sklearn")

        # Then: numpy 배열 그대로 반환 (sklearn은 numpy를 직접 사용)
        assert data is not None


class TestMockAdapterIntegration:
    """Mock 어댑터 통합 테스트"""

    def test_full_workflow_with_mock_adapters(self):
        # Given: Mock 어댑터들
        model_adapter = MockModelAdapter()
        data_adapter = MockDataAdapter()

        # When: 전체 워크플로우 수행
        # 1. 데이터 로드
        X, y = data_adapter.load("data.csv", n_samples=100, n_features=10)

        # 2. 데이터 분할
        X_train, X_val, X_test, y_train, y_val, y_test = data_adapter.split(X, y)

        # 3. 전처리
        X_train, y_train = data_adapter.preprocess(X_train, y_train)

        # 4. 모델 구축
        model_config = ModelConfig(
            name="test_model",
            type=ModelType.CLASSIFICATION,
            architecture="MockNN"
        )
        model = model_adapter.build(model_config)

        # 5. 학습
        train_config = TrainingConfig(epochs=10, batch_size=32, learning_rate=0.001)
        train_metrics = model_adapter.train(model, X_train, X_val, train_config)

        # 6. 평가
        eval_metrics = model_adapter.evaluate(model, X_test)

        # 7. 예측
        predictions = model_adapter.predict(model, X_test)

        # Then: 모든 단계가 성공적으로 완료됨
        assert model is not None
        assert "loss" in train_metrics or "accuracy" in train_metrics
        assert "loss" in eval_metrics or "accuracy" in eval_metrics
        assert len(predictions) == len(X_test)

    def test_save_and_load_workflow(self, tmp_path):
        # Given: 학습된 모델
        adapter = MockModelAdapter()
        model_config = ModelConfig(
            name="saved_model",
            type=ModelType.REGRESSION,
            architecture="MockRegressor"
        )
        model = adapter.build(model_config)

        # When: 저장 후 로드
        model_path = tmp_path / "test_model.pkl"
        adapter.save(model, str(model_path))
        loaded_model = adapter.load(str(model_path))

        # Then: 로드된 모델로 예측 가능
        test_input = np.array([[1, 2, 3]])
        predictions = adapter.predict(loaded_model, test_input)
        assert predictions is not None
        assert len(predictions) == 1


class TestMockAdapterEdgeCases:
    """엣지 케이스 테스트"""

    def test_small_dataset_split(self):
        # Given: 작은 데이터셋
        adapter = MockDataAdapter()
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)

        # When: 분할
        result = adapter.split(X, y)

        # Then: 분할 성공
        X_train, X_val, X_test, y_train, y_val, y_test = result
        assert len(X_train) + len(X_val) + len(X_test) == 10

    def test_single_feature_data(self):
        # Given: 단일 피처 데이터
        adapter = MockDataAdapter()
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 1, 0, 1, 0])

        # When: 전처리
        X_processed, y_processed = adapter.preprocess(X, y)

        # Then: 정상 처리
        assert X_processed.shape == X.shape
        assert len(y_processed) == len(y)

    def test_model_predict_single_sample(self):
        # Given: 단일 샘플
        adapter = MockModelAdapter()
        model = {"type": "mock"}
        single_input = np.array([[1, 2, 3]])

        # When: 예측
        prediction = adapter.predict(model, single_input)

        # Then: 예측 성공
        assert len(prediction) == 1
