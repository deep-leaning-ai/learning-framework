"""
TDD Tests for file data loader.
All tests follow Given-When-Then pattern.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from adapters.loaders.file_data_loader import FileDataLoader
from utils.exceptions import DataException


class TestFileDataLoaderInitialization:
    """데이터 로더 초기화 테스트"""

    def test_create_file_data_loader(self):
        # Given: 데이터 로더
        # When: FileDataLoader 생성
        loader = FileDataLoader()

        # Then: 로더가 올바르게 생성됨
        assert loader is not None


class TestFileDataLoaderCSV:
    """CSV 파일 로드 테스트"""

    def test_load_csv_file(self, tmp_path):
        # Given: CSV 파일
        csv_file = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [0, 1, 0, 1, 0]
        })
        df.to_csv(csv_file, index=False)

        loader = FileDataLoader()

        # When: CSV 로드
        X, y = loader.load(str(csv_file), target_column="target")

        # Then: 데이터가 올바르게 로드됨
        assert X.shape == (5, 2)
        assert y.shape == (5,)
        assert np.array_equal(y, np.array([0, 1, 0, 1, 0]))

    def test_load_csv_without_target(self, tmp_path):
        # Given: 타겟 없는 CSV 파일
        csv_file = tmp_path / "test_features.csv"
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6]
        })
        df.to_csv(csv_file, index=False)

        loader = FileDataLoader()

        # When: 타겟 없이 로드
        X, y = loader.load(str(csv_file))

        # Then: 피처만 로드되고 타겟은 None
        assert X.shape == (3, 2)
        assert y is None

    def test_load_csv_with_header(self, tmp_path):
        # Given: 헤더가 있는 CSV
        csv_file = tmp_path / "header_data.csv"
        df = pd.DataFrame({
            "col1": [10, 20, 30],
            "col2": [40, 50, 60],
            "label": [1, 2, 3]
        })
        df.to_csv(csv_file, index=False)

        loader = FileDataLoader()

        # When: 헤더와 함께 로드
        X, y = loader.load(str(csv_file), target_column="label")

        # Then: 헤더를 제외한 데이터 로드
        assert X.shape == (3, 2)
        assert y.shape == (3,)


class TestFileDataLoaderNumPy:
    """NumPy 파일 로드 테스트"""

    def test_load_numpy_file(self, tmp_path):
        # Given: NumPy 파일
        npy_file = tmp_path / "test_data.npy"
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        np.save(npy_file, data)

        loader = FileDataLoader()

        # When: NumPy 파일 로드
        X, y = loader.load(str(npy_file), target_column=-1)

        # Then: 데이터가 올바르게 분리됨
        assert X.shape == (3, 2)
        assert y.shape == (3,)
        assert np.array_equal(y, np.array([3, 6, 9]))

    def test_load_numpy_without_target(self, tmp_path):
        # Given: NumPy 파일 (타겟 없음)
        npy_file = tmp_path / "features.npy"
        data = np.random.rand(10, 5)
        np.save(npy_file, data)

        loader = FileDataLoader()

        # When: 타겟 없이 로드
        X, y = loader.load(str(npy_file))

        # Then: 전체가 피처로 로드
        assert X.shape == (10, 5)
        assert y is None


class TestFileDataLoaderSplit:
    """데이터 분할 테스트"""

    def test_split_data_default_ratios(self):
        # Given: 데이터
        loader = FileDataLoader()
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, size=100)

        # When: 기본 비율로 분할
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split(X, y)

        # Then: 올바른 비율로 분할됨 (근사치)
        assert 68 <= len(X_train) <= 72
        assert 13 <= len(X_val) <= 17
        assert 13 <= len(X_test) <= 17
        assert len(X_train) + len(X_val) + len(X_test) == 100

    def test_split_data_custom_ratios(self):
        # Given: 사용자 정의 비율
        loader = FileDataLoader()
        X = np.random.rand(100, 3)
        y = np.random.randint(0, 3, size=100)

        # When: 사용자 정의 비율로 분할
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split(
            X, y,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )

        # Then: 올바른 비율로 분할됨
        assert len(X_train) == 60
        assert len(X_val) == 20
        assert len(X_test) == 20

    def test_split_data_with_random_state(self):
        # Given: 동일한 random_state
        loader = FileDataLoader()
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 2, size=50)

        # When: 동일한 random_state로 두 번 분할
        result1 = loader.split(X, y, random_state=42)
        result2 = loader.split(X, y, random_state=42)

        # Then: 결과가 동일함
        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[3], result2[3])


class TestFileDataLoaderPreprocess:
    """데이터 전처리 테스트"""

    def test_preprocess_normalization(self):
        # Given: 정규화할 데이터
        loader = FileDataLoader()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0])

        # When: 정규화
        X_processed, y_processed = loader.preprocess(X, y, method="normalize")

        # Then: 정규화된 데이터
        assert X_processed.shape == X.shape
        assert np.all(X_processed >= 0) and np.all(X_processed <= 1)
        assert np.array_equal(y_processed, y)

    def test_preprocess_standardization(self):
        # Given: 표준화할 데이터
        loader = FileDataLoader()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 2])

        # When: 표준화
        X_processed, y_processed = loader.preprocess(X, y, method="standardize")

        # Then: 표준화된 데이터 (평균 0, 분산 1)
        assert X_processed.shape == X.shape
        assert np.allclose(X_processed.mean(axis=0), 0, atol=1e-10)
        assert np.array_equal(y_processed, y)

    def test_preprocess_without_target(self):
        # Given: 타겟 없는 데이터
        loader = FileDataLoader()
        X = np.random.rand(20, 3)

        # When: 타겟 없이 전처리
        X_processed, y_processed = loader.preprocess(X, method="normalize")

        # Then: 피처만 전처리됨
        assert X_processed.shape == X.shape
        assert y_processed is None


class TestFileDataLoaderFrameworkFormat:
    """프레임워크별 포맷 변환 테스트"""

    def test_convert_to_numpy(self):
        # Given: NumPy 데이터
        loader = FileDataLoader()
        X = np.random.rand(10, 3)
        y = np.random.randint(0, 2, size=10)

        # When: NumPy 포맷으로 변환
        result = loader.to_framework_format(X, y, framework="numpy")

        # Then: 튜플 반환
        assert isinstance(result, tuple)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], X)
        np.testing.assert_array_equal(result[1], y)

    def test_convert_to_tensorflow(self):
        # Given: TensorFlow용 데이터
        loader = FileDataLoader()
        X = np.random.rand(20, 4).astype(np.float32)
        y = np.random.randint(0, 3, size=20)

        # When: TensorFlow 포맷으로 변환
        result = loader.to_framework_format(X, y, framework="tensorflow")

        # Then: TensorFlow Dataset 또는 튜플 반환
        assert result is not None

    def test_convert_to_pandas(self):
        # Given: Pandas용 데이터
        loader = FileDataLoader()
        X = np.random.rand(15, 2)
        y = np.random.randint(0, 2, size=15)

        # When: Pandas 포맷으로 변환
        result = loader.to_framework_format(X, y, framework="pandas")

        # Then: DataFrame과 Series 반환
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestFileDataLoaderErrorHandling:
    """에러 처리 테스트"""

    def test_load_nonexistent_file(self):
        # Given: 존재하지 않는 파일 경로
        loader = FileDataLoader()

        # When/Then: 에러 발생
        with pytest.raises(DataException):
            loader.load("nonexistent_file.csv")

    def test_load_invalid_file_format(self, tmp_path):
        # Given: 지원하지 않는 파일 형식
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("some text")

        loader = FileDataLoader()

        # When/Then: 에러 발생
        with pytest.raises(DataException):
            loader.load(str(invalid_file))

    def test_split_with_invalid_ratios(self):
        # Given: 잘못된 비율
        loader = FileDataLoader()
        X = np.random.rand(10, 2)
        y = np.random.randint(0, 2, size=10)

        # When/Then: 비율 합이 1이 아니면 에러
        with pytest.raises(DataException):
            loader.split(X, y, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_preprocess_with_invalid_method(self):
        # Given: 잘못된 전처리 방법
        loader = FileDataLoader()
        X = np.random.rand(10, 2)

        # When/Then: 에러 발생
        with pytest.raises(DataException):
            loader.preprocess(X, method="invalid_method")


class TestFileDataLoaderIntegration:
    """통합 테스트"""

    def test_full_workflow(self, tmp_path):
        # Given: CSV 파일과 전체 워크플로우
        csv_file = tmp_path / "workflow_data.csv"
        df = pd.DataFrame({
            "f1": np.random.rand(100),
            "f2": np.random.rand(100),
            "f3": np.random.rand(100),
            "target": np.random.randint(0, 2, size=100)
        })
        df.to_csv(csv_file, index=False)

        loader = FileDataLoader()

        # When: 로드 -> 분할 -> 전처리
        X, y = loader.load(str(csv_file), target_column="target")
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split(X, y)
        X_train_processed, y_train_processed = loader.preprocess(X_train, y_train, method="normalize")

        # Then: 모든 단계가 성공적으로 수행됨
        assert 68 <= X_train_processed.shape[0] <= 72
        assert 68 <= y_train_processed.shape[0] <= 72
        assert np.all(X_train_processed >= 0) and np.all(X_train_processed <= 1)

    def test_load_split_preprocess_convert(self, tmp_path):
        # Given: 전체 데이터 파이프라인
        npy_file = tmp_path / "pipeline_data.npy"
        data = np.column_stack([
            np.random.rand(50, 3),
            np.random.randint(0, 2, size=50)
        ])
        np.save(npy_file, data)

        loader = FileDataLoader()

        # When: 전체 파이프라인 실행
        X, y = loader.load(str(npy_file), target_column=-1)
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split(
            X, y, random_state=42
        )
        X_train_norm, y_train_norm = loader.preprocess(
            X_train, y_train, method="standardize"
        )
        train_data = loader.to_framework_format(
            X_train_norm, y_train_norm, framework="numpy"
        )

        # Then: 파이프라인이 성공적으로 완료됨
        assert train_data is not None
        assert len(train_data) == 2
        assert 33 <= train_data[0].shape[0] <= 37
