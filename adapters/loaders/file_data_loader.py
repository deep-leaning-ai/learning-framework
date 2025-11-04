"""
File data loader implementation.

Implements DataPort for loading data from various file formats.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Any
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from core.contracts import DataPort
from utils.exceptions import DataException


class FileDataLoader:
    """
    파일 데이터 로더

    CSV, NumPy 등 다양한 형식의 파일에서 데이터를 로드하고 처리합니다.
    """

    def __init__(self):
        """초기화"""
        self.scaler = None

    def load(self, path: str, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        파일에서 데이터 로드

        Args:
            path: 파일 경로
            **kwargs: 추가 옵션
                - target_column: 타겟 컬럼 이름 또는 인덱스
                - delimiter: CSV 구분자 (기본: ',')
                - header: CSV 헤더 행 번호 (기본: 0)

        Returns:
            (X, y) 튜플 - 피처와 타겟

        Raises:
            DataException: 파일 로드 실패 시
        """
        file_path = Path(path)

        if not file_path.exists():
            raise DataException(
                f"File not found: {path}",
                context={"path": path}
            )

        try:
            # 파일 형식에 따라 로드
            if file_path.suffix == '.csv':
                return self._load_csv(path, **kwargs)
            elif file_path.suffix in ['.npy', '.npz']:
                return self._load_numpy(path, **kwargs)
            else:
                raise DataException(
                    f"Unsupported file format: {file_path.suffix}",
                    context={"path": path, "suffix": file_path.suffix}
                )
        except DataException:
            raise
        except Exception as e:
            raise DataException(
                f"Failed to load file: {str(e)}",
                context={"path": path, "error": str(e)}
            )

    def _load_csv(self, path: str, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        CSV 파일 로드

        Args:
            path: CSV 파일 경로
            **kwargs: CSV 로드 옵션

        Returns:
            (X, y) 튜플
        """
        target_column = kwargs.get('target_column', None)
        delimiter = kwargs.get('delimiter', ',')
        header = kwargs.get('header', 0)

        df = pd.read_csv(path, delimiter=delimiter, header=header)

        if target_column is not None:
            if isinstance(target_column, str):
                y = df[target_column].values
                X = df.drop(columns=[target_column]).values
            elif isinstance(target_column, int):
                y = df.iloc[:, target_column].values
                X = df.drop(df.columns[target_column], axis=1).values
            else:
                raise DataException(
                    f"Invalid target_column type: {type(target_column)}",
                    context={"target_column": target_column}
                )
        else:
            X = df.values
            y = None

        return X, y

    def _load_numpy(self, path: str, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        NumPy 파일 로드

        Args:
            path: NumPy 파일 경로
            **kwargs: NumPy 로드 옵션

        Returns:
            (X, y) 튜플
        """
        target_column = kwargs.get('target_column', None)

        if path.endswith('.npz'):
            data = np.load(path)
            X = data['X'] if 'X' in data else data['features']
            y = data['y'] if 'y' in data else data.get('target', None)
        else:
            data = np.load(path)

            if target_column is not None:
                if isinstance(target_column, int):
                    if target_column == -1:
                        target_column = data.shape[1] - 1
                    y = data[:, target_column]
                    X = np.delete(data, target_column, axis=1)
                else:
                    raise DataException(
                        f"Invalid target_column for NumPy array: {target_column}",
                        context={"target_column": target_column}
                    )
            else:
                X = data
                y = None

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

        Raises:
            DataException: 비율 합이 1이 아닌 경우
        """
        # 비율 검증
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise DataException(
                f"Ratios must sum to 1.0, got {total_ratio}",
                context={
                    "train_ratio": train_ratio,
                    "val_ratio": val_ratio,
                    "test_ratio": test_ratio
                }
            )

        # 첫 번째 분할: train + val vs test
        test_size = test_ratio
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 두 번째 분할: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        method: str = "normalize"
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        데이터 전처리

        Args:
            X: 피처 데이터
            y: 타겟 데이터 (선택)
            method: 전처리 방법 ('normalize', 'standardize', 'none')

        Returns:
            (전처리된 X, y) 튜플

        Raises:
            DataException: 잘못된 전처리 방법
        """
        if method == "none":
            return X, y

        try:
            if method == "normalize":
                scaler = MinMaxScaler()
            elif method == "standardize":
                scaler = StandardScaler()
            else:
                raise DataException(
                    f"Invalid preprocessing method: {method}",
                    context={"method": method, "valid_methods": ["normalize", "standardize", "none"]}
                )

            X_processed = scaler.fit_transform(X)
            self.scaler = scaler

            return X_processed, y

        except DataException:
            raise
        except Exception as e:
            raise DataException(
                f"Preprocessing failed: {str(e)}",
                context={"method": method, "error": str(e)}
            )

    def to_framework_format(
        self,
        X: np.ndarray,
        y: np.ndarray,
        framework: str
    ) -> Any:
        """
        프레임워크별 데이터 포맷으로 변환

        Args:
            X: 피처 데이터
            y: 타겟 데이터
            framework: 프레임워크 이름

        Returns:
            프레임워크별 데이터 객체

        Raises:
            DataException: 지원하지 않는 프레임워크
        """
        framework = framework.lower()

        try:
            if framework == "numpy":
                return (X, y)

            elif framework == "pandas":
                X_df = pd.DataFrame(X)
                y_series = pd.Series(y)
                return (X_df, y_series)

            elif framework == "tensorflow":
                # TensorFlow Dataset으로 변환 (간단한 튜플 반환)
                # 실제로는 tf.data.Dataset을 사용할 수 있음
                return (X.astype(np.float32), y)

            elif framework == "pytorch":
                # PyTorch Tensor로 변환
                try:
                    import torch
                    X_tensor = torch.from_numpy(X).float()
                    y_tensor = torch.from_numpy(y).long()
                    return (X_tensor, y_tensor)
                except ImportError:
                    raise DataException(
                        "PyTorch not installed",
                        context={"framework": framework}
                    )

            else:
                raise DataException(
                    f"Unsupported framework: {framework}",
                    context={
                        "framework": framework,
                        "supported": ["numpy", "pandas", "tensorflow", "pytorch"]
                    }
                )

        except DataException:
            raise
        except Exception as e:
            raise DataException(
                f"Framework conversion failed: {str(e)}",
                context={"framework": framework, "error": str(e)}
            )
