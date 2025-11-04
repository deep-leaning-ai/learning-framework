"""
Experiment service for orchestrating ML experiments.

This service coordinates all components (data, model, metrics, tracking)
to run complete machine learning experiments.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from learning_framework.core.contracts import ModelPort, DataPort, MetricsPort
from learning_framework.core.models import ModelConfig, TrainingConfig, ExperimentResult
from learning_framework.utils.metrics import MetricsCalculator


class ExperimentService:
    """
    실험 서비스

    ML 실험의 전체 파이프라인을 오케스트레이션합니다.
    데이터 준비, 모델 학습, 평가, 결과 저장 등을 담당합니다.
    """

    def __init__(
        self,
        model_adapter: ModelPort,
        data_adapter: DataPort,
        metrics_calculator: Optional[MetricsPort] = None
    ):
        """
        초기화

        Args:
            model_adapter: 모델 어댑터
            data_adapter: 데이터 어댑터
            metrics_calculator: 메트릭 계산기 (선택)
        """
        self.model_adapter = model_adapter
        self.data_adapter = data_adapter
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self.last_result: Optional[ExperimentResult] = None

    def prepare_data(
        self,
        data_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: Optional[int] = None,
        preprocess: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        데이터 준비 (로드, 분할, 전처리)

        Args:
            data_path: 데이터 파일 경로
            train_ratio: 학습 데이터 비율
            val_ratio: 검증 데이터 비율
            test_ratio: 테스트 데이터 비율
            random_state: 난수 시드
            preprocess: 전처리 수행 여부
            **kwargs: 데이터 로드 추가 옵션

        Returns:
            분할된 데이터 딕셔너리
        """
        # 데이터 로드
        X, y = self.data_adapter.load(data_path, **kwargs)

        # 데이터 분할
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_adapter.split(
            X, y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state
        )

        # 전처리
        if preprocess:
            X_train, y_train = self.data_adapter.preprocess(X_train, y_train)
            X_val, y_val = self.data_adapter.preprocess(X_val, y_val)
            X_test, y_test = self.data_adapter.preprocess(X_test, y_test)

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "preprocessed": preprocess
        }

    def train_model(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        train_data: Any,
        val_data: Optional[Any] = None
    ) -> Tuple[Any, Dict[str, float]]:
        """
        모델 학습

        Args:
            model_config: 모델 설정
            training_config: 학습 설정
            train_data: 학습 데이터
            val_data: 검증 데이터 (선택)

        Returns:
            (학습된 모델, 학습 메트릭) 튜플
        """
        # 모델 구축
        model = self.model_adapter.build(model_config)

        # 모델 학습
        metrics = self.model_adapter.train(
            model=model,
            train_data=train_data,
            val_data=val_data,
            config=training_config
        )

        return model, metrics

    def evaluate_model(
        self,
        model: Any,
        test_data: Any,
        test_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        모델 평가

        Args:
            model: 평가할 모델
            test_data: 테스트 데이터
            test_labels: 테스트 레이블 (선택)

        Returns:
            평가 메트릭 딕셔너리
        """
        # 모델 평가
        metrics = self.model_adapter.evaluate(model, test_data)

        # 추가 메트릭 계산 (레이블이 있는 경우)
        if test_labels is not None:
            predictions = self.model_adapter.predict(model, test_data)

            # 정확도 계산
            try:
                accuracy = self.metrics_calculator.calculate_accuracy(
                    test_labels, predictions
                )
                metrics["calculated_accuracy"] = accuracy

                # Precision, Recall, F1 계산
                prf_metrics = self.metrics_calculator.calculate_precision_recall_f1(
                    test_labels, predictions
                )
                metrics.update(prf_metrics)
            except Exception:
                # 메트릭 계산 실패 시 무시
                pass

        return metrics

    def predict(self, model: Any, inputs: np.ndarray) -> np.ndarray:
        """
        예측 수행

        Args:
            model: 예측에 사용할 모델
            inputs: 입력 데이터

        Returns:
            예측 결과
        """
        return self.model_adapter.predict(model, inputs)

    def save_model(self, model: Any, path: str) -> None:
        """
        모델 저장

        Args:
            model: 저장할 모델
            path: 저장 경로
        """
        self.model_adapter.save(model, path)

    def load_model(self, path: str) -> Any:
        """
        모델 로드

        Args:
            path: 모델 파일 경로

        Returns:
            로드된 모델
        """
        return self.model_adapter.load(path)

    def run_experiment(self, experiment_config: Dict[str, Any]) -> ExperimentResult:
        """
        전체 실험 실행

        Args:
            experiment_config: 실험 설정 딕셔너리
                - data_path: 데이터 경로
                - model_config: ModelConfig 객체
                - training_config: TrainingConfig 객체
                - model_save_path: 모델 저장 경로 (선택)
                - train_ratio, val_ratio, test_ratio: 데이터 분할 비율 (선택)

        Returns:
            ExperimentResult 객체
        """
        # 설정 추출
        data_path = experiment_config["data_path"]
        model_config = experiment_config["model_config"]
        training_config = experiment_config["training_config"]
        model_save_path = experiment_config.get("model_save_path")

        # 데이터 분할 비율
        train_ratio = experiment_config.get("train_ratio", 0.7)
        val_ratio = experiment_config.get("val_ratio", 0.15)
        test_ratio = experiment_config.get("test_ratio", 0.15)

        # 1. 데이터 준비
        data = self.prepare_data(
            data_path=data_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        # 2. 모델 학습
        model, train_metrics = self.train_model(
            model_config=model_config,
            training_config=training_config,
            train_data=data["X_train"],
            val_data=data["X_val"]
        )

        # 3. 모델 평가
        eval_metrics = self.evaluate_model(
            model=model,
            test_data=data["X_test"],
            test_labels=data["y_test"]
        )

        # 4. 모델 저장 (경로가 지정된 경우)
        if model_save_path:
            self.save_model(model, model_save_path)

        # 5. 결과 생성
        all_metrics = {**train_metrics, **eval_metrics}

        # DatasetInfo 생성
        from learning_framework.core.models import DatasetInfo
        dataset_info = DatasetInfo(
            name=data_path,
            train_size=len(data["X_train"]),
            val_size=len(data["X_val"]),
            test_size=len(data["X_test"]),
            num_features=data["X_train"].shape[1] if len(data["X_train"].shape) > 1 else 1,
            num_classes=len(np.unique(data["y_train"])) if data["y_train"] is not None else 0
        )

        result = ExperimentResult(
            experiment_id=model_config.name,
            model_config=model_config,
            training_config=training_config,
            dataset_info=dataset_info,
            metrics=all_metrics
        )

        # 마지막 결과 저장
        self.last_result = result

        return result
