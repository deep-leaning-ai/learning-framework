"""
PyTorch Regression Example

Boston Housing 스타일의 회귀 모델 예제
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from adapters.frameworks import PyTorchModelAdapter
from core.models import ModelConfig, TrainingConfig, ModelType, OptimizerType


def create_regression_model():
    """회귀 모델 생성 및 학습 예제"""

    # 1. 어댑터 초기화
    adapter = PyTorchModelAdapter()
    print(f"Using device: {adapter.device}")
    print("-" * 50)

    # 2. 데이터 생성 (회귀 문제)
    print("Generating regression data...")
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=7,
        noise=10,
        random_state=42
    )

    # 데이터 정규화
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X).astype(np.float32)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)

    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print("-" * 50)

    # 3. 모델 설정
    model_config = ModelConfig(
        name="house_price_predictor",
        type=ModelType.REGRESSION,
        architecture="sequential",
        hyperparameters={
            "input_shape": (10,),
            "layers": [
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "batchnormalization"},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "batchnormalization"},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": 16, "activation": "relu"},
                {"type": "dense", "units": 1}  # 회귀는 출력이 1개
            ]
        }
    )

    # 4. 모델 구축
    print("Building model...")
    model = adapter.build(model_config)
    print("Model built successfully!")
    print("-" * 50)

    # 5. 학습 설정
    training_config = TrainingConfig(
        epochs=50,
        batch_size=32,
        optimizer=OptimizerType.ADAM,
        learning_rate=0.001,
        validation_split=0.2
    )

    # 6. 모델 학습
    print("Training model...")
    train_metrics = adapter.train(
        model=model,
        train_data=(X_train, y_train),
        val_data=None,  # validation_split 사용
        config=training_config
    )

    print(f"\nTraining completed!")
    print(f"Final training loss (MSE): {train_metrics['loss']:.4f}")
    if 'rmse' in train_metrics:
        print(f"Final training RMSE: {train_metrics['rmse']:.4f}")
    if 'val_loss' in train_metrics:
        print(f"Final validation loss (MSE): {train_metrics['val_loss']:.4f}")
    if 'val_rmse' in train_metrics:
        print(f"Final validation RMSE: {train_metrics['val_rmse']:.4f}")
    print("-" * 50)

    # 7. 테스트 데이터로 평가
    print("Evaluating on test data...")
    test_metrics = adapter.evaluate(
        model=model,
        test_data=(X_test, y_test)
    )

    print(f"Test loss (MSE): {test_metrics['loss']:.4f}")
    if 'rmse' in test_metrics:
        print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print("-" * 50)

    # 8. 예측 및 시각화
    print("Making predictions...")
    predictions = adapter.predict(model, X_test)

    # 예측값을 원래 스케일로 복원
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_original = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # 샘플 예측 출력
    print("\nSample predictions (original scale):")
    for i in range(min(5, len(X_test))):
        error = abs(y_test_original[i] - predictions_original[i])
        print(f"True: {y_test_original[i]:8.2f}, "
              f"Predicted: {predictions_original[i]:8.2f}, "
              f"Error: {error:6.2f}")

    # R² 스코어 계산
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, predictions)
    print(f"\nR² Score: {r2:.4f}")
    print("-" * 50)

    # 9. 잔차 분석
    residuals = y_test - predictions
    print("Residual Analysis:")
    print(f"Mean residual: {np.mean(residuals):.4f}")
    print(f"Std residual: {np.std(residuals):.4f}")
    print(f"Max absolute residual: {np.max(np.abs(residuals)):.4f}")

    return model, test_metrics, (y_test, predictions)


def experiment_with_architectures():
    """다양한 모델 아키텍처 실험"""

    adapter = PyTorchModelAdapter()

    # 데이터 생성
    X, y = make_regression(
        n_samples=500,
        n_features=5,
        n_informative=4,
        noise=5,
        random_state=42
    )

    # 정규화
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)

    architectures = [
        # 얕은 모델
        {
            "name": "Shallow",
            "layers": [
                {"type": "dense", "units": 16, "activation": "relu"},
                {"type": "dense", "units": 1}
            ]
        },
        # 중간 깊이 모델
        {
            "name": "Medium",
            "layers": [
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": 16, "activation": "relu"},
                {"type": "dense", "units": 1}
            ]
        },
        # 깊은 모델 with BatchNorm
        {
            "name": "Deep+BN",
            "layers": [
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "batchnormalization"},
                {"type": "dropout", "rate": 0.3},
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "batchnormalization"},
                {"type": "dropout", "rate": 0.3},
                {"type": "dense", "units": 16, "activation": "relu"},
                {"type": "dense", "units": 1}
            ]
        }
    ]

    print("\nArchitecture Comparison:")
    print("-" * 60)
    print(f"{'Architecture':<15} {'Parameters':<12} {'Loss (MSE)':<12} {'RMSE':<10}")
    print("-" * 60)

    for arch in architectures:
        config = ModelConfig(
            name=arch["name"],
            type=ModelType.REGRESSION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (5,),
                "layers": arch["layers"]
            }
        )

        model = adapter.build(config)

        # 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())

        training_config = TrainingConfig(
            epochs=30,
            batch_size=32,
            optimizer=OptimizerType.ADAM,
            learning_rate=0.001
        )

        metrics = adapter.train(
            model=model,
            train_data=(X, y),
            val_data=None,
            config=training_config
        )

        rmse = metrics.get('rmse', np.sqrt(metrics['loss']))
        print(f"{arch['name']:<15} {total_params:<12,} {metrics['loss']:<12.4f} {rmse:<10.4f}")

    print("-" * 60)


def learning_rate_experiment():
    """학습률 실험"""

    adapter = PyTorchModelAdapter()

    # 간단한 데이터 생성
    X, y = make_regression(n_samples=500, n_features=10, noise=5, random_state=42)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = y.astype(np.float32)

    # 모델 설정
    config = ModelConfig(
        name="lr_test",
        type=ModelType.REGRESSION,
        architecture="sequential",
        hyperparameters={
            "input_shape": (10,),
            "layers": [
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dense", "units": 1}
            ]
        }
    )

    learning_rates = [0.1, 0.01, 0.001, 0.0001]

    print("\nLearning Rate Comparison:")
    print("-" * 50)

    for lr in learning_rates:
        model = adapter.build(config)

        training_config = TrainingConfig(
            epochs=20,
            batch_size=32,
            optimizer=OptimizerType.ADAM,
            learning_rate=lr
        )

        metrics = adapter.train(
            model=model,
            train_data=(X, y),
            val_data=None,
            config=training_config
        )

        print(f"LR={lr:7.4f}: Loss={metrics['loss']:8.4f}")

    print("-" * 50)


if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch Regression Example")
    print("=" * 60)

    # 메인 회귀 예제 실행
    model, metrics, predictions = create_regression_model()

    # 아키텍처 비교 실험
    experiment_with_architectures()

    # 학습률 실험
    learning_rate_experiment()

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)