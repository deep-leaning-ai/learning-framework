"""
PyTorch Adapter Quick Start

간단한 시작 가이드와 예제
"""

import numpy as np
from adapters.frameworks import PyTorchModelAdapter
from core.models import ModelConfig, TrainingConfig, ModelType, OptimizerType


def quick_classification_example():
    """빠른 분류 예제"""
    print("Quick Classification Example")
    print("-" * 40)

    # 1. 어댑터 생성
    adapter = PyTorchModelAdapter()
    print(f"Device: {adapter.device}")

    # 2. 간단한 데이터 생성
    X_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randint(0, 3, 100)
    X_test = np.random.randn(20, 10).astype(np.float32)
    y_test = np.random.randint(0, 3, 20)

    # 3. 모델 설정
    config = ModelConfig(
        name="quick_classifier",
        type=ModelType.CLASSIFICATION,
        architecture="sequential",
        hyperparameters={
            "input_shape": (10,),
            "layers": [
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": 3}
            ]
        }
    )

    # 4. 모델 구축
    model = adapter.build(config)

    # 5. 학습
    training_config = TrainingConfig(
        epochs=10,
        batch_size=16,
        optimizer=OptimizerType.ADAM,
        learning_rate=0.001
    )

    metrics = adapter.train(
        model=model,
        train_data=(X_train, y_train),
        val_data=None,
        config=training_config
    )

    print(f"Training accuracy: {metrics['accuracy']:.4f}")

    # 6. 평가
    test_metrics = adapter.evaluate(model, (X_test, y_test))
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

    # 7. 예측
    predictions = adapter.predict(model, X_test[:3])
    print(f"Sample predictions: {np.argmax(predictions, axis=1)}")
    print("-" * 40 + "\n")


def quick_regression_example():
    """빠른 회귀 예제"""
    print("Quick Regression Example")
    print("-" * 40)

    adapter = PyTorchModelAdapter()

    # 데이터 생성
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    # 모델 설정
    config = ModelConfig(
        name="quick_regressor",
        type=ModelType.REGRESSION,
        architecture="sequential",
        hyperparameters={
            "input_shape": (5,),
            "layers": [
                {"type": "dense", "units": 16, "activation": "relu"},
                {"type": "dense", "units": 1}
            ]
        }
    )

    # 모델 구축 및 학습
    model = adapter.build(config)

    training_config = TrainingConfig(
        epochs=20,
        batch_size=16,
        optimizer=OptimizerType.ADAM,
        learning_rate=0.001
    )

    metrics = adapter.train(model, (X, y), None, training_config)
    print(f"Final MSE: {metrics['loss']:.4f}")

    # 예측
    X_new = np.random.randn(3, 5).astype(np.float32)
    predictions = adapter.predict(model, X_new)
    print(f"Predictions: {predictions.flatten()}")
    print("-" * 40 + "\n")


def quick_cnn_example():
    """빠른 CNN 예제"""
    print("Quick CNN Example")
    print("-" * 40)

    adapter = PyTorchModelAdapter()

    # 이미지 데이터 생성 (1채널, 28x28)
    X = np.random.randn(100, 1, 28, 28).astype(np.float32)
    y = np.random.randint(0, 10, 100)

    # CNN 모델 설정
    config = ModelConfig(
        name="quick_cnn",
        type=ModelType.CLASSIFICATION,
        architecture="sequential",
        hyperparameters={
            "input_shape": (1, 28, 28),
            "layers": [
                {"type": "conv2d", "filters": 16, "kernel_size": 3, "activation": "relu"},
                {"type": "maxpooling2d", "pool_size": 2},
                {"type": "flatten"},
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dense", "units": 10}
            ]
        }
    )

    # 모델 구축 및 학습
    model = adapter.build(config)

    training_config = TrainingConfig(
        epochs=5,
        batch_size=16,
        optimizer=OptimizerType.ADAM,
        learning_rate=0.001
    )

    metrics = adapter.train(model, (X, y), None, training_config)
    print(f"Training accuracy: {metrics['accuracy']:.4f}")

    # 예측
    X_new = np.random.randn(2, 1, 28, 28).astype(np.float32)
    predictions = adapter.predict(model, X_new)
    print(f"Predicted classes: {np.argmax(predictions, axis=1)}")
    print("-" * 40 + "\n")


def save_load_example():
    """모델 저장 및 로드 예제"""
    print("Save and Load Example")
    print("-" * 40)

    adapter = PyTorchModelAdapter()

    # 간단한 모델 생성
    config = ModelConfig(
        name="save_load_model",
        type=ModelType.CLASSIFICATION,
        architecture="sequential",
        hyperparameters={
            "input_shape": (5,),
            "layers": [
                {"type": "dense", "units": 10, "activation": "relu"},
                {"type": "dense", "units": 3}
            ]
        }
    )

    model = adapter.build(config)

    # 학습 (간단히)
    X = np.random.randn(50, 5).astype(np.float32)
    y = np.random.randint(0, 3, 50)

    training_config = TrainingConfig(
        epochs=5,
        batch_size=16,
        optimizer=OptimizerType.ADAM,
        learning_rate=0.001
    )

    adapter.train(model, (X, y), None, training_config)

    # 저장
    save_path = "models/quick_start_model.pt"
    adapter.save(model, save_path)
    print(f"Model saved to: {save_path}")

    # 로드
    new_model = adapter.build(config)  # 동일한 구조로 모델 생성
    loaded_model = adapter.load(save_path, model_structure=new_model)
    print("Model loaded successfully")

    # 검증
    X_test = np.random.randn(5, 5).astype(np.float32)
    original_pred = adapter.predict(model, X_test)
    loaded_pred = adapter.predict(loaded_model, X_test)

    if np.allclose(original_pred, loaded_pred):
        print("✓ Loaded model produces identical predictions")
    print("-" * 40 + "\n")


if __name__ == "__main__":
    print("=" * 50)
    print("PyTorch Adapter Quick Start")
    print("=" * 50)
    print()

    # 분류 예제
    quick_classification_example()

    # 회귀 예제
    quick_regression_example()

    # CNN 예제
    quick_cnn_example()

    # 저장/로드 예제
    save_load_example()

    print("=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)