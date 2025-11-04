"""
PyTorch Classification Example

MNIST 데이터셋을 사용한 분류 모델 예제
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from learning_framework.adapters.frameworks import PyTorchModelAdapter
from learning_framework.core.models import ModelConfig, TrainingConfig, ModelType, OptimizerType


def create_classification_model():
    """분류 모델 생성 및 학습 예제"""

    # 1. 어댑터 초기화
    adapter = PyTorchModelAdapter()
    print(f"Using device: {adapter.device}")
    print("-" * 50)

    # 2. 데이터 준비 (sklearn의 digits 데이터셋 사용)
    print("Loading data...")
    digits = load_digits()
    X, y = digits.data, digits.target

    # 데이터 정규화
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print("-" * 50)

    # 3. 모델 설정
    model_config = ModelConfig(
        name="digit_classifier",
        type=ModelType.CLASSIFICATION,
        architecture="sequential",
        hyperparameters={
            "input_shape": (64,),  # 8x8 이미지를 평탄화
            "layers": [
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "batchnormalization"},
                {"type": "dropout", "rate": 0.3},
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dropout", "rate": 0.3},
                {"type": "dense", "units": 10}  # 10개 숫자 클래스
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
        epochs=30,
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
    print(f"Final training loss: {train_metrics['loss']:.4f}")
    print(f"Final training accuracy: {train_metrics['accuracy']:.4f}")
    if 'val_loss' in train_metrics:
        print(f"Final validation loss: {train_metrics['val_loss']:.4f}")
        print(f"Final validation accuracy: {train_metrics['val_accuracy']:.4f}")
    print("-" * 50)

    # 7. 테스트 데이터로 평가
    print("Evaluating on test data...")
    test_metrics = adapter.evaluate(
        model=model,
        test_data=(X_test, y_test)
    )

    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print("-" * 50)

    # 8. 예측 예제
    print("Making predictions...")
    sample_size = 5
    X_sample = X_test[:sample_size]
    y_sample = y_test[:sample_size]

    predictions = adapter.predict(model, X_sample)
    predicted_classes = np.argmax(predictions, axis=1)

    print(f"\nSample predictions:")
    for i in range(sample_size):
        confidence = predictions[i, predicted_classes[i]] * 100
        print(f"True: {y_sample[i]}, Predicted: {predicted_classes[i]} "
              f"(confidence: {confidence:.1f}%)")
    print("-" * 50)

    # 9. 모델 저장
    print("Saving model...")
    save_path = "models/pytorch_digit_classifier.pt"
    adapter.save(model, save_path)
    print(f"Model saved to: {save_path}")

    # 10. 모델 로드 및 검증
    print("\nLoading saved model...")
    new_model = adapter.build(model_config)
    loaded_model = adapter.load(save_path, model_structure=new_model)

    # 로드된 모델로 예측
    loaded_predictions = adapter.predict(loaded_model, X_sample)
    loaded_classes = np.argmax(loaded_predictions, axis=1)

    print("Verifying loaded model predictions:")
    assert np.allclose(predictions, loaded_predictions, rtol=1e-5)
    print("✓ Loaded model produces identical predictions")

    return model, test_metrics


def experiment_with_optimizers():
    """다양한 옵티마이저 실험"""

    adapter = PyTorchModelAdapter()

    # 간단한 데이터 생성
    X = np.random.randn(1000, 20).astype(np.float32)
    y = np.random.randint(0, 3, 1000)

    # 모델 설정
    config = ModelConfig(
        name="optimizer_test",
        type=ModelType.CLASSIFICATION,
        architecture="sequential",
        hyperparameters={
            "input_shape": (20,),
            "layers": [
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dense", "units": 3}
            ]
        }
    )

    optimizers = [
        (OptimizerType.ADAM, 0.001, {}),
        (OptimizerType.SGD, 0.01, {"momentum": 0.9}),
        (OptimizerType.RMSPROP, 0.001, {}),
        (OptimizerType.ADAMW, 0.001, {"weight_decay": 0.01})
    ]

    print("\nOptimizer Comparison:")
    print("-" * 60)

    for opt_type, lr, opt_params in optimizers:
        model = adapter.build(config)

        training_config = TrainingConfig(
            epochs=10,
            batch_size=32,
            optimizer=opt_type,
            learning_rate=lr,
            optimizer_params=opt_params
        )

        metrics = adapter.train(
            model=model,
            train_data=(X, y),
            val_data=None,
            config=training_config
        )

        print(f"{opt_type.value:10s} (lr={lr}): "
              f"Loss={metrics['loss']:.4f}, "
              f"Accuracy={metrics['accuracy']:.4f}")

    print("-" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch Classification Example")
    print("=" * 60)

    # 메인 분류 예제 실행
    model, metrics = create_classification_model()

    # 옵티마이저 비교 실험
    experiment_with_optimizers()

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)