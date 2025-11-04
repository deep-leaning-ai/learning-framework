"""
PyTorch CNN Example

Convolutional Neural Network 예제
"""

import numpy as np
from sklearn.model_selection import train_test_split

from learning_framework.adapters.frameworks import PyTorchModelAdapter
from learning_framework.core.models import ModelConfig, TrainingConfig, ModelType, OptimizerType


def generate_image_data(n_samples=1000, img_size=28, n_classes=10):
    """
    간단한 이미지 데이터 생성 (실제로는 MNIST, CIFAR 등 사용)

    Args:
        n_samples: 샘플 수
        img_size: 이미지 크기 (img_size x img_size)
        n_classes: 클래스 수

    Returns:
        X: (n_samples, channels, height, width) 형태의 이미지 데이터
        y: 레이블
    """
    # 간단한 패턴 기반 이미지 생성
    X = []
    y = []

    for i in range(n_samples):
        # 클래스별로 다른 패턴 생성
        class_id = i % n_classes
        img = np.random.randn(1, img_size, img_size) * 0.3  # 노이즈

        # 클래스별 특징 추가
        if class_id < 3:
            # 수직선 패턴
            img[:, :, img_size//2-1:img_size//2+1] += 1.0
        elif class_id < 6:
            # 수평선 패턴
            img[:, img_size//2-1:img_size//2+1, :] += 1.0
        else:
            # 대각선 패턴
            for j in range(img_size):
                if j < img_size:
                    img[:, j, j] += 1.0

        X.append(img)
        y.append(class_id)

    return np.array(X, dtype=np.float32), np.array(y)


def create_cnn_model():
    """CNN 모델 생성 및 학습"""

    # 1. 어댑터 초기화
    adapter = PyTorchModelAdapter()
    print(f"Using device: {adapter.device}")
    print("-" * 50)

    # 2. 이미지 데이터 준비
    print("Generating image data...")
    X, y = generate_image_data(n_samples=1000, img_size=28, n_classes=10)

    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Image shape: (channels={X.shape[1]}, height={X.shape[2]}, width={X.shape[3]})")
    print(f"Number of classes: {len(np.unique(y))}")
    print("-" * 50)

    # 3. CNN 모델 설정
    model_config = ModelConfig(
        name="image_classifier_cnn",
        type=ModelType.CLASSIFICATION,
        architecture="sequential",
        hyperparameters={
            "input_shape": (1, 28, 28),  # (channels, height, width)
            "layers": [
                # First Convolutional Block
                {"type": "conv2d", "filters": 32, "kernel_size": 3, "padding": 1, "activation": "relu"},
                {"type": "conv2d", "filters": 32, "kernel_size": 3, "padding": 1, "activation": "relu"},
                {"type": "maxpooling2d", "pool_size": 2},
                {"type": "dropout", "rate": 0.25},

                # Second Convolutional Block
                {"type": "conv2d", "filters": 64, "kernel_size": 3, "padding": 1, "activation": "relu"},
                {"type": "conv2d", "filters": 64, "kernel_size": 3, "padding": 1, "activation": "relu"},
                {"type": "maxpooling2d", "pool_size": 2},
                {"type": "dropout", "rate": 0.25},

                # Flatten and Dense Layers
                {"type": "flatten"},
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dropout", "rate": 0.5},
                {"type": "dense", "units": 10}  # 10 클래스 출력
            ]
        }
    )

    # 4. 모델 구축
    print("Building CNN model...")
    model = adapter.build(model_config)
    print("Model built successfully!")

    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print("-" * 50)

    # 5. 학습 설정
    training_config = TrainingConfig(
        epochs=20,
        batch_size=32,
        optimizer=OptimizerType.ADAM,
        learning_rate=0.001,
        validation_split=0.2
    )

    # 6. 모델 학습
    print("Training CNN model...")
    train_metrics = adapter.train(
        model=model,
        train_data=(X_train, y_train),
        val_data=None,
        config=training_config
    )

    print(f"\nTraining completed!")
    print(f"Final training loss: {train_metrics['loss']:.4f}")
    print(f"Final training accuracy: {train_metrics['accuracy']:.4f}")
    if 'val_loss' in train_metrics:
        print(f"Final validation loss: {train_metrics['val_loss']:.4f}")
        print(f"Final validation accuracy: {train_metrics['val_accuracy']:.4f}")
    print("-" * 50)

    # 7. 테스트 평가
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

    print("\nSample predictions:")
    for i in range(sample_size):
        confidence = predictions[i, predicted_classes[i]] * 100
        print(f"True: {y_sample[i]}, Predicted: {predicted_classes[i]} "
              f"(confidence: {confidence:.1f}%)")

    return model, test_metrics


def compare_cnn_architectures():
    """다양한 CNN 아키텍처 비교"""

    adapter = PyTorchModelAdapter()

    # 데이터 생성
    X, y = generate_image_data(n_samples=500, img_size=28, n_classes=5)

    architectures = [
        {
            "name": "Simple CNN",
            "layers": [
                {"type": "conv2d", "filters": 16, "kernel_size": 3, "activation": "relu"},
                {"type": "maxpooling2d", "pool_size": 2},
                {"type": "flatten"},
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dense", "units": 5}
            ]
        },
        {
            "name": "Deeper CNN",
            "layers": [
                {"type": "conv2d", "filters": 32, "kernel_size": 3, "padding": 1, "activation": "relu"},
                {"type": "conv2d", "filters": 32, "kernel_size": 3, "padding": 1, "activation": "relu"},
                {"type": "maxpooling2d", "pool_size": 2},
                {"type": "conv2d", "filters": 64, "kernel_size": 3, "padding": 1, "activation": "relu"},
                {"type": "maxpooling2d", "pool_size": 2},
                {"type": "flatten"},
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dense", "units": 5}
            ]
        },
        {
            "name": "CNN + BatchNorm",
            "layers": [
                {"type": "conv2d", "filters": 32, "kernel_size": 3, "padding": 1, "activation": "relu"},
                {"type": "batchnormalization"},
                {"type": "maxpooling2d", "pool_size": 2},
                {"type": "conv2d", "filters": 64, "kernel_size": 3, "padding": 1, "activation": "relu"},
                {"type": "batchnormalization"},
                {"type": "maxpooling2d", "pool_size": 2},
                {"type": "flatten"},
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dropout", "rate": 0.5},
                {"type": "dense", "units": 5}
            ]
        }
    ]

    print("\nCNN Architecture Comparison:")
    print("-" * 70)
    print(f"{'Architecture':<20} {'Parameters':<15} {'Accuracy':<15} {'Loss':<10}")
    print("-" * 70)

    for arch in architectures:
        config = ModelConfig(
            name=arch["name"],
            type=ModelType.CLASSIFICATION,
            architecture="sequential",
            hyperparameters={
                "input_shape": (1, 28, 28),
                "layers": arch["layers"]
            }
        )

        model = adapter.build(config)

        # 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())

        training_config = TrainingConfig(
            epochs=10,  # 빠른 비교를 위해 적은 epoch
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

        print(f"{arch['name']:<20} {total_params:<15,} {metrics['accuracy']:<15.4f} {metrics['loss']:<10.4f}")

    print("-" * 70)


def data_augmentation_example():
    """데이터 증강 효과 비교 (시뮬레이션)"""

    adapter = PyTorchModelAdapter()

    # 기본 데이터
    X_base, y_base = generate_image_data(n_samples=300, img_size=28, n_classes=5)

    # 증강된 데이터 (노이즈 추가, 회전 시뮬레이션)
    X_aug = []
    y_aug = []

    for i in range(len(X_base)):
        # 원본 추가
        X_aug.append(X_base[i])
        y_aug.append(y_base[i])

        # 노이즈 버전 추가
        noisy = X_base[i] + np.random.randn(*X_base[i].shape).astype(np.float32) * 0.1
        X_aug.append(noisy)
        y_aug.append(y_base[i])

        # 약간 이동된 버전 추가
        shifted = np.roll(X_base[i], shift=2, axis=1)
        X_aug.append(shifted)
        y_aug.append(y_base[i])

    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug)

    # 모델 설정
    config = ModelConfig(
        name="augmentation_test",
        type=ModelType.CLASSIFICATION,
        architecture="sequential",
        hyperparameters={
            "input_shape": (1, 28, 28),
            "layers": [
                {"type": "conv2d", "filters": 32, "kernel_size": 3, "activation": "relu"},
                {"type": "maxpooling2d", "pool_size": 2},
                {"type": "flatten"},
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dense", "units": 5}
            ]
        }
    )

    training_config = TrainingConfig(
        epochs=15,
        batch_size=32,
        optimizer=OptimizerType.ADAM,
        learning_rate=0.001
    )

    print("\nData Augmentation Effect:")
    print("-" * 50)

    # 원본 데이터로 학습
    model_base = adapter.build(config)
    metrics_base = adapter.train(
        model=model_base,
        train_data=(X_base, y_base),
        val_data=None,
        config=training_config
    )

    print(f"Without augmentation:")
    print(f"  Samples: {len(X_base)}")
    print(f"  Final accuracy: {metrics_base['accuracy']:.4f}")
    print(f"  Final loss: {metrics_base['loss']:.4f}")

    # 증강 데이터로 학습
    model_aug = adapter.build(config)
    metrics_aug = adapter.train(
        model=model_aug,
        train_data=(X_aug, y_aug),
        val_data=None,
        config=training_config
    )

    print(f"\nWith augmentation:")
    print(f"  Samples: {len(X_aug)}")
    print(f"  Final accuracy: {metrics_aug['accuracy']:.4f}")
    print(f"  Final loss: {metrics_aug['loss']:.4f}")

    improvement = (metrics_aug['accuracy'] - metrics_base['accuracy']) * 100
    print(f"\nAccuracy improvement: {improvement:+.1f}%")
    print("-" * 50)


if __name__ == "__main__":
    print("=" * 70)
    print("PyTorch CNN Example")
    print("=" * 70)

    # 메인 CNN 예제 실행
    model, metrics = create_cnn_model()

    print("\n" + "=" * 70)
    print("Additional Experiments")
    print("=" * 70)

    # CNN 아키텍처 비교
    compare_cnn_architectures()

    # 데이터 증강 효과
    data_augmentation_example()

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)