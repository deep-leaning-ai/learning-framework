# PyTorch Adapter Documentation

## Overview

PyTorch Adapter는 PyTorch 프레임워크를 사용하여 딥러닝 모델을 구축하고 학습할 수 있게 해주는 어댑터입니다.
Hexagonal Architecture의 Port & Adapter 패턴을 따르며, ModelPort 인터페이스를 구현합니다.

## Features

- **자동 디바이스 감지**: CUDA/CPU 자동 선택
- **다양한 레이어 지원**: Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
- **활성화 함수**: ReLU, Sigmoid, Softmax, Tanh
- **옵티마이저**: Adam, SGD, RMSprop, AdamW
- **손실 함수 자동 선택**: 모델 타입에 따른 자동 손실 함수 선택
- **메트릭 추적**: Loss, Accuracy, RMSE 등

## Installation

```bash
# PyTorch extras 설치
pip install -e ".[pytorch]"

# 또는 직접 설치
pip install torch>=1.13.0
```

## Basic Usage

### 1. Adapter 초기화

```python
from adapters.frameworks import PyTorchModelAdapter

adapter = PyTorchModelAdapter()
print(f"Using device: {adapter.device}")  # cuda 또는 cpu
```

### 2. Sequential 모델 구축

```python
from core.models import ModelConfig, ModelType

# 분류 모델 설정
config = ModelConfig(
    name="classifier",
    type=ModelType.CLASSIFICATION,
    architecture="sequential",
    hyperparameters={
        "input_shape": (784,),  # MNIST 입력 크기
        "layers": [
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "dropout", "rate": 0.2},
            {"type": "dense", "units": 64, "activation": "relu"},
            {"type": "dropout", "rate": 0.2},
            {"type": "dense", "units": 10}  # 10개 클래스
        ]
    }
)

# 모델 구축
model = adapter.build(config)
```

### 3. 모델 학습

```python
from core.models import TrainingConfig, OptimizerType
import numpy as np

# 학습 설정
training_config = TrainingConfig(
    epochs=10,
    batch_size=32,
    optimizer=OptimizerType.ADAM,
    learning_rate=0.001,
    validation_split=0.2
)

# 학습 데이터 준비
X_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)

# 모델 학습
metrics = adapter.train(
    model=model,
    train_data=(X_train, y_train),
    val_data=None,  # validation_split 사용 시 자동 분할
    config=training_config
)

print(f"Final loss: {metrics['loss']:.4f}")
print(f"Final accuracy: {metrics['accuracy']:.4f}")
```

### 4. 모델 평가

```python
# 테스트 데이터 준비
X_test = np.random.randn(200, 784).astype(np.float32)
y_test = np.random.randint(0, 10, 200)

# 모델 평가
test_metrics = adapter.evaluate(
    model=model,
    test_data=(X_test, y_test)
)

print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
```

### 5. 예측

```python
# 새로운 데이터에 대한 예측
X_new = np.random.randn(5, 784).astype(np.float32)
predictions = adapter.predict(model, X_new)

print(f"Predictions shape: {predictions.shape}")
print(f"Predicted classes: {np.argmax(predictions, axis=1)}")
```

### 6. 모델 저장 및 로드

```python
# 모델 저장
adapter.save(model, "models/my_pytorch_model.pt")

# 모델 로드
# 주의: PyTorch는 모델 구조를 따로 저장하지 않으므로
# 로드 시 모델 구조를 먼저 생성해야 함
new_model = adapter.build(config)
loaded_model = adapter.load(
    "models/my_pytorch_model.pt",
    model_structure=new_model
)
```

## Advanced Usage

### CNN 모델 구축

```python
# CNN 모델 설정
cnn_config = ModelConfig(
    name="cnn_classifier",
    type=ModelType.CLASSIFICATION,
    architecture="sequential",
    hyperparameters={
        "input_shape": (1, 28, 28),  # (channels, height, width)
        "layers": [
            {"type": "conv2d", "filters": 32, "kernel_size": 3, "activation": "relu"},
            {"type": "maxpooling2d", "pool_size": 2},
            {"type": "conv2d", "filters": 64, "kernel_size": 3, "activation": "relu"},
            {"type": "maxpooling2d", "pool_size": 2},
            {"type": "flatten"},
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "dropout", "rate": 0.5},
            {"type": "dense", "units": 10}
        ]
    }
)

cnn_model = adapter.build(cnn_config)
```

### 회귀 모델

```python
# 회귀 모델 설정
regression_config = ModelConfig(
    name="regressor",
    type=ModelType.REGRESSION,
    architecture="sequential",
    hyperparameters={
        "input_shape": (10,),
        "layers": [
            {"type": "dense", "units": 64, "activation": "relu"},
            {"type": "batchnormalization"},
            {"type": "dense", "units": 32, "activation": "relu"},
            {"type": "dropout", "rate": 0.3},
            {"type": "dense", "units": 1}  # 회귀는 출력이 1개
        ]
    }
)

regression_model = adapter.build(regression_config)
```

### 다양한 옵티마이저 사용

```python
# SGD with momentum
sgd_config = TrainingConfig(
    epochs=20,
    batch_size=64,
    optimizer=OptimizerType.SGD,
    learning_rate=0.01,
    optimizer_params={"momentum": 0.9}
)

# AdamW (weight decay)
adamw_config = TrainingConfig(
    epochs=20,
    batch_size=64,
    optimizer=OptimizerType.ADAMW,
    learning_rate=0.001,
    optimizer_params={"weight_decay": 0.01}
)
```

## Layer Types

### Dense (Fully Connected)

```python
{"type": "dense", "units": 128, "activation": "relu"}
```

### Dropout

```python
{"type": "dropout", "rate": 0.5}
```

### BatchNormalization

```python
{"type": "batchnormalization"}
```

### Conv2D

```python
{
    "type": "conv2d",
    "filters": 32,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "activation": "relu"
}
```

### MaxPooling2D

```python
{
    "type": "maxpooling2d",
    "pool_size": 2,
    "stride": 2
}
```

### Flatten

```python
{"type": "flatten"}
```

## Activation Functions

- `"relu"`: ReLU
- `"sigmoid"`: Sigmoid
- `"tanh"`: Tanh
- `"softmax"`: Softmax (자동으로 마지막 레이어에 적용)
- `None`: 활성화 함수 없음

## Optimizer Types

- `OptimizerType.ADAM`: Adam optimizer
- `OptimizerType.SGD`: Stochastic Gradient Descent
- `OptimizerType.RMSPROP`: RMSProp
- `OptimizerType.ADAMW`: AdamW (Adam with weight decay)

## Loss Functions

손실 함수는 모델 타입에 따라 자동으로 선택됩니다:

- **분류 (Classification)**: CrossEntropyLoss
- **이진 분류 (Binary Classification)**: BCEWithLogitsLoss
- **회귀 (Regression)**: MSELoss

## Error Handling

```python
from core.exceptions import ModelException, DataException

try:
    model = adapter.build(config)
except ModelException as e:
    print(f"Model build failed: {e}")
    print(f"Details: {e.details}")

try:
    metrics = adapter.train(model, train_data, val_data, training_config)
except DataException as e:
    print(f"Training failed: {e}")
```

## Device Management

PyTorch Adapter는 자동으로 사용 가능한 디바이스를 감지합니다:

```python
adapter = PyTorchModelAdapter()
# CUDA가 사용 가능하면 cuda, 아니면 cpu

# 디바이스 확인
print(f"Using device: {adapter.device}")

# 모든 텐서와 모델은 자동으로 적절한 디바이스로 이동
```

## Performance Tips

1. **배치 크기**: GPU 메모리에 맞는 최대 배치 크기 사용
2. **데이터 로더**: 자동으로 DataLoader를 사용하여 효율적인 데이터 처리
3. **Mixed Precision**: 향후 지원 예정
4. **Gradient Accumulation**: optimizer_params로 설정 가능

## Differences from Keras Adapter

1. **ModelConfig 구조**: `hyperparameters` 딕셔너리 사용
   - PyTorch: `hyperparameters={"input_shape": ..., "layers": [...]}`
   - Keras: 직접 속성 사용

2. **모델 저장/로드**: state_dict 방식
   - 모델 구조를 따로 저장하지 않음
   - 로드 시 모델 구조를 먼저 생성 필요

3. **학습 루프**: DataLoader 기반
   - 더 세밀한 제어 가능
   - 배치 단위 처리

## Troubleshooting

### CUDA Out of Memory

```python
# 배치 크기 줄이기
training_config.batch_size = 16

# 또는 CPU 사용
import torch
torch.cuda.empty_cache()
```

### 느린 초기화

PyTorch 첫 import 시 1-3분 소요될 수 있음 (특히 macOS):
- 정상적인 동작
- mutex lock 경고 무시

### Import 오류

```bash
# PyTorch 설치 확인
pip show torch

# 재설치
pip install --upgrade torch
```

## Examples Repository

더 많은 예제는 `examples/` 디렉토리를 참고하세요:

- `examples/pytorch_mnist.py`: MNIST 분류 예제
- `examples/pytorch_regression.py`: 회귀 예제
- `examples/pytorch_cnn.py`: CNN 예제