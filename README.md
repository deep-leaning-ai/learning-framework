# AI Learning Framework

교육용 통합 AI 학습 프레임워크 - Hexagonal Architecture Lite 기반

## 프로젝트 개요

다양한 AI 프레임워크(Keras, PyTorch, HuggingFace)를 통합하여 일관된 인터페이스로 학습 실험을 수행하고 비교할 수 있는 교육용 프레임워크입니다.

### 주요 특징

- **통합 인터페이스**: Keras와 PyTorch를 동일한 API로 사용
- **Hexagonal Architecture**: 비즈니스 로직과 프레임워크 분리
- **자동 디바이스 감지**: CUDA/CPU 자동 선택 (PyTorch)
- **다양한 모델 지원**: Sequential, CNN, 분류, 회귀
- **실험 추적**: MLflow/WandB 통합 지원

## 아키텍처

```
library/
├── core/              # 핵심 비즈니스 로직 (포트 인터페이스)
├── adapters/          # 프레임워크 어댑터 구현
├── config/            # 설정 관리 시스템
├── api/               # API 레이어
├── utils/             # 유틸리티
└── tests/             # TDD 테스트
```

## 설치

```bash
pip install -r requirements.txt
```

### 선택적 의존성

```bash
# Keras/TensorFlow 사용시
pip install tensorflow>=2.10.0

# PyTorch 사용시
pip install torch>=1.13.0 pytorch-lightning>=2.0.0

# HuggingFace 사용시
pip install transformers>=4.25.0 datasets>=2.8.0

# 실험 추적 사용시
pip install mlflow>=2.0.0 wandb>=0.13.0
```

## 사용 예제

### Keras Adapter 사용

```python
from adapters.frameworks import KerasModelAdapter
from core.models import ModelConfig, TrainingConfig, ModelType, OptimizerType

# Keras 어댑터 초기화
adapter = KerasModelAdapter()

# 모델 설정
config = ModelConfig(
    name="keras_classifier",
    type=ModelType.CLASSIFICATION,
    architecture="sequential",
    input_shape=(784,),
    layers=[
        {"type": "dense", "units": 128, "activation": "relu"},
        {"type": "dropout", "rate": 0.2},
        {"type": "dense", "units": 10}
    ]
)

# 모델 구축 및 학습
model = adapter.build(config)
training_config = TrainingConfig(
    epochs=10,
    batch_size=32,
    optimizer=OptimizerType.ADAM,
    learning_rate=0.001
)

metrics = adapter.train(model, (X_train, y_train), None, training_config)
```

### PyTorch Adapter 사용

```python
from adapters.frameworks import PyTorchModelAdapter
from core.models import ModelConfig, TrainingConfig, ModelType, OptimizerType

# PyTorch 어댑터 초기화
adapter = PyTorchModelAdapter()
print(f"Using device: {adapter.device}")  # cuda 또는 cpu

# 모델 설정 (hyperparameters 딕셔너리 사용)
config = ModelConfig(
    name="pytorch_classifier",
    type=ModelType.CLASSIFICATION,
    architecture="sequential",
    hyperparameters={
        "input_shape": (784,),
        "layers": [
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "dropout", "rate": 0.2},
            {"type": "dense", "units": 10}
        ]
    }
)

# 모델 구축 및 학습
model = adapter.build(config)
training_config = TrainingConfig(
    epochs=10,
    batch_size=32,
    optimizer=OptimizerType.ADAM,
    learning_rate=0.001
)

metrics = adapter.train(model, (X_train, y_train), None, training_config)
```

### CNN 모델 예제 (PyTorch)

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

더 많은 예제는 `examples/` 디렉토리를 참고하세요:
- `examples/pytorch_quick_start.py` - PyTorch 빠른 시작
- `examples/pytorch_classification.py` - 분류 예제
- `examples/pytorch_regression.py` - 회귀 예제
- `examples/pytorch_cnn.py` - CNN 예제

## 개발

### 테스트 실행

```bash
pytest
```

### 코드 포맷팅

```bash
black .
```

## 구현 진행 상황

### 완료된 기능
- [x] Unit 1: 프로젝트 초기화
- [x] Unit 2: Core 데이터 모델
- [x] Unit 3: 예외 처리 시스템
- [x] Unit 4: Core 포트 인터페이스
- [x] Unit 5: Config 스키마
- [x] Unit 6: Config 로더
- [x] Unit 7: 메트릭 시스템
- [x] Unit 8: Mock 어댑터
- [x] Unit 9: 실험 서비스
- [x] Unit 10: Keras 어댑터
- [x] Unit 11: 데이터 로더
- [ ] Unit 12: API 엔드포인트

### 추가 구현
- [x] PyTorch Adapter 구현
- [x] PyTorch 단위 테스트
- [x] PyTorch 문서화
- [x] 예제 코드 작성

### 지원 프레임워크
- **Keras/TensorFlow**: 완전 지원 (KerasModelAdapter)
- **PyTorch**: 완전 지원 (PyTorchModelAdapter)
- **HuggingFace**: 개발 예정

## 라이선스

MIT License