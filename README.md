# AI Learning Framework

교육용 통합 AI 학습 프레임워크 - Hexagonal Architecture Lite 기반

## 프로젝트 개요

다양한 AI 프레임워크(Keras, PyTorch, HuggingFace)를 통합하여 일관된 인터페이스로 학습 실험을 수행하고 비교할 수 있는 교육용 프레임워크입니다.

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

```python
# 추후 업데이트 예정
```

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

- [x] Unit 1: 프로젝트 초기화
- [ ] Unit 2: Core 데이터 모델
- [ ] Unit 3: 예외 처리 시스템
- [ ] Unit 4: Core 포트 인터페이스
- [ ] Unit 5: Config 스키마
- [ ] Unit 6: Config 로더
- [ ] Unit 7: 메트릭 시스템
- [ ] Unit 8: Mock 어댑터
- [ ] Unit 9: 실험 서비스
- [ ] Unit 10: Keras 어댑터
- [ ] Unit 11: 데이터 로더
- [ ] Unit 12: API 엔드포인트

## 라이선스

MIT License