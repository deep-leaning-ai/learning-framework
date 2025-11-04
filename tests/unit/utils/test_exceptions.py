"""
TDD Tests for exception handling system.
All tests follow Given-When-Then pattern.
"""

import pytest
from utils.exceptions import (
    AIFrameworkException,
    ErrorSeverity,
    ConfigException,
    ConfigValidationException,
    ConfigLoadException,
    ModelException,
    ModelBuildException,
    ModelLoadException,
    DataException,
    DataLoadException,
    DataValidationException,
    TrainingException,
    TrainingFailedException,
    ResourceException,
    MemoryException,
)


class TestAIFrameworkException:
    """AIFrameworkException 기본 예외 클래스 테스트"""

    def test_basic_exception_creation(self):
        # Given: 기본 에러 메시지
        message = "Test error message"

        # When: AIFrameworkException 생성
        exc = AIFrameworkException(message)

        # Then: 올바르게 생성됨
        assert exc.message == message
        assert exc.severity == ErrorSeverity.MEDIUM
        assert exc.context == {}
        assert exc.recovery_hint is None

    def test_exception_with_all_parameters(self):
        # Given: 모든 파라미터를 포함한 에러 정보
        message = "Test error"
        severity = ErrorSeverity.HIGH
        context = {"key": "value"}
        recovery_hint = "Try this solution"

        # When: AIFrameworkException 생성
        exc = AIFrameworkException(
            message=message,
            severity=severity,
            context=context,
            recovery_hint=recovery_hint
        )

        # Then: 모든 정보가 올바르게 저장됨
        assert exc.message == message
        assert exc.severity == severity
        assert exc.context == context
        assert exc.recovery_hint == recovery_hint

    def test_exception_message_formatting(self):
        # Given: 컨텍스트와 힌트가 있는 예외
        exc = AIFrameworkException(
            message="Error occurred",
            context={"file": "test.py", "line": 42},
            recovery_hint="Check the file"
        )

        # When: 예외 메시지 문자열 조회
        message_str = str(exc)

        # Then: 포맷된 메시지 반환
        assert "Error occurred" in message_str
        assert "Context:" in message_str
        assert "file=test.py" in message_str
        assert "line=42" in message_str
        assert "Hint: Check the file" in message_str

    def test_get_severity_level(self):
        # Given: HIGH severity 예외
        exc = AIFrameworkException(
            message="Test",
            severity=ErrorSeverity.HIGH
        )

        # When: 심각도 레벨 조회
        level = exc.get_severity_level()

        # Then: 올바른 레벨 반환
        assert level == "high"


class TestConfigException:
    """ConfigException 설정 예외 테스트"""

    def test_config_exception_creation(self):
        # Given: 설정 에러 메시지
        message = "Configuration error"

        # When: ConfigException 생성
        exc = ConfigException(message)

        # Then: HIGH severity로 생성됨
        assert exc.message == message
        assert exc.severity == ErrorSeverity.HIGH

    def test_config_validation_exception(self):
        # Given: 검증 실패 정보
        field_name = "epochs"
        invalid_value = -1
        expected = "positive integer"

        # When: ConfigValidationException 생성
        exc = ConfigValidationException(
            field_name=field_name,
            invalid_value=invalid_value,
            expected=expected
        )

        # Then: 올바른 정보가 포함됨
        assert field_name in exc.message
        assert exc.context["field"] == field_name
        assert exc.context["value"] == invalid_value
        assert exc.context["expected"] == expected

    def test_config_load_exception(self):
        # Given: 설정 파일 로드 실패 정보
        file_path = "/path/to/config.yaml"
        reason = "File not found"

        # When: ConfigLoadException 생성
        exc = ConfigLoadException(
            file_path=file_path,
            reason=reason
        )

        # Then: 올바른 정보가 포함됨
        assert file_path in exc.message
        assert exc.context["file_path"] == file_path
        assert exc.context["reason"] == reason
        assert "file exists" in exc.recovery_hint.lower()


class TestModelException:
    """ModelException 모델 예외 테스트"""

    def test_model_exception_creation(self):
        # Given: 모델 에러 메시지
        message = "Model error"

        # When: ModelException 생성
        exc = ModelException(message)

        # Then: HIGH severity로 생성됨
        assert exc.message == message
        assert exc.severity == ErrorSeverity.HIGH

    def test_model_build_exception(self):
        # Given: 모델 빌드 실패 정보
        model_name = "resnet50"
        architecture = "ResNet"
        reason = "Invalid layer configuration"

        # When: ModelBuildException 생성
        exc = ModelBuildException(
            model_name=model_name,
            architecture=architecture,
            reason=reason
        )

        # Then: 올바른 정보가 포함됨
        assert model_name in exc.message
        assert architecture in exc.message
        assert exc.context["model"] == model_name
        assert exc.context["architecture"] == architecture
        assert exc.context["reason"] == reason

    def test_model_load_exception(self):
        # Given: 모델 로드 실패 정보
        model_path = "/path/to/model.pth"
        reason = "Corrupted file"

        # When: ModelLoadException 생성
        exc = ModelLoadException(
            model_path=model_path,
            reason=reason
        )

        # Then: 올바른 정보가 포함됨
        assert model_path in exc.message
        assert exc.context["path"] == model_path
        assert exc.context["reason"] == reason


class TestDataException:
    """DataException 데이터 예외 테스트"""

    def test_data_exception_creation(self):
        # Given: 데이터 에러 메시지
        message = "Data error"

        # When: DataException 생성
        exc = DataException(message)

        # Then: MEDIUM severity로 생성됨
        assert exc.message == message
        assert exc.severity == ErrorSeverity.MEDIUM

    def test_data_load_exception(self):
        # Given: 데이터 로드 실패 정보
        data_path = "/path/to/data.csv"
        reason = "Invalid CSV format"

        # When: DataLoadException 생성
        exc = DataLoadException(
            data_path=data_path,
            reason=reason
        )

        # Then: 올바른 정보가 포함됨
        assert data_path in exc.message
        assert exc.context["path"] == data_path
        assert exc.context["reason"] == reason

    def test_data_validation_exception(self):
        # Given: 데이터 검증 실패 정보
        validation_error = "Missing required column"
        context = {"column": "target"}

        # When: DataValidationException 생성
        exc = DataValidationException(
            validation_error=validation_error,
            context=context
        )

        # Then: 올바른 정보가 포함됨
        assert validation_error in exc.message
        assert exc.context["column"] == "target"


class TestTrainingException:
    """TrainingException 학습 예외 테스트"""

    def test_training_exception_creation(self):
        # Given: 학습 에러 메시지
        message = "Training error"

        # When: TrainingException 생성
        exc = TrainingException(message)

        # Then: HIGH severity로 생성됨
        assert exc.message == message
        assert exc.severity == ErrorSeverity.HIGH

    def test_training_failed_exception(self):
        # Given: 학습 실패 정보
        experiment_id = "exp_001"
        epoch = 10
        reason = "NaN loss detected"

        # When: TrainingFailedException 생성
        exc = TrainingFailedException(
            experiment_id=experiment_id,
            epoch=epoch,
            reason=reason
        )

        # Then: 올바른 정보가 포함됨
        assert experiment_id in exc.message
        assert str(epoch) in exc.message
        assert exc.context["experiment_id"] == experiment_id
        assert exc.context["epoch"] == epoch
        assert exc.context["reason"] == reason


class TestResourceException:
    """ResourceException 리소스 예외 테스트"""

    def test_resource_exception_creation(self):
        # Given: 리소스 에러 메시지
        message = "Resource error"

        # When: ResourceException 생성
        exc = ResourceException(message)

        # Then: CRITICAL severity로 생성됨
        assert exc.message == message
        assert exc.severity == ErrorSeverity.CRITICAL

    def test_memory_exception_without_size(self):
        # Given: 메모리 부족 정보 (크기 없음)
        operation = "model training"

        # When: MemoryException 생성
        exc = MemoryException(operation=operation)

        # Then: 올바른 정보가 포함됨
        assert operation in exc.message
        assert exc.context["operation"] == operation
        assert "batch size" in exc.recovery_hint.lower()

    def test_memory_exception_with_size(self):
        # Given: 메모리 부족 정보 (크기 포함)
        operation = "data loading"
        required_mb = 2048.5

        # When: MemoryException 생성
        exc = MemoryException(
            operation=operation,
            required_mb=required_mb
        )

        # Then: 올바른 정보가 포함됨
        assert exc.context["operation"] == operation
        assert exc.context["required_mb"] == required_mb
        assert exc.severity == ErrorSeverity.CRITICAL


class TestExceptionInheritance:
    """예외 상속 구조 테스트"""

    def test_all_exceptions_inherit_from_base(self):
        # Given: 다양한 예외 클래스들
        # When: 각 예외 생성
        config_exc = ConfigException("test")
        model_exc = ModelException("test")
        data_exc = DataException("test")
        training_exc = TrainingException("test")
        resource_exc = ResourceException("test")

        # Then: 모두 AIFrameworkException의 인스턴스
        assert isinstance(config_exc, AIFrameworkException)
        assert isinstance(model_exc, AIFrameworkException)
        assert isinstance(data_exc, AIFrameworkException)
        assert isinstance(training_exc, AIFrameworkException)
        assert isinstance(resource_exc, AIFrameworkException)

    def test_specific_exceptions_inherit_from_category(self):
        # Given: 특정 예외 클래스들
        # When: 각 예외 생성
        config_val_exc = ConfigValidationException("field", "value", "expected")
        model_build_exc = ModelBuildException("model", "arch", "reason")

        # Then: 해당 카테고리 예외의 인스턴스
        assert isinstance(config_val_exc, ConfigException)
        assert isinstance(model_build_exc, ModelException)


class TestErrorSeverity:
    """ErrorSeverity 열거형 테스트"""

    def test_all_severity_levels_exist(self):
        # Given/When/Then: 모든 심각도 레벨이 존재함
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_severity_comparison(self):
        # Given: 다양한 심각도의 예외들
        low_exc = AIFrameworkException("test", severity=ErrorSeverity.LOW)
        high_exc = AIFrameworkException("test", severity=ErrorSeverity.HIGH)

        # When: 심각도 레벨 조회
        low_level = low_exc.get_severity_level()
        high_level = high_exc.get_severity_level()

        # Then: 올바른 레벨 반환
        assert low_level == "low"
        assert high_level == "high"
        assert low_level != high_level
