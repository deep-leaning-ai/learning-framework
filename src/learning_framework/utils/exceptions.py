"""
Custom exception classes for AI Learning Framework.

All exceptions inherit from AIFrameworkException and provide
structured error handling with context and recovery hints.
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """에러 심각도 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AIFrameworkException(Exception):
    """
    모든 프레임워크 예외의 기본 클래스

    Args:
        message: 에러 메시지
        severity: 에러 심각도
        context: 추가 컨텍스트 정보
        recovery_hint: 복구 힌트
    """

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None,
    ):
        self.message = message
        self.severity = severity
        self.context = context or {}
        self.recovery_hint = recovery_hint
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """사용자 친화적 메시지 포맷"""
        parts = [self.message]

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        if self.recovery_hint:
            parts.append(f"Hint: {self.recovery_hint}")

        return "\n".join(parts)

    def get_severity_level(self) -> str:
        """심각도 레벨 반환"""
        return self.severity.value


class ConfigException(AIFrameworkException):
    """설정 관련 예외"""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_hint=recovery_hint,
        )


class ConfigValidationException(ConfigException):
    """설정 검증 실패 예외"""

    def __init__(
        self,
        field_name: str,
        invalid_value: Any,
        expected: str,
        recovery_hint: Optional[str] = None,
    ):
        super().__init__(
            message=f"Invalid configuration for '{field_name}'",
            context={"field": field_name, "value": invalid_value, "expected": expected},
            recovery_hint=recovery_hint or f"Check {field_name} in your configuration file",
        )


class ConfigLoadException(ConfigException):
    """설정 로드 실패 예외"""

    def __init__(
        self,
        file_path: str,
        reason: str,
        recovery_hint: Optional[str] = None,
    ):
        super().__init__(
            message=f"Failed to load configuration from '{file_path}'",
            context={"file_path": file_path, "reason": reason},
            recovery_hint=recovery_hint or "Check if the file exists and has valid YAML/JSON syntax",
        )


class ModelException(AIFrameworkException):
    """모델 관련 예외"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            context=context,
            recovery_hint=recovery_hint,
        )


class ModelBuildException(ModelException):
    """모델 빌드 실패 예외"""

    def __init__(
        self,
        model_name: str,
        architecture: str,
        reason: str,
        recovery_hint: Optional[str] = None,
    ):
        super().__init__(
            message=f"Failed to build model '{model_name}' with architecture '{architecture}'",
            context={"model": model_name, "architecture": architecture, "reason": reason},
            recovery_hint=recovery_hint or "Check model configuration and architecture parameters",
        )


class ModelLoadException(ModelException):
    """모델 로드 실패 예외"""

    def __init__(
        self,
        model_path: str,
        reason: str,
        recovery_hint: Optional[str] = None,
    ):
        super().__init__(
            message=f"Failed to load model from '{model_path}'",
            context={"path": model_path, "reason": reason},
            recovery_hint=recovery_hint or "Check if the model file exists and is not corrupted",
        )


class DataException(AIFrameworkException):
    """데이터 관련 예외"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            context=context,
            recovery_hint=recovery_hint,
        )


class DataLoadException(DataException):
    """데이터 로드 실패 예외"""

    def __init__(
        self,
        data_path: str,
        reason: str,
        recovery_hint: Optional[str] = None,
    ):
        super().__init__(
            message=f"Failed to load data from '{data_path}'",
            context={"path": data_path, "reason": reason},
            recovery_hint=recovery_hint or "Check if the data file exists and has correct format",
        )


class DataValidationException(DataException):
    """데이터 검증 실패 예외"""

    def __init__(
        self,
        validation_error: str,
        context: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None,
    ):
        super().__init__(
            message=f"Data validation failed: {validation_error}",
            context=context,
            recovery_hint=recovery_hint or "Check data format and values",
        )


class TrainingException(AIFrameworkException):
    """학습 관련 예외"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            context=context,
            recovery_hint=recovery_hint,
        )


class TrainingFailedException(TrainingException):
    """학습 실패 예외"""

    def __init__(
        self,
        experiment_id: str,
        epoch: int,
        reason: str,
        recovery_hint: Optional[str] = None,
    ):
        super().__init__(
            message=f"Training failed for experiment '{experiment_id}' at epoch {epoch}",
            context={"experiment_id": experiment_id, "epoch": epoch, "reason": reason},
            recovery_hint=recovery_hint or "Check training configuration and data",
        )


class ResourceException(AIFrameworkException):
    """리소스 관련 예외"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.CRITICAL,
        context: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            context=context,
            recovery_hint=recovery_hint,
        )


class MemoryException(ResourceException):
    """메모리 부족 예외"""

    def __init__(
        self,
        operation: str,
        required_mb: Optional[float] = None,
        recovery_hint: Optional[str] = None,
    ):
        context = {"operation": operation}
        if required_mb:
            context["required_mb"] = required_mb

        super().__init__(
            message=f"Insufficient memory for operation '{operation}'",
            severity=ErrorSeverity.CRITICAL,
            context=context,
            recovery_hint=recovery_hint or "Try reducing batch size or model size",
        )
