"""
YAML configuration loader implementation.

This module provides a YAML-based configuration loader that implements
the ConfigPort interface and integrates with Pydantic schemas for validation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError

from learning_framework.config.schema import FullConfigSchema
from learning_framework.utils.exceptions import ConfigException


class YAMLConfigLoader:
    """
    YAML 설정 파일 로더

    ConfigPort 프로토콜을 구현하며, YAML 파일을 읽고 쓰며,
    Pydantic 스키마를 사용하여 설정을 검증합니다.
    """

    def load(self, path: str) -> Dict[str, Any]:
        """
        YAML 설정 파일 로드

        Args:
            path: 설정 파일 경로

        Returns:
            설정 딕셔너리

        Raises:
            ConfigException: 파일을 찾을 수 없거나 YAML 파싱 실패 시
        """
        file_path = Path(path)

        # 파일 존재 확인
        if not file_path.exists():
            raise ConfigException(
                f"Configuration file not found: {path}",
                context={"path": path},
                recovery_hint="Check if the file path is correct and the file exists"
            )

        try:
            # YAML 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 빈 파일 체크
            if config is None:
                raise ConfigException(
                    f"Configuration file is empty: {path}",
                    context={"path": path},
                    recovery_hint="Add valid YAML configuration to the file"
                )

            return config

        except yaml.YAMLError as e:
            raise ConfigException(
                f"Failed to parse YAML file: {str(e)}",
                context={"path": path, "error": str(e)},
                recovery_hint="Check YAML syntax for errors"
            ) from e
        except Exception as e:
            raise ConfigException(
                f"Error loading configuration: {str(e)}",
                context={"path": path, "error": str(e)}
            ) from e

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Pydantic 스키마를 사용한 설정 검증

        Args:
            config: 검증할 설정 딕셔너리

        Returns:
            검증 성공 여부
        """
        try:
            # Pydantic 스키마로 검증
            FullConfigSchema(**config)
            return True
        except ValidationError:
            return False
        except Exception:
            return False

    def merge(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        설정 딥 머지 (Deep merge)

        Args:
            base_config: 기본 설정
            override_config: 오버라이드 설정

        Returns:
            병합된 설정 딕셔너리
        """
        def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            """재귀적 딥 머지"""
            result = base.copy()

            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    # 둘 다 딕셔너리면 재귀적으로 병합
                    result[key] = _deep_merge(result[key], value)
                else:
                    # 그 외의 경우 오버라이드 값 사용
                    result[key] = value

            return result

        return _deep_merge(base_config, override_config)

    def save(self, config: Dict[str, Any], path: str) -> None:
        """
        설정을 YAML 파일로 저장

        Args:
            config: 저장할 설정 딕셔너리
            path: 저장 경로

        Raises:
            ConfigException: 파일 저장 실패 시
        """
        file_path = Path(path)

        try:
            # 부모 디렉토리가 없으면 생성
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # YAML 파일로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False
                )

        except Exception as e:
            raise ConfigException(
                f"Failed to save configuration: {str(e)}",
                context={"path": path, "error": str(e)},
                recovery_hint="Check if you have write permissions for the path"
            ) from e
