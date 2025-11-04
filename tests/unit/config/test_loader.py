"""
TDD Tests for YAML configuration loader.
All tests follow Given-When-Then pattern.
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError

from config.loader import YAMLConfigLoader
from config.schema import FullConfigSchema
from utils.exceptions import ConfigException


class TestYAMLConfigLoader:
    """YAMLConfigLoader 기본 기능 테스트"""

    def test_load_valid_yaml_config(self, tmp_path):
        # Given: 유효한 YAML 설정 파일
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
experiment:
  name: test_experiment
  seed: 42

model:
  name: test_model
  framework: pytorch
  task_type: classification
  architecture: ResNet50

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001

data:
  name: test_data
  path: data/test.csv
""")
        loader = YAMLConfigLoader()

        # When: 설정 파일 로드
        config = loader.load(str(config_file))

        # Then: 딕셔너리로 올바르게 로드됨
        assert isinstance(config, dict)
        assert "experiment" in config
        assert config["experiment"]["name"] == "test_experiment"
        assert config["model"]["name"] == "test_model"

    def test_load_nonexistent_file_raises_error(self):
        # Given: 존재하지 않는 파일 경로
        loader = YAMLConfigLoader()
        nonexistent_path = "/nonexistent/config.yaml"

        # When/Then: ConfigException 발생
        with pytest.raises(ConfigException) as exc_info:
            loader.load(nonexistent_path)

        assert "not found" in str(exc_info.value).lower()

    def test_load_invalid_yaml_raises_error(self, tmp_path):
        # Given: 잘못된 YAML 파일
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("""
experiment:
  name: test
  invalid yaml syntax: [
""")
        loader = YAMLConfigLoader()

        # When/Then: ConfigException 발생
        with pytest.raises(ConfigException) as exc_info:
            loader.load(str(config_file))

        assert "yaml" in str(exc_info.value).lower()

    def test_load_empty_file_raises_error(self, tmp_path):
        # Given: 빈 YAML 파일
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        loader = YAMLConfigLoader()

        # When/Then: ConfigException 발생
        with pytest.raises(ConfigException) as exc_info:
            loader.load(str(config_file))

        assert "empty" in str(exc_info.value).lower()


class TestYAMLConfigLoaderValidation:
    """설정 검증 테스트"""

    def test_validate_valid_config_returns_true(self):
        # Given: 유효한 설정 딕셔너리
        loader = YAMLConfigLoader()
        config = {
            "experiment": {"name": "test"},
            "model": {
                "name": "model",
                "framework": "pytorch",
                "task_type": "classification",
                "architecture": "ResNet"
            },
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "data": {
                "name": "dataset",
                "path": "data.csv"
            }
        }

        # When: 검증 수행
        result = loader.validate(config)

        # Then: True 반환
        assert result is True

    def test_validate_invalid_config_returns_false(self):
        # Given: 잘못된 설정 (필수 필드 누락)
        loader = YAMLConfigLoader()
        config = {
            "experiment": {"name": "test"},
            # model, training, data 누락
        }

        # When: 검증 수행
        result = loader.validate(config)

        # Then: False 반환
        assert result is False

    def test_validate_with_invalid_values_returns_false(self):
        # Given: 잘못된 값이 있는 설정
        loader = YAMLConfigLoader()
        config = {
            "experiment": {"name": "test"},
            "model": {
                "name": "model",
                "framework": "invalid_framework",  # 잘못된 프레임워크
                "task_type": "classification",
                "architecture": "ResNet"
            },
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "data": {
                "name": "dataset",
                "path": "data.csv"
            }
        }

        # When: 검증 수행
        result = loader.validate(config)

        # Then: False 반환
        assert result is False


class TestYAMLConfigLoaderMerge:
    """설정 병합 테스트"""

    def test_merge_configs_overwrites_values(self):
        # Given: 기본 설정과 오버라이드 설정
        loader = YAMLConfigLoader()
        base_config = {
            "experiment": {"name": "base", "seed": 42},
            "model": {"name": "base_model"}
        }
        override_config = {
            "experiment": {"name": "override"},
            "model": {"name": "override_model"}
        }

        # When: 설정 병합
        merged = loader.merge(base_config, override_config)

        # Then: 오버라이드 값이 적용됨
        assert merged["experiment"]["name"] == "override"
        assert merged["experiment"]["seed"] == 42  # 오버라이드되지 않은 값 유지
        assert merged["model"]["name"] == "override_model"

    def test_merge_configs_deep_merge(self):
        # Given: 중첩된 딕셔너리를 가진 설정
        loader = YAMLConfigLoader()
        base_config = {
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001
            }
        }
        override_config = {
            "training": {
                "epochs": 50  # epochs만 오버라이드
            }
        }

        # When: 설정 병합
        merged = loader.merge(base_config, override_config)

        # Then: 딥 머지가 수행됨
        assert merged["training"]["epochs"] == 50
        assert merged["training"]["batch_size"] == 32
        assert merged["training"]["learning_rate"] == 0.001

    def test_merge_adds_new_keys(self):
        # Given: 새로운 키를 추가하는 오버라이드
        loader = YAMLConfigLoader()
        base_config = {"experiment": {"name": "test"}}
        override_config = {"experiment": {"name": "test", "description": "new desc"}}

        # When: 설정 병합
        merged = loader.merge(base_config, override_config)

        # Then: 새로운 키가 추가됨
        assert merged["experiment"]["description"] == "new desc"


class TestYAMLConfigLoaderSave:
    """설정 저장 테스트"""

    def test_save_config_creates_yaml_file(self, tmp_path):
        # Given: 저장할 설정
        loader = YAMLConfigLoader()
        config = {
            "experiment": {"name": "test", "seed": 42},
            "model": {"name": "test_model"}
        }
        save_path = tmp_path / "saved_config.yaml"

        # When: 설정 저장
        loader.save(config, str(save_path))

        # Then: YAML 파일이 생성됨
        assert save_path.exists()

    def test_save_and_load_roundtrip(self, tmp_path):
        # Given: 저장할 설정
        loader = YAMLConfigLoader()
        original_config = {
            "experiment": {"name": "roundtrip", "seed": 123},
            "model": {"name": "roundtrip_model", "framework": "pytorch"}
        }
        config_path = tmp_path / "roundtrip.yaml"

        # When: 저장 후 다시 로드
        loader.save(original_config, str(config_path))
        loaded_config = loader.load(str(config_path))

        # Then: 원본과 동일한 설정이 로드됨
        assert loaded_config["experiment"]["name"] == "roundtrip"
        assert loaded_config["experiment"]["seed"] == 123
        assert loaded_config["model"]["name"] == "roundtrip_model"

    def test_save_to_invalid_path_raises_error(self):
        # Given: 잘못된 경로
        loader = YAMLConfigLoader()
        config = {"experiment": {"name": "test"}}
        invalid_path = "/invalid/nonexistent/directory/config.yaml"

        # When/Then: ConfigException 발생
        with pytest.raises(ConfigException) as exc_info:
            loader.save(config, invalid_path)

        assert "save" in str(exc_info.value).lower()


class TestYAMLConfigLoaderWithSchema:
    """Pydantic 스키마와 통합 테스트"""

    def test_load_and_parse_to_schema(self, tmp_path):
        # Given: 완전한 설정 YAML 파일
        config_file = tmp_path / "full_config.yaml"
        config_file.write_text("""
experiment:
  name: schema_test
  seed: 42

model:
  name: test_model
  framework: pytorch
  task_type: classification
  architecture: ResNet50

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001

data:
  name: test_data
  path: data/test.csv
""")
        loader = YAMLConfigLoader()

        # When: 로드 후 Pydantic 스키마로 파싱
        config_dict = loader.load(str(config_file))
        schema = FullConfigSchema(**config_dict)

        # Then: 스키마 객체로 올바르게 변환됨
        assert schema.experiment.name == "schema_test"
        assert schema.model.name == "test_model"
        assert schema.training.epochs == 10

    def test_validate_uses_pydantic_schema(self):
        # Given: YAMLConfigLoader와 설정
        loader = YAMLConfigLoader()
        valid_config = {
            "experiment": {"name": "test"},
            "model": {
                "name": "model",
                "framework": "pytorch",
                "task_type": "classification",
                "architecture": "ResNet"
            },
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "data": {
                "name": "dataset",
                "path": "data.csv"
            }
        }

        # When: Pydantic 스키마로 검증
        is_valid = loader.validate(valid_config)

        # Then: 검증 성공
        assert is_valid is True


class TestYAMLConfigLoaderEdgeCases:
    """엣지 케이스 테스트"""

    def test_load_config_with_comments(self, tmp_path):
        # Given: 주석이 포함된 YAML 파일
        config_file = tmp_path / "with_comments.yaml"
        config_file.write_text("""
# Experiment configuration
experiment:
  name: test  # experiment name
  seed: 42    # random seed

model:
  name: model
  framework: pytorch
  task_type: classification
  architecture: ResNet

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001

data:
  name: data
  path: data.csv
""")
        loader = YAMLConfigLoader()

        # When: 로드
        config = loader.load(str(config_file))

        # Then: 주석을 제외하고 올바르게 로드됨
        assert config["experiment"]["name"] == "test"
        assert config["experiment"]["seed"] == 42

    def test_merge_empty_configs(self):
        # Given: 빈 설정들
        loader = YAMLConfigLoader()
        base_config = {}
        override_config = {}

        # When: 병합
        merged = loader.merge(base_config, override_config)

        # Then: 빈 딕셔너리 반환
        assert merged == {}

    def test_save_creates_parent_directories(self, tmp_path):
        # Given: 존재하지 않는 부모 디렉토리를 가진 경로
        loader = YAMLConfigLoader()
        config = {"experiment": {"name": "test"}}
        nested_path = tmp_path / "nested" / "dir" / "config.yaml"

        # When: 저장 (부모 디렉토리 자동 생성)
        loader.save(config, str(nested_path))

        # Then: 파일이 생성됨
        assert nested_path.exists()
        assert nested_path.parent.exists()
