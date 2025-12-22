#!/usr/bin/env python
from typing import Any

from pytest import fixture, raises

from ai_cr.settings.settings import (
    AiModel,
    Settings,
    get_settings,
    init_settings,
    load_settings_from_yaml,
    reset_settings,
)

"""Tests for the settings module."""


@fixture
def default_settings() -> Settings:
    return Settings(
        models={"qwen3": AiModel(api_base="http://localhost:11434/v1", model_name="qwen3:8b")},
        orchestrator_model_name="qwen3",
        code_analysis_model_name="qwen3",
        code_review_model_name="qwen3",
    )


class TestSettings:
    @staticmethod
    def setup_method(_test_method: Any):
        reset_settings()

    def test_ai_model_sanity(self):
        AiModel(model_name="qwen3:8b", api_base="http://localhost:11434/v1", api_key="ollama")

    def test_settings_missing_orchestrator_model(self):
        with raises(ValueError):
            Settings(
                models={"qwen3": AiModel(api_base="http://localhost:11434/v1", model_name="qwen3:8b")},
                orchestrator_model_name="glm",
                code_analysis_model_name="qwen3",
                code_review_model_name="qwen3",
            )

    def test_settings_empty_model_name(self):
        with raises(ValueError):
            Settings(
                models={"": AiModel(api_base="http://localhost:11434/v1", model_name="qwen3:8b")},
                orchestrator_model_name="qwen3",
                code_analysis_model_name="qwen3",
                code_review_model_name="qwen3",
            )

    def test_settings_empty_model_name_for_category(self):
        with raises(ValueError):
            Settings(
                models={"qwen": AiModel(api_base="http://localhost:11434/v1", model_name="qwen3:8b")},
                orchestrator_model_name="",
                code_analysis_model_name="qwen3",
                code_review_model_name="qwen3",
            )
        with raises(ValueError):
            Settings(
                models={"qwen": AiModel(api_base="http://localhost:11434/v1", model_name="qwen3:8b")},
                orchestrator_model_name="qwen3",
                code_analysis_model_name="",
                code_review_model_name="qwen3",
            )
        with raises(ValueError):
            Settings(
                models={"qwen": AiModel(api_base="http://localhost:11434/v1", model_name="qwen3:8b")},
                orchestrator_model_name="qwen3",
                code_analysis_model_name="qwen3",
                code_review_model_name="",
            )

    def test_get_settings_uninitialized(self):
        with raises(RuntimeError):
            get_settings()

    def test_get_settings(self, default_settings: Settings):
        init_settings(default_settings)
        settings = get_settings()
        assert settings == default_settings

    def test_load_settings_from_yaml(self):
        yaml_content = """
models:
    qwen3:
        model_name: qwen3:8b
        api_base: http://localhost:11434/v1
        api_key: ollama
orchestrator_model_name: qwen3
code_analysis_model_name: qwen3
code_review_model_name: qwen3
        """
        load_settings_from_yaml(yaml_content)
        settings = get_settings()
        assert settings.orchestrator_model_name == "qwen3"
        assert settings.code_analysis_model_name == "qwen3"
        assert settings.code_review_model_name == "qwen3"
        assert settings.models["qwen3"].model_name == "qwen3:8b"
        assert settings.models["qwen3"].api_base == "http://localhost:11434/v1"
        assert settings.models["qwen3"].api_key == "ollama"
