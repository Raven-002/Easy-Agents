#!/usr/bin/env python
from typing import Any

from pytest import fixture, raises

from easy_agents.settings.settings import (
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
        models={
            "qwen3": AiModel(
                runner_type="local_ollama",
                extra_args={"model": "qwen3:8b"},
            )
        },
        orchestrator_model="qwen3",
        code_analysis_model="qwen3",
        code_review_model="qwen3",
    )


class TestSettings:
    @staticmethod
    def setup_method(_test_method: Any):
        reset_settings()

    def test_ai_model_sanity(self):
        AiModel(runner_type="local_ollama", extra_args={})

    def test_settings_missing_orchestrator_model(self):
        with raises(ValueError):
            Settings(
                models={
                    "qwen3": AiModel(
                        runner_type="local_ollama",
                        extra_args={"model": "qwen3:8b"},
                    )
                },
                orchestrator_model="glm",
                code_analysis_model="qwen3",
                code_review_model="qwen3",
            )

    def test_settings_empty_model_name(self):
        with raises(ValueError):
            Settings(
                models={
                    "": AiModel(
                        runner_type="local_ollama",
                        extra_args={"model": "qwen3:8b"},
                    )
                },
                orchestrator_model="",
                code_analysis_model="",
                code_review_model="",
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
        runner_type: local_ollama
        extra_args:
            model: qwen3:8b
orchestrator_model: qwen3
code_analysis_model: qwen3
code_review_model: qwen3
        """
        load_settings_from_yaml(yaml_content)
        settings = get_settings()
        assert settings.orchestrator_model == "qwen3"
        assert settings.models["qwen3"].runner_type == "local_ollama"
        assert settings.models["qwen3"].extra_args == {"model": "qwen3:8b"}
