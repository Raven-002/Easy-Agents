#!/usr/bin/env python
from typing import Any

from pytest import fixture, raises

from ai_cr.settings.settings import (
    AiModel,
    AiRunnerExpertise,
    AiRunnerType,
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
                expertise=[AiRunnerExpertise.ORCHESTRATION],
                runner_type=AiRunnerType.LOCAL_OLLAMA,
                extra_args={"model": "qwen3:8b"},
            )
        },
        orchestrator_model="qwen3",
    )


class TestSettings:
    @staticmethod
    def setup_method(_test_method: Any):
        reset_settings()

    def test_ai_model_no_expertise(self):
        with raises(ValueError):
            AiModel(
                expertise=[],
                runner_type=AiRunnerType.LOCAL_OLLAMA,
                extra_args={"model": "qwen3:8b"},
            )

    def test_ai_model_sanity(self):
        AiModel(expertise=[AiRunnerExpertise.CODE_UNDERSTANDING], runner_type=AiRunnerType.LOCAL_OLLAMA)

    def test_settings_missing_orchestrator_model(self):
        with raises(ValueError):
            Settings(
                models={
                    "qwen3": AiModel(
                        expertise=[AiRunnerExpertise.ORCHESTRATION, AiRunnerExpertise.CODE_REVIEW],
                        runner_type=AiRunnerType.LOCAL_OLLAMA,
                        extra_args={"model": "qwen3:8b"},
                    )
                },
                orchestrator_model="glm",
            )

    def test_settings_orchestrator_missing_expertise(self):
        with raises(ValueError):
            Settings(
                models={
                    "qwen3": AiModel(
                        expertise=[AiRunnerExpertise.CODE_REVIEW],
                        runner_type=AiRunnerType.LOCAL_OLLAMA,
                        extra_args={"model": "qwen3:8b"},
                    )
                },
                orchestrator_model="qwen3",
            )

    def test_settings_empty_model_name(self):
        with raises(ValueError):
            Settings(
                models={
                    "": AiModel(
                        expertise=[AiRunnerExpertise.CODE_REVIEW],
                        runner_type=AiRunnerType.LOCAL_OLLAMA,
                        extra_args={"model": "qwen3:8b"},
                    )
                },
                orchestrator_model="",
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
        expertise:
            - orchestration
            - code_review
        runner_type: local_ollama
        extra_args:
            model: qwen3:8b
orchestrator_model: qwen3
        """
        load_settings_from_yaml(yaml_content)
        settings = get_settings()
        assert settings.orchestrator_model == "qwen3"
        assert settings.models["qwen3"].runner_type == AiRunnerType.LOCAL_OLLAMA
        assert settings.models["qwen3"].extra_args == {"model": "qwen3:8b"}
        assert settings.models["qwen3"].expertise == [AiRunnerExpertise.ORCHESTRATION, AiRunnerExpertise.CODE_REVIEW]
