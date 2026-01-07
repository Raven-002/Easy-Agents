#!/usr/bin/env python
from typing import Any

from pytest import fixture, raises

from easy_agents.settings.settings import (
    AiExpertise,
    Model,
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
        models={"qwen3": Model(api_base="http://localhost:11434/v1", model_name="qwen3:8b", api_key="ollama")},
        model_choices={
            AiExpertise.ORCHESTRATION: "qwen3",
            AiExpertise.CODE_ANALYSIS: "qwen3",
            AiExpertise.CODE_WRITING: "qwen3",
            AiExpertise.CONTEXT_SUMMARIZATION: "qwen3",
        },
    )


class TestSettings:
    @staticmethod
    def setup_method(_test_method: Any):
        reset_settings()

    def test_ai_model_sanity(self):
        Model(model_name="qwen3:8b", api_base="http://localhost:11434/v1", api_key="ollama")

    def test_settings_missing_model_choices(self):
        with raises(ValueError):
            Settings(
                models={"qwen3": Model(api_base="http://localhost:11434/v1", model_name="qwen3:8b", api_key="ollama")},
                model_choices={
                    AiExpertise.ORCHESTRATION: "qwen3",
                    AiExpertise.CODE_ANALYSIS: "qwen3",
                    AiExpertise.CODE_WRITING: "qwen3",
                    # Missing CONTEXT_SUMMARIZATION
                },
            )

    def test_settings_empty_model_name(self):
        with raises(ValueError):
            Settings(
                models={"": Model(api_base="http://localhost:11434/v1", model_name="qwen3:8b", api_key="ollama")},
                model_choices={
                    AiExpertise.ORCHESTRATION: "qwen3",
                    AiExpertise.CODE_ANALYSIS: "qwen3",
                    AiExpertise.CODE_WRITING: "qwen3",
                    AiExpertise.CONTEXT_SUMMARIZATION: "qwen3",
                },
            )

    def test_settings_empty_model_name_for_choice(self):
        with raises(ValueError):
            Settings(
                models={"qwen": Model(api_base="http://localhost:11434/v1", model_name="qwen3:8b", api_key="ollama")},
                model_choices={
                    AiExpertise.ORCHESTRATION: "",
                    AiExpertise.CODE_ANALYSIS: "qwen3",
                    AiExpertise.CODE_WRITING: "qwen3",
                    AiExpertise.CONTEXT_SUMMARIZATION: "qwen3",
                },
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
    qwen3-thinking:
        model_name: qwen3:8b
        api_base: http://localhost:11434/v1
        api_key: ollama
        thinking: true
model_choices:
  orchestration: qwen3
  code_analysis: qwen3-thinking
  code_writing: qwen3
  context_summarization: qwen3
        """
        load_settings_from_yaml(yaml_content)
        settings = get_settings()
        assert settings.models["qwen3"].model_name == "qwen3:8b"
        assert settings.models["qwen3"].api_base == "http://localhost:11434/v1"
        assert settings.models["qwen3"].api_key == "ollama"
        assert settings.models["qwen3"].thinking is False
        assert settings.models["qwen3-thinking"].model_name == "qwen3:8b"
        assert settings.models["qwen3-thinking"].api_base == "http://localhost:11434/v1"
        assert settings.models["qwen3-thinking"].api_key == "ollama"
        assert settings.models["qwen3-thinking"].thinking is True
        assert settings.model_choices == {
            AiExpertise.ORCHESTRATION: "qwen3",
            AiExpertise.CODE_ANALYSIS: "qwen3-thinking",
            AiExpertise.CODE_WRITING: "qwen3",
            AiExpertise.CONTEXT_SUMMARIZATION: "qwen3",
        }
