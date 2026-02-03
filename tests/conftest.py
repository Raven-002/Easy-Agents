from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

from easy_agents.core import Model
from easy_agents.core.router import Router

script_dir = Path(__file__).parent


class ModelsTestExtraSettings(BaseModel):
    skip_tests: bool = False


class ModelsTestSettings(Model):
    test_settings: ModelsTestExtraSettings


class Settings(BaseModel):
    models: dict[str, ModelsTestSettings]
    fast_model_name: str


loaded_settings: Settings | None = None
was_settings_initialized: bool = False


def get_settings() -> Settings | None:
    global loaded_settings
    global was_settings_initialized

    if was_settings_initialized:
        return loaded_settings

    models_path = script_dir / "models.yaml"

    try:
        with open(models_path) as f:
            loaded_settings = Settings(**yaml.safe_load(f))
    except FileNotFoundError:
        loaded_settings = None

    return loaded_settings


@pytest.fixture(scope="session")
def settings_yaml(request: pytest.FixtureRequest) -> Settings:
    settings = get_settings()
    if settings is None:
        pytest.skip("Skipping tests because models.yaml is not found.")
    assert settings is not None
    return settings


@pytest.fixture(scope="session")
def all_models(request: pytest.FixtureRequest, settings_yaml: Settings) -> list[Model]:
    return [m for m in settings_yaml.models.values() if not m.test_settings.skip_tests]


def get_test_models() -> Generator[Any, Any, None]:
    settings = get_settings()
    if settings is None:
        return
    for name, model_obj in settings.models.items():
        yield pytest.param(model_obj, id=name)


@pytest.fixture(params=get_test_models(), scope="module")
def model(request: pytest.FixtureRequest, settings_yaml: Settings) -> Model:
    """A fixture that iterate over all models defined in models.yaml.
    Note: The fixture is scoped to the module level to reduce the amount of model switching during testing, in case
    the models are not available simultaneously, but rather swapped, which takes time.
    """
    requested_model: ModelsTestSettings = request.param
    assert isinstance(requested_model, ModelsTestSettings)
    if requested_model.test_settings.skip_tests:
        pytest.skip("Skipping tests because model is marked as skipped.")
    return requested_model


@pytest.fixture()
def fast_model(request: pytest.FixtureRequest, settings_yaml: Settings) -> Model:
    model = settings_yaml.models[settings_yaml.fast_model_name]
    if model.test_settings.skip_tests:
        pytest.skip("Skipping tests because fast model is marked as skipped.")
    return model


@pytest.fixture()
def simple_router(request: pytest.FixtureRequest, settings_yaml: Settings) -> Router:
    if settings_yaml.models[settings_yaml.fast_model_name].test_settings.skip_tests:
        pytest.skip("Skipping tests because fast model is marked as skipped.")
    router = Router(
        models_pool={settings_yaml.fast_model_name: settings_yaml.models[settings_yaml.fast_model_name]},
        router_pool=[settings_yaml.fast_model_name],
    )
    return router


@pytest.fixture()
def complex_router(request: pytest.FixtureRequest, settings_yaml: Settings) -> Router:
    if settings_yaml.models[settings_yaml.fast_model_name].test_settings.skip_tests:
        pytest.skip("Skipping tests because fast model is marked as skipped.")
    router = Router(
        models_pool={key: m for key, m in settings_yaml.models.items() if not m.test_settings.skip_tests},
        router_pool=[settings_yaml.fast_model_name],
    )
    return router
