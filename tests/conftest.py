from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from easy_agents.core import Model
from easy_agents.core.router import Router

script_dir = Path(__file__).parent


class Settings(BaseModel):
    models: dict[str, Model]
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


@pytest.fixture()
def settings_yaml(request: pytest.FixtureRequest) -> Settings:
    settings = get_settings()
    if settings is None:
        pytest.skip("Skipping tests because models.yaml is not found.")
    assert settings is not None
    return settings


@pytest.fixture()
def all_models(request: pytest.FixtureRequest, settings_yaml: Settings) -> list[Model]:
    return list(settings_yaml.models.values())


def get_test_models() -> Iterator[Model]:
    settings = get_settings()
    if settings is None:
        return
    yield from settings.models.values()


@pytest.fixture(params=get_test_models(), scope="module")
def model(request: pytest.FixtureRequest, settings_yaml: Settings) -> Model:
    """A fixture that iterate over all models defined in models.yaml.
    Note: The fixture is scoped to the module level to reduce the amount of model switching during testing, in case
    the models are not available simultaneously, but rather swapped, which takes time.
    """
    requested_model: Model = request.param
    assert isinstance(requested_model, Model)
    return requested_model


@pytest.fixture()
def fast_model(request: pytest.FixtureRequest, settings_yaml: Settings) -> Model:
    return settings_yaml.models[settings_yaml.fast_model_name]


@pytest.fixture()
def simple_router(request: pytest.FixtureRequest, settings_yaml: Settings) -> Router:
    router = Router(
        models_pool={settings_yaml.fast_model_name: settings_yaml.models[settings_yaml.fast_model_name]},
        router_pool=[settings_yaml.fast_model_name],
    )
    return router


@pytest.fixture()
def complex_router(request: pytest.FixtureRequest, settings_yaml: Settings) -> Router:
    router = Router(
        models_pool=settings_yaml.models,
        router_pool=[settings_yaml.fast_model_name],
    )
    return router
