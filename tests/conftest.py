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


def load_settings() -> Settings:
    models_path = script_dir / "models.yaml"
    try:
        with open(models_path) as f:
            return Settings(**yaml.safe_load(f))
    except FileNotFoundError:
        pytest.skip("Skipping models.yaml because it does not exist.")


@pytest.fixture(scope="session")
def settings_yaml(request: pytest.FixtureRequest) -> Settings:
    return load_settings()


@pytest.fixture(scope="session")
def all_models(request: pytest.FixtureRequest, settings_yaml: Settings) -> list[Model]:
    return list(settings_yaml.models.values())


def get_test_models() -> Iterator[Model]:
    yield from load_settings().models.values()


@pytest.fixture(params=get_test_models(), scope="module")
def model(request: pytest.FixtureRequest, settings_yaml: Settings) -> Model:
    requested_model: Model = request.param
    assert isinstance(requested_model, Model)
    return requested_model


@pytest.fixture(scope="session")
def fast_model(request: pytest.FixtureRequest, settings_yaml: Settings) -> Model:
    return settings_yaml.models[settings_yaml.fast_model_name]


@pytest.fixture(scope="session")
def simple_router(request: pytest.FixtureRequest, settings_yaml: Settings) -> Router:
    router = Router(
        models_pool={settings_yaml.fast_model_name: settings_yaml.models[settings_yaml.fast_model_name]},
        router_pool=[settings_yaml.fast_model_name],
    )
    return router


@pytest.fixture(scope="session")
def complex_router(request: pytest.FixtureRequest, settings_yaml: Settings) -> Router:
    router = Router(
        models_pool=settings_yaml.models,
        router_pool=[settings_yaml.fast_model_name],
    )
    return router
