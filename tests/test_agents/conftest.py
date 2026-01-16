import pytest

from easy_agents.core.router import Router
from tests.test_core.support.helper_routers import complex_models_pool, simple_models_pool


@pytest.fixture(scope="module")
def simple_router(request: pytest.FixtureRequest) -> Router:
    return Router(
        models_pool=simple_models_pool,
        router_pool=["qwen3-instruct"],
    )


@pytest.fixture(scope="module")
def complex_router(request: pytest.FixtureRequest) -> Router:
    return Router(
        models_pool=complex_models_pool,
        router_pool=["qwen3-instruct"],
    )
