#!/usr/bin/env python

import pytest

from easy_agents.core.model import Model
from easy_agents.core.router import ModelId, Router

available_model_name = "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_K_XL"  # Also used for choosing
unavailable_model_name = "qwen-coder-480-abc123"

models_pool: dict[ModelId, Model] = {
    "general-coder": Model(
        model_provider="ollama_chat",
        model_name=available_model_name,
        description="A coding model, run at moderate speeds, only good at writing code, and is average+ in all "
        "programming languages, enough for most tasks. Knows almost only english.",
    ),
    "c-coder": Model(
        model_provider="ollama_chat",
        model_name=available_model_name,
        description="A coding model, run at moderate speeds, Especially good at writing C code.",
    ),
    "python-coder": Model(
        model_provider="ollama_chat",
        model_name=available_model_name,
        description="A coding model, run at moderate speeds, Especially good at writing python code.",
    ),
    "qwen3-coder-30B-A3B": Model(
        model_provider="ollama_chat",
        model_name=available_model_name,
        description="A coding model, run at moderate speeds, good executing well defined tasks.",
    ),
    "qwen3:14b": Model(
        model_provider="ollama_chat",
        model_name=available_model_name,
        description="A model with moderate/slow speed, have hard time with a lot of tools.",
        thinking=True,
    ),
    "glm-z1-9b": Model(
        model_provider="ollama_chat",
        model_name=available_model_name,
        description="A fast model good for deep thinking and analysis",
        thinking=True,
    ),
    "rust-coder": Model(
        model_provider="ollama_chat",
        model_name=unavailable_model_name,
        description="A coding model, run at moderate speeds, Especially good at writing rust code.",
    ),
    "qwen-coder-480": Model(
        model_provider="ollama_chat",
        model_name=unavailable_model_name,
        description="One of the best models for coding tasks. Use this for very complex tasks. The only model that "
        "can handle complex tasks with no bugs and no failures.",
    ),
    "qwen4-multi-language": Model(
        model_provider="ollama_chat",
        model_name=unavailable_model_name,
        description="The only model that knows more than just english. It has deep knowledge in every human language "
        "there is, and is the only valid choice for tasks in foreign languages.",
    ),
}


@pytest.fixture(scope="module")
async def router(request: pytest.FixtureRequest) -> Router:
    router = Router(
        models_pool=models_pool,
        router_pool=["qwen3-coder-30B-A3B"],
    )
    if not await router.models_pool[router.router_pool[0]].is_available():
        pytest.skip("Skipping router because it is not available.")
    return router


@pytest.mark.asyncio
async def test_route_task_python_code(router: Router) -> None:
    model = await router.route_task("Write a python function that returns the sum of two numbers.")
    assert model == models_pool["python-coder"]


@pytest.mark.asyncio
async def test_route_task_general_code(router: Router) -> None:
    model = await router.route_task("Write a function that returns the sum of two numbers.")
    assert model == models_pool["general-coder"]


@pytest.mark.asyncio
async def test_route_task_rust_code(router: Router) -> None:
    model = await router.route_task("Write a simple rust function that returns the sum of two numbers.")
    assert model == models_pool["general-coder"] or model == models_pool["c-coder"]


@pytest.mark.asyncio
async def test_route_task_complex_code(router: Router) -> None:
    model = await router.route_task(
        "Write a very complex function based on the user input. The function is very complex, "
        "and is not allowed to fail or have bugs."
    )
    assert model == models_pool["qwen-coder-480"]


@pytest.mark.asyncio
async def test_route_task_general(router: Router) -> None:
    model = await router.route_task("A fast pace call center core in english.")
    assert (
        model == models_pool["glm-z1-9b"]
        or model == models_pool["qwen3-coder-30B-A3B"]
        or model == models_pool["general-coder"]
    )


@pytest.mark.asyncio
async def test_route_task_multi_language(router: Router) -> None:
    model = await router.route_task("A call center core in hebrew and spanish.")
    assert model == models_pool["qwen4-multi-language"]
