from easy_agents.core.model import Model
from easy_agents.core.router import ModelId

# For speed reason, use a single model
simple_models_pool: dict[ModelId, Model] = {
    "qwen3-coder": Model(
        model_provider="ollama_chat",
        model_name="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_K_XL",
        description="A coding model, run at moderate speeds, good at writing code and general tasks.",
    ),
}

complex_models_pool: dict[ModelId, Model] = {
    "qwen3-coder": Model(
        model_provider="ollama_chat",
        model_name="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_K_XL",
        description="A coding model, run at moderate speeds, good at writing code and general tasks.",
    ),
    "glm-z1-9b": Model(
        model_provider="ollama_chat",
        model_name="glm-z1-9b",
        description="A fast model good for deep thinking and analysis.",
        thinking=True,
    ),
}
