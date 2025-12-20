from .base_ai_runner import AiRunner


class LocalOllamaRunner(AiRunner):
    """Runner pre-configured for local Ollama via LiteLLM."""

    def __init__(self, model: str) -> None:
        # LiteLLM requires 'ollama/' prefix to route to the correct local provider
        formatted_model = f"ollama/{model}" if not model.startswith("ollama/") else model
        super().__init__(model_name=formatted_model, api_base="http://localhost:11434")

    @classmethod
    def create_runner(cls, runner_params) -> "LocalOllamaRunner":
        return cls(**runner_params)
