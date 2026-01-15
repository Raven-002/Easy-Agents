from typing import Any

from pydantic import BaseModel, Field

from .context import AnyChatCompletionMessage, SystemMessage, UserMessage
from .model import Model

type ModelId = str

routing_guidelines: str = """You are an expert at choosing the right model for a task.

# GUIDELINES
-  Prefer available models.
    * Choose an unavailable model only if it is the only good match, and there is no available model that can handle the
      task well enough.
    * If there is a good enough model, choose it and explain why you chose it.
- For complex/critical tasks, prefer the better model, no matter the speed and availability.
- For simple tasks, prefer the faster model.
- For coding tasks, no matter simple or complex, only choose models the are for coding. Do not choose a non coding
  model.
- If a model can do something, but another model is better, consider task complexity, speed requirement, and
  availability to balance the best choice.
"""


class ModelRegistryEntry(BaseModel):
    id: ModelId
    has_thinking: bool
    description: str
    is_available: bool

    @staticmethod
    def from_model(model_id: ModelId, model: Model) -> "ModelRegistryEntry":
        return ModelRegistryEntry(
            id=model_id,
            has_thinking=model.thinking,
            description=model.description,
            is_available=model.is_available(),
        )


class ModelRegistry(BaseModel):
    models: list[ModelRegistryEntry]


class Router(BaseModel):
    models_pool: dict[ModelId, Model]
    router_pool: list[ModelId] = Field(..., description="Priority list of model IDs used to route the task.")

    @staticmethod
    def __validate_models_dictionary(models: dict[ModelId, Model]) -> None:
        if len(models) == 0:
            raise ValueError("Models dictionary must contain at least one model.")
        for model_id in models.keys():
            if model_id == "":
                raise ValueError("Model ID cannot be empty.")

    @staticmethod
    def __validate_router_pool(router_pool: list[ModelId], models: dict[ModelId, Model]) -> None:
        if len(router_pool) == 0:
            raise ValueError("Router pool must contain at least one model ID.")
        if not any([model_id in models for model_id in router_pool]):
            raise ValueError("Router pool must contain at least one valid model ID.")

    def model_post_init(self, _context: Any) -> None:
        self.__validate_models_dictionary(self.models_pool)
        self.__validate_router_pool(self.router_pool, self.models_pool)

    def get_model(self, model_id: ModelId) -> Model:
        return self.models_pool[model_id]

    def get_active_router(self) -> Model:
        for model_id in self.router_pool:
            model = self.get_model(model_id)
            if model.is_available():
                return model
        raise RuntimeError("No available router model found.")

    def get_models_descriptions_and_status(self) -> ModelRegistry:
        entries: list[ModelRegistryEntry] = []
        for model_id, model in self.models_pool.items():
            entries.append(ModelRegistryEntry.from_model(model_id, model))
        if all([entry.is_available is False for entry in entries]):
            raise RuntimeError("All models are unavailable.")
        return ModelRegistry(models=entries)

    def route_task(self, task_description: str) -> Model:
        # TODO: Support no good model, identifying unavailable chosen model.
        router_model = self.get_active_router()
        models_registry = self.get_models_descriptions_and_status()
        messages: list[AnyChatCompletionMessage] = [
            SystemMessage(content=routing_guidelines),
            UserMessage(content=f"Model Pool:\n{models_registry}\n\n Task Description:\n{task_description}"),
        ]

        class ModelChoice(BaseModel):
            model_id: str = Field(..., description="ID of the chosen model.")

        chosen_model_response = router_model.chat_completion(messages, response_format=ModelChoice)
        router_model = self.get_model(chosen_model_response.message.content.model_id)
        return router_model
