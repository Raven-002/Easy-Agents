import json
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from easy_agents.core import (
    AnyChatCompletionMessage,
    AssistantMessage,
    Context,
    ContextRefiner,
    ContextRefinerFactory,
    MessagesEndsWithAssistantMessage,
    Model,
    Router,
    UserMessage,
)

memory_compression_system_prompt = """You are an expert summarizer.
Your task is to summerize messages in the context, without losing important information.

# Guidelines:
- Use short sentences, that preserve all the information, without redundant grammatic fill-ins.
- Do not try to answer any of the questions asked by either the user or the assistant.
- Do not change any information.
- Do not invent new information.
- If you come across a summary in the history, merge it into the new summary without losing any info.
- Do not invent any plans. If the assistant did not plan something, do not make it up. Only keep plans made by the
  assistant.
- Data returned from tools should be an artifact rather than a summarized fact.
- If Tool data is very large, summarize it and mention how to extract more precise data.
- Do not include a summary of system messages, they are handled externally.
"""


class AssistantReply(BaseModel):
    """A final reply made by the assistant."""

    reply_stage: Literal["Final"] = "Final"
    reply: str = Field(description="The reply of the assistant.")


class AssistantReplyProgress(BaseModel):
    """An in-progress reply made by the assistant."""

    reply_stage: Literal["InProgress"] = "InProgress"
    partial_reply: str = Field(
        description="The partial reply of the assistant. Containing only what should stay in the final reply. Does "
        "not include the assistant's progress."
    )
    progress: str = Field(description="The progress of the assistant.")


class UserRequest(BaseModel):
    """A request made by the user that the assistant should reply to, with the given reply/progress."""

    request: str = Field(description="The user request.")
    reply: AssistantReplyProgress | AssistantReply = Field(description="The assistant reply/progress.")


class TechnicalArtifact(BaseModel):
    name: str = Field(description="The name of the file, class, or symbol.")
    type: str = Field(description="e.g., 'Source Code', 'Logs', 'API Spec'")
    summary: str = Field(description="High-level logic/purpose.")
    key_signatures: list[str] = Field(description="Actual text/code snippets/signatures.")
    location_pointer: str = Field(description="How to find the full data again (e.g., 'lines 500-1200 in file.py')")


class SummerizedContext(BaseModel):
    """A summerization of the context."""

    user_requests: list[UserRequest] = Field(
        description="Requests from the user that needs to be addressed by the assistant."
    )
    facts: list[str] = Field(
        description="Facts from the summerized context. Facts can be tool results, or other "
        "acknowledged facts by the user, system, or assistant."
    )
    technical_artifacts: list[TechnicalArtifact] = Field(
        description="Technical artifacts from the summerized context. Names, snippets, etc... They are all factually "
        "correct."
    )
    ideas: list[str] = Field(
        description="Ideas from the summerized context. Things that are not a fact, but still require remembering."
    )
    assistant_plans: list[str] = Field(
        description="Plans from the summerized context. Things that the assistant planned to do, and needs to remember."
    )

    def to_messages(self) -> list[AnyChatCompletionMessage]:
        messages: list[AnyChatCompletionMessage] = []

        if self.user_requests:
            user_requests: list[str] = [r.request for r in self.user_requests]
            user_requests_with_progress: list[str] = [r.model_dump_json() for r in self.user_requests]
            messages.append(
                UserMessage(
                    content=f"I need you to: {json.dumps(user_requests)}",
                )
            )
            messages.append(
                AssistantMessage(
                    content="Current Objective from the user & History of my progress: "
                    f"{json.dumps(user_requests_with_progress)}",
                )
            )

        knowledge_blob = {"verified_facts": self.facts, "current_ideas": self.ideas}
        messages.append(
            AssistantMessage(
                content=f"My knowledge base is: {json.dumps(knowledge_blob)}",
            )
        )

        if self.technical_artifacts:
            artifacts_content = "\n".join(
                [
                    f"Artifact: {a.name} ({a.type})\n"
                    f"Summary: {a.summary}\n"
                    f"Location: {a.location_pointer}\n"
                    f"Signatures: {a.key_signatures}"
                    for a in self.technical_artifacts
                ]
            )
            messages.append(UserMessage(content=f"Technical Reference:\n{artifacts_content}", name="tech_library"))

        if self.assistant_plans:
            messages.append(
                AssistantMessage(
                    content=f"I need to: {json.dumps(self.assistant_plans)}",
                )
            )

        return messages


class MemoryCompressionRefiner(ContextRefiner):
    def __init__(self) -> None:
        self._model: Model | None = None

    async def _get_model(self, router: Router) -> Model:
        if not self._model:
            self._model = await router.route_task("Summerize context.")
        return self._model

    async def refine_new_messages(
        self,
        router: Router,
        raw_messages: Sequence[AnyChatCompletionMessage],
        refined_messages: Sequence[AnyChatCompletionMessage],
    ) -> Sequence[AnyChatCompletionMessage]:
        last_assistant_message_idx = -1
        for i in range(len(refined_messages) - 1, -1, -1):
            if refined_messages[i].role == "assistant":
                last_assistant_message_idx = i
                break
        if last_assistant_message_idx == -1:
            raise ValueError("No assistant message found in the refined messages list.")
        messages_to_summarize = refined_messages[:last_assistant_message_idx]
        messages_to_keep = refined_messages[last_assistant_message_idx:]

        summary_request_context = Context.simple(
            f"Summarize the following messages: [{', '.join([m.model_dump_json() for m in messages_to_summarize])}].",
            system_prompt=memory_compression_system_prompt,
        )
        summerized_context_reply = await (await self._get_model(router)).chat_completion(
            summary_request_context.messages, response_format=SummerizedContext
        )
        print(summerized_context_reply.message.content)
        messages: list[AnyChatCompletionMessage] = []
        messages.extend([m for m in refined_messages if m.role == "system"])
        messages.extend(summerized_context_reply.message.content.to_messages())
        messages.extend(messages_to_keep)
        return messages

    async def refine_pre_tool_assistant_message(
        self,
        router: Router,
        raw_messages: MessagesEndsWithAssistantMessage[str],
        refined_messages: MessagesEndsWithAssistantMessage[str],
    ) -> AssistantMessage[str]:
        """Do not modify anything before the tool call."""
        return refined_messages[-1]


memory_refiner_factory = ContextRefinerFactory[MemoryCompressionRefiner](MemoryCompressionRefiner)
