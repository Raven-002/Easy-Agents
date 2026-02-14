import asyncio
import uuid
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from easy_agents.core import RunContext, Tool

from .deps.tools_registry_deps import ToolsRegistryDeps, tools_registry_deps_type


class AgentTaskRequest(BaseModel):
    request: str
    guidelines: str
    expert: str = Field(
        description="The expert to ask questions to. Use the list_experts tool to get a list of available experts."
    )
    reply_format_description: str


class AgentTaskId(BaseModel):
    id: str


class AgentTaskResult(BaseModel):
    result: str | None
    task_id: AgentTaskId | Literal["current_task"]
    error: str | None = None


class AgentTaskIdList(BaseModel):
    ids: list[AgentTaskId]


class AgentTaskResultList(BaseModel):
    results: list[AgentTaskResult]


class AgentTask:
    def __init__(self, request: AgentTaskRequest) -> None:
        self.request = request

    async def perform_task(self) -> str:
        raise NotImplementedError


@dataclass
class TasksRegistry:
    tasks: dict[str, tuple[AgentTask, asyncio.Task[str]]]


class TasksList(BaseModel):
    tasks: dict[str, AgentTask]


async def get_tasks(ctx: RunContext, _parameters: None) -> TasksRegistry:
    tasks: TasksRegistry | None = ctx.deps.deps_map.get("tasks", None)
    if tasks is None:
        ctx.deps.deps_map["tasks"] = TasksRegistry(tasks={})
        tasks = ctx.deps.deps_map["tasks"]
    assert isinstance(tasks, TasksRegistry)
    return tasks


async def list_tasks(ctx: RunContext, _parameters: None) -> TasksList:
    return TasksList(tasks={k: v[0] for k, v in (await get_tasks(ctx, None)).tasks.items()})


async def start_task(ctx: RunContext, parameters: AgentTaskRequest) -> AgentTaskId:
    tasks = await get_tasks(ctx, None)
    task_id = f"task-{uuid.uuid4()}"
    task = AgentTask(request=parameters)
    task_handle = asyncio.create_task(task.perform_task())
    tasks.tasks[task_id] = task, task_handle
    return AgentTaskId(id=task_id)


async def wait_for_task_result(ctx: RunContext, parameters: AgentTaskId) -> AgentTaskResult:
    tasks = await get_tasks(ctx, None)
    _, task_result = tasks.tasks[parameters.id]
    try:
        result = await task_result
        return AgentTaskResult(result=result, task_id=parameters)
    except Exception as e:
        return AgentTaskResult(result=None, task_id=parameters, error=str(e))


async def run_task_now(_ctx: RunContext, parameters: AgentTaskRequest) -> str:
    return await AgentTask(request=parameters).perform_task()


async def wait_for_first_task_result(ctx: RunContext, parameters: AgentTaskIdList) -> AgentTaskResult:
    tasks_registry = await get_tasks(ctx, None)

    pending_tasks: dict[asyncio.Task[str], AgentTaskId] = {
        tasks_registry.tasks[task_id.id][1]: task_id for task_id in parameters.ids if task_id.id in tasks_registry.tasks
    }

    if not pending_tasks:
        raise ValueError("No tasks found")

    done, _ = await asyncio.wait(pending_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

    finished_coro = done.pop()
    task_id_obj = pending_tasks[finished_coro]

    return await wait_for_task_result(ctx, task_id_obj)


async def wait_for_all_tasks_results(ctx: RunContext, parameters: AgentTaskIdList) -> AgentTaskResultList:
    return AgentTaskResultList(results=[await wait_for_task_result(ctx, task_id) for task_id in parameters.ids])


tasks_tools = [
    Tool[None, TasksList, None](
        name="list_tasks",
        description="Get list of running async tasks.",
        run=list_tasks,
        results_type=TasksList,
    ),
    Tool[AgentTaskRequest, AgentTaskId, ToolsRegistryDeps](
        name="start_tasks",
        description="Start a new async task and return its id.",
        run=start_task,
        parameters_type=AgentTaskRequest,
        results_type=AgentTaskId,
        deps_type=tools_registry_deps_type,
    ),
    Tool[AgentTaskId, AgentTaskResult, None](
        name="wait_for_task_result",
        description="Wait for the result of a running async task.",
        run=wait_for_task_result,
        parameters_type=AgentTaskId,
        results_type=AgentTaskResult,
    ),
    Tool[AgentTaskRequest, str, ToolsRegistryDeps](
        name="run_task_now",
        description="Run a task now and return the result.",
        run=run_task_now,
        parameters_type=AgentTaskRequest,
        results_type=str,
        deps_type=tools_registry_deps_type,
    ),
    Tool[AgentTaskIdList, AgentTaskResult, None](
        name="wait_for_first_task_result",
        description="Wait for the result of the first specified async task to finish running.",
        run=wait_for_first_task_result,
        parameters_type=AgentTaskIdList,
        results_type=AgentTaskResult,
    ),
    Tool[AgentTaskIdList, AgentTaskResultList, None](
        name="wait_for_all_tasks_results",
        description="Wait for the results of all specified async tasks.",
        run=wait_for_all_tasks_results,
        parameters_type=AgentTaskIdList,
        results_type=AgentTaskResultList,
    ),
]
