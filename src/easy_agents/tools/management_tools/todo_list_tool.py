from pydantic import BaseModel

from easy_agents.core import RunContext, Tool


class TodoEntry(BaseModel):
    is_done: bool
    description: str


class TodoEntryId(BaseModel):
    id: int


class TodoList(BaseModel):
    todo_items: dict[TodoEntryId, TodoEntry]


async def get_todo_list(ctx: RunContext, _parameters: None) -> TodoList:
    todo_list: TodoList | None = ctx.deps.deps_map.get("todo_list", None)
    if todo_list is None:
        ctx.deps.deps_map["todo_list"] = TodoList(todo_items={})
        todo_list = ctx.deps.deps_map["todo_list"]
    assert isinstance(todo_list, TodoList)
    return todo_list


async def clear_todo_list(ctx: RunContext, _parameters: None) -> None:
    ctx.deps.deps_map.pop("todo_list", None)


async def set_todo_list(ctx: RunContext, parameters: TodoList) -> None:
    ctx.deps.deps_map["todo_list"] = parameters


async def todo_list_mark_done(ctx: RunContext, parameters: TodoEntryId) -> None:
    todo_list = await get_todo_list(ctx, None)
    todo_list.todo_items[parameters].is_done = True


async def todo_list_mark_undone(ctx: RunContext, parameters: TodoEntryId) -> None:
    todo_list = await get_todo_list(ctx, None)
    todo_list.todo_items[parameters].is_done = False


todo_list_tools = [
    Tool[None, TodoList, None](
        name="get_todo_list",
        description="Get the existing todo list.",
        run=get_todo_list,
        results_type=TodoList,
    ),
    Tool[None, None, None](
        name="clear_todo_list",
        description="Clear the todo list.",
        run=clear_todo_list,
    ),
    Tool[TodoList, None, None](
        name="set_todo_list",
        description="Set the todo list.",
        run=set_todo_list,
        parameters_type=TodoList,
    ),
    Tool[TodoEntryId, None, None](
        name="todo_list_mark_done",
        description="Set a todo list entry as done.",
        run=todo_list_mark_done,
        parameters_type=TodoEntryId,
    ),
    Tool[TodoEntryId, None, None](
        name="todo_list_mark_undone",
        description="Set a todo list entry as undone.",
        run=todo_list_mark_undone,
        parameters_type=TodoEntryId,
    ),
]
