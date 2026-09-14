# -*- coding: utf-8 -*-
"""工具注册与 Function Calling Schema 生成（`docs/agent-prompt-design.md` 的工具规范）。

为什么工具要走注册表而不是在 A7 里写 if-else
--------------------------------------------
A7 对话推荐 Agent 需要"查番剧、看用户历史、按题材筛"这些工具。
如果直接在 prompt 里塞描述、在代码里 if 工具名，会有两个后果：
① prompt 与实现两份描述，改一处忘一处（LLM 会调一个不存在的参数）；
② 工具被 LLM 反复调用，没有统一的次数上限（协议 `60502 TOOL_CALL_LIMIT`）。

所以：**一处声明 → 同时产出 prompt 用的 JSON Schema 与执行用的函数**，
并在执行入口统一计数与限流。

⚠️ 工具**只做数据准备与检索**，不做打分排序。排序是算法层的事（架构 ADR-1）。
"""

from __future__ import annotations

import inspect
import logging
import typing
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from agents.common.errors import AgentError, make

logger = logging.getLogger(__name__)

# 协议 §4.2：`60502 TOOL_CALL_LIMIT`「工具调用超上限（A7 > 3 次）→ 强制收敛输出」
DEFAULT_MAX_TOOL_CALLS = 3

_JSON_TYPES = {int: "integer", float: "number", str: "string", bool: "boolean",
               list: "array", dict: "object"}


def json_type(annotation: Any) -> str:
    """注解 → JSON Schema 类型名。

    ⚠️ 这里踩过一个**静默产错**的坑：最早直接用 `inspect.signature` 拿到的
    `param.annotation`。但本仓库的模块普遍写了 `from __future__ import annotations`，
    此时注解是**字符串**（`"int"` 而不是 `int`），`_JSON_TYPES.get("int", "string")`
    命中兜底 —— 于是 LLM 看到的 schema 里所有参数都是 `string`，
    工具调用传 `"5"` 而不是 `5`，本该在入参处炸掉的问题被推迟到函数体里。
    所以：先 `get_type_hints` 解析真实类型，再对 `Optional[X]` / `list[X]` 做拆解。
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return "string"
    if isinstance(annotation, str):
        # get_type_hints 失败时的兜底：把字符串按名字映射一次
        return _JSON_TYPES.get({"int": int, "float": float, "str": str,
                                "bool": bool, "list": list, "dict": dict}
                               .get(annotation, str), "string")
    origin = typing.get_origin(annotation)
    if origin is typing.Union:                      # Optional[X] == Union[X, None]
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return json_type(args[0]) if args else "string"
    if origin in (list, tuple, set):
        return "array"
    if origin is dict:
        return "object"
    if isinstance(annotation, type):
        return _JSON_TYPES.get(annotation, "string")
    return "string"


@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict = field(default_factory=dict)
    required: tuple[str, ...] = ()

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.required),
                },
            },
        }


_tools: dict[str, ToolSpec] = {}


def tool(name: Optional[str] = None, *, description: str = "",
         required: Optional[tuple[str, ...]] = None) -> Callable:
    """把函数注册成工具。参数类型从签名注解自动推导，避免手写 schema 漂移。

    用法::

        @tool(description="按标题检索番剧")
        def search_anime(keyword: str, limit: int = 5) -> list[dict]: ...
    """

    def deco(func: Callable) -> Callable:
        tool_name = name or func.__name__
        sig = inspect.signature(func)
        # 用 get_type_hints 解析真实注解（见 `json_type` 的说明）
        try:
            hints = typing.get_type_hints(func)
        except Exception:               # 前向引用未解析等 → 退回原始注解
            hints = {}
        params: dict[str, dict] = {}
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            ann = hints.get(pname, param.annotation)
            params[pname] = {"type": json_type(ann)}
            if param.default is inspect.Parameter.empty:
                params[pname]["description"] = ""
        req = required if required is not None else tuple(
            p for p, prm in sig.parameters.items()
            if prm.default is inspect.Parameter.empty and p not in ("self", "cls"))
        _tools[tool_name] = ToolSpec(name=tool_name,
                                     description=description or (func.__doc__ or ""),
                                     func=func, parameters=params, required=req)
        return func

    return deco


def get_tools(names: Optional[list[str]] = None) -> list[ToolSpec]:
    if names is None:
        return list(_tools.values())
    missing = [n for n in names if n not in _tools]
    if missing:
        raise AgentError(make("RESOURCE_NOT_FOUND").code,
                         f"未注册的工具：{missing}",
                         detail={"registered": sorted(_tools)})
    return [_tools[n] for n in names]


def openai_tools(names: Optional[list[str]] = None) -> list[dict]:
    """给 LLM 的 `tools` 参数。"""
    return [t.to_openai_schema() for t in get_tools(names)]


class ToolRunner:
    """一次会话内的工具执行器：统一计数、限流、异常包装。"""

    def __init__(self, *, max_calls: int = DEFAULT_MAX_TOOL_CALLS,
                 allow: Optional[list[str]] = None) -> None:
        self.max_calls = max_calls
        self.allow = allow
        self.calls: list[str] = []

    async def call(self, name: str, arguments: dict) -> Any:
        if self.allow is not None and name not in self.allow:
            raise AgentError(make("FORBIDDEN").code, f"本次会话不允许调用 {name}")
        if len(self.calls) >= self.max_calls:
            # 协议要求"强制收敛输出"，所以这里抛的是可识别的错误码，
            # A7 捕获后立刻让 LLM 基于已有信息作答，而不是继续调工具
            raise AgentError(make("TOOL_CALL_LIMIT").code,
                             f"工具调用已达上限 {self.max_calls} 次，请基于已有信息作答",
                             detail={"calls": list(self.calls)})
        spec = _tools.get(name)
        if spec is None:
            raise AgentError(make("RESOURCE_NOT_FOUND").code, f"未注册的工具：{name}")
        self.calls.append(name)
        try:
            result = spec.func(**arguments)
            if inspect.isawaitable(result):
                result = await result
            return result
        except AgentError:
            raise
        except Exception as exc:
            logger.warning("工具 %s 执行失败：%s", name, exc)
            raise AgentError(make("INTERNAL_ERROR").code,
                             f"工具 {name} 执行失败：{type(exc).__name__}",
                             detail={"tool": name}) from exc

    async def call_llm_tool_call(self, tool_call: Any) -> tuple[str, str]:
        """执行 OpenAI 风格的 tool_call，返回 `(name, json 结果字符串)`。"""
        import json
        name = getattr(getattr(tool_call, "function", None), "name", None) \
            or (tool_call.get("function", {}).get("name") if isinstance(tool_call, dict) else "")
        raw_args = getattr(getattr(tool_call, "function", None), "arguments", None) \
            or (tool_call.get("function", {}).get("arguments") if isinstance(tool_call, dict) else "{}")
        try:
            args = json.loads(raw_args or "{}")
        except ValueError:
            args = {}
        result = await self.call(name, args)
        return name, json.dumps(result, ensure_ascii=False, default=str)


def clear_tools() -> None:
    _tools.clear()


__all__ = [
    "DEFAULT_MAX_TOOL_CALLS",
    "ToolRunner",
    "ToolSpec",
    "clear_tools",
    "get_tools",
    "json_type",
    "openai_tools",
    "tool",
]
