# -*- coding: utf-8 -*-
"""Agent 注册中心与模型注册中心（架构 §5.1 的「模型怎么注册到 Agent」）。

两个注册表，动机不同：

**Agent 注册表** —— A0 调度 Agent 需要"按名字调用另一个 Agent"。
它不能 `from agents.recall.agent import RecallAgent`：那会让编排层硬编码
每个 Agent 的类路径，加减一个 Agent 就要改编排代码。

**模型注册表** —— 关键在**延迟加载**。`torch.load` 一个 SASRec 权重要几百毫秒
并占显存；如果 Agent 模块在 import 期就把模型拉起来，那么
① 单元测试跑不动，② `/health` 也会连带变慢，③ 离线 Agent（A6/A9）
根本不需要模型却被迫加载。
所以注册的是**工厂函数**，`get_model("sasrec")` 第一次调用时才真正加载，
失败则抛 `MODEL_UNAVAILABLE` 让上层降级到 ItemCF / 热门榜。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agents.common.base import BaseAgent
from agents.common.errors import AgentError, make

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- Agent 注册
class AgentRegistry:
    _lock = threading.Lock()
    _agents: dict[str, BaseAgent] = {}

    @classmethod
    def register(cls, agent: BaseAgent | type[BaseAgent]) -> BaseAgent:
        inst = agent() if isinstance(agent, type) else agent
        if not isinstance(inst, BaseAgent):
            raise TypeError(f"{agent!r} 不是 BaseAgent 的子类/实例")
        with cls._lock:
            cls._agents[inst.agent_id] = inst
            cls._agents.setdefault(inst.name, inst)
        return inst

    @classmethod
    def get(cls, key: str) -> BaseAgent:
        inst = cls._agents.get(key) or cls._agents.get(key.upper())
        if inst is None:
            raise AgentError(make("RESOURCE_NOT_FOUND").code,
                             f"未注册的 Agent：{key}",
                             detail={"registered": sorted(set(
                                 a.agent_id for a in cls._agents.values()))})
        return inst

    @classmethod
    def has(cls, key: str) -> bool:
        return key in cls._agents or key.upper() in cls._agents

    @classmethod
    def all(cls) -> list[BaseAgent]:
        seen: dict[int, BaseAgent] = {}
        for a in cls._agents.values():
            seen[id(a)] = a
        return sorted(seen.values(), key=lambda a: a.agent_id)

    @classmethod
    def health(cls) -> list[dict]:
        return [a.health_snapshot() for a in cls.all()]

    @classmethod
    def clear(cls) -> None:
        """测试用。"""
        with cls._lock:
            cls._agents.clear()


# ---------------------------------------------------------------- 模型注册
@dataclass
class ModelSpec:
    """模型的"说明书"。Agent 只认这些字段，不认具体实现。"""

    name: str
    factory: Callable[[], Any]
    version: str = "v1"
    kind: str = "torch"                 # torch / numpy / none
    description: str = ""
    loaded: bool = False
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)


class ModelRegistry:
    _lock = threading.Lock()
    _specs: dict[str, ModelSpec] = {}
    _instances: dict[str, Any] = {}

    @classmethod
    def register(cls, spec: ModelSpec) -> None:
        with cls._lock:
            cls._specs[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> Any:
        """取模型实例（首次调用才真正加载）。失败抛 `MODEL_UNAVAILABLE`。"""
        with cls._lock:
            if name in cls._instances:
                return cls._instances[name]
            spec = cls._specs.get(name)
        if spec is None:
            raise AgentError(make("MODEL_UNAVAILABLE").code,
                             f"未注册的模型：{name}",
                             detail={"registered": sorted(cls._specs)})
        try:
            inst = spec.factory()
        except Exception as exc:
            spec.error = f"{type(exc).__name__}: {exc}"
            logger.warning("模型 %s 加载失败：%s", name, spec.error)
            raise AgentError(make("MODEL_UNAVAILABLE").code,
                             f"模型 {name} 加载失败：{spec.error}",
                             detail={"model": name}) from exc
        with cls._lock:
            cls._instances[name] = inst
            spec.loaded = True
            spec.error = None
        logger.info("模型 %s 已加载（version=%s）", name, spec.version)
        return inst

    @classmethod
    def available(cls, name: str) -> bool:
        return name in cls._instances or name in cls._specs

    @classmethod
    def loaded(cls) -> dict[str, bool]:
        with cls._lock:
            return {n: (n in cls._instances) for n in cls._specs}

    @classmethod
    def describe(cls) -> list[dict]:
        return [{"name": s.name, "version": s.version, "kind": s.kind,
                 "description": s.description, "loaded": s.name in cls._instances,
                 "error": s.error, **s.meta}
                for s in cls._specs.values()]

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._specs.clear()
            cls._instances.clear()


registry = AgentRegistry
models = ModelRegistry

__all__ = ["AgentRegistry", "ModelRegistry", "ModelSpec", "models", "registry"]
