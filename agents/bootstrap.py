# -*- coding: utf-8 -*-
"""把 A0–A9 装配进 `AgentRegistry`（应用层启动时调一次）。

为什么不做成 `agents/__init__.py` 的 import 副作用
-------------------------------------------------
`agents/` 的很多模块会被单独 import（例如 `agents.common.envelope` 的单测）。
若注册依赖 import 副作用，那么"只想要信封工具"的调用方会被迫拖起十个 Agent 包
（含 `models/retrieval` 与 torch）。显式 `register_all()` 把"何时装配"交回调用方，
测试也可以先 `AgentRegistry.clear()` 再只注册需要的子集。
"""

from __future__ import annotations

import importlib
import logging

from agents.common.registry import AgentRegistry

logger = logging.getLogger(__name__)

__all__ = ["AGENT_CLASSES", "WARMUP_AGENTS", "register_all", "warmup_agents"]

# 需要启动期预热的 Agent（其余要么无模型，要么是离线任务）
# A2 要加载 torch 权重、A3 要触碰 32MB 内容向量与索引 —— 不预热的话
# 它们的**第一次**调用会把 1~2s 的加载时间算进 150/200ms 预算而被降级。
WARMUP_AGENTS: tuple[str, ...] = ("A2", "A3")

# (agent_id, "模块:类名") —— 按 agent_id 升序，便于日志与 /admin/agents/health 对照
AGENT_CLASSES: tuple[tuple[str, str], ...] = (
    ("A0", "agents.orchestrator:OrchestratorAgent"),
    ("A1", "agents.profile:ProfileAgent"),
    ("A2", "agents.recall:RecallAgent"),
    ("A3", "agents.coldstart:ColdStartAgent"),
    ("A4", "agents.fusion:FusionRankAgent"),
    ("A5", "agents.explain:ExplainAgent"),
    ("A6", "agents.drift:DriftAgent"),
    ("A7", "agents.chat:ChatAgent"),
    ("A8", "agents.dataops:DataOpsAgent"),
    ("A9", "agents.eval:EvalAgent"),
)


def register_all(*, only: tuple[str, ...] = (), strict: bool = True) -> dict[str, str]:
    """注册全部（或指定的）Agent，返回 `{agent_id: 状态}`。

    `strict=True` 时任一 Agent 导入失败即抛 —— 生产启动期就该炸，
    而不是让 `/recommend` 在第一次请求时才发现"A2 没装上"。
    测试可用 `strict=False` 容忍可选依赖缺失。
    """
    wanted = {a.upper() for a in only} if only else None
    status: dict[str, str] = {}
    for agent_id, path in AGENT_CLASSES:
        if wanted is not None and agent_id not in wanted:
            continue
        module_name, _, cls_name = path.partition(":")
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name)
            AgentRegistry.register(cls)
            status[agent_id] = "ok"
        except Exception as exc:      # noqa: BLE001 —— 启动装配需要看到全部失败
            status[agent_id] = f"failed: {type(exc).__name__}: {exc}"
            logger.exception("注册 %s（%s）失败", agent_id, path)
            if strict:
                raise
    ok = sum(1 for v in status.values() if v == "ok")
    logger.info("Agent 装配完成：%d/%d 成功", ok, len(status))
    return status


async def warmup_agents(*, only: tuple[str, ...] = WARMUP_AGENTS) -> dict[str, str]:
    """启动期预热（FastAPI lifespan 里 `await warmup_agents()`）。

    **为什么必须预热**：协议给 A2/A3 的预算是 200ms / 150ms，这个数字
    隐含"模型已加载、算子已初始化"的前提。冷启动的加载成本（实测 1~2.3s）
    若算进第一次请求，结果不是"慢一点"，而是**直接被超时降级**：
    A2 退热门榜、A3 返空候选，用户看到"第一次推荐没有新番，刷新就好了"。
    这类 bug 只在每个进程的第一次请求出现，最难复现也最难被单测抓。

    `strict=False` 的语义：返回 `{agent_id: "ok" | "skip:..." | "failed:..."}`，
    **不抛** —— 预热失败不该阻止服务启动（运行时还有降级链兜底）。
    """
    status: dict[str, str] = {}
    for agent_id in only:
        try:
            agent = AgentRegistry.get(agent_id)
        except Exception as exc:            # 未注册（比如只注册了子集）
            status[agent_id] = f"not_registered: {exc}"
            continue
        fn = getattr(agent, "warmup", None)
        if fn is None:
            status[agent_id] = "skip:no_warmup"
            continue
        try:
            ok = bool(await fn())
            status[agent_id] = "ok" if ok else "failed:returned_false"
        except Exception as exc:            # noqa: BLE001 —— 预热失败不阻断启动
            status[agent_id] = f"failed: {type(exc).__name__}: {exc}"
            logger.exception("%s 预热失败", agent_id)
    logger.info("Agent 预热完成：%s", status)
    return status
