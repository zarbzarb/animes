# -*- coding: utf-8 -*-
"""M3 运行时装配：把「应用层」与「Agent 层」接起来。

这是分层架构里**唯一**允许双向引用的地方 —— 它对上被 `server/main.py` 调用，
对下通过 `agents.common.ports.bind_runtime` / `agents.common.data.bind_gateway`
把能力**注入**给 Agent 层，而不是让 Agent 去 import `server`。

装配顺序（缺一不可，顺序也有讲究）
--------------------------------
1. `setup_logging()`      —— 后续日志才带 trace_id
2. `init_db()`            —— 表不存在的话连 Agent 装配日志都写不进去
3. `bind_gateway(gw)`     —— Agent 取数唯一通道
4. `bind_runtime(...)`    —— Agent 取 DB 会话 / 缓存 / 模型注册表
5. `register_all()`       —— 注册 A0–A9 实例
6. `warmup_agents()`      —— **把模型加载从「首次请求」挪到「启动期」**

第 6 步不是优化，是**正确性问题**
--------------------------------
A2/A3 的 model registry 是延迟加载：第一次 `predict` 时才读 checkpoint。
实测首次加载 2.3 s（多兴趣 + 内容编码器）。若不预热，这 2.3 s 会落进
A2 的 200ms / A3 的 150ms 预算里 → 两个 Agent 双双超时降级 →
用户第一次点「推荐」看到的是热门兜底而不是个性化结果（且**不报错**）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from server.core.cache import get_cache
from server.core.config import settings
from server.db.gateway import SqlGateway
from server.db.item_index import load_item_index
from server.db.session import init_db, session_scope

logger = logging.getLogger(__name__)

_state: dict[str, Any] = {"ready": False, "degraded": [], "warmup": {}}


def bootstrap(*, create_tables: bool = True, warmup: bool = True) -> dict:
    """**同步**装配入口（脚本 / 测试用）。在事件循环里请用 `await abootstrap()`。"""
    if _running_loop():
        raise RuntimeError(
            "bootstrap() 是同步入口，不能在事件循环里调用（asyncio.run 会抛 "
            "'cannot be called from a running event loop'）。"
            "FastAPI 的 lifespan 请改调 `await abootstrap()`。")
    return asyncio.run(abootstrap(create_tables=create_tables, warmup=warmup))


async def abootstrap(*, create_tables: bool = True, warmup: bool = True) -> dict:
    """装配运行时（**异步**）。幂等 —— uvicorn `--reload` 下 lifespan 会被多次调用。

    ⚠️ 为什么必须有异步版本：预热要 `await warmup_agents()`，而 lifespan 本身
    就在事件循环里。写成 `asyncio.run(...)` 会直接抛 RuntimeError，
    结果**预热被静默跳过**（只留一条 degraded 标记），
    首次推荐又回到"模型冷加载 2.3s → A2/A3 超时降级"的老问题上。
    """
    from agents.bootstrap import register_all, warmup_agents
    from agents.common.data import bind_gateway
    from agents.common.ports import Runtime, bind_runtime

    if _state["ready"]:
        return dict(_state)

    degraded: list[str] = []

    if create_tables:
        try:
            init_db(with_seed=True)
        except Exception as exc:
            logger.exception("建表失败：%s", exc)
            degraded.append("db_init")

    # 物品索引缺失时 A2/A3 会全线不可用 —— 提前检查并给出可执行的修复命令
    try:
        idx = load_item_index()
        logger.info("物品索引：n_items=%s（源 %s）", idx.get("n_items"),
                    idx.get("source", "item_index.json"))
    except Exception as exc:
        logger.error("物品索引加载失败：%s\n  修复：python scripts/export_item_index.py", exc)
        degraded.append("item_index")

    gw = SqlGateway()

    # 缓存：Redis 不可用时自动退化到进程内 TTL 字典（`core/cache.py` 已实现）。
    # 这里探测一次，把结论记下来给 /health 与前端展示 —— 不因缓存不可用中断启动。
    cache = get_cache()
    try:
        await cache.health()
    except Exception as exc:
        logger.debug("缓存健康探测失败（忽略）：%s", exc)
    backend = cache.backend_name
    if backend != "redis":
        degraded.append("cache")
        logger.warning("缓存后端 = %s（Redis 不可用，已降级为进程内缓存）", backend)

    bind_gateway(gw)
    bind_runtime(Runtime(
        session_scope=session_scope,
        cache=cache,
        model_registry=_model_registry(),
        db_ready=True,
        cache_backend=backend,
    ))

    status = register_all()
    logger.info("Agent 装配：%d/%d 就绪",
                sum(1 for v in status.values() if v == "ok"), len(status))

    warm: dict[str, Any] = {}
    if warmup:
        try:
            warm = await warmup_agents()
        except Exception as exc:
            logger.exception("Agent 预热失败：%s", exc)
            degraded.append("warmup")
    _state.update({"ready": True, "degraded": degraded, "warmup": warm,
                   "cache_backend": backend, "agents": status})
    logger.info("运行时装配完成：degraded=%s warmup=%s", degraded, warm)
    return dict(_state)


def _running_loop() -> bool:
    """当前线程有运行中的事件循环时为 True。"""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _model_registry():
    from agents.common.registry import ModelRegistry
    return ModelRegistry


async def ashutdown() -> None:
    """lifespan 退出：关缓存、释放连接池（异步版）。"""
    await _close()


def shutdown() -> None:
    """同步退出（脚本 / 测试用）。"""
    if _running_loop():
        raise RuntimeError("shutdown() 不能在事件循环里调用，请用 await ashutdown()")
    asyncio.run(_close())


async def _close() -> None:
    from server.db.session import get_engine

    try:
        await get_cache().close()
    except Exception as exc:      # 关闭失败不该阻塞进程退出
        logger.debug("关闭缓存失败（忽略）：%s", exc)
    try:
        get_engine().dispose()
    except Exception as exc:
        logger.debug("释放连接池失败（忽略）：%s", exc)
    _state["ready"] = False


def state() -> dict:
    return dict(_state)


__all__ = ["abootstrap", "ashutdown", "bootstrap", "shutdown", "state"]
