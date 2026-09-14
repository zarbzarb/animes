# -*- coding: utf-8 -*-
"""`trace_id` 透传 + 调用链落库（`agent_trace`）+ Agent 健康度累计。

落库策略：**尽力而为，绝不阻塞主链路**。
`agent_trace` 是全系统写入最频繁的表（每次推荐 4–6 行）。如果它的写入
失败了（库挂了、表锁了），正确答案是「丢掉这条 trace」，而不是让用户的
推荐请求 500 —— 观测是为了服务业务，不能反过来。

所以在两条路径上同时记录：
1. **DB**（`agent_trace` / `agent_state`）—— 持久化，管理后台与论文用；
2. **进程内环形缓冲**（默认 200 条）—— 库不可用时 `/admin/traces/{id}`
   仍然能查到最近几次调用，答辩现场不会因为 MySQL 没起而无法展示调用链。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

from agents.common.errors import AgentError
from agents.common.ports import get_runtime

logger = logging.getLogger(__name__)

RING_SIZE = 200
MAX_CALL_PATH = 8


@dataclass
class Span:
    """一次 Agent 调用（= 一行 agent_trace）。"""

    trace_id: str
    msg_id: str
    parent_msg_id: Optional[str]
    from_agent: str
    to_agent: str
    action: str
    priority: str = "P0"
    status: int = 0                     # 0=success 1=degraded 2=error 3=timeout
    error_code: Optional[int] = None
    elapsed_ms: int = 0
    cache_hit: bool = False
    tokens_used: int = 0
    call_path: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id, "msg_id": self.msg_id,
            "parent_msg_id": self.parent_msg_id, "from_agent": self.from_agent,
            "to_agent": self.to_agent, "action": self.action,
            "priority": self.priority, "status": self.status,
            "error_code": self.error_code, "elapsed_ms": self.elapsed_ms,
            "cache_hit": self.cache_hit, "tokens_used": self.tokens_used,
            "call_path": self.call_path, "detail": self.detail,
            "created_at": self.created_at,
        }


class TraceStore:
    def __init__(self, ring_size: int = RING_SIZE) -> None:
        self._ring: Deque[Span] = deque(maxlen=ring_size)
        self._lock = threading.Lock()
        # Agent 健康度累计（进程内；持久化版本在 agent_state 表）
        self._stats: dict[str, dict[str, int]] = {}

    # ---------- 内存 ----------
    def add(self, span: Span) -> None:
        with self._lock:
            self._ring.append(span)

    def get(self, trace_id: str) -> list[Span]:
        with self._lock:
            return [s for s in self._ring if s.trace_id == trace_id]

    def recent(self, limit: int = 20) -> list[Span]:
        with self._lock:
            return list(self._ring)[-limit:]

    def stats(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {k: dict(v) for k, v in self._stats.items()}

    def bump(self, agent_id: str, *, ok: bool, degraded: bool, timeout: bool,
             elapsed_ms: int) -> None:
        with self._lock:
            s = self._stats.setdefault(agent_id, {
                "success": 0, "fail": 0, "timeout": 0, "degrade": 0,
                "total_ms": 0, "calls": 0})
            s["calls"] += 1
            s["total_ms"] += max(0, int(elapsed_ms))
            if timeout:
                s["timeout"] += 1
                s["fail"] += 1
            elif degraded:
                s["degrade"] += 1
                s["success"] += 1
            elif ok:
                s["success"] += 1
            else:
                s["fail"] += 1

    def clear(self) -> None:
        with self._lock:
            self._ring.clear()
            self._stats.clear()


_store = TraceStore()


def get_store() -> TraceStore:
    return _store


# ---------------------------------------------------------------- 落库


def persist_span(span: Span, *, agent_name: str = "") -> bool:
    """把 span 交给应用层注入的落库出口。失败只记 warning。

    ⚠️ 这里**不能** import `server.*` 的 ORM 类 —— 依赖方向不允许
    （`agents/` → `server/` 是反向依赖）。`agent_trace` 怎么写是应用层的知识，
    由 `Runtime.trace_sink` 注入一个闭包，本模块只负责"把 span 递出去"。
    """
    runtime = get_runtime()
    sink = getattr(runtime, "trace_sink", None)
    if sink is None:
        # 未注入（单元测试、纯 Agent 进程）→ 内存环形缓冲已经记了，落库静默跳过
        return False
    try:
        payload = span.to_dict()
        payload["agent_name"] = agent_name
        return bool(sink(payload))
    except Exception as exc:      # 观测失败不影响业务
        logger.warning("trace 落库失败（已忽略）：%s", exc)
        return False


def record_span(
    *,
    trace_id: str,
    msg_id: str,
    parent_msg_id: Optional[str],
    from_agent: str,
    to_agent: str,
    action: str,
    elapsed_ms: int,
    status: int = 0,
    error_code: Optional[int] = None,
    cache_hit: bool = False,
    tokens_used: int = 0,
    call_path: Optional[list] = None,
    detail: Optional[dict] = None,
    persist: bool = True,
) -> Span:
    """记录一次调用。**这是所有 Agent 的唯一 trace 写入口。**"""
    span = Span(
        trace_id=trace_id, msg_id=msg_id, parent_msg_id=parent_msg_id,
        from_agent=from_agent, to_agent=to_agent, action=action,
        status=status, error_code=error_code, elapsed_ms=elapsed_ms,
        cache_hit=cache_hit, tokens_used=tokens_used,
        call_path=list(call_path or []), detail=dict(detail or {}),
    )
    _store.add(span)
    _store.bump(to_agent, ok=(status == 0),
                degraded=(status == 1), timeout=(status == 3),
                elapsed_ms=elapsed_ms)
    if persist:
        persist_span(span)
    return span


def describe_error(exc: BaseException) -> tuple[int, str]:
    """把异常映射成 `(status, 描述)`。超时单独归类，因为它的处置不同（重试/降级）。"""
    if isinstance(exc, AgentError):
        if exc.code in (50301, 60601):
            return 3, exc.name
        return (1 if exc.deniable else 2), exc.name
    if isinstance(exc, (TimeoutError,)):
        return 3, "TIMEOUT"
    return 2, type(exc).__name__


__all__ = [
    "MAX_CALL_PATH",
    "RING_SIZE",
    "Span",
    "TraceStore",
    "describe_error",
    "get_store",
    "persist_span",
    "record_span",
]
