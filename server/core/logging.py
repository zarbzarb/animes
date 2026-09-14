# -*- coding: utf-8 -*-
"""结构化日志 + `trace_id` 注入。

为什么不用 `logging.basicConfig` 了事
------------------------------------
本项目的可观测性要求是「**一次推荐请求的完整 Agent 调用链可追溯**」
（见 `docs/architecture.md` 第一节的「可观测」目标）。这要求日志行里必须带
`trace_id`，且**不需要每个函数都显式传参** —— 用 `ContextVar` 在请求入口
设置一次，其余地方 `logger.info(...)` 自动带上。

用法::

    from server.core.logging import setup_logging, set_trace_id, get_trace_id
    setup_logging()                       # 进程启动时调一次
    set_trace_id("tr_7f3a9c21e0b4")       # 中间件里调
    logger.info("命中缓存")                 # 输出：[tr_7f3a9c21e0b4] 命中缓存
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Optional

_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)

_CONFIGURED = False


def set_trace_id(trace_id: Optional[str]) -> None:
    _trace_id.set(trace_id)


def get_trace_id() -> Optional[str]:
    return _trace_id.get()


class TraceIdFilter(logging.Filter):
    """把当前 `trace_id` 塞进 `record`，供 Formatter 使用。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.trace_id = _trace_id.get() or "-"
        return True


_FORMAT = "%(asctime)s [%(levelname)s] [%(trace_id)s] %(name)s: %(message)s"


def setup_logging(level: Optional[str] = None, *, quiet_third_party: bool = True) -> None:
    """初始化根 logger。重复调用是幂等的（uvicorn reload 会调多次）。"""
    global _CONFIGURED
    from server.core.config import settings

    root = logging.getLogger()
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.addFilter(TraceIdFilter())
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        root.handlers = [handler]
        _CONFIGURED = True

    root.setLevel((level or settings.LOG_LEVEL).upper())

    if quiet_third_party:
        for name in ("uvicorn.access", "httpx", "httpcore", "asyncio"):
            logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["get_logger", "get_trace_id", "set_trace_id", "setup_logging", "TraceIdFilter"]
