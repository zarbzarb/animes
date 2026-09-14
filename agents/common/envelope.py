# -*- coding: utf-8 -*-
"""Agent 间通信的**唯一单位**：Envelope（协议 §三）。

Agent 之间不传裸 dict —— 裸 dict 的问题是「元信息在传递中丢失」：
重试次数、超时预算、trace_id、降级原因，一旦某一跳忘了透传，
排查时就少一段调用链。信封把这些字段固定下来，任何一跳都必须原样带下去。

字段定义严格对齐 `docs/agent-interaction-protocol.md` §3.1 / §3.2，
不做"顺手加一个字段"的扩展 —— 协议文档是 Agent 之间的公开契约，
改它要同时改文档与 `tests/test_agents/`。
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

SCHEMA_VERSION = "1.0"

MSG_TYPES = ("request", "response", "event", "error")
PRIORITIES = ("P0", "P1", "P2")

# 超时预算边界（§3.2：50–30000 ms）。超出即视为把单位写错（如秒当毫秒），
# 这类错误不报错、只是让链路"偶尔慢得莫名其妙"，所以直接夹住。
TIMEOUT_MIN_MS, TIMEOUT_MAX_MS = 50, 30_000

# 允许的 action（§3.3）。新增 action 必须同时改这份表与协议文档，
# tests/test_agents/test_action_registry.py 会对着表校验 Agent 声明的 actions。
ACTION_REGISTRY: dict[str, tuple[str, ...]] = {
    "A0": ("orchestrate.route", "orchestrate.aggregate"),
    "A1": ("profile.get", "profile.refresh", "profile.summarize"),
    "A2": ("recall.sequence", "recall.batch"),
    "A3": ("content.retrieve", "content.match_genre"),
    "A4": ("rank.fusion", "rank.diversify"),
    "A5": ("explain.generate", "explain.stream"),
    "A6": ("drift.analyze", "drift.interpret"),
    "A7": ("chat.reply", "chat.tool_call"),
    "A8": ("dataops.sync_new_anime", "dataops.rebuild_index",
           "dataops.quality_check"),
    "A9": ("eval.offline_metrics", "eval.online_snapshot", "eval.alarm"),
}

AGENT_IDS = tuple(sorted(ACTION_REGISTRY))

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
_TRACE_RE = re.compile(r"^tr_[0-9a-f]{12}$")


def new_msg_id(now_ms: Optional[int] = None) -> str:
    """生成 ULID（26 位 Crockford Base32）—— 前 48 位是毫秒时间戳，天然可按时间排序。"""
    ms = int(now_ms if now_ms is not None else time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    value = (ms << 80) | rand
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def is_valid_msg_id(msg_id: str) -> bool:
    return bool(_ULID_RE.match(msg_id or ""))


def is_valid_trace_id(trace_id: str) -> bool:
    return bool(_TRACE_RE.match(trace_id or ""))


@dataclass
class Header:
    msg_id: str
    trace_id: str
    from_agent: str
    to_agent: str
    action: str
    type: str = "request"
    parent_msg_id: Optional[str] = None
    priority: str = "P0"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    timeout_ms: int = 200
    retry_count: int = 0
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.type not in MSG_TYPES:
            raise ValueError(f"未知消息类型：{self.type}（可选 {MSG_TYPES}）")
        if self.priority not in PRIORITIES:
            raise ValueError(f"未知优先级：{self.priority}（可选 {PRIORITIES}）")
        self.timeout_ms = max(TIMEOUT_MIN_MS, min(TIMEOUT_MAX_MS, int(self.timeout_ms)))
        if not is_valid_msg_id(self.msg_id):
            raise ValueError(f"msg_id 不是合法 ULID：{self.msg_id!r}")
        if not is_valid_trace_id(self.trace_id):
            raise ValueError(f"trace_id 不符合 tr_+12hex：{self.trace_id!r}")

    def to_dict(self) -> dict:
        # 键名与协议一致（from / to 是 Python 关键字，故 dataclass 用 *_agent）
        return {
            "msg_id": self.msg_id,
            "trace_id": self.trace_id,
            "parent_msg_id": self.parent_msg_id,
            "from": self.from_agent,
            "to": self.to_agent,
            "type": self.type,
            "action": self.action,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "timeout_ms": self.timeout_ms,
            "retry_count": self.retry_count,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Header":
        return cls(
            msg_id=d["msg_id"], trace_id=d["trace_id"],
            from_agent=d.get("from", ""), to_agent=d.get("to", ""),
            action=d.get("action", ""), type=d.get("type", "request"),
            parent_msg_id=d.get("parent_msg_id"),
            priority=d.get("priority", "P0"),
            timestamp=int(d.get("timestamp", 0)) or int(time.time() * 1000),
            timeout_ms=int(d.get("timeout_ms", 200)),
            retry_count=int(d.get("retry_count", 0)),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


@dataclass
class Context:
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    locale: str = "zh-CN"
    app_env: str = "dev"

    def to_dict(self) -> dict:
        return {"user_id": self.user_id, "session_id": self.session_id,
                "locale": self.locale, "app_env": self.app_env}

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "Context":
        d = d or {}
        return cls(user_id=d.get("user_id"), session_id=d.get("session_id"),
                   locale=d.get("locale", "zh-CN"),
                   app_env=d.get("app_env", "dev"))


@dataclass
class Meta:
    degraded: bool = False
    degraded_reason: Optional[str] = None
    cache_hit: bool = False
    elapsed_ms: int = 0
    tokens_used: int = 0

    def to_dict(self) -> dict:
        return {"degraded": self.degraded, "degraded_reason": self.degraded_reason,
                "cache_hit": self.cache_hit, "elapsed_ms": self.elapsed_ms,
                "tokens_used": self.tokens_used}

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "Meta":
        d = d or {}
        return cls(degraded=bool(d.get("degraded", False)),
                   degraded_reason=d.get("degraded_reason"),
                   cache_hit=bool(d.get("cache_hit", False)),
                   elapsed_ms=int(d.get("elapsed_ms", 0)),
                   tokens_used=int(d.get("tokens_used", 0)))


@dataclass
class Envelope:
    header: Header
    context: Context = field(default_factory=Context)
    payload: dict = field(default_factory=dict)
    meta: Meta = field(default_factory=Meta)

    # ---------------- 序列化 ----------------
    def to_dict(self) -> dict:
        return {"header": self.header.to_dict(), "context": self.context.to_dict(),
                "payload": self.payload, "meta": self.meta.to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Envelope":
        return cls(header=Header.from_dict(d["header"]),
                   context=Context.from_dict(d.get("context")),
                   payload=dict(d.get("payload") or {}),
                   meta=Meta.from_dict(d.get("meta")))

    # ---------------- 构造 ----------------
    @classmethod
    def request(
        cls,
        *,
        from_agent: str,
        to_agent: str,
        action: str,
        payload: Optional[dict] = None,
        trace_id: str,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
        priority: str = "P0",
        timeout_ms: int = 200,
        parent_msg_id: Optional[str] = None,
    ) -> "Envelope":
        from agents.common.settings import get_settings
        return cls(
            header=Header(msg_id=new_msg_id(), trace_id=trace_id,
                          from_agent=from_agent, to_agent=to_agent,
                          action=action, type="request",
                          parent_msg_id=parent_msg_id, priority=priority,
                          timeout_ms=timeout_ms),
            context=Context(user_id=user_id, session_id=session_id,
                            app_env=get_settings().app_env),
            payload=dict(payload or {}),
        )

    # ---------------- 回包 ----------------
    def reply(self, payload: Optional[dict] = None, *,
              degraded: bool = False, degraded_reason: Optional[str] = None,
              elapsed_ms: int = 0, cache_hit: bool = False,
              tokens_used: int = 0) -> "Envelope":
        return Envelope(
            header=Header(
                msg_id=new_msg_id(), trace_id=self.header.trace_id,
                from_agent=self.header.to_agent, to_agent=self.header.from_agent,
                action=self.header.action, type="response",
                parent_msg_id=self.header.msg_id,
                priority=self.header.priority, timeout_ms=self.header.timeout_ms,
                retry_count=self.header.retry_count,
            ),
            context=self.context,
            payload=dict(payload or {}),
            meta=Meta(degraded=degraded, degraded_reason=degraded_reason,
                      cache_hit=cache_hit, elapsed_ms=elapsed_ms,
                      tokens_used=tokens_used),
        )

    def error(self, *, error_code: int, error_name: str, message: str,
              retryable: bool = True, detail: Optional[dict] = None,
              elapsed_ms: int = 0) -> "Envelope":
        """错误信封（协议 §4.3）。`payload` 结构与成功响应刻意不同 —— 便于上游区分。"""
        return Envelope(
            header=Header(
                msg_id=new_msg_id(), trace_id=self.header.trace_id,
                from_agent=self.header.to_agent, to_agent=self.header.from_agent,
                action=self.header.action, type="error",
                parent_msg_id=self.header.msg_id,
                priority=self.header.priority, timeout_ms=self.header.timeout_ms,
                retry_count=self.header.retry_count,
            ),
            context=self.context,
            payload={"error_code": int(error_code), "error_name": error_name,
                     "message": message, "retryable": bool(retryable),
                     "detail": dict(detail or {})},
            meta=Meta(degraded=True, degraded_reason=error_name,
                      elapsed_ms=elapsed_ms),
        )

    # ---------------- 便捷判定 ----------------
    @property
    def ok(self) -> bool:
        return self.header.type != "error" and self.payload.get("error_code") is None

    @property
    def is_error(self) -> bool:
        return self.header.type == "error"

    def validate(self) -> None:
        """校验 action 是否登记在册（协议 §3.3 的 CI 要求）。"""
        allowed = ACTION_REGISTRY.get(self.header.to_agent)
        if allowed is None:
            raise ValueError(f"未注册的 Agent：{self.header.to_agent}")
        if self.header.action not in allowed:
            raise ValueError(
                f"action {self.header.action!r} 未在协议中登记给 "
                f"{self.header.to_agent}（已登记：{allowed}）")


__all__ = [
    "ACTION_REGISTRY",
    "AGENT_IDS",
    "MSG_TYPES",
    "PRIORITIES",
    "SCHEMA_VERSION",
    "TIMEOUT_MAX_MS",
    "TIMEOUT_MIN_MS",
    "Context",
    "Envelope",
    "Header",
    "Meta",
    "is_valid_msg_id",
    "is_valid_trace_id",
    "new_msg_id",
]
