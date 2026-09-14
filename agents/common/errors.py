# -*- coding: utf-8 -*-
"""Agent 层错误码。

⚠️ **这份表与 `server/core/exceptions.py::ErrCode` 是同一份协议**（§4.2）。
为什么允许重复而不是共享：依赖方向不允许 `agents/` import `server/`。
重复的代价是"可能漂移"，所以用**测试**兜住 ——
`tests/test_api/test_error_codes.py` 会断言两张表逐项相等。
这条经验在本项目已经吃过两次（指标唯一实现、负样本唯一实现）：
**口径可以重复，但必须有测试证明它们一致**，否则早晚分叉。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# 协议 §4.2 的完整清单：(码, 名称, 是否可重试)
CODE_TABLE: tuple[tuple[int, str, bool], ...] = (
    (40001, "INVALID_PARAM", False),
    (40101, "UNAUTHORIZED", False),
    (40301, "FORBIDDEN", False),
    (40401, "RESOURCE_NOT_FOUND", False),
    (40901, "DUPLICATE_REQUEST", False),
    (42901, "RATE_LIMITED", True),
    (50001, "INTERNAL_ERROR", True),
    (50201, "MODEL_UNAVAILABLE", True),
    (50202, "DB_UNAVAILABLE", True),
    (50203, "CACHE_UNAVAILABLE", True),
    (50204, "INDEX_MISSING", False),
    (50301, "AGENT_TIMEOUT", True),
    (60301, "AGENT_CHAIN_BROKEN", False),
    (60401, "AGENT_DEGRADED", False),
    (60501, "DEADLOCK_DETECTED", False),
    (60502, "TOOL_CALL_LIMIT", False),
    (60503, "CONTEXT_OVERFLOW", True),
    (60601, "LLM_TIMEOUT", True),
    (60602, "LLM_RATE_LIMITED", True),
    (60603, "LLM_INVALID_JSON", True),
    (60604, "LLM_HALLUCINATION", True),
)

CODES: dict[str, int] = {name: code for code, name, _ in CODE_TABLE}
RETRYABLE: dict[int, bool] = {code: retry for code, _, retry in CODE_TABLE}

# 这些错误**不返回给前端当错误**，而是降级后返回 200 + meta.degraded（§6 的「重要」）
DEGRADE_SILENTLY = {
    CODES["MODEL_UNAVAILABLE"], CODES["CACHE_UNAVAILABLE"],
    CODES["INDEX_MISSING"], CODES["LLM_TIMEOUT"],
    CODES["LLM_RATE_LIMITED"], CODES["LLM_INVALID_JSON"],
    CODES["LLM_HALLUCINATION"], CODES["AGENT_TIMEOUT"],
    CODES["TOOL_CALL_LIMIT"],
}


@dataclass
class AgentError(Exception):
    """Agent 内部异常。`retryable` 决定上游是否重试（协议 §4.2 最后一列）。"""

    code: int
    message: str = ""
    detail: dict = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = next((n for c, n, _ in CODE_TABLE if c == self.code),
                             "UNKNOWN")
        super().__init__(f"[{self.code} {self.name}] {self.message or self.name}")

    @property
    def retryable(self) -> bool:
        return RETRYABLE.get(self.code, True)

    @property
    def deniable(self) -> bool:
        """能否静默降级（不向前端暴露错误）。"""
        return self.code in DEGRADE_SILENTLY

    def to_payload(self) -> dict:
        return {"error_code": self.code, "error_name": self.name,
                "message": self.message or self.name,
                "retryable": self.retryable, "detail": dict(self.detail)}


def make(name: str, message: str = "", **detail) -> AgentError:
    """`make("MODEL_UNAVAILABLE", "CUDA OOM", agent="A2")`。"""
    return AgentError(code=CODES[name], message=message, detail=dict(detail))


# ---------------- 常用快捷构造 ----------------
def invalid_param(msg: str = "入参错误", **d) -> AgentError:
    return make("INVALID_PARAM", msg, **d)


def timeout(msg: str = "Agent 处理超时", **d) -> AgentError:
    return make("AGENT_TIMEOUT", msg, **d)


def model_unavailable(msg: str = "模型不可用", **d) -> AgentError:
    return make("MODEL_UNAVAILABLE", msg, **d)


def index_missing(msg: str = "索引缺失", **d) -> AgentError:
    return make("INDEX_MISSING", msg, **d)


def chain_broken(msg: str = "关键 Agent 链断", **d) -> AgentError:
    return make("AGENT_CHAIN_BROKEN", msg, **d)


def llm_timeout(msg: str = "LLM 超时", **d) -> AgentError:
    return make("LLM_TIMEOUT", msg, **d)


def internal(msg: str = "未捕获异常", **d) -> AgentError:
    return make("INTERNAL_ERROR", msg, **d)


__all__ = [
    "CODE_TABLE",
    "CODES",
    "DEGRADE_SILENTLY",
    "RETRYABLE",
    "AgentError",
    "chain_broken",
    "index_missing",
    "internal",
    "invalid_param",
    "llm_timeout",
    "make",
    "model_unavailable",
    "timeout",
]
