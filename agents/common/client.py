# -*- coding: utf-8 -*-
"""Agent 之间的调用客户端（**M1 同步请求-响应**，协议 §二）。

为什么需要一个独立的 client，而不是各自 `registry.get(x).handle(env)`
--------------------------------------------------------------------
有三个"只有收口到一处才对"的东西：

1. **循环调用检测**（协议 §5.4）。判据是"`to` 是否已在 `call_path` 里"。
   如果每个 Agent 自己拼信封，就得各自实现一遍这个判断，早晚漏一个。
2. **链路深度上限**（协议 §5.4：全局 6 层）。深度是**链路级**的属性，
   单个 Agent 看不到 —— 只有集中在一处才能拒绝"看起来每一跳都很合理"
   的深链。
3. **超时预算的单调收紧**。子调用拿到的 `timeout_ms` 必须是父预算减掉
   已经花掉的时间，否则一个 200ms 的子预算会挂在只剩 30ms 的父预算上，
   表现为父超时了子还在跑。

失败语义
--------
本函数**不抛异常**（除了编程错误）。返回 `None` 表示"这一步失败了"，
并把原因写进 `SubCallResult`。理由：调用方（A0/A7）需要区分
"这一步降级了"（继续）与"这一步结构性失败"（走 L4 兜底），
用返回值表达比用异常层级表达更清楚。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from agents.common.envelope import Envelope
from agents.common.errors import AgentError, make
from agents.common.registry import AgentRegistry

logger = logging.getLogger(__name__)

__all__ = ["MAX_DEPTH", "SubCallResult", "call_agent", "check_chain"]

# 协议 §5.4：全局链路深度上限 6 层
MAX_DEPTH = 6

# 违规（结构性失败）错误码：这些**不可降级**，调用方应走自己的兜底
_STRUCTURAL = {
    make("DEADLOCK_DETECTED").code,
    make("AGENT_CHAIN_BROKEN").code,
    make("RESOURCE_NOT_FOUND").code,
    make("INVALID_PARAM").code,
}


@dataclass
class SubCallResult:
    ok: bool = False
    payload: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    chain: list[str] = field(default_factory=list)
    error_code: Optional[int] = None
    error_name: str = ""
    message: str = ""
    structural: bool = False
    elapsed_ms: int = 0

    @property
    def degraded(self) -> bool:
        return bool(self.meta.get("degraded"))


def check_chain(call_path: Sequence[str], to_agent: str) -> Optional[str]:
    """返回拒绝原因，`None` 表示可以调用。

    两条规则（协议 §5.4）：
    * `to` 已在路径中 → 环；
    * 路径长度 ≥ `MAX_DEPTH` → 过深。

    ⚠️ 判定用 `to_agent` 与**已访问的 agent id** 比较，而不是与 `action`
    比较：同一对 Agent 之间多次调用（A0 调 A2 取两批候选）是合法的。
    """
    if to_agent in call_path:
        return f"检测到循环调用：{to_agent} 已在链路 {list(call_path)} 中"
    if len(call_path) >= MAX_DEPTH:
        return f"链路深度已达上限 {MAX_DEPTH}：{list(call_path)}"
    return None


async def call_agent(
    *,
    from_agent: str,
    to_agent: str,
    action: str,
    payload: Optional[dict] = None,
    trace_id: str,
    call_path: Sequence[str] = (),
    timeout_ms: int = 200,
    priority: str = "P0",
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
    budget_deadline: Optional[float] = None,
) -> SubCallResult:
    """调用另一个 Agent 并返回结构化结果。

    `budget_deadline` 是父级用 `time.perf_counter()` 记下的截止时刻；
    传入时会**自动收紧** `timeout_ms`，保证子调用不会超出父预算。
    """
    t0 = time.perf_counter()
    path = [str(x) for x in call_path]

    reason = check_chain(path, to_agent)
    if reason:
        logger.warning("%s → %s 被拒绝：%s", from_agent, to_agent, reason)
        name = ("DEADLOCK_DETECTED" if "循环" in reason else "AGENT_CHAIN_BROKEN")
        return SubCallResult(error_code=make(name).code, error_name=name,
                             message=reason, structural=True, chain=path)

    # 预算收紧：至少留 10ms，避免"父预算剩余为负"时子调用拿到负数超时
    budget = int(timeout_ms)
    if budget_deadline is not None:
        remaining_ms = int((budget_deadline - time.perf_counter()) * 1000)
        budget = max(10, min(budget, remaining_ms))

    try:
        agent = AgentRegistry.get(to_agent)
    except AgentError as exc:
        return SubCallResult(error_code=exc.code, error_name=exc.name,
                             message=exc.message, structural=True, chain=path)

    env = Envelope.request(
        from_agent=from_agent, to_agent=to_agent, action=action,
        payload={**(payload or {}), "_call_path": path},
        trace_id=trace_id, user_id=user_id, session_id=session_id,
        priority=priority, timeout_ms=budget)

    try:
        out = await agent.handle(env)
    except Exception as exc:        # handle 理论上不抛；保险丝
        logger.exception("%s → %s 调用异常", from_agent, to_agent)
        return SubCallResult(error_code=make("INTERNAL_ERROR").code,
                             error_name="INTERNAL_ERROR",
                             message=f"{type(exc).__name__}: {exc}",
                             chain=path + [to_agent],
                             elapsed_ms=int((time.perf_counter() - t0) * 1000))

    elapsed = int((time.perf_counter() - t0) * 1000)
    if out.is_error:
        code = out.payload.get("error_code")
        name = str(out.payload.get("error_name") or "")
        structural = (code in _STRUCTURAL)
        return SubCallResult(
            ok=False, payload={}, meta=out.meta.to_dict(),
            chain=path + [to_agent], error_code=code, error_name=name,
            message=str(out.payload.get("message") or ""),
            structural=structural, elapsed_ms=elapsed)

    return SubCallResult(ok=True, payload=dict(out.payload),
                         meta=out.meta.to_dict(), chain=path + [to_agent],
                         elapsed_ms=elapsed)
