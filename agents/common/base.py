# -*- coding: utf-8 -*-
"""`BaseAgent`：所有 Agent 的统一入口与横切关注点（`docs/project-structure.md` §2）。

子类**只实现 `invoke()`**，不覆写 `handle()`。这样以下六件事对所有 Agent
自动生效，不会出现"某个 Agent 忘了记 trace"或"某个 Agent 超时了不降级"：

1. 校验 action 是否在协议里登记（协议 §3.3 的 CI 要求）；
2. 按 `timeout_ms` 强制超时（协议 §3.2：接收方超时必须返回 error 信封）；
3. 计时；
4. 写 `agent_trace` + 更新健康度；
5. 异常 → 错误信封（不把异常抛穿到调用方）；
6. **能降级就降级**：子类实现 `fallback()` 的话，可静默降级的错误会被转成
   `degraded=True` 的正常信封，而不是 error —— 对应协议 §6「降级一律 HTTP 200」。

约定：`invoke()` 返回 `dict`（业务载荷）或 `Envelope`（需要自定义 meta 时）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union

from agents.common.envelope import ACTION_REGISTRY, Envelope
from agents.common.errors import AgentError, make
from agents.common.settings import get_settings
from agents.common.tracing import describe_error, get_store, record_span

logger = logging.getLogger(__name__)

InvokeResult = Union[dict, Envelope]


class BaseAgent(ABC):
    """所有 Agent 的基类。"""

    agent_id: str = "A?"
    name: str = "unnamed"
    actions: tuple[str, ...] = ()
    # 本 Agent 的描述（用在 /admin/agents/health 与 README 生成）
    description: str = ""
    # 是否在主请求链路上（P0）；离线 Agent 设 False
    online: bool = True

    def __init__(self) -> None:
        settings = get_settings()
        self.default_timeout_ms = int(settings.timeout_ms.get(self.agent_id, 200))
        if not self.actions:
            # 允许省略：从协议表反查。但显式声明更清楚，也让测试能对比
            self.actions = ACTION_REGISTRY.get(self.agent_id, ())

    # ------------------------------------------------------------ 子类实现
    @abstractmethod
    async def invoke(self, payload: dict, env: Envelope) -> InvokeResult:
        """业务逻辑。**只管业务**：计时、trace、超时、异常都由 `handle()` 负责。"""

    async def fallback(self, payload: dict, env: Envelope,
                       error: AgentError) -> Optional[dict]:
        """可选的降级实现。返回 `None` 表示"我降不了"，于是返回错误信封。"""
        return None

    # ------------------------------------------------------------ 统一入口
    async def handle(self, env: Envelope) -> Envelope:
        t0 = time.perf_counter()
        # 调用链：`_call_path` 是**内部约定字段**（不是协议字段），由调用方
        # （`agents/common/client.py`）注入，用于循环检测与链路深度上限。
        # 关键点：把**追加过自己**的路径再传给 `invoke`，否则子 Agent 拿不到
        # 上游链路，"A0→A7→A0"这种环就检测不出来。
        call_path = [str(x) for x in (env.payload.get("_call_path") or [])]
        call_path.append(self.agent_id)
        payload_in = {k: v for k, v in env.payload.items() if k != "_call_path"}
        payload_in["_call_path"] = list(call_path)

        # ① action 白名单
        allowed = ACTION_REGISTRY.get(self.agent_id, ())
        if env.header.action not in allowed:
            return self._finish_error(
                env, AgentError(make("INVALID_PARAM").code,
                                f"未登记的 action：{env.header.action}",
                                detail={"allowed": list(allowed)}),
                t0, call_path)

        timeout_s = max(env.header.timeout_ms, 1) / 1000.0
        try:
            result = await asyncio.wait_for(self.invoke(payload_in, env),
                                            timeout=timeout_s)
        except asyncio.TimeoutError:
            # ⚠️ 超时**也要走降级路径**：`AGENT_TIMEOUT`(50301) 在
            # `errors.DEGRADE_SILENTLY` 里，协议 §5.2/§6 要求"能降级就降级、
            # 一律 HTTP 200"。早期实现直接 `_finish_error`，结果是
            # "A2 超时"本可退回热门榜、却给前端抛了一个错误 —— 与协议不一致。
            return await self._degrade_or_error(
                env, AgentError(make("AGENT_TIMEOUT").code,
                                f"{self.agent_id} 超过 {env.header.timeout_ms}ms 预算",
                                detail={"agent": self.agent_id}),
                t0, call_path)
        except AgentError as exc:
            return await self._degrade_or_error(env, exc, t0, call_path)
        except Exception as exc:                    # 未预期异常
            logger.exception("%s 未捕获异常", self.agent_id)
            return self._finish_error(
                env, AgentError(make("INTERNAL_ERROR").code, f"{type(exc).__name__}: {exc}",
                                detail={"agent": self.agent_id}),
                t0, call_path)

        elapsed = int((time.perf_counter() - t0) * 1000)
        if isinstance(result, Envelope):            # 子类自建信封（如流式）
            out = result
        else:
            out = env.reply(result or {}, elapsed_ms=elapsed)

        record_span(
            trace_id=env.header.trace_id, msg_id=out.header.msg_id,
            parent_msg_id=out.header.parent_msg_id,
            from_agent=self.agent_id, to_agent=env.header.from_agent,
            action=env.header.action, elapsed_ms=elapsed,
            status=1 if out.meta.degraded else 0,
            cache_hit=out.meta.cache_hit, tokens_used=out.meta.tokens_used,
            call_path=call_path,
        )
        return out

    # ------------------------------------------------------------ 内部
    async def _degrade_or_error(self, env: Envelope, exc: AgentError,
                                t0: float, call_path: list) -> Envelope:
        """可降级的错误 → 调 `fallback()`；拿不到兜底数据才返回 error 信封。"""
        if exc.deniable:
            try:
                data = await self.fallback(dict(env.payload), env, exc)
            except Exception as fb_exc:            # 兜底自己挂了
                logger.warning("%s 的 fallback 也失败：%s", self.agent_id, fb_exc)
                data = None
            if data is not None:
                elapsed = int((time.perf_counter() - t0) * 1000)
                out = env.reply(data, degraded=True, degraded_reason=exc.name,
                                elapsed_ms=elapsed)
                record_span(
                    trace_id=env.header.trace_id, msg_id=out.header.msg_id,
                    parent_msg_id=out.header.parent_msg_id,
                    from_agent=self.agent_id, to_agent=env.header.from_agent,
                    action=env.header.action, elapsed_ms=elapsed, status=1,
                    error_code=exc.code, call_path=call_path,
                    detail={"degraded_reason": exc.name},
                )
                return out
        return self._finish_error(env, exc, t0, call_path)

    def _finish_error(self, env: Envelope, exc: AgentError, t0: float,
                      call_path: list) -> Envelope:
        elapsed = int((time.perf_counter() - t0) * 1000)
        out = env.error(
            error_code=exc.code, error_name=exc.name,
            message=exc.message or exc.name, retryable=exc.retryable,
            detail={"agent": self.agent_id, **exc.detail}, elapsed_ms=elapsed,
        )
        status, _ = describe_error(exc)
        record_span(
            trace_id=env.header.trace_id, msg_id=out.header.msg_id,
            parent_msg_id=out.header.parent_msg_id,
            from_agent=self.agent_id, to_agent=env.header.from_agent,
            action=env.header.action, elapsed_ms=elapsed, status=status,
            error_code=exc.code, call_path=call_path, detail=exc.detail,
        )
        return out

    # ------------------------------------------------------------ 自检
    @classmethod
    def declared_actions(cls) -> Sequence[str]:
        return ACTION_REGISTRY.get(cls.agent_id, ())

    @classmethod
    def health_snapshot(cls) -> dict[str, Any]:
        s = get_store().stats().get(cls.agent_id, {})
        calls = s.get("calls", 0)
        return {
            "agent_id": cls.agent_id, "name": cls.name,
            "online": cls.online, "actions": list(cls.declared_actions()),
            "calls": calls,
            "success_rate": round(s.get("success", 0) / calls, 4) if calls else None,
            "avg_elapsed_ms": int(s.get("total_ms", 0) / calls) if calls else None,
        }


__all__ = ["BaseAgent", "InvokeResult"]
