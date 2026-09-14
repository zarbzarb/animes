# -*- coding: utf-8 -*-
"""统一 LLM 客户端：**全部 LLM 调用必须走这里**（架构 §5.4）。

它在一条链路上同时做五件事，少做一件都会出问题：

| 职责 | 不做会怎样 |
|---|---|
| 超时（默认 200ms 给主链路、30s 给对话） | LLM 一个抖动就把推荐接口拖到几秒 |
| 重试（指数退避，上限 `LLM_MAX_RETRY`） | 一次网络抖动就退化成模板文案，降级率虚高 |
| Token 计量 | 论文里"LLM 成本"这一节写不出来 |
| 降级（返回 `degraded=True` 而不是抛异常） | 主链路 500 —— 违反架构「任一 Agent 失败不得 500」 |
| **PII 过滤** | 用户手机号/邮箱被发到第三方 API，属于事故 |

另外两条纪律：
1. **未配置（没有 base_url/api_key）时立刻返回降级结果**，不发注定 401 的请求
   —— 那会白等一次超时，把 200ms 预算吃光。
2. **LLM 只负责语言任务**：解释、对话、意图识别、文案、摘要。
   召回与排序绝不允许调这里（架构 ADR-1）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional, Sequence

from agents.common.errors import AgentError, llm_timeout, make
from agents.common.settings import AgentSettings, get_settings

logger = logging.getLogger(__name__)

# ---------------- PII 过滤 ----------------
# 只做"显式标识符"的替换，不做语义判断：宁可有漏网，也不要误伤把正常文本改坏。
_PII_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("phone_cn", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("ipv4", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")),
)


def scrub_pii(text: str) -> str:
    """把显式标识符替换成占位符。**发出去之前**必须调一次。"""
    if not text:
        return text
    for name, pattern in _PII_PATTERNS:
        text = pattern.sub(f"<{name}>", text)
    return text


def contains_pii(text: str) -> bool:
    return any(p.search(text or "") for _, p in _PII_PATTERNS)


# ---------------- 结果结构 ----------------


def _sanitize_message(m: Any) -> dict:
    """把一条消息转成可发出去的 OpenAI 格式：**PII 过滤 + 保留工具协议字段**。

    ⚠️ 一个真实会 400 的坑：早期实现只保留 `role` / `content`。
    Function Calling 的对话里，assistant 轮的 `tool_calls` 与 tool 轮的
    `tool_call_id` 是**协议必需**的 —— 丢任何一个，OpenAI 兼容端点都会
    拒绝整个请求（报"messages 结构非法"），而且报错点看着与工具无关，
    很难定位。另外 assistant 带 tool_calls 时 `content` 通常是 `None`,
    直接 `str(None)` 会变成字符串 `"None"` 发给模型。
    """
    out: dict[str, Any] = {"role": str(m.get("role", "user"))}
    content = m.get("content")
    if content is not None:
        out["content"] = scrub_pii(str(content))
    elif not m.get("tool_calls"):
        out["content"] = ""
    if m.get("tool_calls"):
        out["tool_calls"] = m["tool_calls"]
    if m.get("tool_call_id"):
        out["tool_call_id"] = m["tool_call_id"]
    if m.get("name"):
        out["name"] = m["name"]
    return out


@dataclass
class LLMResult:
    text: str = ""
    model: str = ""
    tokens_used: int = 0
    elapsed_ms: int = 0
    degraded: bool = False
    degraded_reason: Optional[str] = None
    attempts: int = 0
    raw: dict = field(default_factory=dict)
    # Function Calling：本轮模型要求的工具调用（OpenAI 结构，原样透传）
    tool_calls: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.degraded and bool(self.text)

    @property
    def wants_tools(self) -> bool:
        """模型要求执行工具（此时 `text` 通常为空，`ok` 为 False 属正常）。"""
        return not self.degraded and bool(self.tool_calls)


def _estimate_tokens(text: str) -> int:
    """粗略估算（无 tiktoken 依赖）：中文按 ~1.5 字/token，英文按 ~4 字符/token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4) + 1


class LLMClient:
    """OpenAI 兼容客户端（DashScope / DeepSeek / Ollama 均可）。

    `transport` 可注入：测试传一个假函数即可，**任何测试都不许打真实 API**。
    签名：`async (url, headers, payload, timeout) -> dict`
    """

    def __init__(self, settings: Optional[AgentSettings] = None,
                 transport: Optional[Callable[..., Any]] = None) -> None:
        self.settings = settings or get_settings()
        self._transport = transport
        self._client: Any = None
        self.calls = 0              # 进程内计量（/admin/agents/health 用）
        self.tokens_total = 0

    # ---------------- 可用性 ----------------
    @property
    def available(self) -> bool:
        return self.settings.llm_configured

    # ---------------- 内部 ----------------
    async def _default_transport(self, url: str, headers: dict, payload: dict,
                                 timeout: float) -> dict:
        import httpx
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=timeout)
        resp = await self._client.post(url, headers=headers, json=payload,
                                       timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _build(self, messages: Sequence[dict], *, temperature: Optional[float],
               max_tokens: Optional[int], json_mode: bool,
               tools: Optional[Sequence[dict]] = None) -> tuple[str, dict, dict]:
        s = self.settings
        url = s.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {s.llm_api_key}",
                   "Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": s.llm_model,
            # 每次调用前过一遍 PII：**在这里做**而不是让调用方各自记得做
            "messages": [_sanitize_message(m) for m in messages],
            "temperature": s.llm_temperature if temperature is None else temperature,
            "max_tokens": s.llm_max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        return url, headers, payload

    # ---------------- 对外 ----------------
    async def complete(
        self,
        messages: Sequence[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        timeout_ms: Optional[int] = None,
        tools: Optional[Sequence[dict]] = None,
    ) -> LLMResult:
        """一次补全。**绝不抛异常**：任何失败都返回 `degraded=True` 的结果。

        调用方看到 `degraded` 就切模板兜底（协议 §5 的降级链）。

        `tools` 非空时启用 Function Calling：模型可能返回 `tool_calls`
        而 `text` 为空 —— 这**不是失败**，用 `wants_tools` 判断即可。
        """
        if not self.available:
            return LLMResult(degraded=True, degraded_reason="LLM 未配置（缺 base_url/api_key）")

        s = self.settings
        timeout_s = (timeout_ms / 1000.0) if timeout_ms else float(s.llm_timeout)
        url, headers, payload = self._build(messages, temperature=temperature,
                                           max_tokens=max_tokens, json_mode=json_mode,
                                           tools=tools)
        transport = self._transport or self._default_transport

        last_reason = "unknown"
        for attempt in range(1, int(s.llm_max_retry) + 2):     # 首次 + 重试
            t0 = time.perf_counter()
            try:
                data = await asyncio.wait_for(
                    transport(url, headers, payload, timeout_s), timeout=timeout_s)
                try:
                    message = data["choices"][0]["message"]
                    if not isinstance(message, dict):
                        raise TypeError("message 不是对象")
                except (KeyError, IndexError, TypeError):
                    last_reason = "响应结构不符合 OpenAI 协议"
                    raise AgentError(make("LLM_INVALID_JSON").code, last_reason)

                # 带 tool_calls 时 content 常为 None —— 用 or "" 兜住
                text = str(message.get("content") or "")
                tool_calls = list(message.get("tool_calls") or [])

                usage = data.get("usage") or {}
                tokens = int(usage.get("total_tokens")
                             or _estimate_tokens(text))
                self.calls += 1
                self.tokens_total += tokens
                return LLMResult(
                    text=text, model=str(data.get("model") or s.llm_model),
                    tokens_used=tokens,
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                    attempts=attempt, raw=data if isinstance(data, dict) else {},
                    tool_calls=tool_calls,
                )
            except asyncio.TimeoutError:
                last_reason = f"超时（{timeout_s:.1f}s）"
            except AgentError as exc:
                last_reason = exc.message
            except Exception as exc:
                last_reason = f"{type(exc).__name__}: {exc}"

            if attempt <= int(s.llm_max_retry):
                await asyncio.sleep(min(0.2 * 2 ** (attempt - 1), 1.0))

        logger.warning("LLM 调用最终失败：%s（已尝试 %d 次）", last_reason,
                       int(s.llm_max_retry) + 1)
        return LLMResult(degraded=True, degraded_reason=last_reason,
                         attempts=int(s.llm_max_retry) + 1)

    async def complete_json(
        self,
        messages: Sequence[dict],
        *,
        required_keys: Sequence[str] = (),
        timeout_ms: Optional[int] = None,
        **kw: Any,
    ) -> tuple[Optional[dict], LLMResult]:
        """要求 LLM 输出 JSON 并**校验必要字段**。

        返回 `(obj, result)`：`obj is None` 表示必须走模板兜底。
        校验失败**不重试第二次**（协议 §4.2 写的是"重试 1 次 → 降级"，
        而 `complete()` 内部已经做过一次重试），避免把预算耗在注定失败的重试上。
        """
        res = await self.complete(messages, json_mode=True, timeout_ms=timeout_ms, **kw)
        if not res.ok:
            return None, res
        obj = _parse_json_loose(res.text)
        if obj is None:
            res.degraded, res.degraded_reason = True, "输出不是合法 JSON"
            return None, res
        missing = [k for k in required_keys if k not in obj]
        if missing:
            res.degraded = True
            res.degraded_reason = f"JSON 缺少字段 {missing}"
            return None, res
        return obj, res

    async def stream(self, messages: Sequence[dict],
                     **kw: Any) -> AsyncIterator[str]:
        """流式（对话推荐 SSE 用）。

        为了不让主链路依赖流式协议，这里**先整体取回再切片吐字** ——
        对本项目的规模（几十字文案）延迟差异不可感知，
        却避免了 SSE 解析、断流重连等一堆边界情况。
        """
        res = await self.complete(messages, **kw)
        if not res.ok:
            return
        for ch in res.text:
            yield ch
            await asyncio.sleep(0)

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:      # pragma: no cover
                pass


def _parse_json_loose(text: str) -> Optional[dict]:
    """LLM 常把 JSON 包在 ```json 里或前后带解释文字，这里做宽容解析。"""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except ValueError:
            return None
    return None


_client: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def set_llm(client: LLMClient) -> None:
    """测试注入假客户端。"""
    global _client
    _client = client


__all__ = [
    "LLMClient",
    "LLMResult",
    "contains_pii",
    "get_llm",
    "llm_timeout",
    "scrub_pii",
    "set_llm",
]
