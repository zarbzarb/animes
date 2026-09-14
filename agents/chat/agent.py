# -*- coding: utf-8 -*-
"""A7 · 对话推荐智能体（`ChatAgent`）。

职责（`docs/agents.md` §A7）：LLM 驱动 + Function Calling 的多轮对话式推荐。

三条边界
-------
* ❌ 工具调用 ≤ 3 次/轮（防无限循环，协议 `60502`）
* ❌ 不得生成库外动漫（prompt 约束 + `strategy.verify_reply_titles` 机械校验）
* ❌ 不做排序融合（交给 A4，ADR-1）

为什么"卡片"不由 LLM 产出
------------------------
一个很容易做错的设计是"让 LLM 直接输出推荐卡片 JSON"。那样 LLM 就成了
**事实来源**，而它恰恰是最不可靠的一环（会编出真实存在感的番名）。
本实现让 LLM 只产出**自然语言**，卡片一律从工具的真实返回里提取
（`_cards_from`）—— 这样"LLM 幻觉"最多影响措辞，不会污染推荐结果本身。
额外的机械校验（`verify_reply_titles`）只兜住"文案里出现的番名"。

降级链（协议 §5.2：A7 超时 5000ms、失败 → 退化为引导话术）
------------------------------------------------------
LLM 未配置/超时      → 模板话术 + **真实热门卡片**（说清是榜单，不假装个性化）
工具全部失败          → 同上
整体异常（`fallback`）→ 引导话术 + 热门卡片
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any, Optional

from agents.chat import prompts, strategy, tools
from agents.chat.config import ChatConfig
from agents.chat.schemas import ChatInput, ToolCallInput
from agents.common import memory
from agents.common.base import BaseAgent
from agents.common.data import get_gateway
from agents.common.envelope import Envelope
from agents.common.errors import AgentError, invalid_param
from agents.common.llm import get_llm
from agents.common.ports import get_runtime
from agents.common.tools import ToolRunner, openai_tools

logger = logging.getLogger(__name__)

CFG = ChatConfig()


# ---------------------------------------------------------------- LLM tool_call 解析
# OpenAI 兼容端点返回的 tool_call 可能是对象也可能是 dict（不同 SDK/网关），
# 全部按结构鸭子类型读，避免"A 网关能跑、B 网关报 AttributeError"。
def _tool_fn(tc: Any) -> Any:
    return getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)


def _tool_name_of(tc: Any) -> str:
    fn = _tool_fn(tc)
    if fn is None:
        return "unknown"
    return str(getattr(fn, "name", None) or (fn.get("name", "") if isinstance(fn, dict) else "")
               or "unknown")


def _tool_id_of(tc: Any) -> str:
    return str(getattr(tc, "id", None)
               or (tc.get("id") if isinstance(tc, dict) else "") or "")


def _cache_key(user_id: int, session_id: str, message: str) -> str:
    h = hashlib.md5(message.encode("utf-8")).hexdigest()[:10]
    return f"chat:{int(user_id)}:{session_id}:{h}"


class ChatAgent(BaseAgent):
    agent_id = "A7"
    name = "chat"
    description = "对话推荐：多轮上下文 + 工具调用 + 口语化回复"
    online = True

    # ------------------------------------------------------------ 入口
    async def invoke(self, payload: dict, env: Envelope) -> dict:
        action = env.header.action
        if action == "chat.tool_call":
            return await self._tool_call(payload, env)
        if action != "chat.reply":
            raise invalid_param(f"未支持的 action：{action}")
        return await self._reply(payload, env)

    # ------------------------------------------------------------ chat.reply
    async def _reply(self, payload: dict, env: Envelope) -> dict:
        inp = ChatInput.model_validate(payload)
        msg = (inp.message or "").strip()
        if not msg:
            raise invalid_param("message 不能为空")
        msg = msg[: CFG.max_message_len]

        session_id = inp.session_id or f"s{uuid.uuid4().hex[:12]}"
        runtime = get_runtime()
        cache = runtime.cache

        # 防抖：同一 (用户, 会话, 消息内容) 短时间内重复提交直接复用
        key = _cache_key(inp.user_id, session_id, msg)
        try:
            cached = await cache.get_json(key)
        except Exception:
            cached = None
        if isinstance(cached, dict) and cached.get("reply"):
            out = dict(cached)
            out["cache_hit"] = True
            return out

        history = await self._history(inp, session_id)
        shown_src = await self._shown(inp.user_id, session_id)
        profile_hint = self._profile_hint(inp.user_id) if inp.user_id > 0 else ""
        shown_titles = self._titles_of(shown_src) if shown_src else []

        # 自己留 slack：保证在被 `BaseAgent.handle` 强制超时**之前**返回降级话术，
        # 否则前端拿到的是 AGENT_TIMEOUT 错误，而协议要求"退化为引导话术"。
        budget_ms = int(env.header.timeout_ms or CFG.llm_timeout_ms)
        deadline = time.perf_counter() + max(budget_ms - CFG.budget_slack_ms, 200) / 1000.0
        ctx = tools.ToolContext(
            user_id=int(inp.user_id), session_id=session_id,
            trace_id=env.header.trace_id,
            call_path=tuple(str(x) for x in (payload.get("_call_path") or ())),
            deadline=deadline, scene=3)

        with tools.bind_tool_context(ctx):
            reply, pool, tool_names, tokens, degraded, reason = await self._chat_loop(
                inp=inp, msg=msg, ctx=ctx, history=history,
                profile_hint=profile_hint, shown_titles=shown_titles)

        violations: list[str] = []
        if reply and CFG.verify_titles:
            known = [c.get("title") for c in pool if c.get("title")] + list(shown_titles)
            bad, _ = strategy.verify_reply_titles(reply, known)
            if bad:
                # 宁可换成模板话术，也不让编造的番名流到用户眼前
                violations = bad
                logger.warning("A7 回复含库外作品 %s，已整段替换为模板话术", bad)
                reply = prompts.no_llm_reply(
                    [c.get("title") for c in pool if c.get("title")])
                degraded, reason = True, (reason or "LLM_HALLUCINATION")

        cards = strategy.clamp_cards(strategy.dedup_cards(pool), CFG.cards_top_k,
                                     exclude_src=set(shown_src))
        if not cards and pool:
            # 全被"已推过"过滤掉 → 宁可重复，也不返回空列表
            cards = strategy.clamp_cards(strategy.dedup_cards(pool), CFG.cards_top_k)
        if not cards:
            # 工具一次都没成（LLM 没配 / 全部失败）→ 至少给一份真实热门榜，
            # 且回复文案会**明确说是热度榜**，不假装是个性化推荐
            cards = self._hot_cards()

        if not reply:
            reply = prompts.no_llm_reply([c.get("title") for c in cards if c.get("title")])
            degraded = True
            reason = reason or "LLM_UNAVAILABLE"

        await self._write_memory(inp.user_id, session_id, msg, reply, cards, tokens)

        out = {"session_id": session_id, "reply": reply,
               "recommend_cards": cards, "tool_calls": list(tool_names),
               "used_fallback": bool(degraded), "degraded_reason": reason,
               "tokens_used": int(tokens), "violations": violations}
        try:
            await cache.set_json(key, out, ttl=CFG.cache_ttl)
        except Exception as exc:
            logger.debug("A7 写回复缓存失败（已忽略）：%s", exc)
        return out

    # ------------------------------------------------------------ 对话循环
    async def _chat_loop(self, *, inp: ChatInput, msg: str, ctx: tools.ToolContext,
                         history: list[dict], profile_hint: str,
                         shown_titles: list[str],
                         ) -> tuple[str, list[dict], list[str], int, bool, Optional[str]]:
        """返回 `(reply, 卡片池, 工具调用名, tokens, degraded, reason)`。

        循环最多 `max_tool_calls` 轮工具往返；用尽后**摘掉 tools 再问一次**
        （否则模型会继续请求调用，而 `ToolRunner` 会抛 `60502`）。
        """
        llm = get_llm()
        if not llm.available:
            return "", [], [], 0, True, "LLM_UNAVAILABLE"

        max_calls = int(inp.max_tool_calls if inp.max_tool_calls is not None
                        else CFG.max_tool_calls)
        runner = ToolRunner(max_calls=max_calls, allow=list(tools.ALLOWED_TOOLS))
        tool_schemas = (openai_tools(list(tools.ALLOWED_TOOLS)) if max_calls > 0 else None)
        messages = prompts.build_messages(
            message=msg, history=history, profile_hint=profile_hint,
            shown_titles=shown_titles, style=inp.style)

        pool: list[dict] = []
        names: list[str] = []
        tokens = 0
        last_text = ""

        for _round in range(max_calls + 1):
            if ctx.remaining_ms(0) < CFG.min_budget_ms:
                logger.info("A7 预算不足，提前收束（remaining=%sms）", ctx.remaining_ms(0))
                break
            res = await llm.complete(
                messages, tools=tool_schemas, temperature=CFG.temperature,
                max_tokens=CFG.max_tokens,
                timeout_ms=min(ctx.remaining_ms(CFG.llm_timeout_ms), CFG.llm_timeout_ms))
            tokens += res.tokens_used
            if res.degraded:
                return last_text, pool, names, tokens, True, res.degraded_reason
            if not res.tool_calls:
                return res.text.strip(), pool, names, tokens, False, None

            messages.append({"role": "assistant", "content": res.text or None,
                             "tool_calls": res.tool_calls})
            limit_hit = False
            for tc in res.tool_calls:
                try:
                    name, payload_json = await runner.call_llm_tool_call(tc)
                except AgentError as exc:
                    name = _tool_name_of(tc)
                    if exc.name == "TOOL_CALL_LIMIT":
                        limit_hit = True
                        break
                    # 工具级错误交回模型，让它换个策略（比直接失败有用）
                    payload_json = json.dumps({"cards": [], "error": exc.message},
                                              ensure_ascii=False)
                names.append(name)
                pool.extend(self._cards_from(payload_json))
                messages.append({"role": "tool", "tool_call_id": _tool_id_of(tc),
                                 "content": payload_json})
            if limit_hit:
                break

        # 收束轮：不给 tools，强制模型基于已有信息作答
        messages.append({"role": "user", "content": prompts.FINALIZE_PROMPT})
        if ctx.remaining_ms(0) < CFG.min_budget_ms:
            return last_text, pool, names, tokens, True, "AGENT_TIMEOUT"
        res = await llm.complete(
            messages, temperature=CFG.temperature, max_tokens=CFG.max_tokens,
            timeout_ms=min(ctx.remaining_ms(CFG.finalize_timeout_ms),
                           CFG.finalize_timeout_ms))
        tokens += res.tokens_used
        if res.degraded:
            return last_text, pool, names, tokens, True, res.degraded_reason
        return res.text.strip(), pool, names, tokens, False, None

    # ------------------------------------------------------------ chat.tool_call
    async def _tool_call(self, payload: dict, env: Envelope) -> dict:
        """显式触发一次工具（无 LLM）—— 前端做"查看更多"、或测试单点验证工具时用。"""
        inp = ToolCallInput.model_validate(payload)
        if inp.tool not in tools.ALLOWED_TOOLS:
            raise invalid_param(f"未登记的工具：{inp.tool}",
                                allowed=list(tools.ALLOWED_TOOLS))
        budget_ms = int(env.header.timeout_ms or CFG.tool_timeout_ms * 2)
        ctx = tools.ToolContext(
            user_id=int(inp.user_id), session_id=inp.session_id,
            trace_id=env.header.trace_id,
            call_path=tuple(str(x) for x in (payload.get("_call_path") or ())),
            deadline=time.perf_counter() + max(budget_ms - CFG.budget_slack_ms, 200) / 1000.0,
            scene=3)
        runner = ToolRunner(max_calls=1, allow=list(tools.ALLOWED_TOOLS))
        t0 = time.perf_counter()
        with tools.bind_tool_context(ctx):
            try:
                result = await runner.call(inp.tool, dict(inp.arguments or {}))
            except AgentError as exc:
                return {"tool": inp.tool, "result": {}, "error": exc.message,
                        "elapsed_ms": int((time.perf_counter() - t0) * 1000)}
        return {"tool": inp.tool,
                "result": result if isinstance(result, dict) else {"value": result},
                "elapsed_ms": int((time.perf_counter() - t0) * 1000)}

    # ------------------------------------------------------------ 辅助
    @staticmethod
    def _cards_from(payload_json: str) -> list[dict]:
        """从工具返回 JSON 里取真实卡片。**只有带标题的才算卡片**（前端要展示）。"""
        try:
            obj = json.loads(payload_json or "{}")
        except ValueError:
            return []
        if not isinstance(obj, dict):
            return []
        cards = obj.get("cards")
        if not isinstance(cards, list):
            return []
        return [c for c in cards
                if isinstance(c, dict) and str(c.get("title") or "").strip()]

    async def _history(self, inp: ChatInput, session_id: str) -> list[dict]:
        if inp.history is not None:
            items = [h for h in inp.history if isinstance(h, dict)]
            return items[-CFG.history_limit:]
        try:
            return await memory.get_history(inp.user_id, session_id,
                                            limit=CFG.history_limit)
        except Exception as exc:
            logger.warning("A7 读短期记忆失败（按无历史处理）：%s", exc)
            return []

    async def _shown(self, user_id: int, session_id: str) -> list[int]:
        if user_id <= 0:
            return []
        try:
            return await memory.get_shown_anime_ids(user_id, session_id)
        except Exception as exc:
            logger.warning("A7 读已曝光列表失败（按空处理）：%s", exc)
            return []

    @staticmethod
    def _titles_of(src_ids: list[int]) -> list[str]:
        gw = get_gateway()
        if gw is None or not src_ids:
            return []
        try:
            meta = gw.get_animes(list(src_ids)[: CFG.shown_limit]) or {}
        except Exception as exc:
            logger.warning("A7 取已曝光标题失败：%s", exc)
            return []
        out: list[str] = []
        for m in meta.values():
            t = str((m or {}).get("title") or "").strip()
            if t:
                out.append(t)
        return out

    @staticmethod
    def _hot_cards(limit: Optional[int] = None) -> list[dict]:
        """真实热门榜卡片（降级路径唯一的候选来源）。失败返回空列表。"""
        gw = get_gateway()
        if gw is None:
            return []
        k = int(limit or CFG.cards_top_k)
        try:
            rows = gw.list_anime(limit=k, order_by="n_interactions") or []
        except Exception as exc:
            logger.warning("A7 取热门榜失败：%s", exc)
            return []
        return strategy.clamp_cards(
            strategy.dedup_cards(strategy.row_to_card(r) for r in rows), k)

    @staticmethod
    def _profile_hint(user_id: int) -> str:
        """把画像压成一行事实注入 prompt（**只是事实，不作推断**）。"""
        gw = get_gateway()
        if gw is None or user_id <= 0:
            return ""
        try:
            p = gw.get_profile(int(user_id)) or {}
        except Exception as exc:
            logger.warning("A7 读画像失败：%s", exc)
            return ""
        parts: list[str] = []
        for g in (p.get("top_genres") or [])[:3]:
            name = g.get("genre") if isinstance(g, dict) else str(g)
            if name:
                parts.append(str(name))
        if parts:
            parts = ["偏好题材：" + "、".join(parts)]
        label = p.get("activity_label")
        if label:
            parts.append(f"活跃度：{label}")
        summary = str(p.get("summary_text") or "").strip()
        if summary:
            parts.append("画像摘要：" + summary[:80])
        return "；".join(parts)

    async def _write_memory(self, user_id: int, session_id: str, msg: str,
                            reply: str, cards: list[dict], tokens: int) -> None:
        """短期（Redis，30min）+ 长期摘要（MySQL，脱敏）。失败只 warning。"""
        if user_id <= 0:
            return
        src_ids = [int(c["src_anime_id"]) for c in cards if c.get("src_anime_id")]
        try:
            await memory.append_turn(user_id, session_id, "user", msg)
            await memory.append_turn(user_id, session_id, "assistant", reply,
                                     tokens=tokens)
            await memory.record_shown(user_id, session_id, src_ids)
        except Exception as exc:
            logger.warning("A7 写短期记忆失败（已忽略）：%s", exc)
        try:
            memory.persist_summary(user_id, session_id, 0, "assistant", reply,
                                   tokens=tokens, shown_anime_ids=src_ids)
        except Exception as exc:
            logger.debug("A7 写长期摘要失败（已忽略）：%s", exc)

    # ------------------------------------------------------------ 降级
    async def fallback(self, payload: dict, env: Envelope,
                       error: AgentError) -> Optional[dict]:
        """协议 §A7：失败 → 引导话术 + 真实热门卡片（**不假装是个性化推荐**）。"""
        session_id = ""
        try:
            session_id = str(ChatInput.model_validate(payload or {}).session_id or "")
        except Exception:
            pass
        session_id = session_id or f"s{uuid.uuid4().hex[:12]}"

        return {"session_id": session_id,
                "reply": prompts.fallback_reply(error.name),
                "recommend_cards": self._hot_cards(), "tool_calls": [],
                "used_fallback": True, "degraded_reason": error.name,
                "tokens_used": 0, "violations": []}


__all__ = ["ChatAgent"]
