# -*- coding: utf-8 -*-
"""A5 · 解释生成智能体（`ExplainAgent`）。

职责（`docs/agents.md` §A5）
---------------------------
把 A4 的结构化信号转成用户能读懂的推荐理由 —— "可解释推荐"这个创新点
的落地点。

三条边界（第一条是硬约束）
-----------------------
* ❌ **不得新增未在信号中出现的番剧或题材**（`prompts.validate_reason` 兜）
* ❌ 不改变排序
* ❌ 不编造用户行为

降级（协议 §5.2：A5 超时 200ms、失败 L2 换模板文案）
--------------------------------------------------
LLM 超时/未配置/输出不合格 → **模板兜底**（`prompts.fallback_reason`）。
这条路径**必须**能覆盖所有情况：解释可能在主链路的 200ms 预算内失败，
而"没有解释"比"解释不够精彩"严重得多 —— 推荐卡片会空一块。
"""

from __future__ import annotations

import logging
from typing import Optional

from agents.common.base import BaseAgent
from agents.common.data import get_gateway
from agents.common.envelope import Envelope
from agents.common.errors import AgentError, invalid_param
from agents.common.llm import get_llm
from agents.common.ports import get_runtime
from agents.explain import prompts
from agents.explain.config import ExplainConfig
from agents.explain.schemas import ExplainInput

logger = logging.getLogger(__name__)

CFG = ExplainConfig()


class ExplainAgent(BaseAgent):
    agent_id = "A5"
    name = "explain"
    description = "解释生成：结构化信号 → 自然语言理由（LLM + 模板兜底）"
    online = True

    async def invoke(self, payload: dict, env: Envelope) -> dict:
        action = env.header.action
        if action not in ("explain.generate", "explain.stream"):
            raise invalid_param(f"未支持的 action：{action}")
        return await self._generate(payload, env)

    # ------------------------------------------------------------ 主流程
    async def _generate(self, payload: dict, env: Envelope) -> dict:
        inp = ExplainInput.model_validate(payload)
        signals = dict(inp.explain_signals or {})
        top_attn = list(signals.get("top_attn_items") or [])[:CFG.core_items_k]
        matched = list(signals.get("matched_genres") or [])
        overlap = float(signals.get("genre_overlap") or 0.0)
        hit_count = int(signals.get("hit_count") or 1)

        allowed_titles = [t.get("title") for t in top_attn if t.get("title")]
        if inp.target_anime.title:
            allowed_titles.append(inp.target_anime.title)

        pct = prompts.match_percent(overlap, hit_count)
        template = prompts.fallback_reason(top_attn, matched, overlap, style=inp.style)

        timeout = CFG.standalone_timeout_ms if inp.standalone else CFG.timeout_ms
        reason, source, tokens, model, violations, degraded = await self._try_llm(
            inp, signals, top_attn, matched, allowed_titles, pct, template, timeout)

        out = {
            "user_id": inp.user_id,
            "anime_id": inp.target_anime.anime_id,
            "reason": reason,
            "core_items": top_attn,
            "match_percent": pct,
            "confidence": self._confidence(overlap, hit_count, source),
            "source": source,
            "style": inp.style,
            "prompt_ver": prompts.PROMPT_VERSION,
            "llm_model": model,
            "tokens_used": tokens,
            "degraded": degraded,
            "violations": violations,
        }
        # ⚠️ **只持久化 LLM 产出的解释**，模板兜底一律不落库。
        #
        # `recommend_explain` 的主键是 `(user_id, anime_id, style)`，**没有 TTL**
        # —— 它是一张永久缓存。把模板兜底也写进去的后果是：
        # LLM 只要抖动一次（或一开始就没配 key），这条推荐的解释就被
        # **永久冻结**在最空的那句模板上；等 LLM 恢复了，`explain_one`
        # 依然先命中缓存直接返回旧模板，用户再也看不到 LLM 的解释，
        # 而且没有任何报错。实测就是这么踩到的。
        #
        # 不落库的代价接近零：模板是纯字符串拼装（本地 ~1ms、无网络），
        # 每次重算的结果与缓存完全一致，不存在"浪费一次 LLM 调用"的问题。
        if inp.persist and out.get("source") == "llm":
            self._persist(inp, out)
        return out

    # ------------------------------------------------------------ LLM
    async def _try_llm(self, inp: ExplainInput, signals: dict, top_attn: list,
                       matched: list, allowed_titles: list, pct: int,
                       template: str, timeout: int
                       ) -> tuple[str, str, int, Optional[str], list, bool]:
        llm = get_llm()
        if not llm.available:
            return template, "template", 0, None, [], True
        if not allowed_titles and not matched:
            # 信号里既没有番也没有题材 → LLM 只能编，直接模板
            return template, "template", 0, None, [], True

        target = {"title": inp.target_anime.title,
                  "genres": list(inp.target_anime.genres)}
        res = await llm.complete(
            prompts.build_messages(target, {**signals, "top_attn_items": top_attn,
                                            "matched_genres": matched}, inp.style),
            temperature=CFG.llm_temperature, max_tokens=CFG.llm_max_tokens,
            timeout_ms=timeout)
        if not res.ok:
            return template, "template", 0, None, [], True

        ok, cleaned, violations = prompts.validate_reason(
            res.text, allowed_titles=allowed_titles, allowed_genres=matched,
            max_len=CFG.max_reason_len)
        if not ok:
            logger.info("A5 事实校验拦下 LLM 输出（%s），走模板兜底",
                        ",".join(violations))
            return template, "template", res.tokens_used, res.model, violations, True
        return cleaned, "llm", res.tokens_used, res.model, [], False

    # ------------------------------------------------------------ 辅助
    @staticmethod
    def _confidence(overlap: float, hit_count: int, source: str) -> float:
        """置信度：信号强度 × 来源可信度。

        模板兜底的置信度**封顶 0.5** —— 它是保底方案，不是等价方案。
        前端用这个值决定"要不要把理由显示为弱化样式"。
        """
        base = 0.4 * max(0.0, min(1.0, float(overlap))) + 0.15 * min(4, hit_count)
        base = max(0.0, min(1.0, base))
        return round(base if source == "llm" else min(base, 0.5), 4)

    def _persist(self, inp: ExplainInput, out: dict) -> None:
        gw = get_gateway()
        if gw is None or inp.target_anime.anime_id is None:
            return
        try:
            gw.save_explain(inp.user_id, int(inp.target_anime.anime_id), {
                "reason": out["reason"], "core_items": out["core_items"],
                "match_percent": out["match_percent"],
                "matched_genres": (inp.explain_signals or {}).get("matched_genres"),
                "style": inp.style,
                "source": 0 if out["source"] == "llm" else 1,   # EXPLAIN_LLM / TEMPLATE
                "prompt_ver": out["prompt_ver"], "llm_model": out["llm_model"],
                "tokens_used": out["tokens_used"],
            })
        except Exception as exc:
            logger.warning("A5 解释落库失败（已忽略）：%s", exc)

    # ------------------------------------------------------------ 降级
    async def fallback(self, payload: dict, env: Envelope,
                       error: AgentError) -> Optional[dict]:
        """协议 §5.2：A5 失败 → 模板文案（**任何情况下都必须有理由**）。"""
        signals = dict((payload or {}).get("explain_signals") or {})
        target = (payload or {}).get("target_anime") or {}
        style = str((payload or {}).get("style") or "concise")
        top_attn = list(signals.get("top_attn_items") or [])[:CFG.core_items_k]
        matched = list(signals.get("matched_genres") or [])
        overlap = float(signals.get("genre_overlap") or 0.0)
        return {
            "user_id": int((payload or {}).get("user_id")
                           or env.context.user_id or 0),
            "anime_id": target.get("anime_id"),
            "reason": prompts.fallback_reason(top_attn, matched, overlap, style=style),
            "core_items": top_attn,
            "match_percent": prompts.match_percent(overlap),
            "confidence": 0.3, "source": "template", "style": style,
            "prompt_ver": prompts.PROMPT_VERSION, "llm_model": None,
            "tokens_used": 0, "degraded": True,
            "violations": [error.name],
        }


__all__ = ["ExplainAgent"]
