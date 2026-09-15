# -*- coding: utf-8 -*-
"""A0 · 调度智能体（`OrchestratorAgent`）—— 系统的"总机"。

职责（`docs/agents.md` §A0）
---------------------------
解析意图 → 编排下游 → 聚合结果 → 处理失败降级 → 写 `agent_trace`。

五条边界（`docs/agents.md` §五 的速查表）
--------------------------------------
* ❌ 不做任何向量计算（A2/A3 的事，且 ADR-1 禁止）
* ❌ 不直接查业务表（走下游 Agent）
* ❌ 不生成推荐理由文案（A5）
* ❌ 不修改用户数据
* ✅ 只做四件事：路由、编排、聚合、降级

降级（协议 §5.2：A0 800ms，自身失败直接 L4）
------------------------------------------
下游全部失败 → 返回 `fallback_popular` 热门列表并标注 `degraded`。
这是协议的最终兜底路径（L4），**前端永远拿得到一屏内容**。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from agents.common.base import BaseAgent
from agents.common.client import call_agent
from agents.common.data import get_gateway
from agents.common.envelope import Envelope
from agents.common.errors import AgentError, invalid_param
from agents.common.ports import get_runtime
from agents.orchestrator import intent as intent_mod
from agents.orchestrator.config import OrchestratorConfig
from agents.orchestrator.graph import PIPELINES, PipelineResult, Step, run_pipeline
from agents.orchestrator.schemas import OrchestratorInput

logger = logging.getLogger(__name__)

CFG = OrchestratorConfig()

# 意图 → 推荐场景（recommend_result.scene）
_SCENE = {"RECOMMEND_FEED": 0, "RECOMMEND_BY_GENRE": 1,
          "RECOMMEND_NEW_ANIME": 2, "CHAT_RECOMMEND": 3}


class OrchestratorAgent(BaseAgent):
    agent_id = "A0"
    name = "orchestrator"
    description = "调度：意图识别 + 任务编排 + 结果聚合 + 降级兜底"
    online = True

    async def invoke(self, payload: dict, env: Envelope) -> Any:
        action = env.header.action
        if action == "orchestrate.aggregate":
            return await self._aggregate_only(payload, env)
        if action != "orchestrate.route":
            raise invalid_param(f"未支持的 action：{action}")
        return await self._route(payload, env)

    # ------------------------------------------------------------ 路由
    async def _route(self, payload: dict, env: Envelope) -> dict:
        t0 = time.perf_counter()
        inp = OrchestratorInput.model_validate(payload)
        call_path = [str(x) for x in (payload.get("_call_path") or [])]

        it, conf, method, evidence = await intent_mod.classify(
            inp.raw_query or "", inp.intent_hint,
            use_llm=CFG.intent_llm_min_conf > 0,
            timeout_ms=CFG.intent_llm_timeout_ms)

        pipeline = PIPELINES.get(it)
        if pipeline is None:
            logger.warning("意图 %s 没有对应流水线，回落到综合推荐", it)
            it, pipeline = "RECOMMEND_FEED", PIPELINES["RECOMMEND_FEED"]

        chain: list[dict] = []
        deadline = t0 + CFG.total_timeout_ms / 1000.0

        async def _call(step: Step, result: PipelineResult) -> Optional[dict]:
            sub_payload = self._payload_for(step, result, inp, env)
            if sub_payload is None:
                return None
            r = await call_agent(
                from_agent="A0", to_agent=step.agent, action=step.action,
                payload=sub_payload, trace_id=env.header.trace_id,
                call_path=call_path, timeout_ms=step.timeout_ms,
                priority="P0", user_id=inp.user_id or None,
                session_id=env.context.session_id, budget_deadline=deadline)
            chain.append({"agent": step.agent, "action": step.action,
                          "ok": r.ok, "elapsed_ms": r.elapsed_ms,
                          "degraded": r.degraded,
                          "error": r.error_name or None})
            return r.payload if r.ok else None

        result = await run_pipeline(pipeline, call=_call)

        # A5 若是 optional 步骤失败了，用 A4 的信号本地补一条模板理由 ——
        # 不为此再发一次调用（A5 的 fallback 已经在 Agent 内部做过一次了）
        self._ensure_reason(result)

        out = self._aggregate(it, conf, method, result, inp, env)
        out["agent_chain"] = chain
        out["intent_evidence"] = evidence
        out["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
        return out

    async def _aggregate_only(self, payload: dict, env: Envelope) -> dict:
        """只做聚合（`orchestrate.aggregate`）：调用方已跑过流水线，
        把 `pipeline_values` 交回来让 A0 统一成形。"""
        inp = OrchestratorInput.model_validate(payload)
        it = (inp.intent or "RECOMMEND_FEED").upper()
        result = PipelineResult(values=dict(inp.pipeline_values or {}))
        return self._aggregate(it, 1.0, "provided", result, inp, env)

    # ------------------------------------------------------------ 每步入参
    def _payload_for(self, step: Step, result: PipelineResult,
                     inp: OrchestratorInput, env: Envelope) -> Optional[dict]:
        """按上游实际产出构造本步入参。**编排逻辑的唯一一处**。"""
        params = dict(inp.params or {})
        profile = result.get("A1", "profile.get") or {}
        session_id = env.context.session_id or params.get("session_id")

        if step.agent == "A1":
            return {"user_id": inp.user_id, "scope": "both"}

        if step.agent == "A2":
            return {"user_id": inp.user_id,
                    "profile": profile,
                    "exclude_items": list(params.get("exclude_items") or []),
                    "batch": bool(params.get("batch"))}

        if step.agent == "A3":
            return {"user_id": inp.user_id, "profile": profile,
                    "new_anime_only": step.action == "content.retrieve"
                    and params.get("new_anime_only", True),
                    "top_k": int(params.get("content_top_k") or 50),
                    "min_year": params.get("min_year"),
                    "exclude_items": list(params.get("exclude_items") or [])}

        if step.agent == "A4":
            a2 = result.get("A2", "recall.sequence") or {}
            a3 = result.get("A3", "content.retrieve") or {}
            cand2 = a2.get("candidates") or []
            cand3 = a3.get("candidates") or []
            if not cand2 and not cand3 and step.action == "rank.fusion":
                # 两路都空 → 交回 None，让 A0 走 L4 兜底（而不是让 A4 报 no_candidates）
                logger.info("[%s] 双路召回均为空，A0 直接走 L4 兜底",
                            env.header.trace_id)
                return None
            return {"user_id": inp.user_id, "candidate_sets": [cand2, cand3],
                    "profile": profile,
                    "top_n": int(params.get("top_n") or 20),
                    "diversity_lambda": float(params.get("diversity_lambda") or 0.7),
                    "scene": _SCENE.get(inp.intent or "RECOMMEND_FEED", 0),
                    "genre_id": params.get("genre_id"),
                    "batch_id": inp.batch_id,
                    "exclude_items": list(params.get("exclude_items") or [])}

        if step.agent == "A5":
            a4 = result.get("A4", "rank.fusion") or {}
            items = a4.get("items") or []
            if items:
                first = items[0]
                return {"user_id": inp.user_id,
                        "target_anime": {"anime_id": first.get("anime_id"),
                                         "title": first.get("title"),
                                         "genres": list(first.get("explain_signals", {})
                                                        .get("matched_genres") or []),
                                         "year": first.get("year")},
                        "explain_signals": first.get("explain_signals") or {},
                        "style": params.get("style") or "concise",
                        "standalone": False}
            # `EXPLAIN_RECOMMEND` 流水线**不含 A4** —— 单条解释的信号由调用方
            # 从 `recommend_result.explain_signals` 读出来后经 `params` 传入。
            # 没有这条回退分支时，A5 会因为"A4 产出为空"被整体跳过，
            # 表现为「点了为什么推荐却一直没有解释」，且日志里只有一条 returned_none。
            target = params.get("target_anime") or {}
            signals = params.get("explain_signals") or {}
            if not target and not signals:
                return None
            return {"user_id": inp.user_id,
                    "target_anime": target,
                    "explain_signals": signals,
                    "style": params.get("style") or "concise",
                    "standalone": bool(params.get("standalone", True))}

        if step.agent == "A6":
            return {"user_id": inp.user_id, "granularity": params.get("granularity", "quarter"),
                    "start": params.get("start"), "end": params.get("end"),
                    "drift_threshold": float(params.get("drift_threshold") or 0.35)}

        if step.agent == "A7":
            return {"user_id": inp.user_id, "session_id": session_id,
                    "message": inp.raw_query or params.get("message") or "",
                    "history": inp.context.get("history") or []}

        if step.agent == "A9":
            return {"mode": params.get("mode", "online"),
                    "date_range": params.get("date_range")}

        if step.agent == "A8":
            return {"task": params.get("task", "quality_check"), "params": params}

        return dict(params)

    # ------------------------------------------------------------ 聚合
    def _aggregate(self, it: str, conf: float, method: str, result: PipelineResult,
                   inp: OrchestratorInput, env: Envelope) -> dict:
        """把流水线产物拼成**面向业务**的载荷。"""
        profile = result.get("A1", "profile.get") or {}
        a4 = result.get("A4", "rank.fusion") or {}
        a5 = result.get("A5", "explain.generate") or {}
        items = list(a4.get("items") or [])

        # 把 A5 的理由挂到第一条上（主链路只解释 top1；逐条解释走独立接口）
        if items and a5.get("reason"):
            items[0] = {**items[0], "reason": a5["reason"],
                        "match_percent": a5.get("match_percent"),
                        "explain_source": a5.get("source")}

        result_payload: dict[str, Any] = {
            "profile_summary": profile.get("summary_text"),
            "user_tag": profile.get("user_tag"),
            "is_cold_start_user": bool(profile.get("is_cold_start")),
        }

        if it in ("RECOMMEND_FEED", "RECOMMEND_BY_GENRE", "RECOMMEND_NEW_ANIME"):
            result_payload.update({
                "scene": it,
                "items": items,
                "n_items": len(items),
                "batch_id": a4.get("batch_id"),
                # A4 命中结果缓存时它自己知道（`_run` 的早返回分支），
                # 但**只有 A0 能把它透传到面向业务的 meta** —— 不透传的话
                # `server/services/agent_bridge.py::_meta_of` 拿到的永远是
                # 信封上的默认值 False，于是对外契约里的 `meta.cache_hit`
                # 结构性地恒为 false：客户端会以为"从来没命中过缓存"。
                "cache_hit": bool(a4.get("cache_hit")),
                # 同理：A4 走了 L4 全站热门兜底（`used_fallback`）也只有
                # A0 能透传 —— 新注册零记录用户必然走这条路，前端要靠它
                # 显示"热门推荐·加记录后个性化"的横幅。
                "used_fallback": bool(a4.get("used_fallback")),
            })
        elif it == "EXPLAIN_RECOMMEND":
            result_payload.update({
                "anime_id": a5.get("anime_id"), "reason": a5.get("reason"),
                "core_items": a5.get("core_items"),
                "match_percent": a5.get("match_percent"),
                "confidence": a5.get("confidence"),
                "source": a5.get("source"),
            })
        elif it == "ANALYZE_INTEREST":
            a6 = result.get("A6", "drift.analyze") or {}
            result_payload.update({"drift": a6})
        elif it == "CHAT_RECOMMEND":
            a7 = result.get("A7", "chat.reply") or {}
            result_payload.update(a7)
        elif it == "SYSTEM_QUERY":
            a9 = result.get("A9", "eval.online_snapshot") or {}
            result_payload.update({"system": a9})
        elif it == "SEARCH_ANIME":
            result_payload.update({"items": [], "hint": "检索走 /api/v1/anime/search"})

        degraded = list(result.degraded) + list(result.errors)
        return {
            "intent": it,
            "intent_label": intent_mod.INTENT_LABELS.get(it, it),
            "confidence": round(float(conf), 3),
            "intent_method": method,
            "result": result_payload,
            "degraded": degraded,
            "trace_id": env.header.trace_id,
            "lane": list(result.chain),
        }

    def _ensure_reason(self, result: PipelineResult) -> None:
        """A5 缺位时用 A4 的信号本地补模板理由（不额外发调用）。"""
        if result.get("A5", "explain.generate"):
            return
        a4 = result.get("A4", "rank.fusion") or {}
        items = a4.get("items") or []
        if not items:
            return
        from agents.explain import prompts
        sig = items[0].get("explain_signals") or {}
        reason = prompts.fallback_reason(
            list(sig.get("top_attn_items") or []),
            list(sig.get("matched_genres") or []),
            float(sig.get("genre_overlap") or 0.0))
        result.values["A5:explain.generate"] = {
            "user_id": None, "anime_id": items[0].get("anime_id"),
            "reason": reason, "core_items": sig.get("top_attn_items") or [],
            "match_percent": prompts.match_percent(
                float(sig.get("genre_overlap") or 0.0)),
            "confidence": 0.3, "source": "template", "degraded": True,
            "violations": ["a5_step_missing"], "tokens_used": 0,
        }

    # ------------------------------------------------------------ 降级
    async def fallback(self, payload: dict, env: Envelope,
                       error: AgentError) -> Optional[dict]:
        """协议 §5.2：A0 自身失败 → **直接 L4 兜底热门列表**。"""
        gw = get_gateway()
        items: list[dict] = []
        if gw is not None:
            try:
                for rank, src in enumerate(
                        gw.popular_anime_ids(limit=CFG.anonymous_top_n), start=1):
                    items.append({"src_anime_id": int(src), "rank_no": rank,
                                  "final_score": round(1.0 / rank, 6),
                                  "explain_signals": {"fallback": "popular"}})
            except Exception as exc:
                logger.warning("A0 L4 兜底失败：%s", exc)
        return {
            "intent": "RECOMMEND_FEED", "intent_label": "综合推荐（兜底）",
            "confidence": 0.0, "intent_method": "fallback",
            "result": {"scene": "RECOMMEND_FEED", "items": items,
                       "n_items": len(items), "fallback": "popular"},
            "degraded": [{"step": "A0", "reason": error.name}],
            "trace_id": env.header.trace_id, "lane": [],
        }


__all__ = ["OrchestratorAgent"]
