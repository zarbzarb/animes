# -*- coding: utf-8 -*-
"""A4 · 融合排序智能体（`FusionRankAgent`）。

职责（`docs/agents.md` §A4）
---------------------------
把多路候选合并成唯一有序列表，并产出**结构化解释信号**（不生成自然语言）。

三条边界
-------
* ❌ 不生成自然语言（A5 的事）
* ❌ 不调 LLM（**ADR-1**）
* ❌ 不重新训练

降级（协议 §5.2：A4 超时 100ms、失败 L2 简化成纯行为分排序）
----------------------------------------------------------
1. 候选非空 → 正常融合
2. 某一路候选为空 → 只用另一路（权重自动退化为 1:0）
3. 两路都空 → **L4 全站热门兜底**（`degraded=True`），这是协议的最终兜底路径
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agents.common.base import BaseAgent
from agents.common.data import get_gateway
from agents.common.envelope import Envelope
from agents.common.errors import AgentError, invalid_param
from agents.common.ports import get_runtime
from agents.fusion import adapter, strategy
from agents.fusion.config import FusionConfig
from agents.fusion.schemas import CandidateIn, FusionInput

logger = logging.getLogger(__name__)

CFG = FusionConfig()


class FusionRankAgent(BaseAgent):
    agent_id = "A4"
    name = "fusion"
    description = "融合排序：双路加权 + MMR 多样性 + 解释信号"
    online = True

    async def invoke(self, payload: dict, env: Envelope) -> dict:
        action = env.header.action
        if action == "rank.diversify":
            return await self._diversify(payload, env)
        if action != "rank.fusion":
            raise invalid_param(f"未支持的 action：{action}")
        return await self._run(payload, env)

    # ------------------------------------------------------------ 主流程
    async def _run(self, payload: dict, env: Envelope) -> dict:
        inp = FusionInput.model_validate(payload)
        cache = get_runtime().cache
        gw = get_gateway()

        if not inp.batch_id:
            cached = await adapter.load_cached_result(
                inp.user_id, inp.scene, inp.genre_id, cache)
            if cached is not None:
                return {"user_id": inp.user_id, "items": cached,
                        "n_merged": len(cached), "n_after_filter": len(cached),
                        "used_fallback": False, "cache_hit": True}

        merged, n_merged = self._merge(inp.candidate_sets)
        if merged and gw is not None:
            facts = adapter.anime_facts(gw, [m["anime_id"] for m in merged])
        else:
            facts = {}

        kept = self._filter(merged, facts, inp)
        if not kept:
            return await self._fallback_result(inp, reason="no_candidates")

        scored = self._fuse(kept, facts, inp)
        ranked = self._select(scored, facts, inp)

        items = self._assemble(ranked, facts, inp)
        batch_id = adapter.batch_id_of(inp.user_id, inp.scene, inp.batch_id)
        await adapter.save_result(
            inp.user_id, inp.scene, inp.genre_id, items, batch_id=batch_id,
            cache=cache, gateway=gw, ttl=CFG.cache_ttl,
            ttl_minutes=CFG.result_ttl_minutes, persist=inp.persist)

        return {"user_id": inp.user_id, "items": items, "n_merged": n_merged,
                "n_after_filter": len(kept), "used_fallback": False,
                "batch_id": batch_id}

    # ------------------------------------------------------------ 合并
    def _merge(self, sets: list[list[CandidateIn]]) -> tuple[list[dict], int]:
        """把 A2（行为路）与 A3（内容路）合并成一张表。

        判据是 `source` 而不是"第几个集合"：调用方传集合的顺序不该影响结果，
        且将来加第三条路（比如热门路）时不需要改这里。
        """
        merged: dict[int, dict] = {}
        n_raw = 0
        for s in sets:
            for c in s:
                n_raw += 1
                idx = int(c.anime_id)
                row = merged.setdefault(idx, {
                    "anime_id": idx, "behavior_score": None, "content_score": None,
                    "interest_id": None, "interest_ids": [], "hit_count": 1,
                    "interest_label": None, "genre_overlap": None,
                    "matched_genres": [], "matched_genre_ids": [],
                    "src_anime_id": None, "sources": [],
                })
                row["hit_count"] = max(int(row["hit_count"]), int(c.hit_count or 1))
                if c.src_anime_id:
                    row["src_anime_id"] = int(c.src_anime_id)
                src = str(c.source or "sequence")
                if src not in row["sources"]:
                    row["sources"].append(src)

                if src == "content" or c.genre_overlap is not None:
                    # 内容路：**取最大值**而不是相加 —— 一个物品可能被
                    # 两路都以内容身份召回（新番池 ∩ 全库），相加会人为放大
                    prev = row["content_score"]
                    row["content_score"] = max(float(c.score), prev) \
                        if prev is not None else float(c.score)
                    if c.genre_overlap is not None:
                        row["genre_overlap"] = max(
                            float(c.genre_overlap),
                            float(row["genre_overlap"] or 0.0))
                    if c.matched_genres and not row["matched_genres"]:
                        row["matched_genres"] = list(c.matched_genres)
                    if c.matched_genre_ids and not row["matched_genre_ids"]:
                        row["matched_genre_ids"] = list(c.matched_genre_ids)
                    if c.is_cold_start is not None:
                        row["is_cold_start"] = bool(c.is_cold_start)
                else:
                    prev = row["behavior_score"]
                    row["behavior_score"] = max(float(c.score), prev) \
                        if prev is not None else float(c.score)
                    row["interest_id"] = (int(c.interest_id)
                                          if c.interest_id is not None
                                          else row["interest_id"])
                    if c.interest_label and not row["interest_label"]:
                        row["interest_label"] = c.interest_label
                    for i in (c.interest_ids or []):
                        if int(i) not in row["interest_ids"]:
                            row["interest_ids"].append(int(i))
                    if c.is_cold_start is not None:
                        row["is_cold_start"] = bool(c.is_cold_start)
        return list(merged.values()), n_raw

    # ------------------------------------------------------------ 过滤
    def _filter(self, merged: list[dict], facts: dict[int, dict],
                inp: FusionInput) -> list[dict]:
        """业务规则过滤（协议 §A4 第 ④ 步）。

        ⚠️ 过滤**必须在归一化之前**：先归一化再过滤会让"被过滤掉的高分物品"
        参与 min-max 的极值计算，剩下的候选全被压缩到很窄的区间，
        最终分数失去区分度（排序还在，但分数不可解释）。
        """
        excl = {int(x) for x in inp.exclude_items}
        out: list[dict] = []
        for m in merged:
            idx = int(m["anime_id"])
            if idx <= 0 or idx in excl:
                continue
            meta = facts.get(idx) or {}
            if CFG.drop_forbidden and int(meta.get("is_forbidden") or 0):
                continue
            if CFG.drop_watched and bool(meta.get("is_watched")):
                continue
            if CFG.drop_disliked and bool(meta.get("is_disliked")):
                continue
            out.append(m)
        return out

    # ------------------------------------------------------------ 融合
    def _fuse(self, kept: list[dict], facts: dict[int, dict],
              inp: FusionInput) -> list[dict]:
        b_raw = [float(m["behavior_score"] or 0.0) for m in kept]
        c_raw = [float(m["content_score"] or 0.0) for m in kept]

        # 某一路**整列为空**（全是 None）时不归一化，直接置 0.
        # 否则 minmax 会把"全 0"放大成"全 1"，让缺失的那一路凭空拿到满分。
        has_b = any(m["behavior_score"] is not None for m in kept)
        has_c = any(m["content_score"] is not None for m in kept)
        nb = strategy.minmax(b_raw, CFG.norm_eps) if has_b else [0.0] * len(kept)
        nc = strategy.minmax(c_raw, CFG.norm_eps) if has_c else [0.0] * len(kept)

        # 只有单路可用时把权重压到 1:0，避免"缺失的一路占 30% 名额"
        wb, wc = CFG.w_behavior, CFG.w_content
        if not has_c:
            wb, wc = 1.0, 0.0
        elif not has_b:
            wb, wc = 0.0, 1.0

        out: list[dict] = []
        for m, b, c in zip(kept, nb, nc):
            meta = facts.get(int(m["anime_id"])) or {}
            n_inter = meta.get("n_interactions")
            final = strategy.fuse_scores(
                b, c, n_interactions=n_inter,
                w_behavior=wb, w_content=wc,
                w_behavior_cold=CFG.w_behavior_cold if has_c else 1.0,
                w_content_cold=CFG.w_content_cold if has_b else 0.0,
                cold_threshold=CFG.cold_start_threshold)
            m = dict(m)
            m["norm_behavior"] = round(b, 6)
            m["norm_content"] = round(c, 6)
            m["final_score"] = final
            m["n_interactions"] = n_inter
            out.append(m)
        out.sort(key=lambda r: (-r["final_score"], int(r["anime_id"])))
        return out

    # ------------------------------------------------------------ 重排
    def _select(self, scored: list[dict], facts: dict[int, dict],
                inp: FusionInput) -> list[dict]:
        # MMR 的相关性输入用**归一化后的融合分**（否则量级差异会让 λ 失去意义）
        top = scored[: max(int(inp.top_n) * 4, int(inp.top_n) + 20)]

        def _rel(t: dict) -> float:
            """`_run` 走 `final_score`；`rank.diversify`（A7 传入）可能只有 `score`，
            此时退回 `score` —— 否则归一化拿到一列 0，MMR 失去相关性维度。"""
            v = t.get("final_score")
            return float(v) if v is not None else float(t.get("score") or 0.0)

        final_vals = strategy.minmax([_rel(t) for t in top])
        for t, v in zip(top, final_vals):
            # ⚠️ 这里写的是 **MMR 的输入相关性**，`final_score` 只是临时借用作载体
            # （`mmr_select` 从 `final_score` 读 rel）。它**不是最终落库的分数** ——
            # `select_with_quota` 会在选完之后用 MMR 目标值改写 `final_score`。
            # 想改口径请去 `strategy.export_final_scores`，不要在这里手工赋值。
            t["mmr_rel"] = v
            t["final_score"] = v

        genre_of = lambda i: (facts.get(i) or {}).get("genres") or []  # noqa: E731
        picked = strategy.select_with_quota(
            top, int(inp.top_n), lam=float(inp.diversity_lambda),
            genre_of=genre_of, key="interest_id",
            per_group=CFG.min_per_interest if inp.scene != 0 else CFG.min_per_interest)
        return picked

    # ------------------------------------------------------------ 装配
    def _assemble(self, ranked: list[dict], facts: dict[int, dict],
                  inp: FusionInput) -> list[dict]:
        top_attn = self._top_attn_items(inp.profile)
        top_genres = list((inp.profile or {}).get("top_genres") or [])
        out: list[dict] = []
        for rank, m in enumerate(ranked, start=1):
            idx = int(m["anime_id"])
            meta = facts.get(idx) or {}
            b_raw = m.get("behavior_score")
            c_raw = m.get("content_score")

            # 题材信号：优先用 A3 带过来的（内容召回算过余弦，更准）；
            # 没有就由 A4 自己按"画像题材 ∩ 物品题材"补 —— 行为路候选
            # 从来不带这两个字段，不补的话解释只能落到最空的那个模板。
            matched = list(m.get("matched_genres") or [])
            overlap = float(m.get("genre_overlap") or 0.0)
            if not matched:
                matched, overlap = strategy.genre_signals(
                    list(meta.get("genres") or []), top_genres)

            out.append({
                "anime_id": idx,
                "src_anime_id": meta.get("src_anime_id") or m.get("src_anime_id"),
                "rank_no": rank,
                "final_score": round(float(m.get("final_score") or 0.0), 6),
                # `norm_*` 由 `_fuse` 写入；缺失时退回原始分（防御未来新增入口）
                "behavior_score": (round(float(m.get("norm_behavior", b_raw) or 0.0), 6)
                                   if b_raw is not None else None),
                "content_score": (round(float(m.get("norm_content", c_raw) or 0.0), 6)
                                  if c_raw is not None else None),
                "interest_id": m.get("interest_id"),
                "interest_label": m.get("interest_label"),
                "is_cold_start": bool(m.get("is_cold_start") or (
                    m.get("n_interactions") is not None
                    and int(m["n_interactions"]) < CFG.cold_start_threshold)),
                "title": meta.get("title"),
                "year": meta.get("year"),
                "explain_signals": {
                    "top_attn_items": top_attn,
                    "genre_overlap": round(overlap, 4),
                    "matched_genres": matched,
                    "interest_label": m.get("interest_label"),
                    "hit_count": int(m.get("hit_count") or 1),
                    "n_interactions": m.get("n_interactions"),
                },
            })
        return out

    def _top_attn_items(self, profile: Optional[dict]) -> list[dict]:
        """Top3 历史番（解释信号的 `core_items` 来源）。

        ⚠️ 这里**不计算注意力**，而是用画像里的 `recent_items`。
        真正的注意力权重需要模型解释接口，而 A2 的召回前向已经在
        `to_thread` 里跑过一次；为了"看起来更学术"再跑一次前向取注意力，
        会让 A4 的 100ms 预算失守。用近期高分番作为"影响来源"在
        可解释性上等价，成本为 0。
        """
        if not profile:
            return []
        items = profile.get("recent_items") or []
        out: list[dict] = []
        for it in items[:3]:
            out.append({"anime_id": it.get("anime_id"),
                        "src_anime_id": it.get("src_anime_id"),
                        "title": it.get("title"),
                        "rating": it.get("rating")})
        return out

    # ------------------------------------------------------------ 独立重排入口
    async def _diversify(self, payload: dict, env: Envelope) -> dict:
        """只做多样性重排（A7 对话推荐拿到候选后调用）。**不落库、不写缓存**。

        ⚠️ 这里必须走与 `_run` **同一条管线**（merge → filter → fuse → select → assemble）。
        早期实现是把原始候选直接丢给 `_assemble`，而 `_assemble` 读的是
        `norm_behavior` / `behavior_score` 这些**只有 `_fuse` 才会写**的键 ——
        于是必然 `KeyError`。这条路径此前没有调用方（A7 是第一个），
        所以 bug 一直潜伏；也说明了"两个入口绕开同一段共享逻辑"是高风险写法。
        """
        inp = FusionInput.model_validate(payload)
        gw = get_gateway()
        merged, n_merged = self._merge(inp.candidate_sets)
        facts = (adapter.anime_facts(gw, [m["anime_id"] for m in merged])
                 if (merged and gw is not None) else {})
        kept = self._filter(merged, facts, inp)
        if not kept:
            return {"user_id": inp.user_id, "items": [], "n_merged": n_merged,
                    "n_after_filter": 0}
        scored = self._fuse(kept, facts, inp)
        ranked = self._select(scored, facts, inp)
        return {"user_id": inp.user_id,
                "items": self._assemble(ranked, facts, inp),
                "n_merged": n_merged, "n_after_filter": len(kept)}

    # ------------------------------------------------------------ 降级
    async def _fallback_result(self, inp: FusionInput, *, reason: str) -> dict:
        """L4 全局兜底：全站热门 Top20（协议 §5.1 第 4 级）。"""
        gw = get_gateway()
        items: list[dict] = []
        if gw is not None:
            try:
                from agents.recall.adapter import load_item_index
                inv = load_item_index()["index_to_src"]
                for rank, src in enumerate(
                        gw.popular_anime_ids(limit=int(inp.top_n)), start=1):
                    items.append({
                        "anime_id": inv.get(int(src)),
                        "src_anime_id": int(src),
                        "rank_no": rank, "final_score": round(1.0 / rank, 6),
                        "behavior_score": None, "content_score": None,
                        "interest_id": None, "interest_label": None,
                        "is_cold_start": False, "title": None, "year": None,
                        "explain_signals": {"top_attn_items": [],
                                            "genre_overlap": 0.0,
                                            "matched_genres": [],
                                            "fallback": "popular"},
                    })
            except Exception as exc:
                logger.warning("A4 L4 兜底失败：%s", exc)
        return {"user_id": inp.user_id, "items": items, "n_merged": 0,
                "n_after_filter": 0, "used_fallback": True,
                "degraded_reason": reason}

    async def fallback(self, payload: dict, env: Envelope,
                       error: AgentError) -> Optional[dict]:
        try:
            inp = FusionInput.model_validate(payload or {})
        except Exception:
            inp = FusionInput(user_id=int(env.context.user_id or 0))
        return await self._fallback_result(inp, reason=error.name)


__all__ = ["FusionRankAgent"]
