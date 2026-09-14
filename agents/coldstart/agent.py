# -*- coding: utf-8 -*-
"""A3 · 冷启动内容智能体（`ColdStartAgent`）。

职责（`docs/agents.md` §A3）
---------------------------
补足新番（交互数 < 10）无交互数据的召回缺口，走纯内容语义相似。

三条边界
-------
* ❌ 不参与非新番召回（`new_anime_only=False` 时才放宽，供综合推荐的补位路）
* ❌ 不做加权排序（那是 A4；本 Agent 只给 `content_score` 与 `genre_overlap`）
* ❌ 不训练模型，只推理

降级（协议 §5.2：A3 超时 150ms、失败 L3 跳过，不阻塞综合推荐）
----------------------------------------------------------
1. 内容向量 + 索引可用 → 余弦 TopK
2. 向量缺失 → 规则打分（题材重合度 0.7 + 新近度 0.3）
3. 网关不可用 → 空候选 + `degraded`
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from agents.common.base import BaseAgent
from agents.common.data import GatewayNotBound, get_gateway
from agents.common.envelope import Envelope
from agents.common.errors import AgentError, invalid_param
from agents.coldstart import adapter, strategy
from agents.coldstart.config import ColdStartConfig
from agents.coldstart.schemas import ColdStartInput

logger = logging.getLogger(__name__)

CFG = ColdStartConfig()


class ColdStartAgent(BaseAgent):
    agent_id = "A3"
    name = "coldstart"
    description = "冷启动内容召回：DistilBERT 向量 + 余弦近邻 / 规则降级"
    online = True

    async def invoke(self, payload: dict, env: Envelope) -> dict:
        action = env.header.action
        if action == "content.match_genre":
            return await self._match_genre(payload, env)
        if action != "content.retrieve":
            raise invalid_param(f"未支持的 action：{action}")
        return await self._retrieve(payload, env)

    # ------------------------------------------------------------ 召回
    async def _retrieve(self, payload: dict, env: Envelope) -> dict:
        inp = ColdStartInput.model_validate(payload)
        gw = get_gateway()
        if gw is None:
            raise GatewayNotBound("A3 需要网关取历史与新番池")

        min_year = int(inp.min_year) if inp.min_year is not None else CFG.min_year
        pool = adapter.resolve_new_anime_indices(CFG, gw, min_year=min_year)
        if not pool:
            return {"user_id": inp.user_id, "candidates": [], "n_candidates": 0,
                    "n_pool": 0, "used_fallback": True,
                    "degraded_reason": "no_cold_start_pool"}

        seq, high = await asyncio.to_thread(adapter.user_seq_indices,
                                            inp.user_id, CFG, gw)
        if inp.user_seq:
            seq = [int(x) for x in inp.user_seq]
            high = seq
        exclude = sorted(set(seq) | {int(x) for x in inp.exclude_items})

        # 用户偏好题材：优先用 A1 画像，缺省则从高分历史现算
        pref_ids = self._preferred_genre_ids(inp.profile, high, gw)
        genre_map = adapter.genre_map_of(gw)
        cand_genres = adapter.anime_genres_of(gw, pool)
        cand_meta = adapter.anime_meta_of(gw, pool)

        try:
            rows = await asyncio.to_thread(
                self._score_by_content, seq, pool, exclude, inp.top_k)
            used_fallback = False
            reason = None
        except Exception as exc:            # 向量缺失 / torch 不可用
            logger.warning("A3 内容检索不可用，转规则打分：%s", exc)
            rows = self._score_by_rules(pool, exclude, cand_genres, cand_meta,
                                        pref_ids, inp.top_k)
            used_fallback = True
            reason = "content_vector_unavailable"

        cands = self._assemble(rows, cand_genres, cand_meta, pref_ids, genre_map,
                              used_fallback)
        return {"user_id": inp.user_id, "candidates": cands,
                "n_candidates": len(cands), "n_pool": len(pool),
                "used_fallback": used_fallback, "degraded_reason": reason}

    # ------------------------------------------------------------ 题材匹配（独立 action）
    async def _match_genre(self, payload: dict, env: Envelope) -> dict:
        inp = ColdStartInput.model_validate(payload)
        gw = get_gateway()
        if gw is None:
            raise GatewayNotBound("A3 需要网关取题材")
        pool = adapter.resolve_new_anime_indices(CFG, gw)
        cand_genres = adapter.anime_genres_of(gw, pool)
        pref = [int(x) for x in (payload.get("preferred_genre_ids") or [])]
        out = []
        for idx in pool:
            ov, hit = strategy.genre_overlap(cand_genres.get(idx, []), pref)
            out.append({"anime_id": idx, "genre_overlap": ov,
                        "matched_genre_ids": hit})
        out.sort(key=lambda r: (-r["genre_overlap"], r["anime_id"]))
        return {"user_id": inp.user_id, "matches": out[:inp.top_k]}

    # ------------------------------------------------------------ 打分实现
    def _score_by_content(self, seq: list[int], pool: list[int],
                          exclude: list[int], top_k: int) -> list[tuple[int, float]]:
        import torch

        from models.retrieval.catalog import build_input_ids
        from models.retrieval.content import content_topk

        mat = adapter.load_content_matrix()
        input_ids, _ = build_input_ids(seq, CFG.max_history,
                                       device=torch.device("cpu"))
        res = content_topk(mat, input_ids, pool, top_k, exclude=exclude)
        return res[0] if res else []

    def _score_by_rules(self, pool: list[int], exclude: list[int],
                        cand_genres: dict[int, list[int]],
                        cand_meta: dict[int, dict],
                        pref_ids: list[int], top_k: int) -> list[tuple[int, float]]:
        years = [int(m["year"]) for m in cand_meta.values() if m.get("year")]
        latest = max(years) if years else 2026
        oldest = min(years) if years else 2000
        excl = set(exclude)
        out: list[tuple[int, float]] = []
        for idx in pool:
            if idx in excl:
                continue
            ov, _ = strategy.genre_overlap(cand_genres.get(idx, []), pref_ids)
            rec = strategy.recency_score(cand_meta.get(idx, {}).get("year"),
                                        latest=latest, oldest=oldest)
            out.append((idx, strategy.fallback_score(
                ov, rec, w_genre=CFG.fallback_w_genre,
                w_content=CFG.fallback_w_recency)))
        out.sort(key=lambda kv: (-kv[1], kv[0]))
        return out[:top_k]

    # ------------------------------------------------------------ 装配
    def _assemble(self, rows, cand_genres, cand_meta, pref_ids, genre_map,
                  used_fallback: bool) -> list[dict]:
        out: list[dict] = []
        for idx, score in rows:
            genres = cand_genres.get(idx, [])
            ov, hit = strategy.genre_overlap(genres, pref_ids)
            meta = cand_meta.get(idx, {})
            out.append({
                "anime_id": int(idx),
                "src_anime_id": meta.get("src_anime_id"),
                # 降级时 score 是规则分，字段名仍叫 content_score 会误导下游，
                # 所以这里显式分开：正常路径写 content_score，降级路径两者都写
                "content_score": float(score),
                "genre_overlap": ov,
                "matched_genres": [genre_map.get(g, f"genre_{g}") for g in hit],
                "matched_genre_ids": hit,
                "is_cold_start": True,
                "year": meta.get("year"),
                "source": "content_fallback" if used_fallback else "content",
            })
        return out

    def _preferred_genre_ids(self, profile: Optional[dict], high: list[int],
                             gw) -> list[int]:
        """偏好题材 id：A1 画像优先；画像缺失时用高分历史的题材分布现算。"""
        if profile:
            ids = [int(g["genre_id"]) for g in (profile.get("top_genres") or [])
                   if g.get("genre_id")]
            if ids:
                return ids
        if not high:
            return []
        counter: dict[int, int] = {}
        for gids in adapter.anime_genres_of(gw, high).values():
            for g in gids:
                counter[int(g)] = counter.get(int(g), 0) + 1
        return [g for g, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:6]]

    # ------------------------------------------------------------ 预热
    async def warmup(self) -> bool:
        """启动期预热（由 `agents.bootstrap.warmup_agents` 调用）。

        三件事本来都会算进**第一次真实请求**的 150ms 预算里：
        ① `item_index.json` 解析；② 32MB 内容向量的 mmap 首次触碰（缺页读盘）；
        ③ 第一遍 `torch` 前向要初始化线程池与算子缓存。
        实测不预热时 A3 首次响应约 **2.3s** —— 直接撞破预算、被降级成空候选，
        用户看到的现象是"第一次点新番推荐永远没有新番，刷新一下就有了"。
        这类"只在第一次出现的 bug"最难被单测发现，所以必须在启动期预热。
        """
        def _warm() -> bool:
            import torch

            from agents.recall.adapter import load_item_index
            from models.retrieval.catalog import build_input_ids
            from models.retrieval.content import content_topk

            load_item_index()
            mat = adapter.load_content_matrix()
            input_ids, _ = build_input_ids(list(range(1, 40)), CFG.max_history,
                                           device=torch.device("cpu"))
            content_topk(mat, input_ids, list(range(1, 60)), 5, exclude=[])
            return True

        try:
            return await asyncio.to_thread(_warm)
        except Exception as exc:
            logger.warning("A3 预热失败（运行时将走规则降级）：%s", exc)
            return False

    async def fallback(self, payload: dict, env: Envelope,
                       error: AgentError) -> Optional[dict]:
        """协议 §5.2：A3 失败 = L3 跳过 → 返回**空候选**而不是错误。"""
        user_id = int((payload or {}).get("user_id") or env.context.user_id or 0)
        return {"user_id": user_id, "candidates": [], "n_candidates": 0,
                "n_pool": 0, "used_fallback": True,
                "degraded_reason": error.name}


__all__ = ["ColdStartAgent"]
