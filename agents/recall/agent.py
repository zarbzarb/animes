# -*- coding: utf-8 -*-
"""A2 · 序列召回智能体（`RecallAgent`）。

职责（`docs/agents.md` §A2）
---------------------------
主召回源。SASRec / 多兴趣胶囊 从全库物品里筛候选。

三条边界
-------
* ❌ 不做最终排序（A4 的事）
* ❌ 不调 LLM（**架构 ADR-1：LLM 不参与召回与排序**）
* ❌ 不写业务表（本 Agent 只读）

降级链（协议 §5.2：A2 超时 200ms，失败走 L2 换 ItemCF 热门榜）
-----------------------------------------------------------
1. 模型可用 → 纯模型召回
2. 模型/映射文件缺失、CUDA OOM → **热门榜**（`gateway.popular_anime_ids`）
3. 网关也不可用 → 空候选 + `degraded`（A4 会走它自己的 L4 兜底）

⚠️ 第 2 步的降级必须**如实标注** `used_fallback=True`：
论文里"降级率"这个指标就靠它，虚报会让线上表现看起来比离线好。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from agents.common.base import BaseAgent
from agents.common.data import get_gateway
from agents.common.envelope import Envelope
from agents.common.errors import AgentError, invalid_param, make
from agents.recall import adapter, strategy
from agents.recall.config import RecallConfig
from agents.recall.schemas import RecallInput

logger = logging.getLogger(__name__)

CFG = RecallConfig()

# WatchRecord.status：序列口径只看「在看 / 已看」（想看不算看过）
STATUS_SEQUENCE = (1, 2)


class RecallAgent(BaseAgent):
    agent_id = "A2"
    name = "recall"
    description = "序列召回：SASRec + 多兴趣胶囊 / 热门降级"
    online = True

    def __init__(self) -> None:
        super().__init__()
        self._model_name: Optional[str] = None
        self._model_error: Optional[str] = None

    # ------------------------------------------------------------ 入口
    async def invoke(self, payload: dict, env: Envelope) -> dict:
        action = env.header.action
        if action == "recall.batch":
            return await self._run(payload, env, batch=True)
        if action == "recall.sequence":
            return await self._run(payload, env, batch=False)
        raise invalid_param(f"未支持的 action：{action}")

    # ------------------------------------------------------------ 主流程
    async def _run(self, payload: dict, env: Envelope, *, batch: bool) -> dict:
        inp = RecallInput.model_validate(payload)
        user_id = inp.user_id

        seq, exclude = self._build_inputs(inp)
        if len(seq) < CFG.min_seq_len:
            # 历史太短 → 不硬跑模型（冷启动用户由 A3 负责）
            logger.info("[%s] A2 序列过短（%d），走降级",
                        env.header.trace_id, len(seq))
            return self._fallback_result(user_id, exclude, reason="seq_too_short")

        model_name = self._ensure_model()
        if model_name is None:
            return self._fallback_result(user_id, exclude, reason="model_unavailable")

        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(adapter.recall_sequence, seq, CFG,
                                  model_name=model_name,
                                  exclude_indices=exclude),
                timeout=CFG.forward_timeout_ms / 1000.0)
        except asyncio.TimeoutError:
            logger.warning("[%s] A2 前向超时（%dms）",
                           env.header.trace_id, CFG.forward_timeout_ms)
            return self._fallback_result(user_id, exclude, reason="forward_timeout")
        except AgentError:
            raise
        except Exception as exc:            # CUDA OOM / 权重不匹配等
            logger.warning("[%s] A2 前向失败：%s", env.header.trace_id, exc)
            return self._fallback_result(user_id, exclude, reason="forward_failed")

        per_interest = raw["per_interest"]
        cands, n_raw = strategy.merge_candidates(
            per_interest, exclude=exclude, max_candidates=CFG.max_candidates,
            min_per_interest=CFG.min_per_interest)
        if not cands:
            return self._fallback_result(user_id, exclude, reason="empty_after_merge")

        cands = self._attach_src_and_labels(cands, per_interest)

        return {
            "user_id": user_id,
            "candidates": cands,
            "n_raw": n_raw,
            "n_after_dedup": len(cands),
            "model_ver": raw.get("model_ver"),
            "interest_strength": strategy.interest_strength(per_interest),
            "used_fallback": False,
        }

    # ------------------------------------------------------------ 输入构造
    def _build_inputs(self, inp: RecallInput) -> tuple[list[int], list[int]]:
        """返回 `(池内索引序列, 要排除的池内索引)`。"""
        gw = get_gateway()
        if inp.user_seq:
            seq = [int(x) for x in inp.user_seq]
            seen = set(seq)
            exclude = sorted(seen | {int(x) for x in inp.exclude_items})
            return seq[-CFG.input_cap:], exclude

        if gw is None:
            raise AgentError(make("DB_UNAVAILABLE").code,
                             "A2 需要网关取用户序列，但网关未注入")
        records = gw.get_watch_records(inp.user_id, statuses=STATUS_SEQUENCE,
                                       order="asc")
        animes = gw.get_animes([r["anime_id"] for r in records]) if records else {}
        src_ids = [int((animes.get(int(r["anime_id"])) or {}).get("src_anime_id") or 0)
                   for r in records]
        src_ids = [s for s in src_ids if s > 0]
        seq = adapter.src_to_item_index(src_ids)
        exclude = sorted(set(seq) | {int(x) for x in inp.exclude_items})
        return seq[-CFG.input_cap:], exclude

    # ------------------------------------------------------------ 模型
    def _ensure_model(self) -> Optional[str]:
        """注册 + **显式加载**模型，返回模型名（不可用则 None）。

        ⚠️ 必须在**前向预算之外**完成加载：`ModelRegistry.get()` 是延迟加载的，
        一个冻结的 multi_interest 权重在 CPU 上要 ~2s。如果等
        `recall_sequence()` 内部再触发加载，那 2s 会被算进 150ms 的前向预算，
        结果是**每个进程的第一个请求都假降级** —— 这正是本项目刚踩过的坑
        （表现为"日志说前向超时，但直接测明明是 11ms"）。
        """
        if self._model_name:
            return self._model_name
        if self._model_error:
            return None
        try:
            from agents.common.registry import ModelRegistry

            name = adapter.register_recall_model(CFG)
            ModelRegistry.get(name)          # 立即加载（预热的语义）
            self._model_name = name
        except Exception as exc:        # checkpoint 缺失 / 加载失败
            self._model_error = str(exc)
            logger.warning("A2 召回模型不可用（将走热门降级）：%s", exc)
        return self._model_name

    async def warmup(self) -> bool:
        """启动期预热（`server/main.py` 的 lifespan 调用）。

        分两件事：① 加载权重；② **跑一次空序列前向** —— 第一遍 torch 前向会
        初始化线程池与算子缓存，实测冷启动 27ms、热态 11ms。
        不预热的话，第一个真实请求会因为这 16ms 的差额在 200ms 预算里
        边缘抖动。
        """
        name = await asyncio.to_thread(self._ensure_model)
        if name is None:
            return False
        try:
            await asyncio.to_thread(adapter.recall_sequence, list(range(1, 40)),
                                    CFG, model_name=name, exclude_indices=[])
            return True
        except Exception as exc:            # pragma: no cover
            logger.warning("A2 预热失败：%s", exc)
            return False

    # ------------------------------------------------------------ 回译与打标
    def _attach_src_and_labels(self, cands: list[dict],
                               per_interest: list) -> list[dict]:
        idxs = [c["anime_id"] for c in cands]
        src_map = dict(zip(idxs, adapter.item_index_to_src(idxs)))

        labels: list[Optional[str]] = []
        genre_map: dict[int, str] = {}
        anime_genres: dict[int, list[int]] = {}
        gw = get_gateway()
        if gw is not None:
            try:
                genre_map = gw.genre_map() or {}
                src_ids = [s for s in src_map.values() if s]
                # **一次**批量查询后按 idx 重排：逐个查会打 N 次 DB
                by_src = gw.anime_genres(src_ids) if src_ids else {}
                anime_genres = {idx: by_src.get(src, [])
                                for idx, src in src_map.items()}
            except Exception as exc:
                logger.warning("A2 取题材失败（标签置空）：%s", exc)
        labels = strategy.label_interests(per_interest, anime_genres, genre_map)

        out: list[dict] = []
        for c in cands:
            c = dict(c)
            c["src_anime_id"] = src_map.get(c["anime_id"])
            iid = int(c["interest_id"])
            c["interest_label"] = labels[iid] if iid < len(labels) else None
            c["source"] = "sequence"
            out.append(c)
        return out

    # ------------------------------------------------------------ 降级
    def _fallback_result(self, user_id: int, exclude: list[int],
                         *, reason: str) -> dict:
        """热门榜降级。**没有任何模型参与**（`docs/agents.md` §A2 降级策略）。"""
        gw = get_gateway()
        cands: list[dict] = []
        if gw is not None:
            try:
                inv = adapter.load_item_index()["index_to_src"]
                for idx, score in adapter.popular_candidates(CFG, gw, exclude):
                    cands.append({"anime_id": idx, "score": score, "interest_id": 0,
                                  "interest_ids": [0], "hit_count": 1,
                                  "source": "fallback_hot",
                                  "src_anime_id": inv.get(idx),
                                  "interest_label": None})
            except Exception as exc:
                logger.warning("A2 热门降级也失败：%s", exc)
        return {"user_id": user_id, "candidates": cands, "n_raw": len(cands),
                "n_after_dedup": len(cands), "model_ver": None,
                "interest_strength": [], "used_fallback": True,
                "degraded_reason": reason}

    async def fallback(self, payload: dict, env: Envelope,
                       error: AgentError) -> Optional[dict]:
        """协议 §5.2：A2 失败 → L2 换热门榜（而不是把错误抛给 A0）。"""
        user_id = int((payload or {}).get("user_id") or env.context.user_id or 0)
        return self._fallback_result(user_id, [], reason=error.name)


__all__ = ["RecallAgent"]
