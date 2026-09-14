# -*- coding: utf-8 -*-
"""A3 的取数/向量适配层。

本文件负责三件事，都是"拿到数据"而不是"算分"：
1. 装载内容向量矩阵（`mmap_mode="r"`，32MB 不该在导入期就常驻内存）
2. 解析"新番候选子集"（池内索引列表）
3. 把用户历史变成池内索引序列

评分在 `models/retrieval/content.py`，题材重合度在 `strategy.py`。
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, Optional, Sequence

from agents.coldstart.config import ColdStartConfig

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 新番判定与序列口径用到的常量（AGENTS 层不 import server，故按协议抄一份）
STATUS_SEQUENCE = (1, 2)          # 在看 / 已看


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


@lru_cache(maxsize=2)
def load_content_matrix(path: Optional[str] = None):
    """装载内容向量矩阵 `[V, D]`，第 0 行 = PAD 零向量。

    **复用 `fusion.load_content_matrix`**（唯一实现），因为它同时承担了
    「行数与 n_items 校验」与「重新 L2 归一化」两件事 —— 少做任何一件，
    余弦计算会静默给出错的分。
    """
    from models.content_encoder.fusion import load_content_matrix as _load

    from agents.recall.adapter import load_item_index
    p = _abs(path or "data/features/content_vec_512.npy")
    n_items = load_item_index()["n_items"]
    if not os.path.exists(p):
        raise FileNotFoundError(f"内容向量缺失：{p}")
    return _load(p, n_items)


def reset_content_cache() -> None:
    load_content_matrix.cache_clear()


def resolve_new_anime_indices(cfg: ColdStartConfig, gateway: Any,
                              min_year: Optional[int] = None) -> list[int]:
    """新番候选子集（**池内索引**）。

    口径是两者的并集（见 config 注释）。返回空列表 = 没有新番 →
    A3 直接返回空候选，由 A0/A4 决定是否降级。
    """
    from agents.recall.adapter import load_item_index
    m = load_item_index()["src_to_index"]

    src_ids: list[int] = []
    # ① 冷启动池表（A8 维护的权威口径）
    try:
        pool = gateway.list_cold_start_pool(limit=500, active_only=True) or []
        src_ids.extend(int(r["src_anime_id"]) for r in pool
                       if r.get("src_anime_id"))
    except Exception as exc:
        logger.debug("A3 冷启动池不可用：%s", exc)

    # ② 按年份兜底（池表为空时的实验口径）
    year = int(min_year if min_year is not None else cfg.min_year)
    try:
        rows = gateway.list_anime(limit=5000, min_year=year,
                                  cold_only=bool(cfg.new_anime_only)) or []
        src_ids.extend(int(r["src_anime_id"]) for r in rows
                       if r.get("src_anime_id"))
    except Exception as exc:
        logger.warning("A3 新番列表查询失败：%s", exc)

    seen: set[int] = set()
    out: list[int] = []
    for s in src_ids:
        idx = m.get(int(s))
        if idx is not None and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def user_seq_indices(user_id: int, cfg: ColdStartConfig, gateway: Any
                     ) -> tuple[list[int], list[int]]:
    """用户历史 → `(池内索引序列, 高评分索引列表)`。

    高分列表用于构造「兴趣内容向量」：低分番不能代表口味，
    但**rating 为空时按已看计入**（数据集里大量记录没有评分，
    如果严格要求有评分，绝大多数用户的画像会退化成空）。
    """
    from agents.recall.adapter import load_item_index
    m = load_item_index()["src_to_index"]

    recs = gateway.get_watch_records(user_id, statuses=STATUS_SEQUENCE,
                                     order="asc") or []
    animes = gateway.get_animes([r["anime_id"] for r in recs]) if recs else {}

    seq: list[int] = []
    high: list[int] = []
    for r in recs:
        src = int((animes.get(int(r["anime_id"])) or {}).get("src_anime_id") or 0)
        idx = m.get(src)
        if idx is None:
            continue
        seq.append(idx)
        rating = r.get("rating")
        if rating is None or int(rating) >= int(cfg.min_rating_for_profile):
            high.append(idx)
    return seq[-int(cfg.max_history):], high[-int(cfg.max_history):]


def genre_map_of(gateway: Any) -> dict[int, str]:
    try:
        return gateway.genre_map() or {}
    except Exception:
        return {}


def anime_genres_of(gateway: Any, indices: Sequence[int]) -> dict[int, list[int]]:
    """池内索引 → 题材 id 列表。**一次批量查询**后重排。"""
    from agents.recall.adapter import load_item_index
    inv = load_item_index()["index_to_src"]
    src_map = {int(i): inv.get(int(i)) for i in indices}
    src_ids = [s for s in src_map.values() if s]
    try:
        by_src = gateway.anime_genres(src_ids) if src_ids else {}
    except Exception as exc:
        logger.warning("A3 查题材失败：%s", exc)
        by_src = {}
    return {idx: list(by_src.get(src, []) or [])
            for idx, src in src_map.items()}


def anime_meta_of(gateway: Any, indices: Sequence[int]) -> dict[int, dict]:
    """池内索引 → `{year, title, src_anime_id}`（用于新近度与展示）。"""
    from agents.recall.adapter import load_item_index
    inv = load_item_index()["index_to_src"]
    src_map = {int(i): inv.get(int(i)) for i in indices}
    src_ids = [s for s in src_map.values() if s]
    try:
        by_src = gateway.get_animes(src_ids) if src_ids else {}
    except Exception:
        by_src = {}
    return {idx: dict(by_src.get(src) or {}) for idx, src in src_map.items()}
