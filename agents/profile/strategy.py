# -*- coding: utf-8 -*-
"""A1 的画像计算策略 —— **纯函数，无 IO、无 DB、不调 LLM**。

为什么把计算单独拆出来
--------------------
"为什么这个用户的兴趣强度是 0.42" 必须能当场复算。如果计算和取数混在一起
（在同一个函数里边查库边算），那么复核的人要起一套 MySQL 才能验证一个除法。
拆开之后：`adapter` 负责"把库里的东西变成普通 Python 结构"，
本文件负责"结构 → 画像"，测试里直接喂手工构造的 facts 就能覆盖全部分支。

四个口径必须写死的约定
---------------------
**1. strength 是「相对强度」，不是绝对次数占比。**
取用户自己的题材加权次数，除以**最大值**（不是总和）。理由：除以总和时，
一个只看热血番的用户，热血 strength 也只有 0.4 左右，前端雷达图上"最爱的题材"
看起来并不突出；除以最大值让"最爱的题材 = 1.0"，语义与前端雷达图的轴一致。
绝对占比另有 `count` 字段承载。

**2. activity_level 按总记录数分档，不按时间跨度。**
用时间跨度会把"三年前看过 200 部、最近没来"的老用户判成高活跃。
总记录数更接近"这个用户有多爱看番"这个语义。

**3. watch_intensity = 完看率（done / (done + watching + dropped)）。**
不含"想看"—— 想看是意图不是行为。
分母为 0 时返回 0.0 而不是 NaN（NaN 会污染 JSON 序列化）。

**4. trend 比较的是「近期加权」与「历史整体」的差，取值 [-1, 1]。**
近期窗口由 `ProfileConfig.recent_window_days` 决定，且**由 adapter 按数据里的
最后一个时间戳回推**（不是按 `now()`）—— 否则用陈旧的测试数据跑，
所有用户的 recent_count 都会是 0，trend 全是 -1，看起来像"全站用户集体转向"。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from agents.profile.config import ProfileConfig

__all__ = ["compute_profile", "pick_preferred_types", "pick_era_range"]

_LOW, _MEDIUM, _HIGH = 0, 1, 2
_ACTIVITY_LABEL = {_LOW: "low", _MEDIUM: "medium", _HIGH: "high"}


def _strengths(affinity: Sequence[dict]) -> list[dict]:
    """按 `weighted` 取相对强度，除以最大值。"""
    if not affinity:
        return []
    peak = max((float(a.get("weighted", 0.0)) for a in affinity), default=0.0)
    out: list[dict] = []
    for a in affinity:
        w = float(a.get("weighted", 0.0))
        count = int(a.get("count", 0))
        recent = int(a.get("recent_count", 0))
        # trend：近期占比相对历史占比的偏移，压到 [-1,1]
        if count > 0:
            recent_share = recent / max(1, count)
            trend = max(-1.0, min(1.0, round(recent_share * 2.0 - 1.0, 4)))
        else:
            trend = 0.0
        out.append({
            "genre_id": int(a.get("genre_id", 0)),
            "genre": str(a.get("genre", "")),
            "strength": round(w / peak, 4) if peak > 0 else 0.0,
            "count": count,
            "avg_rating": (round(float(a["avg_rating"]), 2)
                           if a.get("avg_rating") is not None else None),
            "trend": trend,
        })
    out.sort(key=lambda g: g["strength"], reverse=True)
    return out


def _activity_level(total: int, cfg: ProfileConfig) -> int:
    if total >= cfg.activity_high_min:
        return _HIGH
    if total >= cfg.activity_medium_min:
        return _MEDIUM
    return _LOW


def pick_preferred_types(records: Sequence[dict]) -> list[str]:
    """按出现次数排序的观看类型（TV/MOVIE/OVA…），最多 3 个。"""
    counter: dict[str, int] = {}
    for r in records:
        t = (r.get("type") or "").strip().upper()
        if t:
            counter[t] = counter.get(t, 0) + 1
    return [t for t, _ in sorted(counter.items(), key=lambda kv: -kv[1])[:3]]


def pick_era_range(records: Sequence[dict]) -> list[int]:
    """偏好年代区间：观看年份的 [P25, P75]，避免被一两部远古番带偏。"""
    years = sorted(int(r["year"]) for r in records
                   if r.get("year") and 1900 < int(r["year"]) < 2100)
    if not years:
        return []
    if len(years) < 4:
        return [years[0], years[-1]]

    def _pct(p: float) -> int:
        idx = min(len(years) - 1, max(0, int(round(p * (len(years) - 1)))))
        return years[idx]

    return [_pct(0.25), _pct(0.75)]


def compute_profile(facts: dict, cfg: ProfileConfig) -> dict:
    """`facts` 由 `adapter.collect_facts()` 产出（普通 dict）。

    期望字段
    --------
    ``user_id, affinity[], activity{}, recent_items[], records_meta[]``
    """
    user_id = int(facts.get("user_id", 0))
    affinity = facts.get("affinity") or []
    activity = facts.get("activity") or {}
    recent_items = list(facts.get("recent_items") or [])
    records_meta = list(facts.get("records_meta") or [])

    total = int(activity.get("total", 0))
    done = int(activity.get("done", 0))
    watching = int(activity.get("watching", 0))
    dropped = int(activity.get("dropped", 0))
    rated = [float(r["rating"]) for r in records_meta if r.get("rating")]
    denom = done + watching + dropped

    genres = _strengths(affinity)[:cfg.top_genres_k]
    drop_rate = round(dropped / denom, 4) if denom else 0.0
    intensity = round(done / denom, 4) if denom else 0.0
    level = _activity_level(total, cfg)

    return {
        "user_id": user_id,
        "top_genres": genres,
        "activity_level": level,
        "activity_label": _ACTIVITY_LABEL[level],
        "watch_intensity": intensity,
        "avg_rating_tendency": round(sum(rated) / len(rated), 2) if rated else None,
        "dropped_rate": drop_rate,
        "preferred_types": pick_preferred_types(records_meta),
        "preferred_era": pick_era_range(records_meta),
        "total_records": total,
        "summary_text": None,
        "user_tag": None,
        "recent_items": recent_items[:cfg.recent_items_k],
        "is_cold_start": total < cfg.cold_start_threshold,
    }
