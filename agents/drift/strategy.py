# -*- coding: utf-8 -*-
"""A6 兴趣漂移分析的统计策略 —— **纯函数**。

两个方法各司其职，不要混
-----------------------
| | JS 散度 | CUSUM |
|---|---|---|
| 回答 | "这段时间的口味与上一段**差多少**" | "从哪一刻起**开始**变了" |
| 输出 | 一个 [0,1] 的距离 | 一个时间点（变点） |
| 阈值 | `drift_threshold`（默认 0.35） | 由 `h` 控制 |

只用 JS 散度的问题是：它把每个季度独立地与前一个季度比，噪声会让
"2024Q3 与 Q2 有一点不同"被误判成漂移；只用 CUSUM 的问题是它给出变点
却给不出"变了多少"。所以**先用 CUSUM 找候选变点，再用 JS 散度确认幅度**
—— 这就是 `detect_drift_points` 的做法。

⚠️ LLM 的边界在这儿：`docs/agents.md` §A6 明写「漂移判定由统计方法定，
LLM 仅解读」。所以本模块里**一个 LLM 调用都没有**，
生成文案是 `prompts.py` + `agent.py` 的事。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

__all__ = [
    "N_GENRES",
    "cusum_changepoints",
    "detect_drift_points",
    "distribution",
    "interpretation_template",
    "js_divergence",
    "radar",
    "trend_series",
]

N_GENRES = 12


def distribution(rows: Sequence[dict], period: str, n_genres: int = N_GENRES
                 ) -> list[float]:
    """某个时间段的题材分布（概率向量，和为 1）。

    无数据的时段返回**全零向量而不是均匀分布** —— 均匀分布会与任何真实
    分布都算出约 0.3 的 JS 散度，凭空制造出"处处都在漂移"的假象。
    """
    counts = [0.0] * n_genres
    for r in rows:
        if str(r.get("period")) != str(period):
            continue
        gid = int(r.get("genre_id") or 0)
        if 1 <= gid <= n_genres:
            counts[gid - 1] += float(r.get("count") or 0)
    total = sum(counts)
    if total <= 0:
        return [0.0] * n_genres
    return [c / total for c in counts]


def _kl(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(p||q)，**跳过 q_i = 0 的位置**（否则 0*log0 → NaN）。"""
    out = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0 or qi <= 0:
            continue
        out += pi * math.log2(pi / qi)
    return out


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon 散度（以 2 为底，值域 [0,1]）。

    ⚠️ 两侧都是全零（都没数据）时返回 0 而不是 0.5。这个坑必须显式处理：
    `m = (p+q)/2` 全零时 `_kl` 返回 0，算式会给出 0 —— 看起来正确，
    但如果只是"一侧为空"，也应该返回 0（而不是把"新用户没有历史"
    说成"兴趣剧烈漂移"）。
    """
    if sum(p) <= 0 or sum(q) <= 0:
        return 0.0
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return max(0.0, min(1.0, round(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 6)))


def cusum_changepoints(series: Sequence[float], *, h: float = 0.4,
                       drift: float = 0.05) -> list[int]:
    """CUSUM 变点检测（双侧），返回变点在 `series` 中的下标。

    对**每维题材**各跑一次太碎，所以输入是"每次观测的分布"，先降成一个
    标量序列（调用方传的是"相邻分布 JS 散度"或"主题材占比"）。

    参数
    ----
    h      判定阈值（以标准差为单位会被标准化，这里用的是绝对量，
           因为输入已是 [0,1] 有界的散度值）
    drift  允许的漂移容忍（典型 0.05），越大越不敏感
    """
    if len(series) < 3:
        return []
    mean = sum(series) / len(series)
    pos = neg = 0.0
    points: list[int] = []
    for i, x in enumerate(series):
        pos = max(0.0, pos + (x - mean) - float(drift))
        neg = min(0.0, neg + (x - mean) + float(drift))
        if pos > h or neg < -h:
            points.append(i)
            pos = neg = 0.0
    return points


def trend_series(rows: Sequence[dict], *, n_genres: int = N_GENRES
                 ) -> list[dict]:
    """按时间升序的趋势序列：每个时段的总量、主题材占比、**全量题材分布**。

    返回 `[{period, total, top_genre_id, top_share, genres}, ...]`。
    `top_share` 是"主题材在当期的占比"，它的上升/下降是用户口味
    **集中化/分散化**的直接信号；`genres` 是 `{genre_id: share}`（占比和为 1），
    给前端画堆叠面积图用 —— 只给 `top_share` 画不出"口味在题材间怎么迁移"。
    """
    periods = sorted({str(r.get("period")) for r in rows if r.get("period")})
    out: list[dict] = []
    for p in periods:
        dist = distribution(rows, p, n_genres)
        total = int(sum(float(r.get("count") or 0) for r in rows
                        if str(r.get("period")) == p))
        if sum(dist) <= 0:
            out.append({"period": p, "total": total, "top_genre_id": None,
                        "top_share": 0.0, "genres": {}})
            continue
        best = max(range(n_genres), key=lambda i: dist[i])
        out.append({"period": p, "total": total, "top_genre_id": best + 1,
                    "top_share": round(dist[best], 4),
                    "genres": {i + 1: round(float(dist[i]), 4)
                               for i in range(n_genres) if dist[i] > 0}})
    return out


def detect_drift_points(rows: Sequence[dict], *, threshold: float = 0.35,
                        n_genres: int = N_GENRES) -> list[dict]:
    """漂移点 = 相邻时段 JS 散度超阈值 且 CUSUM 也认为此处有变点。

    返回 `[{period, prev_period, js, from_genre_id, to_genre_id, description}]`。
    `description` 是**纯统计描述**（不含 LLM 措辞），A6 的 LLM 只在此基础上润色。
    """
    periods = sorted({str(r.get("period")) for r in rows if r.get("period")})
    if len(periods) < 2:
        return []
    dists = [distribution(rows, p, n_genres) for p in periods]
    js_seq = [js_divergence(dists[i - 1], dists[i]) for i in range(1, len(periods))]
    cp_idx = set(cusum_changepoints(js_seq, h=max(0.2, float(threshold) * 0.8)))

    out: list[dict] = []
    for k, js in enumerate(js_seq):
        if js < float(threshold):
            continue
        # CUSUM 的下标 k 对应"第 k+1 个时段"，允许 ±1 的容差（变点常在
        # 突变的前一格被触发，因为 CUSUM 累积的是偏移量）
        if cp_idx and not ({k, k + 1} & cp_idx):
            continue
        prev, cur = dists[k], dists[k + 1]
        frm = max(range(n_genres), key=lambda i: prev[i]) + 1 if sum(prev) else None
        to = max(range(n_genres), key=lambda i: cur[i]) + 1 if sum(cur) else None
        out.append({
            "period": periods[k + 1], "prev_period": periods[k], "js": js,
            "from_genre_id": frm, "to_genre_id": to,
            "description": f"{periods[k]} → {periods[k + 1]} 分布差异 JS={js}",
        })
    return out


def radar(rows: Sequence[dict], *, n_genres: int = N_GENRES,
          latest_periods: int = 4) -> list[float]:
    """12 维雷达图数据：**最近 `latest_periods` 个时段的加权分布**。

    越近的时段权重越高（线性加权 1..n），因为雷达图要回答"**现在**的口味"，
    而不是"历史平均口味"。
    """
    periods = sorted({str(r.get("period")) for r in rows if r.get("period")})[
        -int(latest_periods):]
    if not periods:
        return [0.0] * n_genres
    acc = [0.0] * n_genres
    weight_sum = 0.0
    for w, p in enumerate(periods, start=1):
        d = distribution(rows, p, n_genres)
        if sum(d) <= 0:
            continue
        for i in range(n_genres):
            acc[i] += d[i] * w
        weight_sum += w
    if weight_sum <= 0:
        return [0.0] * n_genres
    return [round(v / weight_sum, 6) for v in acc]


def interpretation_template(radar_vec: Sequence[float],
                            points: Sequence[dict],
                            genre_map: Optional[dict[int, str]] = None) -> str:
    """**模板解读**（LLM 不可用时的兜底，也直接作为论文里的可复现基线）。

    句式与 LLM 版本一致，用户分不出哪条是降级产物。
    """
    gm = genre_map or {}

    def name(gid) -> str:
        return gm.get(int(gid), f"题材{int(gid)}") if gid else "未知"

    if not points:
        top = max(range(len(radar_vec)), key=lambda i: radar_vec[i]) if radar_vec else 0
        if sum(radar_vec) <= 0:
            return "观看记录不足，暂无法分析兴趣分布"
        return f"近期口味集中在{name(top + 1)}，整体分布较为稳定"

    p = points[-1]
    return (f"你在 {p['period']} 前后从{name(p['from_genre_id'])}"
            f"转向{name(p['to_genre_id'])}（分布差异 JS={p['js']}）")
