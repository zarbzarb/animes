# -*- coding: utf-8 -*-
"""A3 的题材重合度与规则降级打分 —— **纯函数**。

两种打分，用途完全不同，不要混
-----------------------------
| | `genre_overlap` | `fallback_score` |
|---|---|---|
| 何时用 | **总是**（正常路径也要） | 只在内容向量/索引缺失时 |
| 含义 | 候选题材与用户偏好题材的重合比例 | 重合度与新近度的加权和 |
| 量纲 | [0, 1] 可比 | 同量纲内部可比，**不与内容余弦混用** |

A3 输出的 `content_score` 与 `genre_overlap` 是**两个独立字段**：
A4 在候选交互数 < 10 时改用 5:5 权重做融合，那时需要知道"哪部分是内容语义分、
哪部分是题材重合分"，混成一个数就再也拆不开了。
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

__all__ = ["fallback_score", "genre_overlap", "matched_genres", "recency_score"]


def genre_overlap(cand_genres: Iterable[int],
                  preferred: Iterable[int]) -> tuple[float, list[int]]:
    """候选的题材里有多少落在用户偏好集合内。

    口径：`|交集| / |候选题材|`（**候选侧归一**）。

    为什么不除以并集（Jaccard）：用户偏好通常有 3~6 个题材，候选只有 1~3 个，
    并集会让分母被偏好侧主导，一个完美命中用户 top1 题材的新番只拿到
    ~0.2 分，区分度被稀释。候选侧归一的语义是"这部番的题材里，
    有多少是我爱看的"，正是冷启动要回答的问题。

    候选无题材 → 返回 `0.0`（而不是 1.0）：无信息应当是中性偏负，
    给 1.0 会让"元数据缺失的番"排到最前面。
    """
    cg = [int(g) for g in cand_genres]
    pref = {int(g) for g in preferred}
    if not cg:
        return 0.0, []
    hit = [g for g in cg if g in pref]
    return round(len(set(hit)) / len(set(cg)), 4), sorted(set(hit))


def matched_genres(cand_genres: Iterable[int], preferred: Iterable[int],
                   genre_map: dict[int, str]) -> list[str]:
    """命中的题材中文名（给 A5 解释链路用）。"""
    _, hit = genre_overlap(cand_genres, preferred)
    return [genre_map.get(g, f"genre_{g}") for g in hit]


def recency_score(year: Optional[int], *, latest: int, oldest: int) -> float:
    """新近度 ∈ [0,1]：越新越高。缺年份 → 0.5（中性，不奖不罚）。"""
    if year is None:
        return 0.5
    if latest <= oldest:
        return 1.0
    y = max(int(oldest), min(int(latest), int(year)))
    return round((y - int(oldest)) / (int(latest) - int(oldest)), 4)


def fallback_score(overlap: float, recency: float, *, w_genre: float,
                   w_content: float) -> float:
    """规则降级分 = `w_genre * 题材重合度 + w_recency * 新近度`。

    ⚠️ 这个分数**只在内容向量/索引缺失时**使用（协议 §5.2 的 A3 降级）。
    正常路径下 A3 给的是余弦相似度，两者量纲不同，**绝不能相加**：
    余弦天然 ∈ [-1,1] 且分布集中在 0.1~0.4，而规则分 ∈ [0,1] 且分布均匀，
    混在一起会让"降级的候选"排到正常候选前面 —— 用户会看到明显的质量断层。
    """
    return round(max(0.0, min(1.0,
                             float(w_genre) * float(overlap)
                             + float(w_content) * float(recency))), 6)
