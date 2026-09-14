# -*- coding: utf-8 -*-
"""A2 的召回合并策略 —— **纯函数**。

多路兴趣召回之后必须解决三个问题
-------------------------------
**1. 同一个物品被多个兴趣召回，该怎么办？**
保留最高分，但**记录它命中的所有兴趣**（用逗号列表）。理由：A4 的多样性重排
需要知道"这个候选代表着哪几路兴趣"，只留一个 interest_id 会让重排
误以为它只属于一路，从而重复曝光同一兴趣。

**2. 分数能不能跨兴趣比较？**
不能。不同兴趣向量的模长不同（`squash` 之后各自归一），内积的绝对值
不可比。所以 `strategy` **不做跨兴趣归一化** —— 归一化放在 A4 统一做
（min-max 是候选集合级的操作，那里才知道最终候选集）。这里只排序、不去分值。

**3. 兴趣标签从哪来？**
不能凭空取"兴趣 0 = 热血"。做法：对每个兴趣，统计它召回的 TopK 里
各题材的分布，取众数作为标签。数据来自 `gateway.anime_genres()`。
这比"随机贴一个题材"诚实得多，也是 A4/A5 解释链路的输入。
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

__all__ = ["merge_candidates", "label_interests", "interest_strength"]


def interest_strength(per_interest: Sequence[Sequence[tuple[int, float]]]) -> list[float]:
    """每路兴趣的**相对强度** ∈ [0, 1]，最强的那一路 = 1.0。

    口径：先取该路 TopK 的**平均分**（用平均而不是 Top1：Top1 受单个高分物品
    影响太大，而"这条路整体强不强"才是要表达的语义），再在**兴趣之间**
    做 min-max 归一。

    ⚠️ 两个必须做的处理，都是实测踩出来的：

    1. **必须减最小值再归一。** 只除以最大值会让"平均分为负"的兴趣路
       得到负强度（实测出现过 `-0.3563`）—— 前端雷达图的扇区宽度不能是负数，
       而且负值会让 A4 的"按强度分配名额"算出负数条数。
    2. **全部相等时返回全 1.0**（而不是 0/0 = NaN）。K 个兴趣都学到同一个
       方向时（训练早期）就会这样，NaN 会一路传进 JSON。

    注意这里**不做跨兴趣的分值比较**去排序候选 —— 只回答"哪路兴趣更活跃"，
    候选的最终顺序由 A4 决定（那里才知道完整候选集）。
    """
    means: list[float] = []
    for pairs in per_interest:
        vals = [float(s) for _, s in pairs]
        means.append(sum(vals) / len(vals) if vals else 0.0)
    if not means:
        return []
    lo, hi = min(means), max(means)
    if hi - lo < 1e-12:                 # 全部相等（含全 0）
        return [1.0 if hi > 0 else 0.0 for _ in means]
    return [round((m - lo) / (hi - lo), 4) for m in means]


def merge_candidates(per_interest: Sequence[Sequence[tuple[int, float]]], *,
                     exclude: Iterable[int] = (),
                     max_candidates: int = 300,
                     min_per_interest: int = 1
                     ) -> tuple[list[dict], int]:
    """合并多路召回并去重。

    返回 `(candidates, n_raw)`。每个 candidate:
    ``{anime_id, score, interest_id, interest_ids[], hit_count}``

    排序规则：`hit_count` 降序 → `score` 降序 → `anime_id` 升序。
    **`hit_count` 优先**是有意的：被多个兴趣同时召回的物品，
    说明它与用户的多个侧面都相关，通常比"某一路的偶然高分"更值得排前面。
    第三键用 `anime_id` 是为了**结果可复现**（同分同命中时排序稳定）。
    """
    excl = {int(x) for x in exclude}
    best: dict[int, dict] = {}
    n_raw = 0

    for ki, pairs in enumerate(per_interest):
        taken = 0
        for idx, score in pairs:
            idx = int(idx)
            if idx <= 0 or idx in excl:
                continue
            n_raw += 1
            taken += 1
            cur = best.get(idx)
            if cur is None:
                best[idx] = {"anime_id": idx, "score": float(score),
                             "interest_id": int(ki), "interest_ids": [int(ki)],
                             "hit_count": 1}
            else:
                cur["hit_count"] += 1
                if int(ki) not in cur["interest_ids"]:
                    cur["interest_ids"].append(int(ki))
                if float(score) > cur["score"]:
                    cur["score"] = float(score)
                    cur["interest_id"] = int(ki)
        # 每路至少保留 min_per_interest 条（若该路整体被过滤空了则忽略）
        if taken == 0 and min_per_interest > 0:
            continue

    ordered = sorted(best.values(),
                     key=lambda c: (-c["hit_count"], -c["score"], c["anime_id"]))
    return ordered[:int(max_candidates)], n_raw


def label_interests(per_interest: Sequence[Sequence[tuple[int, float]]],
                    anime_genres: dict[int, list[int]],
                    genre_map: dict[int, str],
                    top_n: int = 3) -> list[Optional[str]]:
    """给每路兴趣贴题材标签（取该路 TopN 召回里出现最多的题材）。

    `anime_genres` 与 `genre_map` 由调用方从网关取好传进来 ——
    保持本函数**零 IO**，可以在单测里直接喂手工数据。
    """
    labels: list[Optional[str]] = []
    for pairs in per_interest:
        counter: dict[int, int] = {}
        for idx, _score in pairs[:top_n]:
            for gid in anime_genres.get(int(idx), []) or []:
                counter[int(gid)] = counter.get(int(gid), 0) + 1
        if not counter:
            labels.append(None)
            continue
        # 同票数时取 genre_id 小的，保证可复现
        best_gid = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        labels.append(genre_map.get(best_gid, f"genre_{best_gid}"))
    return labels
