# -*- coding: utf-8 -*-
"""内容召回纯计算层（A3 冷启动用）。

复用 `models/content_encoder/fusion.py::ContentScorer` 的**同一份**
「用户内容画像 = 历史向量均值」口径，不另写一套 —— 否则线上冷启动的
内容分与离线 E3 实验的内容分会不一致（本项目已在"指标唯一实现"
上栽过两次，这里的处理方式相同：把口径收敛到一处）。

FAISS 的定位
-----------
文档里写的是 FAISS 近邻检索。**当前实现走精确余弦（矩阵乘）**，原因：
* 池子只有 15,687（新番子集 2,783），精确检索是 `[15688, 512]` 的一次矩阵乘，
  单次 ~5ms（CPU），比 FAISS 建索引 + 调用的固定开销还小；
* FAISS 引入一个二进制依赖，而它带来的收益只在"上百万物品"时才显现。

`docstring` 与 `docs/agents.md` 里保留 FAISS 作为**替换点**：接口是
`topk_over_subset(...)`，将来物品量上来时把内部换成 `faiss.IndexFlatIP`
即可，A3 的 agent.py 一行不用改。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

__all__ = ["content_topk", "load_scorer", "user_profile_vector"]


def load_scorer(content_matrix: torch.Tensor):
    """构造 `ContentScorer`（薄封装：让 A3 的 agent 不必 import 模型子包）。"""
    from models.content_encoder.fusion import ContentScorer
    return ContentScorer(content_matrix)


def user_profile_vector(content_matrix: torch.Tensor,
                        input_ids: torch.Tensor) -> torch.Tensor:
    """用户内容画像 `[B, D]`（历史内容向量的 L2 归一均值）。"""
    return load_scorer(content_matrix).user_profile(input_ids)


def content_topk(content_matrix: torch.Tensor, input_ids: torch.Tensor,
                 candidate_indices: Sequence[int], k: int,
                 *, exclude: Optional[Sequence[int]] = None,
                 chunk: int = 4096) -> list[list[tuple[int, float]]]:
    """在**指定候选子集**内按内容相似度取 TopK。

    参数
    ----
    content_matrix      `[V, D]`，第 0 行必须是 PAD 的零向量
    input_ids           `[B, L]` 左填充的历史（池内索引）
    candidate_indices   候选子集（池内索引）。冷启动传新番子集，
                        综合推荐传全库 —— **这正是"新番专区"与"综合推荐"
                        共用同一个 A3 的唯一区别**
    exclude             要排除的池内索引（已看）

    返回 `[b][(index, score), ...]`，与 `catalog.topk_per_interest` 同形，
    方便 A4 用同一套合并逻辑处理两路候选。

    ⚠️ `candidate_indices` 里的索引会被校验范围：越界直接丢弃而不是 clamp ——
    clamp 会让两个不同的越界物品都变成同一个合法索引，静默产生重复候选。
    """
    scorer = load_scorer(content_matrix)
    with torch.no_grad():
        profile = scorer.user_profile(input_ids)                # [B, D]
        v = scorer.content.shape[0]
        cand = [int(i) for i in candidate_indices if 0 < int(i) < v]
        excl = {int(x) for x in (exclude or [])}

        results: list[list[tuple[int, float]]] = [[] for _ in range(input_ids.shape[0])]
        if not cand:
            return results

        # 分块：候选子集可能接近全库，分块让显存与候选数解耦
        keep_idx: list[int] = []
        keep_score: list[torch.Tensor] = []
        for start in range(0, len(cand), int(chunk)):
            block = cand[start:start + int(chunk)]
            idx_t = torch.tensor(block, dtype=torch.long,
                                 device=scorer.content.device)
            vecs = scorer.content[idx_t]                        # [c, D]
            sc = torch.einsum("bd,cd->bc", profile, vecs)        # [B, c]
            keep_idx.extend(block)
            keep_score.append(sc)

        all_scores = torch.cat(keep_score, dim=1)               # [B, C]

        # 排除：置 -inf（不删列，避免下标错位）
        if excl:
            pos = [i for i, c in enumerate(keep_idx) if c in excl]
            if pos:
                all_scores[:, torch.tensor(pos, dtype=torch.long)] = float("-inf")

        k_eff = max(1, min(int(k), all_scores.shape[1]))
        vals, idxs = torch.topk(all_scores, k_eff, dim=-1)
        vals, idxs = vals.cpu(), idxs.cpu()
        for b in range(all_scores.shape[0]):
            pairs = [(keep_idx[int(idxs[b, j])], float(vals[b, j]))
                     for j in range(k_eff)]
            results[b] = [(i, s) for i, s in pairs if s != float("-inf")]
        return results
