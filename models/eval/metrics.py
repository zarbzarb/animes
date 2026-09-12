# -*- coding: utf-8 -*-
"""排序评价指标的**唯一实现**：HR@K / NDCG@K / Recall@K / MRR。

为什么必须唯一实现
------------------
指标算错一次，整篇实验的所有结论都错。按 `docs/dev-conventions.md` 的
「指标唯一实现」硬约束：所有 HR / NDCG / Recall / MRR 的计算只能来自本文件，
任何脚本、Agent、Notebook 都不得自行实现（CI 会检查其它文件里出现同类公式）。

口径来源：`docs/evaluation-plan.md` 第 5.1 节。**本文件的公式必须与文档逐字一致**，
改公式等于改实验口径，需要同步更新文档与 `experiments/SPLIT_CHANGELOG.md`。

评估协议（来自 evaluation-plan 5.2）
-----------------------------------
    候选集 = 1 个正样本 + 100 个负样本 = 101 个候选（**不是全库排序**）
    负样本 = 预生成的 random-sample_size100-seed98765.pkl，所有模型完全一致
    排名  = 从 1 开始计数（rank=1 表示排第一）
    聚合  = 对测试用户取算术平均（macro），Recall@K 例外，见下

三条硬性设计
------------
1. **纯函数、无 IO**：不读写文件、不查库、不读配置、不打印日志。
   这样才可以被 Agent 层安全复用，也能在单元测试里对拍手算结果。
2. **单一排序口径**：所有需要"取前 K 个"的指标都走同一个 `_stable_order()`，
   避免出现「HR 用一种并列规则、Recall 用另一种」这种极隐蔽的复现性 bug。
3. **并列处理是明确选择，不是偶然**：`torch.argsort(..., descending=True, stable=True)`
   是稳定排序，并列元素**保持列序**，因此列序在前的候选优先。
   评估时把正样本固定放在第 0 列，于是完全并列时正样本得 rank=1（对正样本乐观）。
   实测确认：`[1,1,1]` -> `[0,1,2]`。**这一约定必须写进论文**，
   否则别人换一种并列规则重跑会得到不同的 NDCG。

聚合方式的两处区别（容易搞错）
------------------------------
* HR / NDCG / MRR  → **macro**：先按用户算，再对用户取平均。
* Recall@K        → **micro**：按 evaluation-plan 5.1 的定义是 `Σ命中 / Σ相关`，
  对用户**不**先平均。在标准留一法（每个用户恰 1 个相关项）下两者相等，
  因此可用 `recall@K == hr@K` 作为交叉校验（见 tests/test_metrics/）。

关于空输入
----------
样本集为空时一律返回 `0.0`（而不是 `nan`），以保证 `json.dump` 输出合法 JSON。
但**空集返回 0 与真实得 0 分不可区分**，所以调用方必须同时报告样本量——
`ranking_metrics()` 已把 `n_samples` 一并返回；冷启动等子集的样本量
按 evaluation-plan 6.3 要求必须写进论文。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

__all__ = [
    "stable_order",
    "topk_indices",
    "positive_rank",
    "hr_at_k",
    "ndcg_at_k",
    "mrr",
    "recall_at_k",
    "ranking_metrics",
]


# =====================================================================
# 内部工具
# =====================================================================
def _to_rank_tensor(ranks) -> torch.Tensor:
    """把「排名」输入统一成 1D 的 float64 张量。

    接受 list / tuple / numpy 数组 / torch.Tensor 等任何可迭代形式——
    因为调用方有的来自 numpy 评估循环，有的来自 torch 批量张量。

    用 float64 而不是 int64：后面要直接做 `mean()` 与 `1/rank`，
    提前转浮点可以避免调用方忘记转换而触发整数除法。
    """
    if isinstance(ranks, torch.Tensor):
        t = ranks.detach().reshape(-1).to(torch.float64)
    else:
        t = torch.as_tensor(list(ranks), dtype=torch.float64)
    if t.numel() and bool((t < 1).any()):
        raise ValueError(
            f"rank 从 1 开始计数，出现 <1 的值（最小 {float(t.min())}）说明调用方算错了")
    return t


def stable_order(scores: torch.Tensor) -> torch.Tensor:
    """按分数降序的**稳定**排序索引，返回 [B, N] 的列下标排列。

    这是本模块所有"取前 K 个"行为的唯一来源。并列时保持列序，
    理由见模块 docstring 第 3 条。
    """
    return torch.argsort(scores, dim=-1, descending=True, stable=True)


def topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """取每行分数最高的 k 个候选的**列下标**，返回 [B, k]。

    与 `positive_rank()` 共用 `stable_order()`，保证「TopK 集合」与
    「正样本排名」两个概念用的是同一套并列规则。
    """
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    if scores.dim() != 2:
        raise ValueError(f"期望 [B, N] 的候选打分矩阵，实际 {tuple(scores.shape)}")
    if not 1 <= k <= scores.shape[1]:
        raise ValueError(f"k={k} 超出候选数范围 [1, {scores.shape[1]}]")
    return stable_order(scores)[:, :k]


# =====================================================================
# 从打分矩阵得到正样本排名
# =====================================================================
@torch.no_grad()
def positive_rank(scores: torch.Tensor, pos_index=0) -> torch.Tensor:
    """计算每个样本中正样本的排名（1-based），返回 [B] 的 int64 张量。

    参数
    ----
    scores     [B, N] 候选打分矩阵；每行是同一个用户的 N 个候选（1 正 + N-1 负）
    pos_index  正样本所在列。传 int 表示所有样本共用同一列（评估时固定为 0，
               见 evaluation-plan 5.2）；也可传长度为 B 的张量表示逐样本不同的列。

    说明
    ----
    返回的是排名本身而不是指标值，这样同一批排名可以喂给 HR / NDCG / MRR 三个
    指标，避免"每个指标各排一次序"造成口径漂移与重复计算。
    """
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    if scores.dim() != 2:
        raise ValueError(f"期望 [B, N] 的候选打分矩阵，实际 {tuple(scores.shape)}")

    b, n = scores.shape
    order = stable_order(scores)

    if isinstance(pos_index, (int,)):
        if not 0 <= pos_index < n:
            raise ValueError(f"pos_index={pos_index} 超出候选列范围 [0, {n - 1}]")
        hit = order == pos_index
    else:
        pi = torch.as_tensor(pos_index, dtype=torch.long).reshape(-1)
        if pi.numel() != b:
            raise ValueError(f"逐样本 pos_index 长度 {pi.numel()} 与批大小 {b} 不一致")
        if bool(((pi < 0) | (pi >= n)).any()):
            raise ValueError(f"逐样本 pos_index 存在超出 [0, {n - 1}] 的值")
        hit = order == pi.unsqueeze(-1)

    # argmax 返回第一个 True 的位置（0-based），+1 得到 1-based 排名
    found = hit.any(dim=-1)
    if not bool(found.all()):
        raise RuntimeError("有样本在候选矩阵中找不到正样本列，说明 pos_index 传错")
    return hit.to(torch.long).argmax(dim=-1) + 1


# =====================================================================
# 指标：输入为排名
# =====================================================================
def hr_at_k(ranks, k: int) -> float:
    """HR@K（命中率）：正样本排名 <= K 的用户占比。

        HR@K = (1/|U|) · Σ_{u∈U} 1( rank(positive_u) <= K )

    边界口径：`rank == K` **算命中**（`<=` 而不是 `<`）。
    例：`hr_at_k([10], k=10) == 1.0`，`hr_at_k([11], k=10) == 0.0`。
    """
    t = _to_rank_tensor(ranks)
    if t.numel() == 0:
        return 0.0
    return float((t <= k).to(torch.float64).mean())


def ndcg_at_k(ranks, k: int) -> float:
    """NDCG@K（归一化折损累计增益）。

        DCG@K  = Σ_{i=1..K} (2^rel_i - 1) / log2(i + 1)   rel_i = 1 当且仅当位置 i 是正样本
        IDCG@K = 1 / log2(2) = 1                          （理想：正样本排第 1）
        NDCG@K = DCG@K / IDCG@K

    在留一法（每个用户恰 1 个正样本）下上式化简为

        NDCG@K = 1 / log2(rank + 1)   若 rank <= K，否则 0

    所以本函数直接按化简式实现。例：正样本排第 2 位时 `1/log2(3) ≈ 0.6309`。
    """
    t = _to_rank_tensor(ranks)
    if t.numel() == 0:
        return 0.0
    gain = torch.where(t <= k,
                       1.0 / torch.log2(t + 1.0),
                       torch.zeros_like(t))
    return float(gain.mean())


def mrr(ranks) -> float:
    """MRR（平均倒数排名，辅助指标）：`(1/|U|) · Σ 1 / rank(positive_u)`。

    与 NDCG 的区别：MRR 不打折到 K（不设 K 上限），NDCG 有 K 与对数折损。
    """
    t = _to_rank_tensor(ranks)
    if t.numel() == 0:
        return 0.0
    return float((1.0 / t).mean())


# =====================================================================
# 指标：输入为打分矩阵（需要 TopK 集合的指标）
# =====================================================================
def recall_at_k(scores: torch.Tensor, relevant: torch.Tensor, k: int) -> float:
    """Recall@K，**micro 聚合**，用于 E4 分题材实验。

        Recall@K = Σ_u |TopK_u ∩ Relevant_u| / Σ_u |Relevant_u|

    注意是「先求和再相除」，不是「逐用户算 Recall 再平均」——
    两者在多相关项场景下结果不同，文档 5.1 的定义是前者。

    参数
    ----
    scores    [B, N] 候选打分矩阵
    relevant  [B, N] 布尔张量，True 表示该候选属于该用户的「相关集」。
              在 E4 里 `Relevant_u` 被限定为该题材的测试正样本。

    在标准留一法（每行恰 1 个 True）下本函数与 `hr_at_k` 等价，
    可用作交叉校验。
    """
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    rel = relevant if isinstance(relevant, torch.Tensor) else torch.as_tensor(relevant)
    if rel.dtype != torch.bool:
        rel = rel.to(torch.bool)
    if rel.shape != scores.shape:
        raise ValueError(
            f"relevant {tuple(rel.shape)} 与 scores {tuple(scores.shape)} 形状不一致")

    total = int(rel.sum())
    if total == 0:
        return 0.0
    idx = topk_indices(scores, k)
    hits = int(torch.gather(rel, 1, idx).sum())
    return hits / total


# =====================================================================
# 便捷入口：一次算出实验表格需要的全部指标
# =====================================================================
def ranking_metrics(ranks, ks: Sequence[int] = (5, 10)) -> dict:
    """把一批排名一次性算成实验表格用的指标字典。

    键名与 `configs/experiment.yaml` 的 `metrics` 字段对齐
    （`hr@5` / `hr@10` / `ndcg@5` / `ndcg@10` / `mrr`），
    可直接 `json.dump` 成 `metrics.json`。

    额外返回 `n_samples`：按 evaluation-plan 6.3，样本量必须随指标一起报告，
    否则无法判断「空集返回 0」还是「真的得 0 分」。
    """
    out = {}
    for k in ks:
        out[f"hr@{k}"] = hr_at_k(ranks, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranks, k)
    out["mrr"] = mrr(ranks)
    out["n_samples"] = int(_to_rank_tensor(ranks).numel())
    return out
