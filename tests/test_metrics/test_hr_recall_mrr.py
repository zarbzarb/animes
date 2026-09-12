# -*- coding: utf-8 -*-
"""MRR、Recall@K 与「指标口径一致性」测试。

重点验证三件容易出错的事：
1. Recall@K 是 **micro**（Σ命中/Σ相关），不是逐用户平均 —— 两者在多相关项时结果不同。
2. 留一法下 `Recall@K` 必须恒等于 `HR@K`（互为交叉校验）。
3. 需要 TopK 集合的指标与 `positive_rank` 用的是同一套并列规则。
"""

import math

import torch

from models.eval.metrics import (
    hr_at_k,
    mrr,
    positive_rank,
    recall_at_k,
    topk_indices,
)


# ---------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------
def test_mrr_hand_calculation():
    """MRR = 平均倒数排名：ranks=[1,2,4] -> (1 + 1/2 + 1/4)/3"""
    expected = (1.0 + 0.5 + 0.25) / 3.0
    assert abs(mrr([1, 2, 4]) - expected) < 1e-12


def test_mrr_perfect_and_worst():
    assert mrr([1]) == 1.0
    assert abs(mrr([2]) - 0.5) < 1e-12
    assert abs(mrr([100]) - 0.01) < 1e-12


def test_mrr_has_no_k_cutoff():
    """MRR 不打折到 K：rank=100 仍得 1/100，而 NDCG@10 为 0（与 NDCG 的区别）"""
    from models.eval.metrics import ndcg_at_k
    assert mrr([100]) > 0
    assert ndcg_at_k([100], k=10) == 0.0


# ---------------------------------------------------------------------
# Recall@K（micro）
# ---------------------------------------------------------------------
def test_recall_micro_hand_calculation():
    """单用户：4 个候选里 2 个相关，Top2 命中 1 个 -> 1/2"""
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    relevant = torch.tensor([[True, False, False, True]])
    assert abs(recall_at_k(scores, relevant, k=2) - 0.5) < 1e-12


def test_recall_is_micro_not_macro():
    """本用例专门区分 micro 与 macro —— 两者的结果不同，文档定义的是 micro。

    用户 1：1 个相关项，Top2 命中 → 用户级 recall = 1.0
    用户 2：2 个相关项，Top2 全没命中 → 用户级 recall = 0.0

        micro = (1 + 0) / (1 + 2) = 1/3 ≈ 0.3333   ← 文档 5.1 的定义
        macro = (1.0 + 0.0) / 2   = 0.5            ← 错误的做法

    如果实现里先对用户平均，这个断言会失败。
    """
    scores = torch.tensor([
        [0.9, 0.1, 0.1, 0.1],      # 用户 1：Top2 = [0,1]
        [0.9, 0.8, 0.1, 0.1],      # 用户 2：Top2 = [0,1]
    ])
    relevant = torch.tensor([
        [True,  False, False, False],   # 用户 1 的相关项在第 0 列
        [False, False, True,  True],    # 用户 2 的相关项在第 2、3 列
    ])
    got = recall_at_k(scores, relevant, k=2)
    assert abs(got - 1.0 / 3.0) < 1e-12
    assert abs(got - 0.5) > 1e-6, "若等于 0.5 说明实现成了 macro 平均"


def test_recall_equals_hr_under_leave_one_out():
    """留一法（每用户恰 1 个相关项）下 Recall@K 必须恒等于 HR@K。

    这是两条独立代码路径的交叉校验：HR 走 positive_rank 得到的排名，
    Recall 走 topk_indices 得到的 TopK 集合。若两者并列规则不一致会在此暴露。
    """
    torch.manual_seed(0)
    for _ in range(20):
        b, n = 8, 101
        scores = torch.randn(b, n)
        pos_col = torch.randint(0, n, (b,))
        # 构造「正样本在第 pos_col 列」的候选矩阵，并让正样本排进前若干名
        relevant = torch.zeros(b, n, dtype=torch.bool)
        relevant[torch.arange(b), pos_col] = True

        ranks = positive_rank(scores, pos_col)
        for k in (5, 10):
            assert abs(recall_at_k(scores, relevant, k) - hr_at_k(ranks, k)) < 1e-12


def test_recall_empty_relevant_returns_zero():
    scores = torch.tensor([[0.1, 0.2]])
    relevant = torch.tensor([[False, False]])
    assert recall_at_k(scores, relevant, k=1) == 0.0


def test_recall_shape_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        recall_at_k(torch.zeros(2, 3), torch.zeros(2, 4, dtype=torch.bool), k=1)


def test_recall_accepts_non_bool_dtype():
    """relevant 传 0/1 整数也应可用（调用方可能来自 numpy bool 或 int 掩码）"""
    scores = torch.tensor([[0.9, 0.8, 0.7]])
    assert abs(recall_at_k(scores, torch.tensor([[1, 0, 0]]), k=1) - 1.0) < 1e-12


# ---------------------------------------------------------------------
# TopK 与排名的口径一致性
# ---------------------------------------------------------------------
def test_topk_and_rank_share_tie_policy():
    """全并列时：Top2 = [0,1]，因此正样本若在第 0 或 1 列，rank 必 <= 2"""
    scores = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    idx = topk_indices(scores, 2)
    assert idx.tolist() == [[0, 1]]
    assert positive_rank(scores, 0).item() == 1
    assert positive_rank(scores, 1).item() == 2
    assert positive_rank(scores, 2).item() == 3


def test_topk_k_out_of_range_raises():
    import pytest
    with pytest.raises(ValueError):
        topk_indices(torch.zeros(1, 4), k=5)
    with pytest.raises(ValueError):
        topk_indices(torch.zeros(1, 4), k=0)
