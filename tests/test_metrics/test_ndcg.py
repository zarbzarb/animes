# -*- coding: utf-8 -*-
"""NDCG@K 与 HR@K 的正确性测试（指标算错则整篇实验全错，见 dev-conventions 4.2）。

测试策略是**两路对拍**：
1. 手工算例：数字由人按公式算好写死在断言里，可读、可复核。
2. 暴力实现对拍：把 `evaluation-plan.md` 5.1 的**通用公式**（含 IDCG）
   直接翻译成循环实现，与 `metrics.py` 里的化简式 `1/log2(rank+1)` 全量比对。
   化简式一旦推错，这里就会暴露——这是本文件最重要的一个测试。
"""

import math

import pytest

from models.eval.metrics import hr_at_k, ndcg_at_k


# ---------------------------------------------------------------------
# 1. 手工算例（与 dev-conventions.md 4.3 的示例保持同名同值）
# ---------------------------------------------------------------------
def test_ndcg_hand_calculation():
    """与手工计算对齐：正样本排第 2 位时 NDCG@5 = 1/log2(3)"""
    ranks = [2]           # 正样本排名（1-based）
    expected = 1.0 / math.log2(3)
    assert abs(ndcg_at_k(ranks, k=5) - expected) < 1e-9


def test_ndcg_perfect_ranking():
    """正样本排第 1 位时应为 1.0"""
    assert abs(ndcg_at_k([1], k=10) - 1.0) < 1e-9


def test_hr_at_k_boundary():
    assert hr_at_k([10], k=10) == 1.0
    assert hr_at_k([11], k=10) == 0.0


# ---------------------------------------------------------------------
# 2. 更多手工算例（覆盖 K 的两个档位与多用户平均）
# ---------------------------------------------------------------------
def test_ndcg_rank_equals_k_is_counted():
    """rank == K 要算命中，折损为 1/log2(K+1)"""
    assert abs(ndcg_at_k([10], k=10) - 1.0 / math.log2(11)) < 1e-12


def test_ndcg_beyond_k_is_zero():
    """rank > K 时折损为 0（不是"很小的正数"）"""
    assert ndcg_at_k([11], k=10) == 0.0
    assert ndcg_at_k([100], k=5) == 0.0


def test_hr_macro_average():
    """HR 是 macro：先按用户判定，再对用户取平均"""
    # 3 个用户，rank 1/2/11，K=10 时前两个命中
    assert abs(hr_at_k([1, 2, 11], k=10) - 2.0 / 3.0) < 1e-12


def test_ndcg_macro_average():
    ranks = [1, 2, 6]                       # 第 3 个用户的 rank=6 > K=5，记 0
    expected = (1.0 + 1.0 / math.log2(3) + 0.0) / 3.0
    assert abs(ndcg_at_k(ranks, k=5) - expected) < 1e-12


def test_perfect_ranking_all_users():
    """全部用户都排第 1 → HR 与 NDCG 都为 1"""
    assert hr_at_k([1] * 50, k=5) == 1.0
    assert abs(ndcg_at_k([1] * 50, k=5) - 1.0) < 1e-12


# ---------------------------------------------------------------------
# 3. 暴力对拍：通用 DCG/IDCG 公式 vs 化简式
# ---------------------------------------------------------------------
def _ndcg_general_formula(rank: int, k: int, n_candidates: int = 200) -> float:
    """按 evaluation-plan.md 5.1 的通用公式暴力实现（测试专用，不复用产品代码）。

        DCG@K  = Σ_{i=1..K} (2^rel_i - 1) / log2(i + 1)
        IDCG@K = 理想排序（正样本在第 1 位）下的 DCG@K
        NDCG@K = DCG@K / IDCG@K
    """
    n = max(n_candidates, k, rank)
    rel = [1 if (i + 1) == rank else 0 for i in range(n)]   # 位置 i+1 是否正样本
    lim = min(k, n)

    dcg = sum((2 ** rel[i] - 1) / math.log2(i + 2) for i in range(lim))
    # 理想排序：第 1 位是正样本，其余为负
    idcg = sum(((2 ** 1 - 1) if i == 0 else 0) / math.log2(i + 2) for i in range(lim))
    return dcg / idcg if idcg > 0 else 0.0


@pytest.mark.parametrize("k", [5, 10])
@pytest.mark.parametrize("rank", list(range(1, 21)))
def test_ndcg_matches_general_formula(rank, k):
    """化简式 NDCG = 1/log2(rank+1) 必须等于通用公式的结果。

    这是本文件的核心测试：如果化简推导有误（例如折损分母写成 log2(rank)），
    手工算例可能只覆盖到某一两个 rank，而这里把 rank 1~20 × K∈{5,10} 全覆盖。
    """
    assert abs(ndcg_at_k([rank], k=k) - _ndcg_general_formula(rank, k)) < 1e-12


def test_ndcg_is_rank_monotonic():
    """排名越靠前得分越高（同一组用户内不应出现反转）"""
    scores = [ndcg_at_k([r], k=10) for r in range(1, 11)]
    assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 1))


# ---------------------------------------------------------------------
# 4. 输入形式与边界
# ---------------------------------------------------------------------
def test_accepts_list_tuple_and_tensor():
    import torch
    for form in ([3], (3,), torch.tensor([3])):
        assert abs(ndcg_at_k(form, k=5) - 1.0 / math.log2(4)) < 1e-12


def test_empty_input_returns_zero_not_nan():
    """空集返回 0.0 而不是 nan —— 保证 json.dump 输出合法 JSON。

    注意：这与"真的得 0 分"不可区分，所以调用方必须同时报告样本量
    （ranking_metrics 已返回 n_samples）。
    """
    for fn in (hr_at_k, ndcg_at_k):
        v = fn([], k=10)
        assert v == 0.0 and not math.isnan(v)


def test_invalid_rank_raises():
    """rank 从 1 开始，出现 0 或负数说明调用方算错了"""
    with pytest.raises(ValueError):
        hr_at_k([0], k=10)
    with pytest.raises(ValueError):
        ndcg_at_k([-1], k=10)
