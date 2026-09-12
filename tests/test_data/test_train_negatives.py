# -*- coding: utf-8 -*-
"""`sample_train_negatives()` 的单元测试（训练侧负采样）。

它与评估侧的 `sample_negatives()` 是**两个不同口径**（见
`models/data/negatives.py` 模块 docstring 的对照表），所以测试也分开写。
这一份只锁训练侧的四条性质：

1. 索引合法（落在 `[1, n_items]`，永不为 PAD）；
2. **永不等于正样本**（否则同一物品既正又负，BCE 目标自相矛盾）；
3. 同 seed 可复现、传 rng 时按序消耗；
4. 分布大致均匀（否则会系统性偏向某些物品，等价于引入热度先验）。
"""

from __future__ import annotations

import numpy as np
import pytest

from models.data.negatives import DEFAULT_NEG_SEED, sample_train_negatives

N_ITEMS = 100


# =====================================================================
# 1. 形状与索引合法性
# =====================================================================
def test_shape_and_dtype():
    out = sample_train_negatives([5, 9, 11], n_items=N_ITEMS, seed=0)
    assert out.shape == (3,)
    assert out.dtype == np.int64


def test_indices_within_pool():
    t = np.random.default_rng(0).integers(1, N_ITEMS + 1, size=5000)
    out = sample_train_negatives(t, n_items=N_ITEMS, seed=1)
    assert out.min() >= 1
    assert out.max() <= N_ITEMS


def test_never_selects_pad_item():
    """PAD = 0 绝不能作为负样本 —— 它在嵌入表里是"空位"，不是物品。"""
    t = np.ones(2000, dtype=np.int64)          # 全是同一个 target，压力更集中
    out = sample_train_negatives(t, n_items=N_ITEMS, seed=2)
    assert (out != 0).all()


def test_empty_input_returns_empty():
    out = sample_train_negatives([], n_items=N_ITEMS, seed=0)
    assert out.shape == (0,)
    assert out.dtype == np.int64


# =====================================================================
# 2. 永不等于正样本（最重要）
# =====================================================================
def test_never_equals_target():
    """随机 10 万条，逐条核对负样本 != 正样本。

    若这条失败，最小化 BCE 会让模型学到"正负样本是同一个物品"，
    loss 会卡在一个无意义的常数上，而且**不会报错**。
    """
    t = np.random.default_rng(3).integers(1, N_ITEMS + 1, size=100_000)
    out = sample_train_negatives(t, n_items=N_ITEMS, seed=3)
    assert (out != t).all(), f"{(out == t).sum()} 条负样本撞上了正样本"


def test_boundary_target_at_n_items():
    """target == n_items（上界）时同样不能被抽中，且索引不越界。"""
    t = np.full(5000, N_ITEMS, dtype=np.int64)
    out = sample_train_negatives(t, n_items=N_ITEMS, seed=4)
    assert (out != N_ITEMS).all()
    assert out.min() >= 1 and out.max() <= N_ITEMS


def test_single_item_pool_is_rejected():
    """池里只有 1 个物品时，"避开正样本"在数学上不可能，必须显式报错。"""
    with pytest.raises(ValueError, match="至少为 2"):
        sample_train_negatives([1], n_items=1, seed=0)


# =====================================================================
# 3. 可复现性与 rng 语义
# =====================================================================
def test_same_seed_is_reproducible():
    t = [3, 7, 11, 42, 99]
    a = sample_train_negatives(t, n_items=N_ITEMS, seed=7)
    b = sample_train_negatives(t, n_items=N_ITEMS, seed=7)
    assert np.array_equal(a, b)


def test_different_seed_differs():
    t = list(range(1, N_ITEMS + 1))
    a = sample_train_negatives(t, n_items=N_ITEMS, seed=7)
    b = sample_train_negatives(t, n_items=N_ITEMS, seed=8)
    assert not np.array_equal(a, b)


def test_default_seed_is_used_when_nothing_given():
    """不传 rng、不传 seed 时，结果由 DEFAULT_NEG_SEED 决定 —— 依然可复现。"""
    t = [1, 2, 3]
    a = sample_train_negatives(t, n_items=N_ITEMS)
    b = sample_train_negatives(t, n_items=N_ITEMS, seed=DEFAULT_NEG_SEED)
    assert np.array_equal(a, b)


def test_passed_rng_is_consumed_in_order():
    """显式传 rng 时，两次调用应当**不同**（rng 状态被推进）。

    这是训练侧有意为之的语义：一个 epoch 内每个 batch 的负样本都不重复。
    若这里失败，说明函数内部自行新建了 rng，训练会看到同一批负样本。
    """
    rng = np.random.default_rng(123)
    t = list(range(1, 101))
    a = sample_train_negatives(t, n_items=N_ITEMS, rng=rng)
    b = sample_train_negatives(t, n_items=N_ITEMS, rng=rng)
    assert not np.array_equal(a, b)

    # 重建同样的 rng 状态后，两次调用的结果必须逐位复现
    rng2 = np.random.default_rng(123)
    a2 = sample_train_negatives(t, n_items=N_ITEMS, rng=rng2)
    b2 = sample_train_negatives(t, n_items=N_ITEMS, rng=rng2)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)


# =====================================================================
# 4. 越界输入必须报错（不静默产出坏索引）
# =====================================================================
def test_pad_target_is_rejected():
    """target 收到 PAD(0) 时当场报错，而不是产出一个越界/无意义的索引。"""
    with pytest.raises(ValueError, match="越界"):
        sample_train_negatives([0, 5], n_items=N_ITEMS, seed=0)


def test_out_of_range_target_is_rejected():
    with pytest.raises(ValueError, match="越界"):
        sample_train_negatives([1, N_ITEMS + 1], n_items=N_ITEMS, seed=0)


# =====================================================================
# 5. 分布均匀性
# =====================================================================
def test_distribution_is_roughly_uniform():
    """10 万条样本、100 个物品，各物品被抽中的次数应接近均值。

    判据用相对偏差而不是卡方检验：目标是挡住"负采样偏向热门物品"
    这类**系统性**错误（那会让模型靠热度先验而非序列偏好得分），
    而不是把统计噪声判成 bug。均值 1000 时，±30% 已是极宽的容忍带。
    """
    n, c = 100_000, N_ITEMS
    t = np.random.default_rng(9).integers(1, c + 1, size=n)
    out = sample_train_negatives(t, n_items=c, seed=9)
    counts = np.bincount(out, minlength=c + 1)[1:]     # 丢掉 PAD 位
    expect = n / c
    assert counts.sum() == n
    assert counts.min() > expect * 0.7, f"最少 {counts.min()}，期望约 {expect:.0f}"
    assert counts.max() < expect * 1.3, f"最多 {counts.max()}，期望约 {expect:.0f}"


def test_target_exclusion_does_not_bias_the_rest():
    """当所有 target 都相同时，被排除的那个物品出现 0 次，**其余仍应均匀**。

    ⚠️ 这条测试实际抓到过一个 bug：早期实现用「环绕 +1」避让正样本，
    于是所有撞车样本都被转移给了 `target + 1`，使该物品被抽中的次数
    变成其他物品的两倍。真实数据下（target 分布多样）这点偏差小到
    可忽略，所以指标上看不出来；但只要某个 target 在批内高度集中，
    偏差就会成规模。实测本用例：`target+1` 命中 2036 次 vs 期望 1020。
    现实现改为**重抽**，本测试即为它的回归防线。
    """
    c = 50
    t = np.full(50_000, 7, dtype=np.int64)
    out = sample_train_negatives(t, n_items=c, seed=11)
    counts = np.bincount(out, minlength=c + 1)
    assert counts[7] == 0                      # 正样本被完全排除
    others = np.delete(counts[1:], 6)          # 去掉 target=7 那一格
    expect = out.size / (c - 1)
    assert others.min() > expect * 0.8, \
        f"最少 {others.min()}，期望约 {expect:.0f}（避让引入的偏差）"
    assert others.max() < expect * 1.2, \
        f"最多 {others.max()}，期望约 {expect:.0f}（避让引入的偏差）"


def test_adjacent_item_is_not_over_sampled():
    """专门盯 `target ± 1` 这两个"最容易被迫害"的物品。

    "环绕 +1" 那版实现会让 `target+1` 翻倍；任何"只往某一侧避让"的写法
    也都会在这里露馅。这是上一条测试的定向加强版，因为相邻偏移
    正是这类实现最典型的失败模式。
    """
    c = 50
    t = np.full(40_000, 25, dtype=np.int64)
    out = sample_train_negatives(t, n_items=c, seed=13)
    counts = np.bincount(out, minlength=c + 1)
    expect = out.size / (c - 1)
    for item in (24, 26):                      # target 的两个邻居
        ratio = counts[item] / expect
        assert 0.8 < ratio < 1.2, \
            f"物品 {item} 被抽中 {counts[item]} 次，期望约 {expect:.0f}（比值 {ratio:.2f}）"
