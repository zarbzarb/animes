# -*- coding: utf-8 -*-
"""`models/data/negatives.py` 的单元测试。

这个模块的失败模式很隐蔽：抽出来的负样本看着「像那么回事」，
但要么与用户训练物品重叠（指标虚高），要么换个评估子集就变了
（跨模型共用负样本的前提被破坏）。所以测试重点不在"抽得均匀不均匀"，
而在**这些不变量**：

1. 确定性：同 seed 同结果。
2. 顺序无关：打乱输入顺序，每个用户拿到的负样本逐字节不变。
3. 子集无关：只评一部分用户，与评全部时对应用户的负样本一致。
4. 排除规则：负样本不含该用户的训练物品，也不含正样本本身。
5. 无放回：100 个负样本互不相同。
6. 池限定：冷启动场景负样本只来自指定池。
7. 边界：池小于负样本数时降级为有放回并如实上报，不静默。
"""

from __future__ import annotations

import numpy as np
import pytest

from models.data.negatives import NegativeSampleResult, sample_negatives


# =====================================================================
# 测试夹具
# =====================================================================
def _make_train_seqs(n_users: int = 8, seed: int = 0):
    """构造一份可控的小规模训练序列。

    训练物品只从 `1..50` 里取，物品池却有 `1..80`，于是 `51..80` 这一段
    保证「不属于任何用户的训练集」—— 测试里用它当正样本，
    才能干净地区分「排除训练物品」和「排除正样本」两条规则。
    """
    rng = np.random.default_rng(seed)
    out = {}
    for u in range(n_users):
        k = int(rng.integers(12, 21))
        out[u] = rng.choice(np.arange(1, 51), size=k, replace=False).tolist()
    return out


N_ITEMS = 80
# 一定不在任何训练序列里的正样本（见 _make_train_seqs 的说明）
POS_BASE = 51


def _rows(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64)


# =====================================================================
# 1. 确定性
# =====================================================================
def test_same_seed_gives_identical_result():
    train = _make_train_seqs()
    rows = _rows(8)
    pos = np.arange(POS_BASE, POS_BASE + 8, dtype=np.int64)

    a = sample_negatives(rows, pos, N_ITEMS, n_negatives=10, seed=42, train_seqs=train)
    b = sample_negatives(rows, pos, N_ITEMS, n_negatives=10, seed=42, train_seqs=train)

    assert np.array_equal(a.negatives, b.negatives)


def test_different_seed_gives_different_result():
    """不同种子应当抽出不同结果——否则说明 seed 根本没接进随机流。"""
    train = _make_train_seqs()
    rows = _rows(8)
    pos = np.full(8, POS_BASE, dtype=np.int64)

    a = sample_negatives(rows, pos, N_ITEMS, n_negatives=10, seed=42, train_seqs=train)
    b = sample_negatives(rows, pos, N_ITEMS, n_negatives=10, seed=43, train_seqs=train)

    assert not np.array_equal(a.negatives, b.negatives)


# =====================================================================
# 2. 顺序无关 / 3. 子集无关
# =====================================================================
def test_row_order_does_not_change_per_row_negatives():
    """打乱输入顺序后，同一个用户行拿到的负样本必须不变。

    这是「落盘缓存后跨模型复用」的前提：如果负样本跟遍历顺序绑定，
    评估时按用户分块并行就会得到与串行不同的结果。
    """
    train = _make_train_seqs()
    rows = _rows(8)
    pos = np.arange(POS_BASE, POS_BASE + 8, dtype=np.int64)

    straight = sample_negatives(rows, pos, N_ITEMS, n_negatives=10, seed=7, train_seqs=train)

    perm = np.array([5, 0, 7, 2, 1, 6, 3, 4], dtype=np.int64)
    shuffled = sample_negatives(rows[perm], pos[perm], N_ITEMS,
                                n_negatives=10, seed=7, train_seqs=train)

    # 逐用户比对：shuffled 的第 k 行对应 straight 的第 perm[k] 行
    for k, src in enumerate(perm):
        assert np.array_equal(shuffled.negatives[k], straight.negatives[src]), \
            f"用户行 {src} 的负样本在打乱顺序后发生了变化"


def test_subset_gives_identical_rows_as_full_run():
    """只评估一部分用户时，这些用户的负样本必须与全量评估时完全一致。"""
    train = _make_train_seqs()
    rows = _rows(8)
    pos = np.arange(POS_BASE, POS_BASE + 8, dtype=np.int64)

    full = sample_negatives(rows, pos, N_ITEMS, n_negatives=10, seed=11, train_seqs=train)

    keep = np.array([2, 5, 7], dtype=np.int64)
    sub = sample_negatives(rows[keep], pos[keep], N_ITEMS,
                           n_negatives=10, seed=11, train_seqs=train)

    for k, src in enumerate(keep):
        assert np.array_equal(sub.negatives[k], full.negatives[src])


# =====================================================================
# 4. 排除规则
# =====================================================================
def test_excludes_user_train_items():
    train = _make_train_seqs()
    rows = _rows(8)
    pos = np.arange(POS_BASE, POS_BASE + 8, dtype=np.int64)

    res = sample_negatives(rows, pos, N_ITEMS, n_negatives=10, seed=3, train_seqs=train)

    for i, row in enumerate(rows):
        overlap = set(res.negatives[i].tolist()) & set(train[int(row)])
        assert not overlap, f"用户行 {row} 的负样本含其训练物品：{sorted(overlap)}"


def test_excludes_positive_item():
    """正样本不能同时作为负样本，否则 BCE 的监督信号自相矛盾。

    这里正样本取自 `POS_BASE` 段（保证不属于任何用户的训练集），
    因此一旦它出现在负样本里，只可能是「排除正样本」这条规则失效，
    不会与「排除训练物品」的失败混淆。
    """
    train = _make_train_seqs()
    rows = _rows(8)
    pos = np.arange(POS_BASE, POS_BASE + 8, dtype=np.int64)

    res = sample_negatives(rows, pos, N_ITEMS, n_negatives=10, seed=3, train_seqs=train)

    for i in range(len(rows)):
        assert int(pos[i]) not in res.negatives[i].tolist()


def test_exclude_switches_can_be_turned_off():
    """关掉排除后，负样本必然与训练物品重叠——用来证明排除确实生效。

    训练集刻意覆盖 1..60（池 1..80 的 75%），使抽 20 个负样本时
    一次都不撞上的概率低到可以忽略（约 1e-12），避免测试本身变成碰运气。
    """
    train = {u: list(range(1, 61)) for u in range(4)}
    rows = _rows(4)
    pos = np.array([71, 72, 73, 74], dtype=np.int64)

    res = sample_negatives(rows, pos, N_ITEMS, n_negatives=20, seed=5,
                           train_seqs=train, exclude_train=False, exclude_positive=False)

    overlapped = any(set(res.negatives[i].tolist()) & set(train[i]) for i in range(4))
    assert overlapped, "关闭排除后仍无重叠，说明排除逻辑可能没被真正触发过"


def test_missing_train_seqs_raises():
    with pytest.raises(ValueError, match="train_seqs"):
        sample_negatives(_rows(2), np.array([61, 62], dtype=np.int64), N_ITEMS)


# =====================================================================
# 5. 无放回
# =====================================================================
def test_negatives_are_distinct_within_a_sample():
    train = _make_train_seqs()
    rows = _rows(8)
    pos = np.arange(POS_BASE, POS_BASE + 8, dtype=np.int64)

    res = sample_negatives(rows, pos, N_ITEMS, n_negatives=20, seed=9, train_seqs=train)

    for i in range(len(rows)):
        vals = res.negatives[i].tolist()
        assert len(set(vals)) == len(vals), f"第 {i} 行存在重复负样本"
    assert res.n_shortfall == 0
    assert res.as_report()["replacement"] == "without"


# =====================================================================
# 6. 池限定（冷启动）
# =====================================================================
def test_pool_restricts_candidates():
    """E3 冷启动要求负样本与新番同分布，池必须被严格限定。"""
    train = _make_train_seqs()
    rows = _rows(8)
    pos = np.arange(POS_BASE, POS_BASE + 8, dtype=np.int64)
    new_pool = np.arange(POS_BASE, POS_BASE + 20, dtype=np.int64)   # 20 个「新番」

    res = sample_negatives(rows, pos, N_ITEMS, n_negatives=10, seed=4,
                           train_seqs=train, pool=new_pool)

    assert res.pool_size == 20
    assert set(res.negatives.ravel().tolist()) <= set(new_pool.tolist())


def test_pool_out_of_range_raises():
    with pytest.raises(ValueError, match="越界"):
        sample_negatives(_rows(2), np.array([1, 2], dtype=np.int64), N_ITEMS,
                         pool=np.array([1, 2, 999], dtype=np.int64),
                         train_seqs=_make_train_seqs())


# =====================================================================
# 7. 边界
# =====================================================================
def test_small_pool_falls_back_to_with_replacement_and_reports_it():
    """池比负样本数还小时，「无放回」不可能成立，必须降级并如实上报。"""
    train = {0: [1, 2, 3], 1: [4, 5, 6]}
    rows = _rows(2)
    pos = np.array([7, 8], dtype=np.int64)
    tiny_pool = np.array([11, 12, 13, 14, 15], dtype=np.int64)   # 仅 5 个

    res = sample_negatives(rows, pos, N_ITEMS, n_negatives=10, seed=2,
                           train_seqs=train, pool=tiny_pool)

    assert res.n_shortfall == 2
    assert res.as_report()["replacement"] == "partially_with"
    assert res.negatives.shape == (2, 10)
    # 即便降级，仍然只能来自池内
    assert set(res.negatives.ravel().tolist()) <= set(tiny_pool.tolist())


def test_shortfall_after_excluding_many_items():
    """池够大但被训练物品挤掉后不足，也应降级并上报。"""
    # 用户训练集吃掉池里 90 个物品中的 85 个，只剩 5 个可选
    train = {0: list(range(1, 86))}
    pool = np.arange(1, 91, dtype=np.int64)
    res = sample_negatives(_rows(1), np.array([86], dtype=np.int64), 90, n_negatives=10,
                           seed=1, train_seqs=train, pool=pool)
    assert res.n_shortfall == 1


def test_empty_rows_returns_empty_array():
    res = sample_negatives([], [], N_ITEMS, n_negatives=10, train_seqs={})
    assert isinstance(res, NegativeSampleResult)
    assert res.negatives.shape == (0, 10)
    assert res.n_rows == 0


def test_index_range_is_within_pool():
    train = _make_train_seqs()
    rows = _rows(8)
    pos = np.arange(POS_BASE, POS_BASE + 8, dtype=np.int64)
    res = sample_negatives(rows, pos, N_ITEMS, n_negatives=10, seed=6, train_seqs=train)
    assert int(res.negatives.min()) >= 1
    assert int(res.negatives.max()) <= N_ITEMS
    assert res.negatives.dtype == np.int32


def test_dtype_is_int32_to_keep_cache_small():
    """1.31 亿个负样本，int64 会白占 524MB；索引上限 15,687，int32 足够。"""
    res = sample_negatives(_rows(3), np.arange(POS_BASE, POS_BASE + 3, dtype=np.int64),
                           N_ITEMS, n_negatives=5, train_seqs=_make_train_seqs())
    assert res.negatives.dtype == np.int32


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="长度必须一致"):
        sample_negatives(_rows(3), np.array([1, 2], dtype=np.int64),
                         N_ITEMS, train_seqs=_make_train_seqs())


def test_non_positive_n_negatives_raises():
    with pytest.raises(ValueError, match="n_negatives"):
        sample_negatives(_rows(1), np.array([1], dtype=np.int64), N_ITEMS,
                         n_negatives=0, train_seqs=_make_train_seqs())
