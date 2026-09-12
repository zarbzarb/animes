"""嵌套用户抽样的单元测试 —— tests/test_data/test_user_subset.py

这组测试锁住四件事，任何一条被破坏，档位制就失效：

  1. 确定性   —— 同 (user_ids, ratio, seed) 必然得到同一个掩码
  2. 嵌套性   —— 小比例选中的用户必须是大比例选中用户的子集
  3. 顺序无关 —— 打乱输入顺序，被选中的用户【集合】不变
  4. 边界收敛 —— ratio=0 / 1 / 超范围、单用户、极小比例

第 2 条是"小档调参、大档直接用"的全部依据；第 3 条是历史实验可比的依据。
"""

import numpy as np
import pytest

from models.data.user_subset import (
    is_nested,
    kept_user_ids,
    mask_from_order,
    nested_user_subset,
    subset_order,
    user_hash_keys,
)


def _make_uids(n: int = 5000, upper: int = 2_000_000, seed: int = 0) -> np.ndarray:
    """生成"真实风格"的用户 id：不连续、量级与真实 uid 相近。

    真实场景下 uid 是重编号后的 0 ~ 1,306,690，中间有空洞。
    用连续整数做测试会掩盖「按位置抽样」与「按值抽样」的差异，
    因此这里特意生成不连续的 id。
    """
    rng = np.random.default_rng(seed)
    uids = np.unique(rng.integers(0, upper, size=n))
    return uids


# --------------------------------------------------------------------------
# 1. 确定性
# --------------------------------------------------------------------------
def test_same_input_same_result():
    uids = _make_uids()
    a = nested_user_subset(uids, 0.05, seed=42)
    b = nested_user_subset(uids, 0.05, seed=42)
    assert np.array_equal(a, b)


def test_hash_keys_are_reproducible():
    uids = _make_uids(200)
    assert np.array_equal(user_hash_keys(uids, seed=42), user_hash_keys(uids, seed=42))


def test_different_seed_gives_different_subset():
    uids = _make_uids()
    a = nested_user_subset(uids, 0.05, seed=42)
    b = nested_user_subset(uids, 0.05, seed=43)
    # 规模一致，但选中的不是同一批人
    assert a.sum() == b.sum()
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------
# 2. 嵌套性（本模块存在的理由）
# --------------------------------------------------------------------------
def test_nested_containment_chain():
    """0.5% ⊂ 2% ⊂ 5% ⊂ 20% ⊂ 100%"""
    uids = _make_uids()
    ratios = [0.005, 0.02, 0.05, 0.20, 1.0]
    masks = [nested_user_subset(uids, r, seed=42) for r in ratios]

    for smaller, larger in zip(masks[:-1], masks[1:]):
        assert np.all(larger[smaller]), "嵌套关系被破坏，档位制的前提失效"

    assert is_nested(masks)


def test_nesting_holds_across_many_ratios():
    """任意两个比例之间都必须满足包含关系，不只是相邻档位。"""
    uids = _make_uids()
    ratios = [0.01, 0.05, 0.1, 0.3, 0.7, 1.0]
    masks = [nested_user_subset(uids, r, seed=7) for r in ratios]

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            assert np.all(masks[j][masks[i]])


def test_is_nested_detects_violation():
    """不同 seed 抽样出来的集合不嵌套，is_nested 必须能识别。"""
    uids = _make_uids()
    small = nested_user_subset(uids, 0.05, seed=42)
    big = nested_user_subset(uids, 0.20, seed=43)
    assert not is_nested([small, big])


def test_subset_order_is_ratio_independent():
    """嵌套的根源：排序结果只依赖 (uids, seed)，与 ratio 无关。"""
    uids = _make_uids()
    assert np.array_equal(subset_order(uids, seed=42), subset_order(uids, seed=42))


# --------------------------------------------------------------------------
# 3. 顺序无关性
# --------------------------------------------------------------------------
def test_order_invariance():
    """打乱输入顺序后，选中的用户集合必须完全相同。

    这是「预处理重跑后历史实验仍然可比」的保证。
    """
    uids = _make_uids()
    mask_a = nested_user_subset(uids, 0.05, seed=42)
    selected_a = set(uids[mask_a].tolist())

    perm = np.random.default_rng(123).permutation(len(uids))
    shuffled = uids[perm]
    mask_b = nested_user_subset(shuffled, 0.05, seed=42)
    selected_b = set(shuffled[mask_b].tolist())

    assert selected_a == selected_b
    assert len(selected_a) == len(selected_b)


def test_order_invariance_for_kept_user_ids():
    uids = _make_uids()
    kept_direct = kept_user_ids(uids, 0.1, seed=42)

    perm = np.random.default_rng(9).permutation(len(uids))
    shuffled = uids[perm]
    kept_shuffled = kept_user_ids(shuffled, 0.1, seed=42)

    assert set(kept_direct.tolist()) == set(kept_shuffled.tolist())


# --------------------------------------------------------------------------
# 4. 比例精度与边界
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ratio", [0.005, 0.02, 0.05, 0.20, 0.50])
def test_ratio_accuracy(ratio):
    uids = _make_uids()
    mask = nested_user_subset(uids, ratio, seed=42)
    assert mask.sum() == int(round(len(uids) * ratio))


def test_empty_input():
    mask = nested_user_subset(np.array([], dtype=np.int64), 0.05, seed=42)
    assert mask.shape == (0,)
    assert mask.sum() == 0


def test_ratio_zero_returns_empty():
    uids = _make_uids(100)
    assert nested_user_subset(uids, 0.0, seed=42).sum() == 0


def test_ratio_one_returns_all():
    uids = _make_uids(100)
    mask = nested_user_subset(uids, 1.0, seed=42)
    assert mask.sum() == len(uids)
    assert np.all(mask)


def test_ratio_above_one_clamps_to_all():
    uids = _make_uids(100)
    assert nested_user_subset(uids, 3.0, seed=42).sum() == len(uids)


def test_tiny_ratio_keeps_at_least_one_user():
    """n * ratio 四舍五入到 0 时，仍须保留 1 个用户，不能静默产出空集。"""
    uids = _make_uids(10)
    mask = nested_user_subset(uids, 0.01, seed=42)
    assert mask.sum() == 1


def test_single_user():
    uids = np.array([12345])
    assert nested_user_subset(uids, 1.0, seed=42).tolist() == [True]
    assert nested_user_subset(uids, 0.5, seed=42).tolist() == [True]


def test_list_input_is_accepted():
    """调用方可能传 list 而非 ndarray，必须同样可用。"""
    mask = nested_user_subset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5, seed=42)
    assert isinstance(mask, np.ndarray)
    assert mask.sum() == 5


# --------------------------------------------------------------------------
# 5. 便捷封装的行为
# --------------------------------------------------------------------------
def test_kept_user_ids_matches_mask():
    uids = _make_uids(1000)
    mask = nested_user_subset(uids, 0.1, seed=42)
    assert np.array_equal(kept_user_ids(uids, 0.1, seed=42), uids[mask])


def test_kept_user_ids_preserves_input_order():
    """保序：返回的用户在输入数组中必须按下标递增出现。"""
    uids = _make_uids(1000)
    mask = nested_user_subset(uids, 0.1, seed=42)
    positions = np.flatnonzero(mask)
    kept = kept_user_ids(uids, 0.1, seed=42)

    assert len(kept) == len(positions)
    assert np.array_equal(kept, uids[positions])
    assert np.all(np.diff(positions) > 0)


def test_is_nested_single_mask_is_trivially_true():
    uids = _make_uids(100)
    assert is_nested([nested_user_subset(uids, 0.05, seed=42)])
    assert is_nested([])


# --------------------------------------------------------------------------
# 6. 复用全序（mask_from_order）—— 训练脚本会同时要训练掩码与评估掩码
# --------------------------------------------------------------------------
def test_mask_from_order_matches_direct_call():
    """复用全序的结果必须与直接调用完全一致，否则会引出难查的口径分叉。"""
    uids = _make_uids(3000)
    order = subset_order(uids, seed=42)

    for ratio in (0.02, 0.05, 0.20, 1.0):
        reused = mask_from_order(len(uids), order, ratio)
        direct = nested_user_subset(uids, ratio, seed=42)
        assert np.array_equal(reused, direct)


def test_mask_from_order_keeps_nesting():
    """一次算序、多次取比例，结果之间仍必须严格嵌套。"""
    uids = _make_uids(3000)
    order = subset_order(uids, seed=42)
    ratios = [0.005, 0.02, 0.05, 0.20, 1.0]
    masks = [mask_from_order(len(uids), order, r) for r in ratios]

    for smaller, larger in zip(masks[:-1], masks[1:]):
        assert np.all(larger[smaller])
    assert is_nested(masks)


def test_mask_from_order_edge_cases():
    uids = _make_uids(100)
    order = subset_order(uids, seed=42)
    assert mask_from_order(len(uids), order, 0.0).sum() == 0
    assert mask_from_order(len(uids), order, 1.0).all()
