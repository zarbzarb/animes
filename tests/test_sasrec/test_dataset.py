# -*- coding: utf-8 -*-
"""`models/sasrec/dataset.py` 的单元测试。

本文件按重要性排序：

1. **因果性**（最重要）—— 第 t 个样本不得看到 t 及其后的任何物品。
   这里用「只改未来、断言过去样本逐字节不变」的方式锁死，
   而不是「人工检查一眼切片下标」：后者在重构时防不住回归。
2. 窗口口径 —— 总数 = Σ train_len；窗口号 ↔ (行号, t) 一一对应。
3. 填充与截断 —— 左填充、尾部对齐、超长取最近 cap 条。
4. 评估输入构造 —— val 用 train，test 用 train+val。
5. 剥离（E3 冷启动）—— 新番从序列中消失，窗口数随之下降。
"""

from __future__ import annotations

import numpy as np
import torch

from models.sasrec.dataset import (
    PAD_ITEM,
    TARGET_SLOTS,
    SlidingWindowDataset,
    build_eval_inputs,
    count_windows,
    make_padded_id_array,
)

# 测试统一用小 max_seq_len 以便手算：cap = 6 - 2 = 4
MAX_SEQ_LEN = 6
CAP = MAX_SEQ_LEN - TARGET_SLOTS


def _seqs():
    """三个可控序列：短（4）、长（10）、刚好等于 cap（4）。"""
    return {
        0: [1, 2, 3, 4],
        1: list(range(1, 11)),      # 1..10
        2: [11, 12, 13, 14],
    }


# =====================================================================
# 1. 因果性 —— 本文件最重要的测试
# =====================================================================
def test_no_sample_can_see_the_future():
    """把序列的**未来**整段改掉，前面每个样本的输入与 target 都必须不变。

    这是对「在线滑窗」最容易写错的地方（比如误用 `seq[:t+1]`、
    或把 target 顺手拼进输入）的硬拦截。若这条失败，离线指标会虚高，
    而且因为指标"看起来更好了"，人工 review 很难发现。
    """
    base = list(range(1, 11))                 # 1..10
    mutated = [1, 2, 3, 4, 5, 999, 998, 997, 996, 995]   # 从第 6 个物品起全部不同

    ds_base = SlidingWindowDataset({0: base}, max_seq_len=MAX_SEQ_LEN)
    ds_mut = SlidingWindowDataset({0: mutated}, max_seq_len=MAX_SEQ_LEN)

    assert len(ds_base) == len(ds_mut) == 10

    # t = 0..4 这 5 个样本只依赖前 5 个物品，必须逐字节相同
    for t in range(5):
        x_a, y_a = ds_base[t]
        x_b, y_b = ds_mut[t]
        assert torch.equal(x_a, x_b), f"第 {t} 个样本的输入被未来改动影响了"
        assert int(y_a) == int(y_b) == base[t]

    # t = 5 的 target 就是被改动的位置，必须不同（否则说明上面比的是错东西）
    assert int(ds_base[5][1]) == 6
    assert int(ds_mut[5][1]) == 999


def test_target_never_appears_in_its_own_input():
    """逐样本断言：target 不在输入里，且输入元素都来自 target 之前。"""
    seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ds = SlidingWindowDataset({0: seq}, max_seq_len=MAX_SEQ_LEN)

    for t in range(len(seq)):
        x, y = ds[t]
        vals = x.tolist()
        target = int(y)
        assert target == seq[t]
        assert target not in vals, f"第 {t} 个样本的输入里出现了自己的 target {target}"
        # 输入里的非 PAD 元素必须都是 t 之前的物品（即值小于 target 的位置）
        seen = [v for v in vals if v != PAD_ITEM]
        assert set(seen) <= set(seq[:t]), \
            f"第 {t} 个样本输入含 t 之前的序列之外的元素：{seen}"


def test_first_window_has_empty_prefix():
    """t=0 时没有任何历史，整行都是 PAD —— 这条约定让"冷序列"行为可预测。"""
    ds = SlidingWindowDataset(_seqs(), max_seq_len=MAX_SEQ_LEN)
    x, y = ds[0]
    assert (x == PAD_ITEM).all()
    assert int(y) == 1


# =====================================================================
# 2. 窗口口径
# =====================================================================
def test_window_count_equals_sum_of_train_len():
    """窗口总数必须等于 Σ train_len —— 这是与阶段一 109,081,471 对账的口径。"""
    ds = SlidingWindowDataset(_seqs(), max_seq_len=MAX_SEQ_LEN)
    assert len(ds) == 4 + 10 + 4 == ds.n_windows
    assert ds.summary()["n_windows"] == 18


def test_row_of_maps_window_back_to_user_and_position():
    ds = SlidingWindowDataset(_seqs(), max_seq_len=MAX_SEQ_LEN)
    expect = [(0, 0), (0, 1), (0, 2), (0, 3),
              (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
              (2, 0), (2, 1), (2, 2), (2, 3)]
    for i, (row, t) in enumerate(expect):
        assert ds.row_of(i) == (row, t), f"窗口 {i} 的行/位置映射错误"


def test_row_of_out_of_range_raises():
    ds = SlidingWindowDataset(_seqs(), max_seq_len=MAX_SEQ_LEN)
    try:
        ds.row_of(len(ds))
    except IndexError:
        return
    raise AssertionError("越界窗口号应当抛 IndexError")


def test_getitem_out_of_range_raises():
    ds = SlidingWindowDataset(_seqs(), max_seq_len=MAX_SEQ_LEN)
    try:
        _ = ds[len(ds)]
    except IndexError:
        return
    raise AssertionError("越界下标应当抛 IndexError")


def test_user_rows_subset_keeps_window_mapping_consistent():
    """档位抽样：只取部分用户时，(行号, t) 的对应关系不变。"""
    ds_all = SlidingWindowDataset(_seqs(), max_seq_len=MAX_SEQ_LEN)
    ds_sub = SlidingWindowDataset(_seqs(), max_seq_len=MAX_SEQ_LEN, user_rows=[1])

    assert len(ds_sub) == 10
    for t in range(10):
        assert torch.equal(ds_sub[t][0], ds_all[4 + t][0])
        assert int(ds_sub[t][1]) == int(ds_all[4 + t][1])
    assert ds_sub.row_of(0) == (1, 0)


def test_user_rows_out_of_range_raises():
    try:
        SlidingWindowDataset(_seqs(), max_seq_len=MAX_SEQ_LEN, user_rows=[99])
    except ValueError:
        return
    raise AssertionError("越界 user_rows 应当抛 ValueError")


def test_empty_sequence_contributes_no_window():
    seqs = {0: [], 1: [7, 8]}
    ds = SlidingWindowDataset(seqs, max_seq_len=MAX_SEQ_LEN)
    assert len(ds) == 2
    assert ds.row_of(0) == (1, 0)


# =====================================================================
# 3. 填充与截断
# =====================================================================
def test_left_padding_and_tail_alignment():
    """历史不足时左侧补 PAD，右侧最后一个位置恒为最新物品。"""
    ds = SlidingWindowDataset({0: [1, 2, 3]}, max_seq_len=MAX_SEQ_LEN)
    x0, y0 = ds[0]
    x1, y1 = ds[1]
    x2, y2 = ds[2]

    assert x0.tolist() == [0, 0, 0, 0] and int(y0) == 1
    assert x1.tolist() == [0, 0, 0, 1] and int(y1) == 2
    assert x2.tolist() == [0, 0, 1, 2] and int(y2) == 3


def test_long_prefix_is_truncated_to_most_recent_cap():
    """超长序列只取「最近的 cap 条」：更早的历史被丢弃，不是从头部截断长度。"""
    ds = SlidingWindowDataset({0: list(range(1, 11))}, max_seq_len=MAX_SEQ_LEN)
    assert CAP == 4
    # t=9（target=10）时，历史应是最近的 4 个：6,7,8,9
    x, y = ds[9]
    assert x.tolist() == [6, 7, 8, 9]
    assert int(y) == 10
    # t=5（target=6）时，历史 2..5
    x, y = ds[5]
    assert x.tolist() == [2, 3, 4, 5]
    assert int(y) == 6


def test_input_cap_is_max_seq_len_minus_target_slots():
    """输入上限必须是 max_seq_len - 2，否则位置编码会越界。"""
    ds = SlidingWindowDataset(_seqs(), max_seq_len=50)
    assert ds.input_cap == 48
    assert ds.max_seq_len == 50
    s = ds.summary()
    assert s["target_slots"] == 2 and s["input_cap"] == 48


def test_max_seq_len_too_small_raises():
    for bad in (2, 1, 0, -1):
        try:
            SlidingWindowDataset(_seqs(), max_seq_len=bad)
        except ValueError:
            continue
        raise AssertionError(f"max_seq_len={bad} 应当抛 ValueError")


def test_shapes_and_dtypes():
    ds = SlidingWindowDataset(_seqs(), max_seq_len=MAX_SEQ_LEN)
    x, y = ds[0]
    assert x.dtype == torch.long and x.shape == (CAP,)
    assert y.dtype == torch.long and y.dim() == 0


# =====================================================================
# 4. 批量接口
# =====================================================================
def test_getitems_matches_single_item_calls():
    """`__getitems__` 是 DataLoader 的批量入口，必须与逐个取完全一致。"""
    ds = SlidingWindowDataset(_seqs(), max_seq_len=MAX_SEQ_LEN)
    idx = [0, 3, 5, 13, 17]
    batch = ds.__getitems__(idx)
    assert len(batch) == len(idx)
    for j, i in enumerate(idx):
        x_s, y_s = ds[i]
        x_b, y_b = batch[j]
        assert torch.equal(x_s, x_b)
        assert int(y_s) == int(y_b)


def test_data_loader_collate_works():
    """真实 DataLoader 走一遍，确认能与默认 collate 组合。"""
    from torch.utils.data import DataLoader

    ds = SlidingWindowDataset(_seqs(), max_seq_len=MAX_SEQ_LEN)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    xb, yb = next(iter(dl))
    assert xb.shape == (4, CAP)
    assert yb.shape == (4,)
    # 前 4 个窗口都来自用户 0，target 应为 1,2,3,4
    assert yb.tolist() == [1, 2, 3, 4]


# =====================================================================
# 5. 评估输入构造
# =====================================================================
def test_eval_input_val_uses_train_only():
    train = {0: [1, 2, 3], 1: [4, 5, 6, 7, 8, 9]}
    x = build_eval_inputs(train, [0, 1], max_seq_len=MAX_SEQ_LEN, split="val")
    assert x.shape == (2, CAP)
    # 用户 0 的 train 是 [1,2,3]，cap=4 -> 左补一个 PAD
    assert x[0].tolist() == [0, 1, 2, 3]
    # 用户 1 的历史 4..9 共 6 条，cap=4，应取最近 4 条
    assert x[1].tolist() == [6, 7, 8, 9]


def test_eval_input_test_includes_val_item():
    """预测 test 时历史里要含 val —— 这是留一法协议的要求。"""
    train = {0: [1, 2, 3]}
    val_items = np.array([4], dtype=np.int64)
    x = build_eval_inputs(train, [0], max_seq_len=MAX_SEQ_LEN,
                          split="test", val_items=val_items)
    # train=[1,2,3] 拼上 val=4 后恰好 4 条，正好填满 cap，不需要 PAD
    assert x[0].tolist() == [1, 2, 3, 4]


def test_eval_input_test_requires_val_items():
    try:
        build_eval_inputs({0: [1]}, [0], max_seq_len=MAX_SEQ_LEN, split="test")
    except ValueError:
        return
    raise AssertionError("split='test' 缺少 val_items 时应当抛 ValueError")


def test_eval_input_rejects_unknown_split():
    try:
        build_eval_inputs({0: [1]}, [0], max_seq_len=MAX_SEQ_LEN, split="train")
    except ValueError:
        return
    raise AssertionError("未知 split 应当抛 ValueError")


def test_eval_input_never_leaks_the_target():
    """test 输入含 val 但**不含 test 目标** —— 否则就是答案泄漏。"""
    train = {0: list(range(1, 9))}      # 1..8
    val_items = np.array([9], dtype=np.int64)
    test_item = 10
    x = build_eval_inputs(train, [0], max_seq_len=50, split="test", val_items=val_items)
    assert test_item not in x[0].tolist()
    assert int(val_items[0]) in x[0].tolist()


# =====================================================================
# 6. 剥离（E3 冷启动）
# =====================================================================
def test_strip_items_removes_them_and_shrinks_window_count():
    seqs = {0: [1, 2, 3, 4, 5, 6]}
    ds = SlidingWindowDataset(seqs, max_seq_len=MAX_SEQ_LEN)
    assert len(ds) == 6

    ds_strip = SlidingWindowDataset(seqs, max_seq_len=MAX_SEQ_LEN,
                                    strip_items=[2, 4], n_items=10)
    # 剥离后序列变成 [1,3,5,6]，窗口数降为 4
    assert ds_strip.n_windows == 4
    assert ds_strip.n_stripped_items == 2

    seen_targets = [int(ds_strip[i][1]) for i in range(4)]
    assert seen_targets == [1, 3, 5, 6]
    for i in range(4):
        x, _ = ds_strip[i]
        assert 2 not in x.tolist() and 4 not in x.tolist()


def test_strip_items_out_of_range_raises():
    try:
        SlidingWindowDataset({0: [1, 2]}, max_seq_len=MAX_SEQ_LEN,
                             strip_items=[99], n_items=10)
    except ValueError:
        return
    raise AssertionError("越界 strip_items 应当抛 ValueError")


def test_eval_input_strips_new_items_too():
    """冷启动评估：历史序列里不能有已经"上线"的新番。"""
    train = {0: [1, 2, 3, 4]}
    val_items = np.array([4], dtype=np.int64)
    x = build_eval_inputs(train, [0], max_seq_len=MAX_SEQ_LEN, split="test",
                          val_items=val_items, strip_items=[2, 4], n_items=10)
    # 剥离 2 和 4 后历史只剩 [1,3]
    assert x[0].tolist() == [0, 0, 1, 3]


def test_count_windows_matches_dataset():
    seqs = {0: [1, 2, 3], 1: list(range(4, 14))}
    ds = SlidingWindowDataset(seqs, max_seq_len=MAX_SEQ_LEN)
    stats = count_windows(seqs)
    assert stats["n_windows"] == ds.n_windows
    assert stats["n_users"] == 2
    assert stats["max_windows"] == 10


def test_count_windows_respects_strip():
    seqs = {0: [1, 2, 3, 4, 5, 6]}
    stats = count_windows(seqs, strip_items=[2, 4], n_items=10)
    assert stats["n_windows"] == 4


# =====================================================================
# 7. 纯函数工具
# =====================================================================
def test_make_padded_id_array_takes_tail():
    out = make_padded_id_array([[1, 2, 3, 4, 5], [9]], cap=3)
    assert out.tolist() == [[3, 4, 5], [0, 0, 9]]
    assert out.dtype == np.int64


def test_make_padded_id_array_handles_empty_and_bad_cap():
    out = make_padded_id_array([[], [1]], cap=2)
    assert out.tolist() == [[0, 0], [0, 1]]
    try:
        make_padded_id_array([[1]], cap=0)
    except ValueError:
        return
    raise AssertionError("cap<=0 应当抛 ValueError")
