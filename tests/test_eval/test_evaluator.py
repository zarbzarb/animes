# -*- coding: utf-8 -*-
"""`models/eval/evaluator.py` 的单元测试。

评估器最容易出的三类错，测试按这三类组织：

1. **契约错**：候选矩阵第 0 列不是正样本、打分函数返回形状不对
   —— 这类错不会崩，只会让指标悄悄偏掉，所以必须显式报错并测试。
2. **分块不一致**：batch_size 一变指标就变（比如把并列时的列序搞乱）
   —— 用「不同 batch_size 结果必须逐位相同」锁死。
3. **答案泄漏**：正样本出现在模型可见的输入里，指标虚高且"越改越好看"。

手算基准全部写死在断言里（不拿 metrics 的输出当期望值），
否则 metrics 一起算错时测试会跟着一起错。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.eval.evaluator import (
    EvalData,
    collect_ranks,
    drop_leaked_samples,
    evaluate,
    evaluate_cold_start,
    evaluate_grouped,
    find_leaked_mask,
)

# 手算场景：3 个样本、每个 1 正 + 2 负（共 3 个候选）
# 分数表刻意让第 2 个样本的正样本排到第 2 位，以便区分 HR@1 与 HR@2
SCORES = {1: 100.0, 2: 5.0, 3: 1.0, 4: 90.0, 5: 80.0, 6: 9.0, 7: 3.0, 8: 0.5, 9: 0.1}


def _score_by_item() -> callable:
    """打分函数：候选分数只由物品 id 决定，与输入历史无关。

    这样期望排名可以完全手算，不受模型结构影响。
    """

    def fn(input_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        b, c = candidate_ids.shape
        out = torch.empty((b, c), dtype=torch.float32)
        for i in range(b):
            for j in range(c):
                out[i, j] = SCORES[int(candidate_ids[i, j])]
        return out

    return fn


def _data(label: str = "unit") -> EvalData:
    """基准评估数据：3 个样本，每个 1 正 + 2 负。

    刻意让 `input_ids` **不含任何正样本**（正样本是 1/2/3，历史里只出现 4~9），
    这样它是"干净"的：除泄漏专项外，其余测试都不受答案泄漏干扰。
    """
    return EvalData(
        input_ids=np.array([[0, 0, 0, 4], [0, 0, 6, 7], [0, 8, 9, 5]], dtype=np.int64),
        positives=np.array([1, 2, 3], dtype=np.int64),
        negatives=np.array([[4, 5], [6, 7], [8, 9]], dtype=np.int64),
        user_rows=np.array([10, 20, 30], dtype=np.int64),
        genres=np.array(["A", "B", "A"]),
        label=label,
    )


# =====================================================================
# 1. EvalData 契约
# =====================================================================
def test_candidates_put_positive_in_column_zero():
    """第 0 列必须是正样本 —— 这是全项目并列口径的前提。"""
    d = _data()
    assert d.n_candidates == 3
    assert d.candidates[:, 0].tolist() == [1, 2, 3]
    assert d.candidates[:, 1:].tolist() == [[4, 5], [6, 7], [8, 9]]


def test_shape_validation():
    with pytest.raises(ValueError, match="positives"):
        EvalData(input_ids=np.zeros((2, 4), dtype=np.int64),
                 positives=np.array([1], dtype=np.int64),
                 negatives=np.zeros((2, 2), dtype=np.int64))

    with pytest.raises(ValueError, match="negatives"):
        EvalData(input_ids=np.zeros((2, 4), dtype=np.int64),
                 positives=np.array([1, 2], dtype=np.int64),
                 negatives=np.zeros((3, 2), dtype=np.int64))

    with pytest.raises(ValueError, match="input_ids"):
        EvalData(input_ids=np.zeros(4, dtype=np.int64),
                 positives=np.array([1], dtype=np.int64),
                 negatives=np.zeros((1, 2), dtype=np.int64))

    with pytest.raises(ValueError, match="genres"):
        EvalData(input_ids=np.zeros((2, 4), dtype=np.int64),
                 positives=np.array([1, 2], dtype=np.int64),
                 negatives=np.zeros((2, 2), dtype=np.int64),
                 genres=np.array(["A"]))


def test_subset_keeps_consistency():
    d = _data()
    sub = d.subset(np.array([True, False, True]), label="only_A")
    assert sub.n_samples == 2
    assert sub.label == "only_A"
    assert sub.positives.tolist() == [1, 3]
    assert sub.genres.tolist() == ["A", "A"]
    assert sub.user_rows.tolist() == [10, 30]
    assert sub.candidates[:, 0].tolist() == [1, 3]


def test_subset_bad_mask_raises():
    with pytest.raises(ValueError, match="掩码"):
        _data().subset(np.array([True, False]))


def test_summary_reports_distinct_users():
    s = _data().summary()
    assert s["n_samples"] == 3
    assert s["n_candidates"] == 3
    assert s["n_distinct_users"] == 3
    assert s["window_len"] == 4


# =====================================================================
# 2. 排名与指标（手算对拍）
# =====================================================================
def test_collect_ranks_hand_computed():
    """手算：样本 0 正样本最高分(100) -> rank1；
    样本 1 正样本 5 分，输给负样本 6 的 9 分 -> rank2；
    样本 2 正样本 1 分，全场最高 -> rank1。"""
    ranks = collect_ranks(_score_by_item(), _data())
    assert ranks.tolist() == [1, 2, 1]


def test_evaluate_matches_hand_computed_metrics():
    res = evaluate(_score_by_item(), _data(), ks=(1, 2))

    # HR@1 = 2/3（第 2 个样本正样本排第 2）
    assert res["hr@1"] == pytest.approx(2 / 3, abs=1e-9)
    # HR@2 = 3/3
    assert res["hr@2"] == pytest.approx(1.0, abs=1e-9)
    # NDCG@1: rank=2 的样本折损后为 0
    assert res["ndcg@1"] == pytest.approx(2 / 3, abs=1e-9)
    # NDCG@2 = (1 + 1/log2(3) + 1) / 3
    assert res["ndcg@2"] == pytest.approx((1 + 1 / np.log2(3) + 1) / 3, abs=1e-9)
    # MRR = (1 + 1/2 + 1) / 3
    assert res["mrr"] == pytest.approx(2.5 / 3, abs=1e-9)
    assert res["n_samples"] == 3
    assert res["n_candidates"] == 3
    assert res["n_negatives"] == 2


def test_ties_follow_stable_column_order():
    """全体候选同分时，正样本（第 0 列）必须排第 1 —— 乐观口径。

    这条是 metrics 的约定，但评估器把它用在真实调用路径上，
    所以在这里也钉一次：一旦有人把候选矩阵的拼法改掉（比如把负样本放前面），
    名次会立刻变化，指标却"看起来正常"。
    """

    def all_equal(input_ids, candidate_ids):
        return torch.ones(candidate_ids.shape, dtype=torch.float32)

    ranks = collect_ranks(all_equal, _data())
    assert ranks.tolist() == [1, 1, 1]


def test_batch_size_does_not_change_result():
    """分块必须无损：不同 batch_size 结果逐位相同。

    若评估器不小心把批次间的顺序或统计量串起来（例如累积计数写错），
    这条会立刻抓住。
    """
    d = _data()
    fn = _score_by_item()
    base = evaluate(fn, d, ks=(1, 2), batch_size=1)
    for bs in (2, 3, 10, 100):
        other = evaluate(fn, d, ks=(1, 2), batch_size=bs)
        for key in ("hr@1", "hr@2", "ndcg@1", "ndcg@2", "mrr", "n_samples"):
            assert other[key] == pytest.approx(base[key], abs=1e-12), \
                f"batch_size={bs} 时 {key} 与 batch_size=1 不一致"


def test_empty_dataset_returns_zeros_with_zero_sample_count():
    """空集返回 0.0（而不是 nan），但 n_samples 必须同时为 0 才能区分。"""
    empty = EvalData(input_ids=np.zeros((0, 4), dtype=np.int64),
                     positives=np.zeros(0, dtype=np.int64),
                     negatives=np.zeros((0, 2), dtype=np.int64))
    res = evaluate(_score_by_item(), empty, ks=(5, 10))
    assert res["n_samples"] == 0
    assert res["hr@5"] == 0.0 and res["ndcg@10"] == 0.0 and res["mrr"] == 0.0


# =====================================================================
# 3. 契约校验
# =====================================================================
def test_wrong_return_shape_raises():
    def bad(input_ids, candidate_ids):
        return torch.zeros(candidate_ids.shape[0], dtype=torch.float32)  # 少了一维

    with pytest.raises(ValueError, match="score_fn 返回形状"):
        collect_ranks(bad, _data())


def test_wrong_return_type_raises():
    def bad(input_ids, candidate_ids):
        return np.zeros(candidate_ids.shape, dtype=np.float32)

    with pytest.raises(TypeError, match="torch.Tensor"):
        collect_ranks(bad, _data())


def test_ranks_boundaries():
    """排名恒在 [1, C] —— 越界说明 positive_rank 用错了列。"""
    ranks = collect_ranks(_score_by_item(), _data())
    assert ranks.min() >= 1 and ranks.max() <= _data().n_candidates


# =====================================================================
# 4. 答案泄漏
# =====================================================================
def test_find_leaked_mask_detects_positive_in_input():
    d = EvalData(
        input_ids=np.array([[0, 0, 7, 8], [0, 0, 1, 2]], dtype=np.int64),
        positives=np.array([8, 2], dtype=np.int64),      # 两个都在输入里
        negatives=np.array([[3, 4], [5, 6]], dtype=np.int64),
    )
    assert find_leaked_mask(d).tolist() == [True, True]


def test_find_leaked_mask_ignores_pad_and_absence():
    d = EvalData(
        input_ids=np.array([[0, 0, 0, 0], [0, 0, 1, 2]], dtype=np.int64),
        # 第一行输入全 PAD，正样本 3 未出现 -> 不算泄漏
        positives=np.array([3, 9], dtype=np.int64),
        negatives=np.array([[4, 5], [6, 7]], dtype=np.int64),
    )
    assert find_leaked_mask(d).tolist() == [False, False]


def test_drop_leaked_samples_removes_and_reports_count():
    d = EvalData(
        input_ids=np.array([[0, 0, 7, 8], [0, 0, 1, 2], [0, 3, 4, 5]], dtype=np.int64),
        positives=np.array([8, 30, 40], dtype=np.int64),   # 只有第 1 个泄漏
        negatives=np.array([[3, 4], [5, 6], [9, 10]], dtype=np.int64),
    )
    kept, n = drop_leaked_samples(d)
    assert n == 1
    assert kept.n_samples == 2
    assert kept.positives.tolist() == [30, 40]


def test_drop_leaked_is_noop_when_clean():
    d = _data()
    kept, n = drop_leaked_samples(d)
    assert n == 0
    assert kept is d


# =====================================================================
# 5. 分组（E4 分题材）
# =====================================================================
def test_evaluate_grouped_only_forwards_once_and_slices_ranks():
    """分组评估必须复用同一批排名，且各组指标等于对该组排名单独计算的结果。"""
    calls = {"n": 0}
    inner = _score_by_item()

    def counting(input_ids, candidate_ids):
        calls["n"] += 1
        return inner(input_ids, candidate_ids)

    d = _data()
    res = evaluate_grouped(counting, d, ks=(1, 2), group_field="genres", batch_size=2)

    # 3 个样本 / batch=2 -> 2 个批次；分组不许再触发额外前向
    assert calls["n"] == 2
    assert res["n_groups"] == 2
    assert set(res["groups"].keys()) == {"A", "B"}

    # 组 A = 样本 0 与 2，rank 均为 1
    assert res["groups"]["A"]["n_samples"] == 2
    assert res["groups"]["A"]["hr@1"] == pytest.approx(1.0, abs=1e-9)
    assert res["groups"]["A"]["ndcg@1"] == pytest.approx(1.0, abs=1e-9)

    # 组 B = 样本 1，rank = 2
    assert res["groups"]["B"]["n_samples"] == 1
    assert res["groups"]["B"]["hr@1"] == pytest.approx(0.0, abs=1e-9)
    assert res["groups"]["B"]["hr@2"] == pytest.approx(1.0, abs=1e-9)

    # 总体与 evaluate() 一致
    plain = evaluate(inner, d, ks=(1, 2))
    assert res["overall"]["hr@1"] == pytest.approx(plain["hr@1"], abs=1e-12)
    assert res["overall"]["n_samples"] == 3


def test_evaluate_grouped_without_labels_returns_empty_groups():
    d = EvalData(input_ids=np.zeros((2, 4), dtype=np.int64),
                 positives=np.array([1, 2], dtype=np.int64),
                 negatives=np.array([[3, 4], [5, 6]], dtype=np.int64))
    res = evaluate_grouped(_score_by_item(), d, group_field="genres")
    assert res["n_groups"] == 0
    assert res["overall"]["n_samples"] == 2


def test_evaluate_grouped_min_samples_filters_tiny_groups():
    d = EvalData(input_ids=np.zeros((3, 4), dtype=np.int64),
                 positives=np.array([1, 2, 3], dtype=np.int64),
                 negatives=np.array([[4, 5], [6, 7], [8, 9]], dtype=np.int64),
                 genres=np.array(["A", "A", "rare"]))
    res = evaluate_grouped(_score_by_item(), d, group_field="genres", min_samples=2)
    assert set(res["groups"].keys()) == {"A"}


# =====================================================================
# 6. 冷启动
# =====================================================================
def test_evaluate_cold_start_tags_protocol_and_reports_sample_count():
    res = evaluate_cold_start(_score_by_item(), _data(label="cold_start_2021"), ks=(1, 2))
    assert res["protocol"] == "cold_start_holdout"
    assert res["label"] == "cold_start_2021"
    assert res["n_samples"] == 3           # 样本量必须随指标一起报
    assert res["n_negatives"] == 2


# =====================================================================
# 7. 与 metrics 的口径一致性（交叉校验）
# =====================================================================
def test_hr_and_ndcg_consistent_with_metrics_module():
    """评估器算出的排名，喂给 metrics 必须与 evaluate() 的输出一致。"""
    from models.eval.metrics import ranking_metrics

    d = _data()
    ranks = collect_ranks(_score_by_item(), d)
    manual = ranking_metrics(ranks, ks=(1, 2, 5))
    res = evaluate(_score_by_item(), d, ks=(1, 2, 5))
    for key, val in manual.items():
        assert res[key] == pytest.approx(val, abs=1e-12), f"{key} 不一致"
