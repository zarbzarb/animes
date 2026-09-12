# -*- coding: utf-8 -*-
"""`positive_rank()` 与 `ranking_metrics()` 的测试。

`positive_rank` 是唯一把「打分矩阵」翻译成「排名」的地方，HR/NDCG/MRR 全部依赖它，
所以它的并列口径、批量处理、错误输入都必须被测到。
"""

import math

import pytest
import torch

from models.eval.metrics import hr_at_k, positive_rank, ranking_metrics, stable_order


# ---------------------------------------------------------------------
# 基本排名
# ---------------------------------------------------------------------
def test_rank_positive_at_first_column():
    """标准评估协议：正样本固定放第 0 列，分数最高时 rank=1"""
    scores = torch.tensor([[0.9, 0.1, 0.2]])
    assert positive_rank(scores, pos_index=0).tolist() == [1]


def test_rank_counts_negatives_above_positive():
    """有 2 个负样本分数高于正样本时 rank=3"""
    scores = torch.tensor([[0.5, 0.6, 0.7, 0.4]])   # 正样本在第 0 列
    assert positive_rank(scores, pos_index=0).tolist() == [3]


def test_rank_batch_is_independent_per_row():
    """批量内各行互不影响（容易写成用整个 batch 全局排序的低级错误）"""
    scores = torch.tensor([
        [0.9, 0.1],     # 正样本在第 0 列 -> rank 1
        [0.1, 0.9],     # 正样本在第 0 列 -> rank 2
    ])
    assert positive_rank(scores, pos_index=0).tolist() == [1, 2]


def test_rank_positive_at_last_column():
    scores = torch.tensor([[0.9, 0.8, 0.1]])
    assert positive_rank(scores, pos_index=2).tolist() == [3]


def test_rank_at_101_candidate_scale():
    """按 evaluation-plan 5.2 的真实规模（1 正 + 100 负 = 101 候选）验证"""
    n = 101
    scores = torch.zeros(1, n)
    scores[0, 0] = -1.0                     # 正样本最低分 -> 排最后
    assert positive_rank(scores, pos_index=0).item() == 101
    scores[0, 0] = 1.0                      # 正样本最高分 -> 排第 1
    assert positive_rank(scores, pos_index=0).item() == 1


# ---------------------------------------------------------------------
# 并列口径（必须与模块 docstring 的声明一致）
# ---------------------------------------------------------------------
def test_tie_breaks_keep_column_order():
    """稳定排序：并列时列序在前的优先（实测 [3,3,1,3] -> [0,1,3,2]）"""
    scores = torch.tensor([[3.0, 3.0, 1.0, 3.0]])
    assert stable_order(scores).tolist() == [[0, 1, 3, 2]]


def test_fully_tied_positive_at_col0_gets_rank_1():
    """完全并列时正样本（第 0 列）得 rank=1 —— 对正样本的乐观口径，需写进论文"""
    scores = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    assert positive_rank(scores, pos_index=0).tolist() == [1]
    assert positive_rank(scores, pos_index=3).tolist() == [4]


def test_tie_with_negative_above_positive():
    """正样本与负样本并列但负样本列序在前时，正样本名次靠后"""
    scores = torch.tensor([[5.0, 1.0, 5.0]])    # 列 0 与列 2 并列最高
    assert positive_rank(scores, pos_index=2).tolist() == [2]


# ---------------------------------------------------------------------
# 逐样本 pos_index 与错误输入
# ---------------------------------------------------------------------
def test_per_sample_pos_index_tensor():
    scores = torch.tensor([
        [0.9, 0.1, 0.5],
        [0.9, 0.1, 0.5],
    ])
    # 用户 1 的正样本在第 0 列（第 1 名），用户 2 的在第 2 列（第 2 名）
    got = positive_rank(scores, pos_index=torch.tensor([0, 2]))
    assert got.tolist() == [1, 2]


def test_pos_index_out_of_range_raises():
    scores = torch.zeros(1, 3)
    with pytest.raises(ValueError):
        positive_rank(scores, pos_index=3)
    with pytest.raises(ValueError):
        positive_rank(scores, pos_index=-1)


def test_per_sample_pos_index_length_mismatch_raises():
    scores = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        positive_rank(scores, pos_index=torch.tensor([0, 1, 2]))


def test_three_dim_input_raises():
    with pytest.raises(ValueError):
        positive_rank(torch.zeros(2, 3, 4))


def test_rank_has_no_grad():
    """评估不建计算图（显存与速度都重要），已加 @torch.no_grad()"""
    scores = torch.zeros(1, 3, requires_grad=True)
    out = positive_rank(scores, pos_index=0)
    assert not out.requires_grad


# ---------------------------------------------------------------------
# ranking_metrics：实验表格的入口
# ---------------------------------------------------------------------
def test_ranking_metrics_keys_match_experiment_yaml():
    """键名必须与 configs/experiment.yaml 的 metrics 字段一致，可直接落 metrics.json"""
    m = ranking_metrics([1, 2, 3], ks=(5, 10))
    assert set(m) == {"hr@5", "hr@10", "ndcg@5", "ndcg@10", "mrr", "n_samples"}


def test_ranking_metrics_values_hand_calculation():
    ranks = [1, 2, 3]
    m = ranking_metrics(ranks, ks=(5, 10))
    assert abs(m["hr@5"] - 1.0) < 1e-12                       # 3 个都 <= 5
    assert abs(m["hr@10"] - 1.0) < 1e-12
    expected_ndcg = (1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)) / 3.0
    assert abs(m["ndcg@5"] - expected_ndcg) < 1e-12
    assert abs(m["ndcg@10"] - expected_ndcg) < 1e-12
    assert abs(m["mrr"] - (1.0 + 0.5 + 1.0 / 3.0) / 3.0) < 1e-12
    assert m["n_samples"] == 3


def test_ranking_metrics_reports_n_samples():
    """样本量必须随指标报告（evaluation-plan 6.3），否则无法区分空集与真 0 分"""
    assert ranking_metrics([])["n_samples"] == 0
    assert ranking_metrics([1, 2])["n_samples"] == 2


def test_ranking_metrics_is_deterministic():
    """纯函数：同一输入重复调用结果完全一致（可复现性的前提）"""
    ranks = [1, 5, 11, 2, 7]
    assert ranking_metrics(ranks) == ranking_metrics(ranks)


def test_ranking_metrics_custom_ks():
    m = ranking_metrics([1, 6], ks=(1, 3, 10))
    assert set(m) == {"hr@1", "hr@3", "hr@10", "ndcg@1", "ndcg@3", "ndcg@10",
                      "mrr", "n_samples"}
    assert abs(m["hr@1"] - 0.5) < 1e-12     # 只有 rank=1 命中
    assert abs(m["hr@3"] - 0.5) < 1e-12     # rank=6 仍未命中
    assert abs(m["hr@10"] - 1.0) < 1e-12


# ---------------------------------------------------------------------
# 稳定排序：与纯 Python 参考实现对拍（本文件最关键的一组测试）
# ---------------------------------------------------------------------
def _python_reference_order(row) -> list:
    """独立的稳定降序参考实现：按 `(-分数, 列号)` 排序。

    刻意用纯 Python 写，不依赖 torch 的 `stable` 语义——否则等于拿 torch 验证 torch。
    """
    return sorted(range(len(row)), key=lambda i: (-float(row[i]), i))


def _tie_heavy_scores(n: int, seed: int = 0) -> torch.Tensor:
    """构造并列极多的候选分数（取值域只有 3 档），用来暴露排序稳定性问题。"""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 3, (1, n), generator=gen).float()


@pytest.mark.parametrize("n", [8, 32, 64, 101])
def test_stable_order_matches_python_reference(n):
    """在并列密集的输入上与纯 Python 参考实现逐列比对。

    ⚠️ 为什么必须包含 n >= 64 的用例（这是踩过的坑）：
    实测本机 torch 2.14.0，当并列元素较少（n <= 32）时，**非稳定**排序也会"碰巧"
    按列序输出，于是看起来 `stable=True` 可有可无；但当 n >= 64 且并列密集时，
    非稳定排序会给出任意次序（n=101 时 30/30 个随机种子都出现差异）。
    这意味着一旦漏掉 `stable=True`，NDCG 会随实现/线程数/设备变化而**不可复现**，
    而这类 bug 在指标数字上完全看不出来。

    本用例是测试套件里唯一能抓住「把 stable=True 改成 False」的断言
    （由变异测试发现，见 docs/progress.md 的 M2.1 记录）。
    """
    scores = _tie_heavy_scores(n)
    expect = _python_reference_order(scores[0].tolist())
    assert stable_order(scores).tolist()[0] == expect


def test_stable_order_reference_at_real_101_scale():
    """真实评估规模（1 正 + 100 负 = 101 候选）下的稳定性，并覆盖多个种子"""
    for seed in range(10):
        scores = _tie_heavy_scores(101, seed=seed)
        assert stable_order(scores).tolist()[0] == _python_reference_order(scores[0].tolist())


def test_positive_rank_matches_python_reference_all_columns():
    """任意列作为正样本时，positive_rank 都必须与参考实现的排名一致。

    这是比"挑几个位置断言"强得多的性质测试：它把 101 个位置全测了。
    """
    scores = _tie_heavy_scores(101)
    order = _python_reference_order(scores[0].tolist())
    rank_of = {col: pos + 1 for pos, col in enumerate(order)}
    for col in range(101):
        assert positive_rank(scores, pos_index=col).item() == rank_of[col]


def test_hr_uses_stable_order_at_scale():
    """端到端：101 候选中正样本恰好排第 3 时，HR@2=0 且 HR@3=1。"""
    scores = torch.zeros(1, 101)
    # 让列 50 成为第 3 名：列 7、列 90 分数更高（并列也无所谓，列序已定）
    scores[0, 7] = 2.0
    scores[0, 90] = 2.0
    scores[0, 50] = 1.0
    rank = positive_rank(scores, pos_index=50).item()
    assert rank == 3
    assert hr_at_k([rank], k=2) == 0.0
    assert hr_at_k([rank], k=3) == 1.0

