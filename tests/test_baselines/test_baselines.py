# -*- coding: utf-8 -*-
"""对比基线（M2.7）单元测试 —— `tests/test_baselines/test_baselines.py`

覆盖四类东西，按"出错时有多难发现"排序：

1. **公平性契约**：三个基线的 `score()` 都必须满足
   `models/eval/evaluator.py` 的 `ScoreFn` 契约（形状 `[B,1+C]`、
   第 0 列恒正样本、分越大越相关）—— 不满足时评估会直接报错，好抓。
2. **数值正确性**：热度频次、ItemCF 余弦相似度都用**手算值**对拍。
3. **隐性错误**（最重要）：GRU4Rec 的**左填充不变性**。
   若不用 `pack_padded_sequence`，同一个用户的历史会随"前面垫了几个 PAD"
   而得到不同表示 —— 不报错、指标也算得出来，只是数字是错的。
4. **并列口径的假象**：ItemCF 的"101 个候选全为 0 分"的行会让
   正样本按本项目并列约定白拿 rank=1；这条必须能被测试观测到。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.baselines.gru4rec import GRU4Rec, GRU4RecConfig
from models.baselines.itemcf import (
    ItemCFScorer,
    build_cooccurrence,
    build_itemcf_similarity,
)
from models.baselines.popularity import (
    PopularityScorer,
    count_train_frequency,
    normalize_frequency,
)
from models.eval.evaluator import EvalData, evaluate

# ---------------------------------------------------------------------
# 公用玩具数据
# ---------------------------------------------------------------------
# 四个用户的训练序列（物品池内索引，PAD = 0）
TRAIN_SEQS = [[1, 2], [1, 2], [2, 3], [1, 2, 3]]
N_ITEMS = 3
# 手算的共现：C[1,1]=3 C[2,2]=4 C[3,3]=2；C[1,2]=3 C[1,3]=1 C[2,3]=2
COS_12 = 3.0 / np.sqrt(3.0 * 4.0)
COS_13 = 1.0 / np.sqrt(3.0 * 2.0)
COS_23 = 2.0 / np.sqrt(4.0 * 2.0)


def _toy_eval_data(input_ids, positives, negatives) -> EvalData:
    return EvalData(
        input_ids=np.asarray(input_ids, dtype=np.int64),
        positives=np.asarray(positives, dtype=np.int64),
        negatives=np.asarray(negatives, dtype=np.int64),
    )


# =====================================================================
# 1. 热度基线
# =====================================================================
class TestPopularity:
    def test_count_is_hand_checkable(self):
        freq = count_train_frequency(TRAIN_SEQS, N_ITEMS)
        # 物品 1 出现在用户 0,1,3 → 3 次；2 出现在 0,1,2,3 → 4 次；3 → 2 次
        assert freq.tolist() == [0.0, 3.0, 4.0, 2.0]

    def test_pad_slot_is_always_zero(self):
        freq = count_train_frequency([[0, 1, 0, 2]], N_ITEMS)
        assert freq[0] == 0.0
        assert freq[1] == 1.0 and freq[2] == 1.0

    def test_rows_argument_selects_users(self):
        # 只统计前两个用户 → 物品 3 不应该出现
        freq = count_train_frequency(TRAIN_SEQS, N_ITEMS, rows=[0, 1])
        assert freq.tolist() == [0.0, 2.0, 2.0, 0.0]

    def test_normalize_maps_max_to_one_and_keeps_pad_zero(self):
        # 下标 0 = PAD，恒 0；最大频次 4（物品 2）归一到 1.0
        f = normalize_frequency(np.array([0.0, 3.0, 4.0, 2.0]))
        assert f[0] == 0.0
        assert f[2] == pytest.approx(1.0)
        assert f[1] == pytest.approx(0.75)
        assert f[3] == pytest.approx(0.5)

    def test_normalize_power_variant(self):
        f = normalize_frequency(np.array([0.0, 4.0, 16.0]), power=0.5)
        assert f[2] == pytest.approx(1.0)
        assert f[1] == pytest.approx(0.5)

    def test_score_ignores_input_ids_completely(self):
        """零信息基线的定义：同一个候选集，无论历史是什么，分数都一样。"""
        scorer = PopularityScorer(normalize_frequency(
            count_train_frequency(TRAIN_SEQS, N_ITEMS)))
        cand = torch.tensor([[1, 2, 3], [1, 2, 3]])
        s1 = scorer.score(torch.tensor([[0, 0, 1]]), cand)
        s2 = scorer.score(torch.tensor([[2, 3, 3]]), cand)
        assert torch.equal(s1, s2)

    def test_score_orders_by_frequency(self):
        scorer = PopularityScorer(normalize_frequency(
            count_train_frequency(TRAIN_SEQS, N_ITEMS)))
        s = scorer.score(torch.zeros(1, 3, dtype=torch.long),
                         torch.tensor([[1, 2, 3]]))
        # 频次 2 > 1 > 3  ⇒ 分数 s[0,1] > s[0,0] > s[0,2]
        assert s[0, 1] > s[0, 0] > s[0, 2]

    def test_contract_shape_and_no_training(self):
        scorer = PopularityScorer(np.arange(5, dtype=np.float32))
        out = scorer.score(torch.zeros(4, 6, dtype=torch.long),
                           torch.randint(1, 5, (4, 101)))
        assert out.shape == (4, 101)
        assert out.dtype == torch.float32
        assert scorer.n_params == 0
        assert scorer.config_snapshot()["trained"] is False

    def test_out_of_range_candidate_is_clipped_not_crashed(self):
        """候选越界时宁可裁剪（且打低分），也不要让评估跑一半崩在基线。"""
        scorer = PopularityScorer(normalize_frequency(
            count_train_frequency(TRAIN_SEQS, N_ITEMS)))
        out = scorer.score(torch.zeros(1, 2, dtype=torch.long),
                           torch.tensor([[2, 99]]))
        assert out.shape == (1, 2)


# =====================================================================
# 2. ItemCF
# =====================================================================
class TestItemCFSimilarity:
    def test_cooccurrence_is_hand_checkable(self):
        C = build_cooccurrence(TRAIN_SEQS, N_ITEMS).toarray()
        assert C.shape == (4, 4)
        assert C[1, 1] == 3 and C[2, 2] == 4 and C[3, 3] == 2
        assert C[1, 2] == 3 and C[1, 3] == 1 and C[2, 3] == 2
        # 对称
        assert C[2, 1] == C[1, 2]
        # PAD 行/列恒 0
        assert C[0, :].sum() == 0 and C[:, 0].sum() == 0

    def test_repeated_item_within_user_counts_once(self):
        """同一用户重复消费同一部番只算一次（二值化）。"""
        C = build_cooccurrence([[1, 1, 1, 2], [1, 2]], N_ITEMS).toarray()
        assert C[1, 2] == 2          # 两个用户，不是 4 次共现
        assert C[1, 1] == 2          # 分子分母都用"用户数"，自洽

    def test_cosine_matches_hand_computed(self):
        sim, stats = build_itemcf_similarity(TRAIN_SEQS, N_ITEMS, topk=200)
        S = sim.toarray()
        assert S[1, 2] == pytest.approx(COS_12, abs=1e-6)
        assert S[1, 3] == pytest.approx(COS_13, abs=1e-6)
        assert S[2, 3] == pytest.approx(COS_23, abs=1e-6)
        assert stats["n_items"] == N_ITEMS and stats["topk"] == 200

    def test_diagonal_is_zero(self):
        """自己不是自己的邻居：否则 sim(i,i)=1 会永远占据 TopK 第一位。"""
        sim, _ = build_itemcf_similarity(TRAIN_SEQS, N_ITEMS, topk=200)
        assert np.allclose(sim.diagonal(), 0.0)

    def test_topk_truncates_by_cosine_not_by_count(self):
        sim, stats = build_itemcf_similarity(TRAIN_SEQS, N_ITEMS, topk=1)
        S = sim.toarray()
        # 物品 1 的两个邻居 sim: (2)=0.866 > (3)=0.408 ⇒ 只留 2
        assert S[1, 2] == pytest.approx(COS_12, abs=1e-6)
        assert S[1, 3] == 0.0
        # 物品 2: (1)=0.866 > (3)=0.707 ⇒ 只留 1
        assert S[2, 1] == pytest.approx(COS_12, abs=1e-6)
        assert S[2, 3] == 0.0
        # 3 个物品各留 1 条有向边 ⇒ 均值邻居数 = 1.0
        assert stats["avg_neighbors"] == pytest.approx(1.0)

    def test_deterministic_for_ties(self):
        """并列邻居时输出顺序必须确定（浮点并列很常见）。"""
        sim1, _ = build_itemcf_similarity(TRAIN_SEQS, N_ITEMS, topk=1)
        sim2, _ = build_itemcf_similarity(TRAIN_SEQS, N_ITEMS, topk=1)
        assert np.array_equal(sim1.indices, sim2.indices)
        assert np.array_equal(sim1.data, sim2.data)


class TestItemCFScoring:
    def _scorer(self, topk=200, **kw):
        sim, _ = build_itemcf_similarity(TRAIN_SEQS, N_ITEMS, topk=topk)
        return ItemCFScorer(sim, n_items=N_ITEMS, topk=topk, **kw)

    def test_score_is_sum_of_similarity_over_history(self):
        scorer = self._scorer()
        # 历史 = [1]，候选 = [2, 3] ⇒ [sim(2,1), sim(3,1)]
        out = scorer.score(torch.tensor([[0, 0, 1]]), torch.tensor([[2, 3]]))
        assert out[0, 0].item() == pytest.approx(COS_12, abs=1e-6)
        assert out[0, 1].item() == pytest.approx(COS_13, abs=1e-6)

    def test_history_order_and_duplicates_do_not_matter(self):
        """分数是 Σ sim，与历史顺序无关；重复物品只算一次（对应二值化）。"""
        scorer = self._scorer()
        cand = torch.tensor([[2, 3]])
        a = scorer.score(torch.tensor([[1, 1, 2]]), cand)
        b = scorer.score(torch.tensor([[2, 1, 0]]), cand)
        assert torch.allclose(a, b, atol=1e-6)

    def test_empty_history_gives_zero_scores(self):
        scorer = self._scorer()
        out = scorer.score(torch.tensor([[0, 0, 0]]), torch.tensor([[2, 3]]))
        assert torch.equal(out, torch.zeros_like(out))
        assert scorer.last_stats["n_zero_rows"] == 1

    def test_chunking_does_not_change_result(self):
        """内部分块大小是性能参数，绝不能影响数值（否则批大小会变成变量）。"""
        x = torch.tensor([[0, 1], [0, 2], [1, 2], [0, 0]])
        cand = torch.tensor([[2, 3], [1, 3], [3, 1], [1, 2]])
        s1 = self._scorer(score_chunk=1).score(x, cand)
        s2 = self._scorer(score_chunk=512).score(x, cand)
        assert torch.allclose(s1, s2, atol=1e-6)

    def test_stats_accumulate_across_calls_until_reset(self):
        """诊断量必须**跨 batch 累计**。

        回归保护：`evaluator.collect_ranks` 会按 batch 反复调用 `score()`，
        若只记最后一次调用，main 档会报出最后一个残缺 batch 的行数
        （实测 2,161 / 129,137，少报 98%）—— 数字看着像模像样。
        """
        scorer = self._scorer(topk=1)
        cand = torch.tensor([[1, 2]])
        scorer.reset_stats()
        scorer.score(torch.tensor([[0, 3]]), cand)      # 全零行
        scorer.score(torch.tensor([[0, 3]]), cand)      # 再来一批
        assert scorer.last_stats["n_rows"] == 2
        assert scorer.last_stats["n_zero_rows"] == 2
        scorer.reset_stats()
        assert scorer.last_stats == {"n_rows": 0, "n_zero_rows": 0,
                                     "zero_row_share": 0.0, "tiebreak": None}

    def test_zero_row_is_reported_and_tiebreak_only_fills_them(self):
        """全零行必须被上报；tiebreak 只填这些行，不能改动其他行的排序。"""
        freq = normalize_frequency(count_train_frequency(TRAIN_SEQS, N_ITEMS))
        plain = self._scorer(topk=1)
        tb = self._scorer(topk=1, popularity=freq, tiebreak="popularity")
        x = torch.tensor([[0, 3]])                     # 历史 = [3]
        cand = torch.tensor([[1, 2]])                  # 两个候选都与 3 无邻居（topk=1）

        p = plain.score(x, cand)
        assert torch.equal(p, torch.zeros_like(p))
        assert plain.last_stats["n_zero_rows"] == 1
        assert plain.last_stats["zero_row_share"] == pytest.approx(1.0)

        t = tb.score(x, cand)
        # 填充值 = 候选的热度：物品 1 频次 3/4，物品 2 频次 4/4
        assert t[0, 0].item() == pytest.approx(freq[1], abs=1e-6)
        assert t[0, 1].item() == pytest.approx(freq[2], abs=1e-6)

    def test_tiebreak_requires_popularity_array(self):
        with pytest.raises(ValueError, match="popularity"):
            self._scorer(topk=1, tiebreak="popularity")

    def test_shape_mismatch_raises(self):
        sim, _ = build_itemcf_similarity(TRAIN_SEQS, N_ITEMS, topk=200)
        with pytest.raises(ValueError, match="形状"):
            ItemCFScorer(sim, n_items=N_ITEMS + 5, topk=200)

    def test_contract_and_no_training(self):
        scorer = self._scorer()
        out = scorer.score(torch.tensor([[0, 1, 2], [0, 0, 2]]),
                           torch.tensor([[2, 3, 1], [1, 3, 2]]))
        assert out.shape == (2, 3)
        assert scorer.n_params == 0
        snap = scorer.config_snapshot()
        assert snap["arch"] == "itemcf" and snap["trained"] is False


# =====================================================================
# 3. GRU4Rec
# =====================================================================
def _gru_cfg(**kw) -> GRU4RecConfig:
    # max_seq_len=8 ⇒ input_cap=6，给"不同左填充数"的对照留出余量
    base = dict(n_items=8, max_seq_len=8, hidden_size=16, num_layers=1, dropout=0.0)
    base.update(kw)
    return GRU4RecConfig(**base)


class TestGRU4Rec:
    def test_contract_shape(self):
        m = GRU4Rec(_gru_cfg()).eval()
        x = torch.tensor([[0, 0, 1, 2], [0, 3, 4, 5]])
        cand = torch.randint(1, 9, (2, 101))
        out = m.score(x, cand)
        assert out.shape == (2, 101)
        assert torch.isfinite(out).all()

    def test_repr_invariant_to_left_padding(self):
        """★ 核心回归：同一个历史 + 不同数量的左填充 ⇒ 表示必须逐位相同。

        只取 `hidden[:, -1]`（或不用 pack_padded_sequence）时这条会失败：
        GRU 在零输入上仍会更新状态，于是"垫了几个 PAD"会改变结果。
        """
        m = GRU4Rec(_gru_cfg()).eval()
        short = torch.tensor([[0, 1, 2, 3]])            # L=4
        long_ = torch.tensor([[0, 0, 0, 1, 2, 3]])      # L=6，同样历史、多垫 2 个 PAD
        with torch.no_grad():
            a = m.encode(short)
            b = m.encode(long_)
        # pad_packed_sequence 输出长度按最长序列对齐，取同一个表示比较
        assert torch.allclose(a, b, atol=1e-6)

    def test_all_pad_row_is_zero_and_finite(self):
        m = GRU4Rec(_gru_cfg()).eval()
        with torch.no_grad():
            r = m.encode(torch.zeros(1, 5, dtype=torch.long))
        assert torch.isfinite(r).all()
        assert torch.equal(r, torch.zeros_like(r))

    def test_batch_with_mixed_lengths_keeps_short_row_unaffected(self):
        """短序列与长序列同批时，短序列的表示不能因为"同批有长序列"而变。"""
        m = GRU4Rec(_gru_cfg()).eval()
        alone = torch.tensor([[0, 0, 0, 1, 2]])
        batched = torch.tensor([[0, 0, 0, 1, 2], [1, 2, 3, 4, 5]])
        with torch.no_grad():
            r1 = m.encode(alone)[0]
            r2 = m.encode(batched)[0]
        assert torch.allclose(r1, r2, atol=1e-6)

    def test_param_count_matches_formula(self):
        m = GRU4Rec(_gru_cfg())
        assert m.n_params == m.expected_n_params()
        # 手算一遍，防止 expected_ 与实现一起写错
        h, V = 16, 9
        assert m.n_params == V * h + (3 * h * h + 3 * h) * 2 + h * h

    def test_pad_row_gets_no_gradient(self):
        m = GRU4Rec(_gru_cfg())
        loss = m.score(torch.tensor([[0, 0, 1, 2]]), torch.tensor([[3, 4]])).sum()
        loss.backward()
        assert m.item_emb.weight.grad[0].abs().sum().item() == 0.0

    def test_input_longer_than_cap_raises(self):
        m = GRU4Rec(_gru_cfg(max_seq_len=6))       # input_cap = 4
        with pytest.raises(ValueError, match="超过上限"):
            m.encode(torch.zeros(1, 5, dtype=torch.long))

    def test_non_suffix_mask_raises_instead_of_silently_wrong(self):
        """滚动对齐的正确性依赖「有效位置是后缀」；不满足必须立刻报错。

        这条防的是"中间挖空"式输入（例如上游出现行内 PAD）——
        静默把有效物品当成 PAD 处理，指标照样算得出来。
        """
        m = GRU4Rec(_gru_cfg()).eval()
        seq = torch.tensor([[0, 5, 6, 0]])          # 第 0 与第 3 位是 PAD ⇒ 挖空
        with pytest.raises(ValueError, match="后缀"):
            m.encode(seq)

    def test_empty_and_batch0_inputs_are_safe(self):
        m = GRU4Rec(_gru_cfg()).eval()
        assert m.encode(torch.zeros(0, 4, dtype=torch.long)).shape == (0, 16)
        assert m.encode(torch.zeros(2, 0, dtype=torch.long)).shape == (2, 16)

    def test_config_snapshot_carries_arch(self):
        snap = GRU4Rec(_gru_cfg()).config_snapshot()
        assert snap["arch"] == "gru4rec"
        assert snap["num_layers"] == 1
        assert snap["n_params"] == GRU4Rec(_gru_cfg()).n_params

    def test_from_dict_ignores_unknown_keys(self):
        """`model` 段里的 SASRec 专属字段（num_heads/ffn_ratio）不能被吃进来报错。"""
        cfg = GRU4RecConfig.from_dict(
            {"hidden_size": 32, "num_layers": 2, "num_heads": 2,
             "ffn_ratio": 4, "share_item_emb": True}, n_items=10)
        assert cfg.hidden_size == 32 and cfg.num_layers == 2

    def test_trains_and_loss_decreases(self):
        """接线检查：单样本过拟合，损失必须真的下降（不验证泛化）。"""
        torch.manual_seed(0)
        m = GRU4Rec(_gru_cfg(hidden_size=32))
        opt = torch.optim.Adam(m.parameters(), lr=0.05)
        crit = nn.BCEWithLogitsLoss()
        x = torch.tensor([[0, 0, 1, 2]])
        cand = torch.tensor([[3, 4]])
        y = torch.tensor([[1.0, 0.0]])
        losses = []
        for _ in range(60):
            opt.zero_grad()
            loss = crit(m.score(x, cand), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        assert losses[-1] < losses[0] * 0.9
        assert losses[-1] < 0.5

    def test_checkpoint_roundtrip_rebuilds_by_arch(self, tmp_path):
        """meta 里的 `arch: gru4rec` 必须让加载端重建出同一个结构。"""
        from models.checkpoint.io import load_model_from_checkpoint, save_checkpoint

        torch.manual_seed(3)
        m = GRU4Rec(_gru_cfg()).eval()
        path = str(tmp_path / "gru4rec_x_best.pt")
        save_checkpoint(path, m, meta={"arch": "gru4rec",
                                       "model_config": m.config_snapshot()})
        loaded, meta = load_model_from_checkpoint(path)
        assert isinstance(loaded, GRU4Rec)
        assert loaded.cfg.hidden_size == m.cfg.hidden_size
        assert loaded.cfg.num_layers == m.cfg.num_layers
        assert meta["model_config"]["arch"] == "gru4rec"

        # 逐位一致（重建路径错了会在这里暴露）
        loaded.eval()
        x = torch.tensor([[0, 0, 1, 2, 3]])
        cand = torch.randint(1, 9, (1, 20))
        with torch.no_grad():
            assert torch.allclose(m.score(x, cand), loaded.score(x, cand), atol=1e-6)


# =====================================================================
# 4. 三个基线都能被同一套评估器直接吃（公平性契约的最终验收）
# =====================================================================
class TestEvaluatorIntegration:
    def _data(self) -> EvalData:
        # ⚠️ 必须写成**规整的左填充**（有效位置是行的后缀）：
        # GRU4Rec 会校验这条前提，行内出现 PAD 会直接报错。
        # 真实数据由 `dataset.build_eval_inputs` 保证（`out[j, cap-take:] = ...`）。
        return _toy_eval_data(
            input_ids=[[0, 0, 1], [0, 0, 2], [0, 1, 2]],
            positives=[2, 1, 2],
            negatives=[[3, 1], [3, 2], [1, 3]],
        )

    def test_popularity_through_evaluator(self):
        scorer = PopularityScorer(normalize_frequency(
            count_train_frequency(TRAIN_SEQS, N_ITEMS)))
        res = evaluate(scorer.score, self._data(), ks=(1, 2), batch_size=2)
        assert res["n_samples"] == 3
        assert 0.0 <= res["hr@1"] <= 1.0 and 0.0 <= res["ndcg@2"] <= 1.0

    def test_itemcf_through_evaluator(self):
        sim, _ = build_itemcf_similarity(TRAIN_SEQS, N_ITEMS, topk=200)
        scorer = ItemCFScorer(sim, n_items=N_ITEMS, topk=200)
        res = evaluate(scorer.score, self._data(), ks=(1, 2), batch_size=2)
        assert res["n_samples"] == 3
        assert 0.0 <= res["hr@2"] <= 1.0

    def test_gru4rec_through_evaluator(self):
        m = GRU4Rec(_gru_cfg()).eval()
        res = evaluate(m.score, self._data(), ks=(1, 2), batch_size=2)
        assert res["n_samples"] == 3 and res["n_candidates"] == 3
        # 未训练模型在 3 候选上不应出现非法值
        assert np.isfinite(res["ndcg@2"])

    def test_all_three_agree_on_sample_count(self):
        """同一份 EvalData 上三个基线的样本量必须一致（否则表不可比）。"""
        data = self._data()
        pop = PopularityScorer(normalize_frequency(
            count_train_frequency(TRAIN_SEQS, N_ITEMS)))
        sim, _ = build_itemcf_similarity(TRAIN_SEQS, N_ITEMS, topk=200)
        icf = ItemCFScorer(sim, n_items=N_ITEMS, topk=200)
        gru = GRU4Rec(_gru_cfg()).eval()
        counts = {evaluate(s.score, data, ks=(2,))["n_samples"]
                  for s in (pop, icf, gru)}
        assert counts == {3}
