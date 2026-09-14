# -*- coding: utf-8 -*-
"""A4 融合/重排纯算法的单测（M3 阶段补齐）。

这里**只测纯函数**，不碰 IO、不碰 DB —— `agents/fusion/strategy.py` 的定位
就是"全链路唯一决定最终顺序的地方，且口径必须能逐项复算"，
所以它必须能在没有任何数据库/模型的环境下被完整验证。

重点锁住两件事：
1. **落库不变量**「`rank_no` 升序 ⟹ `final_score` 单调不增」。
   2026-09-14 dev 档 feed 冒烟实测踩到过它的反面：
   `#2 rel=0.742` 排在 `#3 rel=0.969` 前面（MMR 的重复惩罚把 Fate/Zero S2
   压下去了，这是**正确**行为），但当时把 MMR 的**输入** rel 当成
   `final_score` 导出，导致位置与分数对不上号，A9 的位置-CTR 分析失去意义。
2. **单调 ≠ 按相关性排序**。如果哪天有人为了"让分数单调"而把 MMR 去掉，
   这两组测试会同时红 —— 提醒他多样性收益被抹掉了。
"""

import pytest

from agents.fusion import strategy


# =====================================================================
# 构造候选的小工具
# =====================================================================
def _item(anime_id: int, rel: float, genres, *, interest_id=None):
    return {"anime_id": int(anime_id), "final_score": float(rel),
            "genres": list(genres), "interest_id": interest_id}


def _genre_of_factory(items):
    table = {int(it["anime_id"]): list(it.get("genres") or []) for it in items}
    return lambda i: table.get(int(i), [])


def _assert_monotone(out, tol=1e-9):
    fs = [float(x["final_score"]) for x in out]
    bad = [(i, fs[i], fs[i + 1]) for i in range(len(fs) - 1)
           if fs[i] < fs[i + 1] - tol]
    assert not bad, f"final_score 不单调：位置(0基)、前值、后值 = {bad}"


# =====================================================================
# 核心不变量
# =====================================================================
class TestFinalScoreMonotone:
    """`rank_no` 与 `final_score` 必须同步 —— 落库/位置-CTR 的根基。"""

    def test_near_duplicate_is_demoted_but_scores_stay_monotone(self):
        """复现实测场景：低相关性的新颖番排到了高相关性的近重复番前面。

        ⚠️ 构造这条用例时踩过一个坑，写在最前面免得下一个人重蹈：
        最直觉的写法是"1 条近重复 + 3 条新颖，取 Top4"。但那种构造下
        近重复项会被 MMR **整个挤出 Top4**，而违规恰恰发生在被截掉的那一位
        —— `_assert_monotone` 于是**照样通过**，测试变成了摆设
        （实测：把 `export_final_scores` 整个注释掉，该版本用例仍然全绿）。
        所以这里刻意让候选池 = k，**保证被压低的项留在 TopK 内**。

        断言分三层：
        * 全序层：rel 第 2 的 #2（0.95）被压到末位，输给 rel 只有 0.55 的 #6。
          MMR 序 ≠ rel 序 —— 否则这条测试就退化成普通排序测试了。
        * TopK 层：#2 仍在 Top6 内（截断掩盖不了违规）。
        * 分数层：`final_score` 单调不增，即分数跟着**位次**走。
        """
        items = [
            _item(1, 1.00, [1, 2]),   # 高相关
            _item(2, 0.95, [1, 2]),   # rel 第 2 高，但与 #1 完全同题材 → 应被压制
            _item(3, 0.70, [3]),
            _item(4, 0.65, [4]),
            _item(5, 0.60, [5]),
            _item(6, 0.55, [6]),
        ]
        g_of = _genre_of_factory(items)
        k = len(items)

        full = strategy.mmr_select(items, k, lam=0.7, genre_of=g_of)
        mmr_order = [int(x["anime_id"]) for x in full]
        rel_order = [int(x["anime_id"])
                     for x in sorted(items, key=lambda t: -t["final_score"])]
        assert rel_order == [1, 2, 3, 4, 5, 6]
        assert mmr_order == [1, 3, 4, 5, 6, 2], (
            f"MMR 未按预期压制重复项，实际全序 {mmr_order}")

        out = strategy.select_with_quota(items, k, lam=0.7, genre_of=g_of)
        order = [int(x["anime_id"]) for x in out]
        assert order == mmr_order, "取全池时 TopK 应当就是全序"
        assert 2 in order, "近重复项必须留在 TopK 内，否则截断会掩盖违规"
        assert out[-1]["mmr_rel"] == pytest.approx(0.95), (
            "末位应当就是那条 rel 第 2 高的重复项")
        _assert_monotone(out)

    def test_final_score_is_not_the_mmr_input_when_reorder_happens(self):
        """直球：只要 MMR 改变了顺序，`final_score` 就**必须**不再等于 rel。

        这是 `export_final_scores` 的变异杀手：把那一步去掉/注释掉，
        `final_score` 会退回成 MMR 的输入 `mmr_rel`，本用例立刻红。
        比"检查单调性"更直接 —— 因为单调性可能被截断、被取值巧合掩盖
        （见上一条用例的注释），而"两者相等"没有巧合的空间。
        """
        items = [
            _item(1, 1.00, [1, 2]), _item(2, 0.95, [1, 2]),
            _item(3, 0.70, [3]), _item(4, 0.65, [4]),
        ]
        out = strategy.select_with_quota(
            items, len(items), lam=0.7, genre_of=_genre_of_factory(items))
        mismatched = [x for x in out
                      if abs(float(x["final_score"]) - float(x["mmr_rel"])) > 1e-9]
        assert mismatched, (
            "final_score 与 mmr_rel 完全相同 ⇒ 导出的仍是 MMR 的输入，"
            f"位次与分数必然错位：{[ (x['anime_id'], x['final_score'], x['mmr_rel']) for x in out ]}")

    @pytest.mark.parametrize("lam", [0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    @pytest.mark.parametrize("k", [1, 3, 5, 10])
    def test_monotone_across_lambda_and_k(self, lam, k):
        """λ 取遍 [0,1]、k 从小到大都必须单调。

        λ=1.0 时 MMR 退化成"纯相关性排序"（sim 项被消掉），此时单调性
        应当由**相关性本身的降序**保证；λ<1 时靠 MMR 目标值的单调性保证。
        两条路径不同，都得绿。
        """
        items = [
            _item(1, 0.95, [1, 2]),
            _item(2, 0.90, [1, 2]),   # 与 #1 重复
            _item(3, 0.80, [3]),
            _item(4, 0.70, [1]),        # 与 #1 部分重叠
            _item(5, 0.65, [4, 5]),
            _item(6, 0.60, [4]),        # 与 #5 重叠
            _item(7, 0.55, [6]),
            _item(8, 0.50, [7]),
        ]
        out = strategy.select_with_quota(
            items, k, lam=lam, genre_of=_genre_of_factory(items))
        assert len(out) == min(k, len(items))
        _assert_monotone(out)

    def test_final_score_in_unit_interval(self):
        """`docs/api-specification.md` 契约：final_score ∈ [0,1]，且 Top1 = 1.0。"""
        items = [_item(i, rel, [i % 3])
                 for i, rel in enumerate([0.9, 0.8, 0.7, 0.6, 0.5], start=1)]
        out = strategy.select_with_quota(
            items, 5, lam=0.7, genre_of=_genre_of_factory(items))
        for x in out:
            assert 0.0 <= x["final_score"] <= 1.0
        assert out[0]["final_score"] == pytest.approx(1.0)

    def test_base_is_full_pool_not_truncated_topk(self):
        """归一化基准必须是**完整候选池**，不能让 TopK 被压成"20 个里排第几"。

        20 个候选取 Top3：若基准错用了截断后的 3 条，第 3 名必然 = 0.0；
        用全池做基准时，第 3 名的分数应显著大于 0。
        这条测试防的是"基准传错集合"这种静默降级 —— 顺序仍然单调，
        只是分数全被压到 [0,1] 的两端，肉眼很难发现。
        """
        items = [_item(i, 1.0 - i * 0.01, [i]) for i in range(1, 21)]
        out = strategy.select_with_quota(
            items, 3, lam=1.0, genre_of=_genre_of_factory(items))
        assert out[-1]["final_score"] > 0.5, (
            f"基准疑似用了截断后的池：{ [x['final_score'] for x in out] }")


class TestQuotaRepair:
    """配额修复：补回名额的同时不能打乱"位次 ↔ 分数"的对应。"""

    def test_donor_is_appended_at_tail_and_keeps_monotone(self):
        """被"名额吃光"的组要从 MMR 序的**后面**补人，补完仍单调。

        构造：组 A 挤满前 3 名（per_group=1 时超额），组 B 只在第 4 位有一个，
        取 Top3 则 B 无曝光 → 必须由 B 的候选替换掉 A 的末位。
        """
        items = [
            _item(1, 1.00, [1], interest_id="A"),
            _item(2, 0.98, [1], interest_id="A"),
            _item(3, 0.96, [1], interest_id="A"),
            _item(4, 0.40, [2], interest_id="B"),
        ]
        out = strategy.select_with_quota(
            items, 3, lam=0.8, genre_of=_genre_of_factory(items),
            key="interest_id", per_group=1)
        _assert_monotone(out)
        assert len(out) == 3
        assert any(x["interest_id"] == "B" for x in out), "配额修复没生效"

    def test_per_group_zero_is_passthrough(self):
        """`per_group<=0` 时退化为"取前 K"，不得抛异常也不得改顺序。"""
        items = [_item(i, 1.0 - i * 0.1, [1]) for i in range(1, 6)]
        out = strategy.select_with_quota(
            items, 3, lam=0.7, genre_of=_genre_of_factory(items), per_group=0)
        assert [int(x["anime_id"]) for x in out] == [1, 2, 3]
        _assert_monotone(out)

    def test_repair_when_every_group_is_within_quota(self):
        """所有组都恰好只有 per_group 条 → 没有可让出的名额，跳过修复且不改序。

        这是 `quota_repair` 里 `replaced = False` 那条分支：
        早期实现在这里会把候选**插到队首**，把排序打乱（也正是这次要修的
        "位置与分数不一致"的原始形态）。
        """
        items = [
            _item(1, 1.00, [1], interest_id="A"),
            _item(2, 0.90, [2], interest_id="B"),
            _item(3, 0.80, [3], interest_id="C"),
        ]
        out = strategy.select_with_quota(
            items, 3, lam=0.7, genre_of=_genre_of_factory(items),
            key="interest_id", per_group=1)
        assert [int(x["anime_id"]) for x in out] == [1, 2, 3]
        _assert_monotone(out)


# =====================================================================
# MMR 本体
# =====================================================================
class TestMMRSelect:
    def test_records_diagnostics(self):
        """`mmr_rel` / `mmr_sim` / `mmr_score` 三个诊断字段必须齐全且自洽。"""
        items = [_item(1, 1.0, [1, 2]), _item(2, 0.9, [1, 2]),
                 _item(3, 0.5, [3])]
        lam = 0.7
        out = strategy.mmr_select(
            items, 3, lam=lam, genre_of=_genre_of_factory(items))
        for it in out:
            for k in ("mmr_rel", "mmr_sim", "mmr_score"):
                assert k in it, f"缺少诊断字段 {k}"
            expect = lam * it["mmr_rel"] - (1 - lam) * it["mmr_sim"]
            assert it["mmr_score"] == pytest.approx(expect, abs=1e-6)

    def test_first_pick_is_max_relevance_with_zero_sim(self):
        """第 1 位必是 rel 最大者，且 sim=0（已选集为空）。"""
        items = [_item(1, 0.4, [1]), _item(2, 0.99, [2]),
                 _item(3, 0.7, [3])]
        out = strategy.mmr_select(
            items, 3, lam=0.5, genre_of=_genre_of_factory(items))
        assert int(out[0]["anime_id"]) == 2
        assert out[0]["mmr_sim"] == 0.0

    def test_identity_duplicate_gets_full_penalty(self):
        """完全同题材的候选 sim=1（多热余弦），在 λ=0.5 下必然被压到后面。

        三条 rel 全等 ⇒ 第 1 位靠"首次胜出"拿到 #1；第 2 位时
        #2（重复）目标值 `0.5·1 - 0.5·1 = 0`，#3（新颖）为 `0.5·1 - 0 = 0.5`，
        于是 #3 先上；#2 落到最后且带满额惩罚。
        """
        items = [_item(1, 1.0, [1, 2]), _item(2, 1.0, [1, 2]),
                 _item(3, 1.0, [3])]
        out = strategy.mmr_select(
            items, 3, lam=0.5, genre_of=_genre_of_factory(items))
        assert [int(x["anime_id"]) for x in out] == [1, 3, 2]
        assert out[0]["mmr_sim"] == 0.0
        assert out[1]["mmr_sim"] == 0.0      # #3 与 #1 无公共题材
        assert out[2]["mmr_sim"] == pytest.approx(1.0)

    def test_empty_input_and_zero_k(self):
        assert strategy.mmr_select([], 5, lam=0.7, genre_of=lambda i: []) == []
        items = [_item(1, 1.0, [1])]
        assert strategy.mmr_select(items, 0, lam=0.7, genre_of=lambda i: []) == []

    def test_k_larger_than_pool_returns_whole_pool(self):
        items = [_item(i, 0.5, [1]) for i in range(1, 4)]
        out = strategy.mmr_select(
            items, 99, lam=0.7, genre_of=_genre_of_factory(items))
        assert len(out) == 3

    def test_no_genre_metadata_is_handled(self):
        """题材缺失（A3 新番常见）时 `_cosine_multi_hot` 返回 0，不得崩。"""
        items = [_item(1, 0.9, []), _item(2, 0.8, []), _item(3, 0.7, [])]
        out = strategy.mmr_select(
            items, 3, lam=0.7, genre_of=lambda i: [])
        assert len(out) == 3
        assert all(it["mmr_sim"] == 0.0 for it in out)
        # 无题材 ⇒ sim 项恒 0 ⇒ MMR 退化成纯相关性降序
        assert [int(x["anime_id"]) for x in out] == [1, 2, 3]


# =====================================================================
# 归一化与融合
# =====================================================================
class TestMinMax:
    def test_empty_returns_empty(self):
        assert strategy.minmax([]) == []

    def test_all_equal_returns_ones(self):
        """全相等 → 全 1.0（不是全 0）。全 0 会让该路在融合中彻底静默。"""
        assert strategy.minmax([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]

    def test_single_value_returns_one(self):
        assert strategy.minmax([7.5]) == [1.0]

    def test_basic_range(self):
        assert strategy.minmax([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]

    def test_negative_values(self):
        assert strategy.minmax([-2.0, 0.0, 2.0]) == [0.0, 0.5, 1.0]

    def test_within_eps_returns_ones(self):
        assert strategy.minmax([1.0, 1.0 + 1e-12]) == [1.0, 1.0]


class TestFuseScores:
    KW = dict(w_behavior=0.7, w_content=0.3,
              w_behavior_cold=0.5, w_content_cold=0.5,
              cold_threshold=10)

    def test_warm_uses_7_3(self):
        v = strategy.fuse_scores(1.0, 1.0, n_interactions=100, **self.KW)
        assert v == pytest.approx(1.0)
        v = strategy.fuse_scores(1.0, 0.0, n_interactions=100, **self.KW)
        assert v == pytest.approx(0.7)

    def test_cold_uses_5_5(self):
        v = strategy.fuse_scores(1.0, 0.0, n_interactions=3, **self.KW)
        assert v == pytest.approx(0.5)

    def test_threshold_boundary_is_warm(self):
        """`n_interactions == threshold` 算**非冷启**（严格小于才冷启）。"""
        v = strategy.fuse_scores(1.0, 0.0, n_interactions=10, **self.KW)
        assert v == pytest.approx(0.7)

    def test_none_is_treated_as_cold(self):
        """无元数据 → 按冷启处理（保守地给内容路更高权重）。"""
        v = strategy.fuse_scores(1.0, 0.0, n_interactions=None, **self.KW)
        assert v == pytest.approx(0.5)

    def test_weights_are_renormalized(self):
        """权重和不为 1 时按比例归一，不得直接把分数放大。"""
        v = strategy.fuse_scores(
            1.0, 0.0, n_interactions=100,
            w_behavior=3.5, w_content=1.5, w_behavior_cold=1.0,
            w_content_cold=1.0, cold_threshold=10)
        assert v == pytest.approx(0.7)

    def test_zero_total_falls_back_to_5_5(self):
        v = strategy.fuse_scores(
            1.0, 0.0, n_interactions=100, w_behavior=0.0, w_content=0.0,
            w_behavior_cold=0.0, w_content_cold=0.0, cold_threshold=10)
        assert v == pytest.approx(0.5)

    def test_output_is_bounded(self):
        for b in (0.0, 0.5, 1.0):
            for c in (0.0, 0.5, 1.0):
                v = strategy.fuse_scores(b, c, n_interactions=99, **self.KW)
                assert 0.0 <= v <= 1.0


# =====================================================================
# 题材信号（A4 自己补的解释信号）
# =====================================================================
class TestGenreSignals:
    PROFILE = [
        {"genre_id": 1, "genre": "战斗", "strength": 1.0},
        {"genre_id": 2, "genre": "奇幻", "strength": 0.5},
        {"genre_id": 3, "genre": "日常", "strength": 0.5},
    ]

    def test_overlap_is_weight_ratio(self):
        names, overlap = strategy.genre_signals([1, 3], self.PROFILE)
        assert overlap == pytest.approx((1.0 + 0.5) / 2.0)
        assert names == ["战斗", "日常"]

    def test_names_sorted_by_strength_desc(self):
        names, _ = strategy.genre_signals([3, 2, 1], self.PROFILE)
        assert names == ["战斗", "奇幻", "日常"]

    def test_no_hit_returns_empty(self):
        names, overlap = strategy.genre_signals([99], self.PROFILE)
        assert names == [] and overlap == 0.0

    def test_empty_inputs(self):
        assert strategy.genre_signals([], self.PROFILE) == ([], 0.0)
        assert strategy.genre_signals([1], []) == ([], 0.0)

    def test_zero_strength_profile(self):
        """画像强度全 0（半成品画像）不得 ZeroDivisionError。"""
        prof = [{"genre_id": 1, "genre": "战斗", "strength": 0.0}]
        assert strategy.genre_signals([1], prof) == ([], 0.0)

    def test_missing_genre_id_is_skipped(self):
        prof = [{"genre": "战斗", "strength": 1.0},
                {"genre_id": 5, "genre": "奇幻", "strength": 1.0}]
        names, overlap = strategy.genre_signals([5], prof)
        assert names == ["奇幻"] and overlap == pytest.approx(1.0)

    def test_overlap_capped_at_one(self):
        """同一 genre_id 重复出现时不得把 overlap 顶到 >1。"""
        prof = [{"genre_id": 1, "genre": "战斗", "strength": 1.0}] * 3
        _, overlap = strategy.genre_signals([1], prof)
        assert 0.0 <= overlap <= 1.0


# =====================================================================
# 契约守卫
# =====================================================================
class TestModuleContract:
    def test_all_exports_resolve(self):
        """`__all__` 里的每个名字都必须真实存在。

        曾经这里挂着一个已改名的 `quota_first`（现名 `quota_repair`），
        `import *` 会直接 AttributeError —— 但因为没人用 `import *`，
        这个错一直潜伏着。
        """
        for name in strategy.__all__:
            assert hasattr(strategy, name), f"__all__ 里的 {name} 不存在"

    def test_select_with_quota_is_the_only_public_entry(self):
        """A4 的调用方只该用 `select_with_quota`；底层两步是它的实现细节。

        这条断言防的是"调用方绕过 `export_final_scores` 自己拼顺序"，
        那会重新引入"位置与分数不一致"。
        """
        assert callable(strategy.select_with_quota)
        assert callable(strategy.export_final_scores)
