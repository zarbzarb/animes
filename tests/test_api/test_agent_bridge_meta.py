# -*- coding: utf-8 -*-
"""`server/services/agent_bridge.py` 的响应成形测试。

这一层的职责是**把 Agent 的产物翻译成对外接口契约**（`docs/api-specification.md`）。
它最容易出的错不是逻辑错，而是**取错层级**：A0 的产物是两层的 ——
`{intent, confidence, result:{items, cache_hit, batch_id, ...}, degraded, lane}`，
业务字段全在 `result` 里，而 `intent / agent_chain` 在第一层。
取错层不会报错，只会让某个字段**恒为默认值**。

实测案例（2026-09-14）：`meta.cache_hit` 一直在读第一层。
A4 明明命中了推荐缓存，接口却永远返回 `false` ——
客户端会以为"缓存从来没生效过"，从而去排查一个不存在的问题。
"""

from server.services.agent_bridge import _meta_of


def _a0(**result_keys):
    """构造一个形状正确的 A0 载荷（业务字段在 `result` 下）。"""
    return {
        "intent": "RECOMMEND_FEED",
        "intent_label": "综合推荐",
        "confidence": 1.0,
        "intent_method": "hint",
        "result": dict(result_keys),
        "degraded": [],
        "lane": ["A1", "A2", "A4"],
        "elapsed_ms": 123,
    }


class TestCacheHitPropagation:
    def test_read_from_nested_result_true(self):
        """回归：`cache_hit` 必须从 `result` 里取，不能取第一层。"""
        assert _meta_of(_a0(cache_hit=True))["cache_hit"] is True

    def test_read_from_nested_result_false(self):
        assert _meta_of(_a0(cache_hit=False))["cache_hit"] is False

    def test_missing_key_defaults_false(self):
        assert _meta_of(_a0())["cache_hit"] is False

    def test_envelope_level_flag_wins(self):
        """信封层的显式 `cache_hit=True`（本轮编排自身命中）也要能生效。"""
        assert _meta_of({"result": {}}, cache_hit=True)["cache_hit"] is True

    def test_top_level_cache_hit_is_ignored(self):
        """反向断言：把 `cache_hit` 放在**第一层**不该被采纳。

        这条把"正确的层级"锁死 —— 否则有人把字段挪到第一层也能过，
        两种写法并存，下一个人又会写错。
        """
        a0 = {"cache_hit": True, "result": {}}
        assert _meta_of(a0)["cache_hit"] is False

    def test_missing_result_does_not_crash(self):
        """`result` 缺位（A0 走 L4 兜底等）时不得 KeyError。"""
        assert _meta_of({})["cache_hit"] is False
        assert _meta_of({"result": None})["cache_hit"] is False


class TestMetaShape:
    def test_meta_contract_keys(self):
        m = _meta_of(_a0(cache_hit=True))
        for k in ("cache_hit", "agent_chain", "degraded", "elapsed_ms",
                  "intent", "intent_label", "intent_method", "confidence"):
            assert k in m, f"meta 少了契约字段 {k}"

    def test_agent_chain_flattened_from_step_dicts(self):
        """`agent_chain` 对外是**字符串数组**（Agent 名），不是 step 对象数组。"""
        a0 = _a0()
        a0["agent_chain"] = [{"agent": "A1", "ok": True},
                             {"agent": "A2", "ok": False}]
        assert _meta_of(a0)["agent_chain"] == ["A1", "A2"]

    def test_agent_chain_falls_back_to_lane(self):
        a0 = _a0()
        assert _meta_of(a0)["agent_chain"] == ["A1", "A2", "A4"]

    def test_degraded_flattened_to_step_names(self):
        a0 = _a0()
        a0["degraded"] = [{"step": "A3", "reason": "timeout"}]
        assert _meta_of(a0)["degraded"] == ["A3"]
