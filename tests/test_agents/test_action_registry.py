# -*- coding: utf-8 -*-
"""`action` 注册表的三方一致性守卫。

`agents/common/envelope.py::ACTION_REGISTRY` 与
`docs/agent-interaction-protocol.md` §3.3 的表在注释里互相承诺"改一个要改另一个"，
但**注释不是机制**。`BaseAgent.handle()` 在真正派发前会查这张表做白名单
（`未登记的 action：xxx`），所以表一旦漂移，症状是"代码里明明实现了某个 action，
线上却报未登记"，而两边都看不出哪边对。

这份测试把三方钉在一起：

    ACTION_REGISTRY(代码)  ↔  AGENT_CLASSES(可注册的 Agent)  ↔  协议文档 §3.3 表格

另外锁住一个容易在复制粘贴里出错的点：**每个 Agent 的 action 必须同属一个域**
（A4 只能是 `rank.*`），否则 A0 的路由表按"域"匹配时会静默路由到错的 Agent。
"""

import asyncio
import importlib
import re
from pathlib import Path

import pytest

from agents.bootstrap import AGENT_CLASSES
from agents.common.base import BaseAgent
from agents.common.envelope import ACTION_REGISTRY
from agents.common.errors import make

DOC = Path(__file__).resolve().parents[2] / "docs" / "agent-interaction-protocol.md"

EXPECTED_AGENTS = tuple(f"A{i}" for i in range(10))     # A0..A9

# 「域」→ 该域只能属于一个 Agent。域不是 agent_id 的函数，所以单独列出来锁住。
DOMAIN_OWNER = {
    "orchestrate": "A0", "profile": "A1", "recall": "A2", "content": "A3",
    "rank": "A4", "explain": "A5", "drift": "A6", "chat": "A7",
    "dataops": "A8", "eval": "A9",
}


def _load_agent_classes():
    """按 `AGENT_CLASSES` 动态载入类对象（不调用 `register_all()`，避免全局副作用）。

    ⚠️ 不用 `BaseAgent.__subclasses__()` 取类：它只反映**当前已 import** 的子类，
    于是测试结果取决于其它测试有没有先导入某个模块（顺序敏感、偶发红）。
    按注册表显式载入才是确定的。
    """
    out = {}
    for agent_id, path in AGENT_CLASSES:
        module_name, cls_name = path.split(":")
        out[agent_id] = getattr(importlib.import_module(module_name), cls_name)
    return out


def _cls(agent_id: str) -> type:
    return _load_agent_classes()[agent_id]


def _doc_action_table() -> dict[str, set[str]]:
    """解析协议文档 §3.3 的表格 → `{agent_id: {action, ...}}`。

    行格式：`| A4 | `rank.fusion`、`rank.diversify` |`
    （用 `、` 分隔、反引号包裹；表里也可能出现不带反引号的裸词，一并接受）
    """
    text = DOC.read_text(encoding="utf-8")
    # 只取 §3.3 到下一个二级标题之间的内容，避免误吃别的表格
    m = re.search(r"### 3\.3.*?\n(.*?)\n---", text, re.S)
    assert m, f"在 {DOC} 里找不到 §3.3 段落 —— 文档结构改了，本解析器需要同步更新"
    table: dict[str, set[str]] = {}
    for line in m.group(1).splitlines():
        row = re.match(r"^\|\s*(A\d)\s*\|(.+?)\|\s*$", line.strip())
        if not row:
            continue
        agent_id, cells = row.group(1), row.group(2)
        table[agent_id] = set(re.findall(r"`([^`]+)`", cells)) or {
            w.strip() for w in re.split(r"[、,]", cells) if w.strip()
        }
    return table


# =====================================================================
class TestRegistryShape:
    def test_covers_a0_to_a9(self):
        assert tuple(sorted(ACTION_REGISTRY)) == EXPECTED_AGENTS

    def test_every_action_is_unique(self):
        """没有哪个 action 被两个 Agent 同时认领 —— 否则 A0 不知道该发给谁。"""
        seen: dict[str, str] = {}
        dup = []
        for agent_id, actions in ACTION_REGISTRY.items():
            assert actions, f"{agent_id} 没有任何 action"
            for a in actions:
                if a in seen:
                    dup.append(f"{a}（{seen[a]} 与 {agent_id}）")
                seen[a] = agent_id
        assert not dup, f"action 重复登记：{dup}"

    def test_no_duplicate_within_agent(self):
        for agent_id, actions in ACTION_REGISTRY.items():
            assert len(actions) == len(set(actions)), f"{agent_id} 内部有重复 action"

    def test_naming_convention(self):
        """`<域>.<动词>`、全小写、只允许小写字母/数字/下划线。"""
        for agent_id, actions in ACTION_REGISTRY.items():
            for a in actions:
                assert re.fullmatch(r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*", a), (
                    f"{agent_id} 的 {a!r} 不符合 `<域>.<动词>` 全小写规范")

    def test_each_agent_has_exactly_one_domain(self):
        for agent_id, actions in ACTION_REGISTRY.items():
            domains = {a.split(".")[0] for a in actions}
            assert len(domains) == 1, f"{agent_id} 混用了多个域：{sorted(domains)}"

    def test_domain_ownership(self):
        for agent_id, actions in ACTION_REGISTRY.items():
            domain = next(iter(actions)).split(".")[0]
            assert DOMAIN_OWNER.get(domain) == agent_id, (
                f"域 {domain} 应属于 {DOMAIN_OWNER.get(domain)}，实际记在 {agent_id}")


class TestAgentsAgreeWithRegistry:
    def test_agent_classes_exist_and_are_agents(self):
        for agent_id, cls in _load_agent_classes().items():
            assert issubclass(cls, BaseAgent), f"{agent_id} 不是 BaseAgent 子类"
            assert cls.agent_id == agent_id, (
                f"{agent_id} 的类里写的是 {cls.agent_id} —— 注册表/类不一致")

    def test_agent_classes_cover_registry(self):
        assert set(_load_agent_classes()) == set(ACTION_REGISTRY)

    @pytest.mark.parametrize("agent_id", EXPECTED_AGENTS)
    def test_explicit_actions_match_registry(self, agent_id):
        """类里若**显式**写了 `actions`，就必须与注册表逐项相同。

        （不显式写的类由 `BaseAgent.__init__` 从表里兜底，见 base.py 第 50 行 ——
        这是允许的，但不能与表冲突。）
        """
        declared = _cls(agent_id).__dict__.get("actions")
        if declared:
            assert tuple(declared) == tuple(ACTION_REGISTRY[agent_id]), (
                f"{agent_id} 类里声明 {tuple(declared)}，注册表是 "
                f"{tuple(ACTION_REGISTRY[agent_id])}")

    @pytest.mark.parametrize("agent_id", EXPECTED_AGENTS)
    def test_declared_actions_accessor(self, agent_id):
        assert tuple(_cls(agent_id).declared_actions()) == \
            tuple(ACTION_REGISTRY[agent_id])


class TestDocumentationParity:
    def test_doc_table_matches_registry(self):
        """协议文档 §3.3 必须与代码里的表一致。

        文档那句"新增 action 必须在本文档登记，否则 CI 会失败"靠的就是这条 ——
        没有它，那句话只是愿望。
        """
        doc = _doc_action_table()
        assert set(doc) == set(ACTION_REGISTRY), (
            f"文档少登记 {sorted(set(ACTION_REGISTRY) - set(doc))}；"
            f"多出 {sorted(set(doc) - set(ACTION_REGISTRY))}")
        for agent_id in EXPECTED_AGENTS:
            assert doc[agent_id] == set(ACTION_REGISTRY[agent_id]), (
                f"{agent_id} 文档写的是 {sorted(doc[agent_id])}，"
                f"代码里是 {sorted(ACTION_REGISTRY[agent_id])}")

    def test_doc_parser_is_not_vacuous(self):
        """解析器必须真的解析出了内容 —— 否则上一条会"空对空"地通过。

        同类教训：一个只会返回空集的解析器，能让任何"集合相等"断言变绿。
        """
        doc = _doc_action_table()
        assert len(doc) == 10
        assert sum(len(v) for v in doc.values()) >= 20
        assert "orchestrate.route" in doc["A0"]


class TestUnknownActionIsRejected:
    """白名单必须在 `invoke()` 之前生效，并回协议规定的错误码。"""

    @pytest.mark.parametrize("agent_id", EXPECTED_AGENTS)
    def test_unknown_action_returns_error_envelope(self, agent_id):
        from agents.common.envelope import Envelope
        env = Envelope.request(from_agent="TEST", to_agent=agent_id,
                               action="definitely.not_registered", payload={},
                               trace_id="tr_00000000000f", timeout_ms=500)
        out = asyncio.run(_cls(agent_id)().handle(env))

        assert out.header.type == "error", f"{agent_id} 未拦截未登记的 action"
        p = out.payload or {}
        assert p.get("error_code") == make("INVALID_PARAM").code
        assert p.get("error_name") == "INVALID_PARAM"
        # `allowed` 要回带可选值，方便调用方自查（协议 §4.2）
        assert set(p.get("detail", {}).get("allowed") or []) == \
            set(ACTION_REGISTRY[agent_id])

    def test_registered_action_passes_the_whitelist(self):
        """反向对照：已登记的 action 不该被白名单拦住。

        只断言"不是 INVALID_PARAM"，**不**断言业务成功 ——
        那需要真实网关/模型，属于端到端冒烟的职责（`scripts/smoke_api.py`）。
        """
        from agents.common.envelope import Envelope
        from agents.orchestrator.agent import OrchestratorAgent
        env = Envelope.request(from_agent="TEST", to_agent="A0",
                               action="orchestrate.aggregate", payload={"results": {}},
                               trace_id="tr_00000000000e", timeout_ms=800)
        out = asyncio.run(OrchestratorAgent().handle(env))
        body = out.payload or {}
        assert body.get("error_name") != "INVALID_PARAM"
