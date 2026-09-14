# -*- coding: utf-8 -*-
"""A0 的意图识别（`docs/agents.md` §A0：8 类意图）。

两级策略：**规则优先，LLM 兜底**
-------------------------------
为什么不是"全部交给 LLM"：
* 主链路总预算 800ms，而一次 LLM 往返就是 150~300ms，占掉 1/4 以上；
  常见的 6 类意图（"推荐"、"新番"、"类似XX"）用关键词就能 95% 命中；
* LLM 不可用时系统**必须仍能工作** —— 如果意图识别只能靠 LLM，
  那就等于把整条推荐链路绑死在外部 API 上。

为什么还要 LLM：余下的 2 类（`SEARCH_ANIME` / `SYSTEM_QUERY`）与
口语化表达（"有没有那种看了就停不下来的"）规则覆盖不了，
但它们是**非主链路**，慢一点可以接受。

⚠️ 一个必须写清的边界：本模块**只决定"走哪条流程"**，不做任何排序，
也不生成文案（ADR-1）。
"""

from __future__ import annotations

import re
from typing import Optional

from agents.common.llm import get_llm

__all__ = ["INTENTS", "INTENT_LABELS", "classify_rules", "classify"]

INTENTS: tuple[str, ...] = (
    "RECOMMEND_FEED",        # 综合推荐（首页）
    "RECOMMEND_BY_GENRE",    # 按题材推荐
    "RECOMMEND_NEW_ANIME",   # 新番专区
    "EXPLAIN_RECOMMEND",     # 解释某条推荐
    "CHAT_RECOMMEND",        # 对话推荐
    "ANALYZE_INTEREST",      # 兴趣漂移分析
    "SEARCH_ANIME",          # 检索动漫
    "SYSTEM_QUERY",          # 系统状态/指标查询
)

INTENT_LABELS = {
    "RECOMMEND_FEED": "综合推荐",
    "RECOMMEND_BY_GENRE": "按题材推荐",
    "RECOMMEND_NEW_ANIME": "新番推荐",
    "EXPLAIN_RECOMMEND": "推荐解释",
    "CHAT_RECOMMEND": "对话推荐",
    "ANALYZE_INTEREST": "兴趣分析",
    "SEARCH_ANIME": "动漫检索",
    "SYSTEM_QUERY": "系统查询",
}

# 关键词表：**顺序即优先级**（先匹配到的赢）。
# 顺序经过刻意安排：`EXPLAIN` 在 `RECOMMEND` 之前 ——
# "为什么推荐这部" 同时含"推荐"，若按推荐优先会被错误路由到首页推荐。
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("EXPLAIN_RECOMMEND", ("为什么推荐", "为什么给我推", "推荐理由", "解释一下",
                            "为何推荐", "凭什么推")),
    ("ANALYZE_INTEREST", ("兴趣变化", "口味变化", "兴趣漂移", "我的偏好",
                          "兴趣分析", "品味变化", "兴趣雷达")),
    ("RECOMMEND_NEW_ANIME", ("新番", "新作", "本季", "这个季度", "最新番",
                              "刚出的番")),
    ("RECOMMEND_BY_GENRE", ("题材", "类型", "看个", "来点", "有没有")),
    ("CHAT_RECOMMEND", ("类似", "像", "差不多", "但更", "轻松点", "换换口味",
                        "推荐点", "聊聊")),
    ("SEARCH_ANIME", ("搜索", "找一下", "查一下", "叫什么的", "那部")),
    ("SYSTEM_QUERY", ("系统状态", "健康", "指标", "降级", "监控", "在线人数")),
    ("RECOMMEND_FEED", ("推荐", "首页", "给我推", "有什么好看", "看点什么")),
)

_GENRE_WORDS = ("热血", "战斗", "冒险", "奇幻", "日常", "治愈", "恋爱", "校园",
                "悬疑", "推理", "科幻", "机战", "喜剧", "搞笑", "运动", "竞技",
                "超自然", "灵异", "剧情", "文艺", "音乐", "偶像", "后宫", "福利")

# 编号《》里的番名 → 对话推荐（"有没有类似《咒术回战》但轻松点的"）
_SEQUEL_RE = re.compile(r"(类似|像|差不多)[《〈]?([^》〉，,。?？]{2,30})")


def _hits(text: str, words: tuple[str, ...]) -> int:
    return sum(1 for w in words if w in text)


def classify_rules(raw_query: str) -> tuple[Optional[str], float, dict]:
    """规则识别。返回 `(intent | None, confidence, evidence)`。

    置信度按命中词数与是否唯一命中给出 —— 不是"猜一个数"，
    而是 `0.6 + 0.1*命中数`（上限 0.9），命中数 ≥3 才算高置信。
    `None` 表示规则无法判断，交给 LLM。
    """
    text = (raw_query or "").strip()
    if not text:
        return "RECOMMEND_FEED", 0.5, {"reason": "empty_query_default"}

    evidence: dict = {"matched": {}, "genre_words": []}
    for intent, words in _RULES:
        n = _hits(text, words)
        if n:
            evidence["matched"][intent] = n
    if not evidence["matched"]:
        return None, 0.0, evidence

    best_n = max(evidence["matched"].values())
    # 按 _RULES 顺序取第一个达到最高命中的意图 = 优先级语义
    for intent, words in _RULES:
        if evidence["matched"].get(intent) == best_n:
            evidence["genre_words"] = [w for w in _GENRE_WORDS if w in text]
            conf = min(0.9, 0.6 + 0.1 * best_n)
            return intent, round(conf, 2), evidence
    return None, 0.0, evidence


async def classify(raw_query: str, intent_hint: Optional[str] = None,
                   *, use_llm: bool = True, timeout_ms: int = 300
                   ) -> tuple[str, float, str, dict]:
    """`(intent, confidence, method, evidence)`。`method ∈ {hint, rule, llm, default}`。

    `intent_hint` 来自前端（用户点了"新番"标签），**最高优先**：
    显式意图永远比推断可靠。

    ⚠️ 本函数是 `async`：LLM 兜底必须 `await`。早期版本在这里用了
    `asyncio.run()`，在协程里会直接抛 `RuntimeError: asyncio.run() cannot be
    called from a running event loop` —— 而它被 `except RuntimeError` 吞掉后，
    表现为"LLM 兜底永远不生效"，属于最难查的静默失效。
    """
    if intent_hint and intent_hint.upper() in INTENTS:
        return intent_hint.upper(), 1.0, "hint", {"reason": "explicit_hint"}

    intent, conf, evidence = classify_rules(raw_query)
    if intent is not None and conf >= 0.7:
        return intent, conf, "rule", evidence

    if use_llm:
        intent_llm = await _classify_llm(raw_query, timeout_ms)
        if intent_llm:
            return intent_llm, 0.75, "llm", evidence

    if intent is not None:
        return intent, conf, "rule", evidence
    return "RECOMMEND_FEED", 0.3, "default", evidence


_SYSTEM = """你是动漫推荐系统的意图分类器。把用户输入归到唯一一个意图上。

可选意图（只输出英文标识符，不要解释）：
RECOMMEND_FEED      综合推荐（想看点推荐，没有特别限定）
RECOMMEND_BY_GENRE  按题材推荐（明确说了题材/类型）
RECOMMEND_NEW_ANIME 新番推荐（提到新番/本季/最新）
EXPLAIN_RECOMMEND   推荐解释（问"为什么推荐"）
CHAT_RECOMMEND      对话推荐（"有没有类似X但更Y的"这类多轮对话）
ANALYZE_INTEREST    兴趣分析（问自己口味/兴趣的变化）
SEARCH_ANIME        检索动漫（找某一部，问叫什么）
SYSTEM_QUERY        系统查询（问系统状态、指标、健康度）

只输出那一个标识符。"""


async def _classify_llm(raw_query: str, timeout_ms: int) -> Optional[str]:
    """LLM 兜底。超时 300ms —— 最多吃掉 800ms 主链路预算的 3/8。

    返回 `None` 表示"识别不出/不可用"，由调用方回落到规则结果，
    **绝不抛异常**（意图识别失败不该让整个请求 500）。
    """
    llm = get_llm()
    if not llm.available or not (raw_query or "").strip():
        return None
    try:
        res = await llm.complete(
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": raw_query.strip()[:200]}],
            temperature=0.0, max_tokens=16, timeout_ms=timeout_ms)
    except Exception:               # LLM 客户端理论上不抛，这里兜底
        return None
    if not res.ok:
        return None
    parts = res.text.strip().upper().split()
    token = re.sub(r"[^A-Z_]", "", parts[0]) if parts else ""
    return token if token in INTENTS else None
