# -*- coding: utf-8 -*-
"""A5 解释生成的提示词与事实校验（`docs/agent-prompt-design.md`）。

这是全项目**唯一**由 LLM 生成、且会直接展示给用户的文案。所以这里
同时是「生成器」和「守门人」：

生成 → 校验 → 不合格就丢
-----------------------
`validate_reason()` 是本模块的核心。它拦三类问题：

1. **库外番剧名**：输出的书名号内容必须属于 `core_items` 里的番。
   LLM 极爱补充"类似《XXX》"这类训练数据里的常见搭配 —— 那是幻觉。
2. **库外题材**：出现的题材词必须来自 `matched_genres`。
3. **标识信息**：邮箱/手机号（复用 `llm.scrub_pii`）。

拦截后**不重试**（协议 §4.2 的 60604 是"丢弃重生成 → 降级"，
而 `complete()` 内部已经有一次重试），直接走模板兜底 —— 200ms 预算
不允许为了文案做第二轮。
"""

from __future__ import annotations

import re

from agents.common.llm import scrub_pii

__all__ = [
    "PROMPT_VERSION", "SYSTEM_PROMPT", "build_messages",
    "fallback_reason", "match_percent", "validate_reason",
]

PROMPT_VERSION = "a5.explain.v1"

SYSTEM_PROMPT = """你是动漫推荐系统的推荐理由生成器。
根据给定的**结构化信号**，为用户解释"为什么推荐这部动漫"。

硬性要求（违反即作废）：
1. 只能引用信号中出现过的番剧名称。**严禁**提及任何其它番剧。
2. 只能使用信号中出现的题材名称，不得引入新题材。
3. 不得编造用户行为（例如信号里没有"收藏"，就不能说"你收藏了"）。
4. 1~2 句话，总长不超过 60 个汉字，不要表情、不要换行、不要引号。
5. 句末以"。"结束。
6. 只输出理由本身，不要"理由："之类的前缀。

句式参考：
- 「你近期看过的《{番A}》与它同属{题材}，题材匹配度 {N}%」
- 「结合你对{题材}的偏好，这部与你常看的《{番A}》《{番B}》很接近」
"""

_STYLE_HINT = {
    "concise": "简洁陈述，一句话为主。",
    "detailed": "可以两句，先说命中题材再说依据的番剧。",
    "casual": "口语化、轻松，像朋友推荐；仍不得超过 60 字。",
}

# 书名号包裹的番剧名（中文《》与日式〈〉都要认）
_TITLE_RE = re.compile(r"[《〈]([^》〉]{1,40})[》〉]")
_PREFIX_RE = re.compile(r"^\s*(理由|推荐理由|原因)\s*[:：]\s*")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


def build_messages(target: dict, signals: dict, style: str = "concise") -> list[dict]:
    top = signals.get("top_attn_items") or []
    names = "、".join(f"《{t.get('title')}》" for t in top if t.get("title")) or "无"
    matched = "、".join(signals.get("matched_genres") or []) or "无"
    overlap = float(signals.get("genre_overlap") or 0.0)
    label = signals.get("interest_label") or "无"
    user = (
        f"目标动漫：《{target.get('title') or '（无标题）'}》\n"
        f"目标动漫题材：{'、'.join(target.get('genres') or []) or '无'}\n"
        f"用户看过的相关番剧（只能引用这些）：{names}\n"
        f"命中的题材（只能用这些）：{matched}\n"
        f"题材重合度：{round(overlap * 100)}%\n"
        f"命中的兴趣维度：{label}\n"
        f"相关番剧数量：{int(signals.get('hit_count') or 1)}\n"
        f"风格要求：{_STYLE_HINT.get(style, _STYLE_HINT['concise'])}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def match_percent(overlap: float, hit_count: int = 1) -> int:
    """matched 百分比。

    口径：`round(overlap * 100)` 为主，**不**因为 hit_count 多就给 100%。
    刻意加上限 95：把 `100%` 留给"完全重合"这种几乎不会发生的极端，
    否则界面上到处都是"匹配度 100%"，用户立刻不信这个数字了。
    """
    pct = int(round(max(0.0, min(1.0, float(overlap))) * 100))
    return max(0, min(95, pct))


def fallback_reason(top_attn: list[dict], matched: list[str],
                    overlap: float, *, style: str = "concise") -> str:
    """模板兜底（协议 §5.2 明写的那句式）。

    「你近期观看了《{top1}》《{top2}》等{genre}番（题材匹配度{match}%）」

    没有 `top_attn` 时退化成不带番名的版本 —— 而不是空字符串。
    空理由在界面上是一个孤零零的卡片，比"因为题材相近"更糟。
    """
    pct = match_percent(overlap)
    names = [str(t.get("title")) for t in top_attn if t.get("title")]
    genre = matched[0] if matched else "同类"

    if len(names) >= 2:
        return f"你近期观看了《{names[0]}》《{names[1]}》等{genre}番（题材匹配度{pct}%）"
    if len(names) == 1:
        return f"你近期观看了《{names[0]}》，它的题材与这部{genre}番相近（匹配度{pct}%）"
    if matched:
        return f"结合你对{genre}题材的偏好，为你推荐这部（题材匹配度{pct}%）"
    return "根据你近期的观看偏好，为你推荐这部作品"


def validate_reason(text: str, *, allowed_titles: list[str],
                    allowed_genres: list[str],
                    max_len: int = 60) -> tuple[bool, str, list[str]]:
    """事实校验。返回 `(是否通过, 清洗后的文本, 违规项列表)`。

    通过后仍然会把文本裁剪到 `max_len` —— 宁可截断也不要因为超长就丢弃
    （超长主要是措辞啰嗦，不是事实错误）。
    """
    violations: list[str] = []
    if not text:
        return False, "", ["empty"]

    cleaned = scrub_pii(str(text)).strip()
    cleaned = _PREFIX_RE.sub("", cleaned)
    cleaned = _CTRL_RE.sub("", cleaned).replace("\n", "").strip().strip('"“”')
    if not cleaned:
        return False, "", ["empty_after_clean"]

    # ① 库外番剧名
    allowed_t = [str(t).strip() for t in allowed_titles if str(t).strip()]
    for m in _TITLE_RE.findall(cleaned):
        name = m.strip()
        if not any(name == a or name in a or a in name for a in allowed_t):
            violations.append(f"unknown_title:{name}")

    # ② 库外题材
    allowed_g = [str(g).strip() for g in allowed_genres if str(g).strip()]
    for token in re.findall(r"[\u4e00-\u9fff]{2,8}", cleaned):
        if token in ("题材", "推荐", "匹配", "偏好", "观看", "近期", "因为", "所以",
                     "这部", "一部", "同类", "作品", "动漫", "番剧", "结合", "非常",
                     "适合", "值得", "热度", "口碑", "评分"):
            continue
        if token.endswith("番") and len(token) > 1:
            base = token[:-1]
            if allowed_g and not any(base in g or g in base for g in allowed_g):
                violations.append(f"unknown_genre:{token}")

    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("，、, ") + "。"
    if not cleaned.endswith("。"):
        cleaned = cleaned.rstrip("。.") + "。"
    return (not violations), cleaned, violations
