# -*- coding: utf-8 -*-
"""A7 的纯函数层：题材名解析、卡片归一化、事实校验。

**没有任何 IO**：不查库、不调 Agent、不碰 LLM。IO 全在 `tools.py` 与 `agent.py`，
这样这里的规则可以单测覆盖（这些规则是本项目最容易被 LLM 输出带偏的地方）。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Sequence

__all__ = [
    "TITLE_RE",
    "card_of",
    "clamp_cards",
    "dedup_cards",
    "extract_titles",
    "item_to_card",
    "resolve_genre_ids",
    "row_to_card",
    "verify_reply_titles",
]

# 《书名号》里的番名 —— 事实校验的抓取目标
TITLE_RE = re.compile(r"《([^》]{1,60})》")

# 卡片只保留这些键（多余字段不进输出，避免把内部信号泄给前端）
_CARD_KEYS = ("src_anime_id", "anime_id", "title", "year", "score", "reason")


def _to_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def card_of(*, src_anime_id: Any = None, anime_id: Any = None, title: Any = None,
            year: Any = None, score: Any = None, reason: Any = None) -> dict:
    """构造一张卡片。**只保留白名单键**，`None` 值也保留（前端能区分"无"与"0"）。"""
    return {
        "src_anime_id": _to_int(src_anime_id),
        "anime_id": _to_int(anime_id),
        "title": str(title or "").strip(),
        "year": _to_int(year),
        "score": _to_float(score),
        "reason": (str(reason).strip() or None) if reason is not None else None,
    }


def row_to_card(row: dict, *, score: Any = None, reason: Any = None) -> dict:
    """DB 行 → 卡片。

    ⚠️ `src_anime_id` 优先于 `anime_id`：`recommend_result` 表里
    `anime_id` 存的是**池内索引**（A4 的输入空间），而前端要的是
    数据集原始 id。两者不等（实测 51.7% 不同），拿错会让前端打开错的番。
    """
    src = row.get("src_anime_id") or row.get("anime_src_id")
    idx = row.get("anime_id")
    return card_of(src_anime_id=src if src is not None else None,
                   anime_id=idx,
                   title=row.get("title"),
                   year=row.get("year"),
                   score=score if score is not None else (
                       row.get("final_score") if row.get("final_score") is not None
                       else row.get("score")),
                   reason=reason)


def item_to_card(item: dict, *, reason: Any = None) -> dict:
    """A4 的 `RankedItem` → 卡片。"""
    sig = item.get("explain_signals") or {}
    if reason is None:
        reason = sig.get("interest_label") or None
    return card_of(src_anime_id=item.get("src_anime_id"),
                   anime_id=item.get("anime_id"),
                   title=item.get("title"),
                   year=item.get("year"),
                   score=item.get("final_score"),
                   reason=reason)


def dedup_cards(cards: Iterable[dict]) -> list[dict]:
    """按 `src_anime_id`（缺失则 `anime_id`）去重，**保留首次出现的顺序**。

    为什么不用 `title` 去重：同名不同季/不同版本是真实存在的，
    按标题合并会把它们误杀。
    """
    out: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for c in cards:
        if not isinstance(c, dict):
            continue
        src, idx = c.get("src_anime_id"), c.get("anime_id")
        if src is None and idx is None:
            continue
        key = ("src", int(src)) if src is not None else ("idx", int(idx))
        if key in seen:
            continue
        seen.add(key)
        out.append({k: c.get(k) for k in _CARD_KEYS})
    return out


def clamp_cards(cards: Sequence[dict], limit: int,
                exclude_src: Optional[set[int]] = None) -> list[dict]:
    """截断 + 排除已曝光（`exclude_src` 用 **src id** 比较）。"""
    ex = {int(x) for x in (exclude_src or set())}
    out: list[dict] = []
    for c in cards:
        src = c.get("src_anime_id")
        if src is not None and int(src) in ex:
            continue
        out.append(dict(c))
        if len(out) >= int(limit):
            break
    return out


def resolve_genre_ids(names: Sequence[Any],
                      genre_map: dict[int, str]) -> tuple[list[int], list[str]]:
    """中文题材名 → `genre_id`。返回 `(命中 id 列表, 未识别名列表)`。

    三级匹配：精确 → 包含（"热血" 命中 "热血战斗"）→ 反向包含。
    只做字符串匹配，**不做同义词表** —— 同义词属于数据问题，
    应该在 `genre` 表里加别名而不是在代码里硬编码一张表（否则两边会漂移）。
    """
    if not genre_map:
        return [], [str(n) for n in names if str(n).strip()]

    exact = {str(v).strip(): int(k) for k, v in genre_map.items()}
    hit: list[int] = []
    miss: list[str] = []
    for raw in names:
        name = str(raw or "").strip()
        if not name:
            continue
        gid = exact.get(name)
        if gid is None:
            for label, cand in exact.items():
                if name in label or label in name:
                    gid = cand
                    break
        if gid is None:
            miss.append(name)
        elif gid not in hit:
            hit.append(gid)
    return hit, miss


def extract_titles(text: str) -> list[str]:
    """取出回复里所有《...》里的番名（去重保序）。"""
    out: list[str] = []
    for m in TITLE_RE.finditer(text or ""):
        t = m.group(1).strip()
        if t and t not in out:
            out.append(t)
    return out


def verify_reply_titles(reply: str,
                        known_titles: Iterable[str]) -> tuple[list[str], list[str]]:
    """校验回复里的《番名》是否都在已知集合里。

    返回 `(违规名列表, 已知名列表)`。**判据用包含关系**而不是相等：
    LLM 常把《进击的巨人 第三季》简写成《进击的巨人》，
    这属于正常表达，不该判违规；反之把《进击的巨人》写成
    《巨人的进击》则两者互不包含，会被抓住。
    """
    known = [str(t).strip() for t in known_titles if str(t).strip()]
    violations: list[str] = []
    for t in extract_titles(reply):
        if not any(t in k or k in t for k in known):
            violations.append(t)
    return violations, known
