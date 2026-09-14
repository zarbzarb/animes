# -*- coding: utf-8 -*-
"""A7 的工具集（Function Calling，`docs/agents.md` §A7 的五个工具）。

设计要点
--------
**1. 上下文用 `ContextVar` 而不是参数传递。**
工具是全局注册的（`agents/common/tools.py` 的 `@tool`），签名必须干净
（LLM 只看得到业务参数），但工具内部又需要 `user_id` / `trace_id` / `call_path`
才能回调 A2/A4/A5。把这些塞进参数会让 LLM 有机会篡改调用链与身份 ——
所以用 `ContextVar` 存"环境事实"：A7 在进入对话循环前绑一次，
工具通过 `current_context()` 读取。`ContextVar` 天然按协程隔离，
并发请求不会串味（`to_thread` 会丢上下文，因此工具必须全程 async、不落线程池）。

**2. 工具返回 `{"cards": [...], "error": ...}`，不抛异常。**
LLM 拿到 `error` 字段能自己改策略（换个题材再试），比一个 500 有用得多。
只有两种情形抛：工具未注册、调用次数超限（`60502`，由 `ToolRunner` 抛，
A7 捕获后强制收束）。

**3. 一切候选都要过 A4 `rank.diversify`。**
协议 §A7 明写"不做排序融合（交给 A4）"。工具只负责"取到候选"，
顺序与多样性由 A4 决定；A4 不可用时按原序返回并在 `note` 里说明。
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from agents.chat import strategy
from agents.common.client import call_agent
from agents.common.data import get_gateway
from agents.common.tools import tool

logger = logging.getLogger(__name__)

__all__ = [
    "ALLOWED_TOOLS",
    "ToolContext",
    "bind_tool_context",
    "current_context",
]

# 只允许这五个（协议 §A7「可调用工具」），也是 `ToolRunner(allow=...)` 的白名单
ALLOWED_TOOLS: tuple[str, ...] = (
    "search_anime", "recall_by_genre", "get_similar",
    "get_recommendation", "explain_recommendation",
)


# ---------------------------------------------------------------- 上下文
@dataclass
class ToolContext:
    user_id: int = 0
    session_id: str = ""
    trace_id: str = ""
    call_path: tuple[str, ...] = ()
    # `time.perf_counter()` 的截止时刻；0 表示不限制
    deadline: float = 0.0
    scene: int = 3

    def remaining_ms(self, default: int) -> int:
        """剩余预算（毫秒）。子调用必须用它，否则会挂在一个已经超时的父预算上。"""
        if not self.deadline:
            return int(default)
        return max(10, int((self.deadline - time.perf_counter()) * 1000))


_CTX: ContextVar[Optional[ToolContext]] = ContextVar("a7_tool_ctx", default=None)


@contextmanager
def bind_tool_context(ctx: ToolContext) -> Iterator[ToolContext]:
    token = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(token)


def current_context() -> ToolContext:
    """未绑定时返回一个匿名空上下文 —— 工具会返回明确的 error，而不是崩。"""
    return _CTX.get() or ToolContext()


# ---------------------------------------------------------------- 小工具
def _ok(cards: Optional[list] = None, **extra: Any) -> dict:
    out = {"cards": list(cards or [])}
    out.update(extra)
    return out


def _err(msg: Any, **extra: Any) -> dict:
    out = {"cards": [], "error": str(msg)}
    out.update(extra)
    return out


def _as_int(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _as_name_list(v: Any) -> list[str]:
    """LLM 有时给数组、有时给 "热血,战斗" 这种字符串，两种都收。"""
    if v is None:
        return []
    if isinstance(v, str):
        parts: list[Any] = re.split(r"[,，、/\s]+", v)
    elif isinstance(v, (list, tuple, set)):
        parts = list(v)
    else:
        parts = [v]
    return [str(p).strip() for p in parts if str(p).strip()]


def _fill_titles(gw: Any, cards: list[dict]) -> list[dict]:
    """补全标题/年份（A4 不可用时的退化路径用）。"""
    if gw is None or not cards:
        return cards
    srcs = [c["src_anime_id"] for c in cards if c.get("src_anime_id")]
    if not srcs:
        return cards
    try:
        meta = gw.get_animes(srcs) or {}
    except Exception as exc:
        logger.warning("A7 批量取动漫元数据失败：%s", exc)
        return cards
    out: list[dict] = []
    for c in cards:
        m = meta.get(c.get("src_anime_id")) or {}
        out.append({**c,
                    "title": c.get("title") or m.get("title"),
                    "year": (c.get("year") if c.get("year") is not None
                             else m.get("year"))})
    return out


# ---------------------------------------------------------------- A4 重排
async def _diversify(ctx: ToolContext, candidates: list[dict], *,
                     top_n: int) -> Optional[list[dict]]:
    """调 A4 `rank.diversify`。返回 `None` = A4 不可用，调用方自行退化。"""
    if not candidates or ctx.user_id <= 0:
        return None
    res = await call_agent(
        from_agent="A7", to_agent="A4", action="rank.diversify",
        payload={"user_id": int(ctx.user_id), "candidate_sets": [candidates],
                 "top_n": int(top_n), "scene": int(ctx.scene)},
        trace_id=ctx.trace_id, call_path=ctx.call_path,
        timeout_ms=ctx.remaining_ms(600), user_id=int(ctx.user_id),
        session_id=ctx.session_id or None)
    if not res.ok:
        logger.info("A7 → A4 重排失败（退化原序）：%s", res.error_name or res.message)
        return None
    items = res.payload.get("items") or []
    return items or None


async def _rank_cards(ctx: ToolContext, cards: list[dict], *,
                      top_n: int) -> list[dict]:
    """卡片（src id 空间）→ 池内索引 → A4 重排 → 卡片。

    池外作品（新番等不在训练池里的）**不参与模型重排**（没有索引），
    但也不能丢 —— 它们被排到重排结果之后。
    """
    from agents.recall.adapter import load_item_index   # 同层依赖，允许

    try:
        src_to_idx = load_item_index()["src_to_index"]
    except Exception as exc:
        logger.info("A7 无物品索引映射，跳过重排：%s", exc)
        return strategy.dedup_cards(cards)[:top_n]

    inside: list[dict] = []
    outside: list[dict] = []
    for rank, c in enumerate(cards):
        src = c.get("src_anime_id")
        idx = src_to_idx.get(int(src)) if src is not None else None
        if idx is None:
            outside.append(c)
            continue
        # 没有显式分数时用位次的倒数保序（A4 会 min-max 归一化，量级无关）
        score = c.get("score")
        score = float(score) if score is not None else 1.0 / (1 + rank)
        inside.append({"anime_id": int(idx), "score": score, "final_score": score,
                       "source": "sequence", "interest_id": 0, "hit_count": 1,
                       "src_anime_id": int(src),
                       "title": c.get("title"), "year": c.get("year")})

    items = await _diversify(ctx, inside, top_n=int(top_n))
    if items is None:
        return strategy.dedup_cards(cards)[:top_n]

    out = [strategy.item_to_card(it) for it in items]
    for c in outside:
        if len(out) >= top_n:
            break
        out.append(c)
    return strategy.dedup_cards(out)[:top_n]


# ---------------------------------------------------------------- 工具
@tool(name="search_anime",
      description="按关键词检索番剧库，返回匹配作品（标题/年份/原始 id）。"
                  "用户提到某部具体作品时先调它拿到 anime_id。")
async def search_anime(keyword: str, limit: int = 5) -> dict:
    gw = get_gateway()
    if gw is None:
        return _err("数据网关未注入")
    kw = str(keyword or "").strip()
    if not kw:
        return _err("关键词为空")
    try:
        rows = gw.search_anime(kw, limit=max(1, min(_as_int(limit) or 5, 20)))
    except Exception as exc:
        logger.warning("A7 search_anime 失败：%s", exc)
        return _err(f"检索失败：{type(exc).__name__}")
    cards = strategy.dedup_cards(strategy.row_to_card(r) for r in (rows or []))
    return _ok(cards, note=None if cards else f"库里没有匹配「{kw}」的作品")


@tool(name="recall_by_genre",
      description="按题材推荐作品。genres 传中文题材名数组，"
                  "如 ['热血战斗','悬疑推理']（必须是库里的题材分类）。")
async def recall_by_genre(genres: list, top_k: int = 10) -> dict:
    ctx = current_context()
    gw = get_gateway()
    if gw is None:
        return _err("数据网关未注入")
    names = _as_name_list(genres)
    if not names:
        return _err("未给出题材名")
    try:
        gmap = gw.genre_map() or {}
    except Exception as exc:
        logger.warning("A7 读题材表失败：%s", exc)
        gmap = {}
    ids, miss = strategy.resolve_genre_ids(names, gmap)
    if not ids:
        return _err("题材名无法识别",
                    unknown=miss, available=sorted(set(gmap.values())))

    k = max(1, min(_as_int(top_k) or 10, 30))
    cards: list[dict] = []
    for gid in ids:
        try:
            rows = gw.list_anime(limit=k * 3, genre_id=int(gid),
                                 order_by="n_interactions")
        except Exception as exc:
            logger.warning("A7 按题材取数失败(genre_id=%s)：%s", gid, exc)
            continue
        for r in (rows or []):
            cards.append(strategy.row_to_card(r))
    cards = strategy.dedup_cards(cards)
    if not cards:
        return _err("该题材下没有可推荐的作品", unknown=miss)
    ranked = await _rank_cards(ctx, cards, top_n=k)
    return _ok(ranked, unknown=miss or None,
               available=sorted(set(gmap.values())) if miss else None)


@tool(name="get_similar",
      description="找与某部作品相似的番剧。anime_id 传 search_anime 返回的 src_anime_id。")
async def get_similar(anime_id: int, top_k: int = 10) -> dict:
    ctx = current_context()
    gw = get_gateway()
    if gw is None:
        return _err("数据网关未注入")
    src = _as_int(anime_id)
    if src <= 0:
        return _err("anime_id 无效")
    k = max(1, min(_as_int(top_k) or 10, 30))

    src_ids: list[int] = []
    try:
        src_ids = [int(x) for x in (gw.similar_anime_ids(src, top_k=k * 2) or [])]
    except Exception as exc:
        logger.warning("A7 similar_anime_ids 失败：%s", exc)

    note: Optional[str] = None
    if not src_ids:
        # 没有预计算共现表时的诚实退化：用同题材的热门近似"相似"
        src_ids = _same_genre_src_ids(gw, src, limit=k * 2)
        note = ("无共现近邻表，已退化为同题材热门" if src_ids
                else "库中没有共现近邻表，无法计算相似")
    if not src_ids:
        return _err(note or "没找到相似作品")

    cards = _fill_titles(gw, [strategy.card_of(src_anime_id=s) for s in src_ids])
    ranked = await _rank_cards(ctx, cards, top_n=k)
    return _ok(ranked, note=note)


def _same_genre_src_ids(gw: Any, src_anime_id: int, limit: int) -> list[int]:
    """同题材热门（`get_similar` 的退化数据源）。"""
    try:
        gmap = gw.anime_genres([src_anime_id]) or {}
        gids = [int(g) for g in (gmap.get(src_anime_id) or gmap.get(int(src_anime_id)) or [])]
    except Exception as exc:
        logger.warning("A7 取题材失败：%s", exc)
        return []
    out: list[int] = []
    for gid in gids[:3]:
        try:
            rows = gw.list_anime(limit=limit, genre_id=gid,
                                 order_by="n_interactions") or []
        except Exception as exc:
            logger.warning("A7 同题材取数失败(gid=%s)：%s", gid, exc)
            continue
        for r in rows:
            s = strategy.row_to_card(r)["src_anime_id"]
            if s and int(s) != int(src_anime_id) and int(s) not in out:
                out.append(int(s))
        if len(out) >= limit:
            break
    return out[:limit]


@tool(name="get_recommendation",
      description="获取某用户的个性化推荐（基于其观看历史与兴趣）。"
                  "user_id 传 0 表示当前对话用户。")
async def get_recommendation(user_id: int = 0, top_k: int = 10) -> dict:
    ctx = current_context()
    uid = _as_int(user_id) or ctx.user_id
    if uid <= 0:
        return _err("匿名会话没有个性化推荐，请改用 recall_by_genre 按题材推荐")
    k = max(1, min(_as_int(top_k) or 10, 30))
    gw = get_gateway()

    # ① 优先复用 A0 主链路已落库的结果（零额外算力）
    if gw is not None:
        try:
            rows = gw.get_recommendations(int(uid), 0, limit=k) or []
        except Exception as exc:
            logger.warning("A7 读已落库推荐失败：%s", exc)
            rows = []
        if rows:
            cards = strategy.dedup_cards(strategy.row_to_card(r) for r in rows)
            if cards:
                return _ok(cards[:k], source="cached")

    # ② 没有缓存 → 现算：A2 召回 → A4 重排
    res = await call_agent(
        from_agent="A7", to_agent="A2", action="recall.sequence",
        payload={"user_id": int(uid)}, trace_id=ctx.trace_id,
        call_path=ctx.call_path, timeout_ms=ctx.remaining_ms(800),
        user_id=int(uid), session_id=ctx.session_id or None)
    if not res.ok:
        return _err(f"召回不可用（{res.error_name or 'unknown'}）")
    cands = res.payload.get("candidates") or []
    if not cands:
        return _err("召回结果为空")

    cand_set = [{"anime_id": int(c["anime_id"]),
                 "score": float(c.get("score") or 0.0),
                 "final_score": float(c.get("score") or 0.0),
                 "source": "sequence",
                 "interest_id": c.get("interest_id"),
                 "interest_ids": c.get("interest_ids"),
                 "hit_count": int(c.get("hit_count") or 1),
                 "interest_label": c.get("interest_label"),
                 "src_anime_id": c.get("src_anime_id")}
                for c in cands if c.get("anime_id") is not None]
    items = await _diversify(ctx, cand_set, top_n=k)
    if items is not None:
        return _ok([strategy.item_to_card(it) for it in items], source="fresh")

    cards = _fill_titles(gw, [strategy.card_of(src_anime_id=c.get("src_anime_id"),
                                              anime_id=c.get("anime_id"),
                                              score=c.get("score"))
                              for c in cands[:k]])
    return _ok(cards, source="recall_only")


@tool(name="explain_recommendation",
      description="生成某部作品的推荐理由（基于用户画像与推荐信号）。"
                  "anime_id 传 src_anime_id；title 可选。")
async def explain_recommendation(anime_id: int = 0, title: str = "") -> dict:
    ctx = current_context()
    if ctx.user_id <= 0:
        return _err("匿名会话没有画像，无法生成个性化理由")
    src = _as_int(anime_id)
    if src <= 0 and not str(title or "").strip():
        return _err("需要 anime_id 或 title")
    res = await call_agent(
        from_agent="A7", to_agent="A5", action="explain.generate",
        payload={"user_id": int(ctx.user_id), "style": "casual",
                 "standalone": True, "persist": False,
                 "target_anime": {"src_anime_id": src or None,
                                  "title": str(title or "")}},
        trace_id=ctx.trace_id, call_path=ctx.call_path,
        timeout_ms=ctx.remaining_ms(1200), user_id=int(ctx.user_id),
        session_id=ctx.session_id or None)
    if not res.ok:
        return _err(f"解释不可用（{res.error_name or 'unknown'}）")
    return {"cards": [], "reason": str(res.payload.get("reason") or ""),
            "source": res.payload.get("source")}
