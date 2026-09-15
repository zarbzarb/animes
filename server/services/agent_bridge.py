# -*- coding: utf-8 -*-
"""业务服务层：把「HTTP 语义」翻译成「Agent 信封」，再把「Agent 产物」翻译回「接口形状」。

为什么需要这一层
----------------
Agent 层用的是**池内索引**（模型候选空间）与内部字段名（`rank_no` / `explain_signals`），
而 `docs/api-specification.md` 定义的是**面向前端**的形状（`rank` / `anime{}` / `explain{}`，
且 `anime.id` 是 `anime.id` 主键、`anime.src_anime_id` 是数据集 id）。
这层翻译如果散落在 41 个 handler 里，必然出现"某条接口少补了 genres"这种错，
而且前端只有在渲染时才发现。

**唯一入口**：所有 handler 只通过本模块访问 Agent，不自己拼信封。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Sequence

from agents.common.client import call_agent
from server.core.response import new_trace_id
from server.deps import get_gw

logger = logging.getLogger(__name__)

__all__ = [
    "call_agent_direct",
    "chat_once",
    "drift_of",
    "explain_one",
    "feed",
    "profile_of",
    "radar_of",
    "route_intent",
    "shape_items",
    "system_snapshot",
]

# 前端意图 → API 场景号（recommend_result.scene）
SCENE = {"RECOMMEND_FEED": 0, "RECOMMEND_BY_GENRE": 1,
         "RECOMMEND_NEW_ANIME": 2, "CHAT_RECOMMEND": 3}

# A0 的总预算（协议 §5.2）。API 层给够余量，避免"服务端先超时、Agent 再降级"。
A0_TIMEOUT_MS = 9000


async def _call(agent: str, action: str, payload: dict, *,
                trace_id: Optional[str] = None, timeout_ms: int = 3000,
                user_id: Optional[int] = None,
                session_id: Optional[str] = None):
    tid = trace_id or new_trace_id()
    return await call_agent(from_agent="API", to_agent=agent, action=action,
                            payload=payload, trace_id=tid,
                            call_path=["API"], timeout_ms=int(timeout_ms),
                            priority="P0", user_id=user_id,
                            session_id=session_id)


async def call_agent_direct(agent: str, action: str, payload: dict, *,
                            timeout_ms: int = 8000,
                            user_id: Optional[int] = None):
    """调试接口用：直接打某个 Agent（`/analysis/agent-invoke`）。"""
    return await _call(agent, action, payload, timeout_ms=timeout_ms,
                       user_id=user_id)


async def route_intent(user_id: int, *, intent_hint: Optional[str] = None,
                       raw_query: Optional[str] = None, params: Optional[dict] = None,
                       context: Optional[dict] = None,
                       timeout_ms: int = A0_TIMEOUT_MS):
    """走 A0 完成一次完整的意图路由 + 编排。"""
    payload = {
        "user_id": int(user_id),
        "intent_hint": intent_hint,
        "raw_query": raw_query,
        "params": dict(params or {}),
        "context": dict(context or {}),
    }
    return await _call("A0", "orchestrate.route", payload,
                       timeout_ms=timeout_ms, user_id=user_id)


# ---------------------------------------------------------------- 形态转换


def shape_items(items: Sequence[dict], *, explain_from_item: bool = True) -> list[dict]:
    """A4 产物 → 接口 `data.items[]`。

    需要补的是 `anime{}` 整块（标题/类型/评分/集数/图片/题材名）——
    A4 只带 `src_anime_id`（它不该为了展示字段去查库），
    所以这里**一次批量**取回，而不是逐条查（N+1）。
    """
    rows = list(items or [])
    if not rows:
        return []
    srcs = [int(it["src_anime_id"]) for it in rows if it.get("src_anime_id")]
    meta = get_gw().get_animes(srcs) if srcs else {}

    out: list[dict] = []
    for it in rows:
        src = it.get("src_anime_id")
        m = meta.get(int(src)) if src else None
        sig = it.get("explain_signals") or {}
        explain = None
        if explain_from_item:
            reason = it.get("reason")
            if reason:
                explain = {
                    "reason": reason,
                    "core_items": list(sig.get("top_attn_items") or []),
                    "match_percent": it.get("match_percent"),
                    "matched_genres": list(sig.get("matched_genres") or []),
                    "source": it.get("explain_source") or "template",
                }
        anime = {
            "id": (m or {}).get("id"),
            "src_anime_id": src,
            "title": (m or {}).get("title") or it.get("title"),
            "title_cn": (m or {}).get("title_cn"),
            "type": (m or {}).get("type"),
            "year": (m or {}).get("year") if m else it.get("year"),
            "score": (m or {}).get("score"),
            "episodes": (m or {}).get("episodes"),
            "image_url": (m or {}).get("image_url"),
            "genres": list((m or {}).get("genre_names") or []),
        }
        out.append({
            "rank": int(it.get("rank_no") or len(out) + 1),
            "anime": anime,
            "final_score": it.get("final_score"),
            "behavior_score": it.get("behavior_score"),
            "content_score": it.get("content_score"),
            "interest_id": it.get("interest_id"),
            "interest_label": it.get("interest_label"),
            "is_cold_start": bool(it.get("is_cold_start")),
            "cold_start_badge": "冷启动推荐" if it.get("is_cold_start") else None,
            "explain": explain,
        })
    return out


def _meta_of(a0_result: dict, cache_hit: bool = False) -> dict:
    # ⚠️ A0 的产物是**两层**的：`{intent, result:{...}, degraded, lane}`
    # —— 业务字段在 `result` 里，而 `intent/agent_chain` 在第一层。
    # `cache_hit` 属于业务字段，必须从 `result` 里取（第一版就是取错了层，
    # 结果"修好了但看起来没生效"：A4 明明命中了缓存，对外仍是 false）。
    res = a0_result.get("result") or {}
    return {
        # 两个来源：A0 信封 meta（本轮编排自身）与 A0 载荷 `result.cache_hit`
        # （A4 的推荐结果缓存）。业务关心后者；早期只读前者，而 A0 从不设置它，
        # 于是这个契约字段对客户端恒为 false（"从来没命中过缓存"）。
        "cache_hit": bool(cache_hit or res.get("cache_hit")),
        # A4 走了 L4 全站热门兜底（新注册零记录用户）→ 前端显示"热门推荐"横幅
        "used_fallback": bool(res.get("used_fallback")),
        "profile_version": a0_result.get("profile_version"),
        "agent_chain": [c.get("agent") for c in (a0_result.get("agent_chain") or [])]
                       or list(a0_result.get("lane") or []),
        "degraded": [d.get("step") for d in (a0_result.get("degraded") or [])],
        "elapsed_ms": a0_result.get("elapsed_ms"),
        "intent": a0_result.get("intent"),
        "intent_label": a0_result.get("intent_label"),
        "intent_method": a0_result.get("intent_method"),
        "confidence": a0_result.get("confidence"),
    }


async def feed(user_id: int, *, size: int = 20, with_explain: bool = True,
               intent_hint: str = "RECOMMEND_FEED",
               params: Optional[dict] = None,
               raw_query: Optional[str] = None) -> tuple[dict, dict]:
    """`/recommend/feed | by-genre | new-anime` 的公共实现。

    返回 `(data, meta)`。`data` 已按接口规范成形（`items` / `meta`）。
    """
    p = {"top_n": int(size), **(params or {})}
    if not with_explain:
        # 不要解释就别让 A5 跑：省一次 LLM 调用（协议里 A5 是可跳步骤）
        p.setdefault("skip_explain", True)

    r = await route_intent(user_id, intent_hint=intent_hint, raw_query=raw_query,
                           params=p)
    if not r.ok:
        logger.warning("A0 路由失败：%s %s", r.error_name, r.message)
        raise _to_biz(r)

    a0 = r.payload or {}
    res = a0.get("result") or {}
    items = shape_items(res.get("items") or [])
    data = {
        "items": items,
        "profile_summary": res.get("profile_summary"),
        "user_tag": res.get("user_tag"),
        "is_cold_start_user": bool(res.get("is_cold_start_user")),
    }
    meta = _meta_of(a0, cache_hit=bool(r.meta.get("cache_hit")))
    meta["n_items"] = len(items)
    if res.get("batch_id"):
        meta["batch_id"] = res["batch_id"]
    if intent_hint == "RECOMMEND_NEW_ANIME":
        try:
            pool = get_gw().list_cold_start_pool(limit=500, active_only=True) or []
            meta["pool_size"] = len(pool)
        except Exception:
            pass
    return data, meta


async def explain_one(user_id: int, anime_id: int, *,
                      style: str = "concise") -> tuple[dict, dict]:
    """`GET /recommend/{anime_id}/explain`。

    复用链：`recommend_explain` 命中 → 直接返回；否则从 `recommend_result`
    取该番的 `explain_signals`（A4 落的结构化信号，唯一事实来源）→ 交 A5 生成。
    """
    gw = get_gw()

    # 只认 LLM 产出的缓存。`recommend_explain` 没有 TTL，早期版本把**模板兜底**
    # 也写了进去，于是解释被永久冻结在最空的那句模板上（LLM 恢复也不会更新）。
    # 现在 A5 不再落库模板，这里再把历史遗留的模板行当未命中处理 ——
    # 不然数据库里已有的脏行会一直生效，得手工 DELETE 才能修好。
    cached = gw.get_explain(user_id, int(anime_id), style=style)
    if cached and cached.get("source") != "template":
        return _explain_data(cached), {"cache_hit": True, "source": cached.get("source")}

    anime = gw.get_anime(int(anime_id))
    if anime is None:
        from server.core.exceptions import not_found
        raise not_found(f"动漫 {anime_id} 不存在")

    signals: dict = {}
    try:
        for scene in (0, 1, 2):
            for row in gw.get_recommendations(user_id, scene, limit=100):
                if int(row.get("src_anime_id") or 0) != int(anime_id):
                    continue
                sig = row.get("explain_signals") or {}
                # 只认**带信息**的信号：`recommend_result` 里可能残留旧口径写入的
                # 批次（两个字段都空）。用它拼出来的是最空的那句模板，
                # 反而不如走下面的"按题材相似"兜底 —— 后者至少能说出题材名。
                if sig.get("matched_genres") or sig.get("top_attn_items"):
                    signals = sig
                break
            if signals:
                break
    except Exception as exc:
        logger.warning("读取历史推荐信号失败：%s", exc)

    if not signals:
        # 协议 §3.3：不在当前用户的推荐结果中 → 40401。
        # 但**不要因此让用户看不到任何东西**：退化为"按题材相似"的通用解释。
        signals = {"matched_genres": list(anime.get("genre_names") or [])[:2],
                   "genre_overlap": 0.0, "top_attn_items": [],
                   "fallback": "no_recommendation_context"}

    r = await route_intent(user_id, intent_hint="EXPLAIN_RECOMMEND",
                           params={"target_anime": {
                                       "anime_id": int(anime_id),
                                       "src_anime_id": int(anime_id),
                                       "title": anime.get("title") or "",
                                       "genres": list(anime.get("genre_names") or []),
                                       "year": anime.get("year")},
                                   "explain_signals": signals,
                                   "style": style, "standalone": True},
                           timeout_ms=15000)
    if not r.ok:
        raise _to_biz(r)
    res = (r.payload or {}).get("result") or {}
    data = {
        "anime_id": int(anime_id),
        "title": anime.get("title"),
        "reason": res.get("reason"),
        "core_items": res.get("core_items") or [],
        "match_percent": res.get("match_percent"),
        "matched_genres": list(signals.get("matched_genres") or []),
        "source": res.get("source") or "template",
        "prompt_ver": res.get("prompt_ver"),
        "degraded": bool(res.get("degraded")),
    }
    return data, _meta_of(r.payload or {})


def _explain_data(row: dict) -> dict:
    return {
        "anime_id": row.get("anime_id"),
        "title": row.get("title"),
        "reason": row.get("reason"),
        "core_items": row.get("core_items") or [],
        "match_percent": row.get("match_percent"),
        "matched_genres": row.get("matched_genres") or [],
        "source": row.get("source") or "template",
        "prompt_ver": row.get("prompt_ver"),
        "style": row.get("style"),
    }


async def profile_of(user_id: int) -> tuple[dict, dict]:
    """`/analysis/interest-radar` 的画像部分（A1）。"""
    r = await _call("A1", "profile.get",
                    {"user_id": int(user_id), "scope": "both"},
                    timeout_ms=8000, user_id=user_id)
    if not r.ok:
        raise _to_biz(r)
    return r.payload or {}, {"agent": "A1", "elapsed_ms": r.elapsed_ms}


async def radar_of(user_id: int) -> tuple[dict, dict]:
    """兴趣雷达：A1 画像 → 12 项题材强度 + 兴趣胶囊。

    12 项是**固定槽位**（`value=0` 也要返回）—— 前端雷达图靠固定 12 轴渲染，
    稀疏返回会让图形随数据变形，看起来像"口味变了"。
    """
    prof, meta = await profile_of(user_id)
    gw = get_gw()
    gmap: dict[int, str] = {}
    try:
        gmap = {int(k): v for k, v in (gw.genre_map() or {}).items()}
    except Exception as exc:
        logger.warning("题材表不可用：%s", exc)

    tops = {int(g["genre_id"]): g for g in (prof.get("top_genres") or [])
            if g.get("genre_id") is not None}
    radar = []
    for gid in sorted(gmap):
        g = tops.get(int(gid))
        radar.append({
            "genre_id": int(gid),
            "genre": gmap[gid],
            "value": round(float((g or {}).get("strength") or 0.0), 4),
            "count": int((g or {}).get("count") or 0),
            "trend": (g or {}).get("trend"),
        })

    # 兴趣胶囊：权威来源是 `user_interest_capsule`（多兴趣模型离线写入）。
    # 该表为空时用 A1 的 Top4 题材降级 —— 前端只需要"标签+强度"来画扇区，
    # 用真实画像数据降级比返回空列表好（空列表会让雷达图整块消失）。
    capsules: list[dict] = []
    try:
        capsules = gw.get_capsules(user_id) or []
    except Exception as exc:
        logger.warning("读取兴趣胶囊失败：%s", exc)
    capsule_source = "table"
    if not capsules:
        capsules = [{"capsule_id": i, "label": g["genre"], "strength": g["value"]}
                    for i, g in enumerate(radar_sorted(radar)[:4])]
        capsule_source = "profile_fallback"

    meta.update({"agent": "A1", "n_genres": len(radar),
                 "capsule_source": capsule_source})
    return {
        "radar": radar,
        "summary_text": prof.get("summary_text"),
        "user_tag": prof.get("user_tag"),
        "activity_level": prof.get("activity_level"),
        "activity_label": prof.get("activity_label"),
        "is_cold_start": bool(prof.get("is_cold_start")),
        "watch_intensity": prof.get("watch_intensity"),
        "dropped_rate": prof.get("dropped_rate"),
        "preferred_types": prof.get("preferred_types") or [],
        "preferred_era": prof.get("preferred_era"),
        "total_records": prof.get("total_records"),
        "recent_items": prof.get("recent_items") or [],
        "interest_capsules": capsules,
    }, meta


def radar_sorted(radar: list[dict]) -> list[dict]:
    """按强度降序（胶囊取 Top4 用）。"""
    return sorted(radar, key=lambda r: -float(r.get("value") or 0.0))


async def drift_of(user_id: int, *, granularity: str = "quarter") -> tuple[dict, dict]:
    """A6 兴趣漂移（趋势 + 漂移点一次返回，接口层再拆）。"""
    r = await _call("A6", "drift.analyze",
                    {"user_id": int(user_id), "granularity": granularity},
                    timeout_ms=12000, user_id=user_id)
    if not r.ok:
        raise _to_biz(r)
    return r.payload or {}, {"agent": "A6", "elapsed_ms": r.elapsed_ms}


async def chat_once(user_id: int, message: str, *, session_id: Optional[str] = None,
                    history: Optional[list] = None) -> tuple[dict, dict]:
    """A7 对话推荐（非流式；SSE 版见 `api/v1/chat.py`）。"""
    r = await _call("A7", "chat.reply",
                    {"user_id": int(user_id), "message": message,
                     "session_id": session_id, "history": list(history or [])},
                    timeout_ms=20000, user_id=user_id, session_id=session_id)
    if not r.ok:
        raise _to_biz(r)
    return r.payload or {}, {"agent": "A7", "elapsed_ms": r.elapsed_ms}


async def system_snapshot() -> tuple[dict, dict]:
    """A9 在线快照（`/admin/agents/health` 的数据源之一）。"""
    r = await _call("A9", "eval.online_snapshot", {"mode": "online"},
                    timeout_ms=12000)
    if not r.ok:
        raise _to_biz(r)
    return r.payload or {}, {"agent": "A9", "elapsed_ms": r.elapsed_ms}


def _to_biz(r) -> Exception:
    """`SubCallResult` → 本层可抛的业务异常（错误码原样透传，不重新编号）。"""
    from server.core.exceptions import BizError

    code = int(r.error_code or 50001)
    return BizError(code, r.message or r.error_name or "Agent 调用失败",
                    detail={"agent_error": r.error_name,
                            "chain": list(r.chain or [])})
