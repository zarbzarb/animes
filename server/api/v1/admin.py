# -*- coding: utf-8 -*-
"""管理后台（api-specification.md §3.6）。全部要求管理员角色。

这一层是**只读为主**：它把 Agent 的运行时状态、调用链、冷启动看板暴露给答辩演示，
不参与推荐链路。唯一的写操作是动漫的增删改（A8 的数据运营接口的 HTTP 入口）。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from server.core.exceptions import not_found
from server.core.response import make_router, Enveloped, paginate
from server.deps import current_admin, get_gw
from server.services.agent_bridge import shape_items

logger = logging.getLogger(__name__)
router = make_router(prefix="/admin", tags=["管理"])


# ---------------------------------------------------------------- 动漫管理


class AnimeIn(BaseModel):
    src_anime_id: int = Field(..., ge=1, description="数据集 animeID（唯一）")
    title: str = Field(..., min_length=1, max_length=255)
    alt_title: str | None = None
    anime_type: str | None = Field(None, alias="type")
    year: int | None = Field(None, ge=1900, le=2100)
    score: float | None = Field(None, ge=0, le=10)
    episodes: int | None = Field(None, ge=0)
    image_url: str | None = None
    summary: str | None = None
    genre_ids: list[int] = Field(default_factory=list)
    primary_genre_id: int | None = Field(None, ge=1, le=12)

    model_config = {"populate_by_name": True, "extra": "ignore"}


@router.get("/animes", summary="动漫管理列表")
async def admin_animes(
    keyword: str = Query(""),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    admin: dict = Depends(current_admin),
) :
    gw = get_gw()
    rows = gw.list_anime(limit=size, offset=(page - 1) * size, keyword=keyword,
                         order_by="id")
    total = gw.count_anime(keyword=keyword)
    return Enveloped(data=paginate(rows, page, size, total))


@router.post("/animes", summary="新增动漫")
async def admin_create_anime(body: AnimeIn,
                             admin: dict = Depends(current_admin)) :
    gw = get_gw()
    data = body.model_dump(exclude_none=True, by_alias=False)
    row = gw.upsert_anime(data)
    logger.info("管理员 %s 新增/更新动漫 %s", admin.get("username"), body.src_anime_id)
    return Enveloped(data=row, code=0, message="已保存")


@router.put("/animes/{anime_id}", summary="编辑动漫")
async def admin_update_anime(anime_id: int, body: AnimeIn,
                             admin: dict = Depends(current_admin)) :
    gw = get_gw()
    if gw.get_anime(anime_id) is None:
        raise not_found(f"动漫 {anime_id} 不存在")
    data = body.model_dump(exclude_none=True, by_alias=False)
    data["src_anime_id"] = int(anime_id)
    return Enveloped(data=gw.upsert_anime(data), code=0, message="已更新")


@router.put("/animes/{anime_id}/offline", summary="下架动漫")
async def admin_offline_anime(anime_id: int,
                              admin: dict = Depends(current_admin)) :
    gw = get_gw()
    ok = gw.set_anime_offline(int(anime_id))
    if not ok:
        raise not_found(f"动漫 {anime_id} 不存在")
    return Enveloped(data={"anime_id": int(anime_id), "is_online": 0},
                     code=0, message="已下架")


@router.get("/users", summary="用户列表")
async def admin_users(
    keyword: str = Query(""),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    admin: dict = Depends(current_admin),
) :
    gw = get_gw()
    rows = gw.list_users(limit=size, offset=(page - 1) * size, keyword=keyword)
    total = gw.count_users(keyword=keyword)
    return Enveloped(data=paginate(rows, page, size, total))


# ---------------------------------------------------------------- 监控


@router.get("/agents/health", summary="Agent 健康状态")
async def agents_health(admin: dict = Depends(current_admin)) :
    """合并两个来源：

    * `agent_state` —— 权威状态（含熔断、版本），由 `tracing.persist_span` 维护；
    * `agent_trace` 的聚合 —— 真实调用量/耗时，`agent_state` 在无调用时是全 0。

    只读一个来源都会给出误导性的面板：只看 `agent_state` 会显示"0 次调用"，
    只看 `agent_trace` 会丢掉从未被调用的 Agent（它们恰恰最需要被发现）。
    """
    gw = get_gw()
    states = {s["agent_id"]: s for s in (gw.list_agent_states() or [])}
    try:
        stats = {t["agent_id"]: t for t in (gw.trace_stats(since=None) or [])}
    except Exception as exc:
        logger.warning("trace 聚合失败（健康面板将缺少调用量）：%s", exc)
        stats = {}

    label = {0: "下线", 1: "正常", 2: "降级中", 3: "熔断"}
    agents = []
    for aid in sorted(set(states) | set(stats)):
        st = states.get(aid) or {}
        tr = stats.get(aid) or {}
        calls = int(tr.get("calls") or 0)
        success = int(tr.get("success") or 0)
        agents.append({
            "agent_id": aid,
            "name": st.get("agent_name") or aid.lower(),
            "status": int(st.get("status", 1)),
            "status_label": label.get(int(st.get("status", 1)), "未知"),
            "health_score": (round(success / calls, 4) if calls
                             else float(st.get("health_score") or 1.0)),
            "avg_elapsed_ms": int(tr.get("avg_elapsed_ms") or st.get("avg_elapsed_ms") or 0),
            "p95_elapsed_ms": int(tr.get("p95_elapsed_ms") or st.get("p95_elapsed_ms") or 0),
            "calls": calls,
            "success_cnt": success or int(st.get("success_cnt") or 0),
            "fail_cnt": int(tr.get("fail") or st.get("fail_cnt") or 0),
            "timeout_cnt": int(tr.get("timeout") or st.get("timeout_cnt") or 0),
            "degrade_cnt": int(tr.get("degrade") or st.get("degrade_cnt") or 0),
            "last_active_at": st.get("last_active_at"),
        })

    total_calls = sum(a["calls"] for a in agents) or 1
    total_deg = sum(a["degrade_cnt"] for a in agents)
    total_tmo = sum(a["timeout_cnt"] for a in agents)
    return Enveloped(data={
        "agents": agents,
        "global": {
            "degrade_rate": round(total_deg / total_calls, 4),
            "llm_timeout_rate": round(total_tmo / total_calls, 4),
            "circuit_open": [a["agent_id"] for a in agents if a["status"] == 3],
        },
    })


@router.get("/traces/{trace_id}", summary="调用链详情")
async def trace_detail(trace_id: str,
                       admin: dict = Depends(current_admin)) :
    gw = get_gw()
    spans = gw.get_trace(trace_id) or []
    if not spans:
        raise not_found(f"没有找到 trace {trace_id}（只保留最近 30 天）")
    return Enveloped(data={
        "trace_id": trace_id,
        "total_elapsed_ms": max(int(s.get("elapsed_ms") or 0) for s in spans),
        "n_spans": len(spans),
        "spans": spans,
    })


@router.get("/metrics", summary="推荐效果监控")
async def admin_metrics(
    metric_type: str = Query("hr10"),
    days: int = Query(30, ge=1, le=365),
    scene: int | None = Query(None, ge=0, le=3),
    admin: dict = Depends(current_admin),
) :
    """离线指标（`metric_snapshot`）+ 在线反馈转化（`user_feedback`）。

    两者必须一起看：离线 HR@10 涨了但在线 CTR 掉了，说明指标与体验脱节，
    只看其中一个会得出相反的结论。
    """
    gw = get_gw()
    series = gw.query_metrics(metric_type=metric_type, days=days, scene=scene) or []
    try:
        online = gw.feedback_counts(scene=scene)
    except Exception as exc:
        logger.warning("在线反馈统计失败：%s", exc)
        online = {}
    try:
        daily = gw.feedback_daily(days=min(days, 30), scene=scene)
    except Exception:
        daily = []
    return Enveloped(data={
        "metric_type": metric_type,
        "series": series,
        "latest_model_ver": gw.latest_model_ver(),
        "online": {"counts": {str(k): int(v) for k, v in (online or {}).items()}},
        "daily": daily,
        "note": ("离线指标来自 `metric_snapshot`（M2.x 实验产物），"
                 "为空表示尚未跑过 `scripts/run_experiments.py`"),
    })


@router.get("/cold-start", summary="冷启动看板")
async def admin_cold_start(admin: dict = Depends(current_admin)) :
    gw = get_gw()
    data = gw.cold_start_stats()
    data["note"] = ("本数据集无 n_interactions<COLD_START_THRESHOLD 的番"
                    "（H4），池内容量由 `cold_start_pool` 决定")
    data["pool_items"] = shape_items([
        {"src_anime_id": r["src_anime_id"], "rank_no": i + 1}
        for i, r in enumerate(gw.list_cold_start_pool(limit=24, active_only=True))],
        explain_from_item=False)
    return Enveloped(data=data)


@router.post("/tasks/offline-recommend", summary="手动触发离线推荐")
async def trigger_offline(
    user_id: int | None = Query(None, description="留空=全部活跃用户"),
    limit: int = Query(50, ge=1, le=500),
    admin: dict = Depends(current_admin),
) :
    """投递一次离线重排任务（A8 的入口）。

    ⚠️ 生产上这里应当投递到调度器（APScheduler / 队列）后**立即返回**，
    否则一个 500 用户的批量重排会让 HTTP 请求挂几分钟。
    当前实现同步跑（演示规模下 5 个用户 < 1s），并在响应里如实标注 `sync=True`。
    """
    from server.tasks.offline import run_offline_recommend

    result = await run_offline_recommend(user_id=user_id, limit=limit)
    return Enveloped(data=result, code=0, message="离线推荐已执行")
