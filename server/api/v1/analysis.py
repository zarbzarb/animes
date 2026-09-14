# -*- coding: utf-8 -*-
"""分析模块（api-specification.md §3.5）：兴趣雷达 / 漂移趋势 / 漂移点 / Agent 调试。

`drift-trend` 与 `drift-points` 都来自 A6 的**同一次** `drift.analyze`，
只是切成两个视图 —— 分两次调用会让两次结果落在不同的数据快照上，
前端出现"趋势图与漂移点对不上"这种很难查的错。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from server.core.config import settings
from server.core.exceptions import forbidden, invalid_param
from server.core.response import make_router, Enveloped
from server.deps import current_user, get_gw, rate_limit
from server.services.agent_bridge import call_agent_direct, drift_of, radar_of

logger = logging.getLogger(__name__)
router = make_router(prefix="/analysis", tags=["分析"])


@router.get("/interest-radar", summary="兴趣雷达图",
            dependencies=[Depends(rate_limit("analysis"))])
async def interest_radar(
    user: dict = Depends(current_user),
) :
    data, meta = await radar_of(int(user["id"]))
    return Enveloped(data=data | {"meta": meta})


async def _drift(user_id: int, granularity: str) -> tuple[dict, dict]:
    if granularity not in ("month", "quarter", "year"):
        raise invalid_param("granularity 必须是 month / quarter / year")
    return await drift_of(user_id, granularity=granularity)


@router.get("/drift-trend", summary="题材漂移趋势",
            dependencies=[Depends(rate_limit("analysis"))])
async def drift_trend(
    granularity: str = Query("quarter"),
    user: dict = Depends(current_user),
) :
    raw, meta = await _drift(int(user["id"]), granularity)
    labels = {i + 1: n for i, n in enumerate(raw.get("radar_labels") or [])}
    trend = []
    for row in (raw.get("trend") or []):
        dist = row.get("genres") or {}
        trend.append({
            "period": row.get("period"),
            "total": row.get("total"),
            "top_genre": labels.get(int(row["top_genre_id"]))
                        if row.get("top_genre_id") else None,
            "top_share": row.get("top_share"),
            # key 用**题材名**而不是 id：前端图例直接可用，不必再查一次题材表
            "genres": {labels.get(int(g), str(g)): v for g, v in dist.items()},
        })
    return Enveloped(data={"trend": trend, "granularity": granularity,
                           "n_periods": raw.get("n_periods"), "meta": meta})


@router.get("/drift-points", summary="漂移点标注",
            dependencies=[Depends(rate_limit("analysis"))])
async def drift_points(
    granularity: str = Query("quarter"),
    threshold: float = Query(0.35, ge=0.05, le=0.95),
    user: dict = Depends(current_user),
) :
    raw, meta = await _drift(int(user["id"]), granularity)
    labels = {i + 1: n for i, n in enumerate(raw.get("radar_labels") or [])}
    points = []
    for p in (raw.get("drift_points") or []):
        js = float(p.get("js") or 0.0)
        points.append({
            "period": p.get("period"),
            "prev_period": p.get("prev_period"),
            "js_divergence": round(js, 4),
            "from": labels.get(int(p["from_genre_id"]))
                    if p.get("from_genre_id") else None,
            "to": labels.get(int(p["to_genre_id"]))
                  if p.get("to_genre_id") else None,
            "severity": ("high" if js >= float(threshold) * 1.5
                         else "medium" if js >= float(threshold) else "low"),
            "description": p.get("description"),
        })
    return Enveloped(data={
        "drift_points": points,
        "interpretation": raw.get("interpretation"),
        "is_stable": bool(raw.get("is_stable")),
        "threshold": float(threshold),
        "granularity": granularity,
        "meta": meta,
    })


class AgentInvokeIn(BaseModel):
    agent: str = Field(..., min_length=2, max_length=16,
                      description="A0-A9 或包名（orchestrator/profile/recall/…）")
    action: str = Field(..., min_length=3, max_length=48)
    payload: dict = Field(default_factory=dict)


# 包名 → agent id（答辩演示时用包名更好读）
_NAME_TO_ID = {"orchestrator": "A0", "profile": "A1", "recall": "A2",
               "coldstart": "A3", "fusion": "A4", "explain": "A5",
               "drift": "A6", "chat": "A7", "dataops": "A8", "eval": "A9"}


@router.post("/agent-invoke", summary="直接调 Agent（调试用）")
async def agent_invoke(body: AgentInvokeIn,
                       user: dict = Depends(current_user)) :
    """⚠️ 仅开发/调试环境启用（协议 §3.5 明确 `APP_ENV != prod`）。

    生产环境暴露它等于把「任意 Agent × 任意 action」开成公网接口：
    调用方可以绕过 A0 的校验直接让 A2 拿别的用户的序列算召回事。
    """
    if settings.APP_ENV == "prod":
        raise forbidden("调试接口在生产环境已关闭")

    aid = body.agent.upper()
    if aid not in {f"A{i}" for i in range(10)}:
        aid = _NAME_TO_ID.get(body.agent.lower(), "")
    if not aid:
        raise invalid_param(f"未知 Agent：{body.agent}")

    payload = dict(body.payload or {})
    # 演示时几乎总是想用「我自己」的数据；未显式给 user_id 就填当前用户
    if "user_id" not in payload and aid in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        payload["user_id"] = int(user["id"])

    r = await call_agent_direct(aid, body.action, payload, timeout_ms=15000,
                                user_id=int(user["id"]))
    return Enveloped(
        data={"agent": aid, "action": body.action, "ok": r.ok,
              "payload": r.payload, "chain": list(r.chain or [])},
        code=0 if r.ok else int(r.error_code or 50001),
        message="ok" if r.ok else (r.message or r.error_name),
        meta={"elapsed_ms": r.elapsed_ms, "error_name": r.error_name or None},
    )
