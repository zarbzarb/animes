# -*- coding: utf-8 -*-
"""v1 路由聚合。`server/main.py` 只需要 `include_router(api_v1_router)`。"""

from __future__ import annotations

from fastapi import Depends

from server.api.v1 import (admin, analysis, animes, auth, chat, internal,
                           recommend, records, users)
from server.core.response import Enveloped, make_router
from server.deps import get_gw, optional_user
from server.runtime import state as runtime_state

# ⚠️ 必须用 `make_router`：聚合路由**自己**也挂着 `/health`、`/health/deep`
# 两条直接装饰的路由，而 `APIRouter()` 的默认 `route_class` 是 FastAPI 原版 ——
# 不换成 `EnvelopeRoute` 的话这两条会退回裸 dict，响应里没有
# `trace_id`/`elapsed_ms`（实测过，其余 40 条正常、只有它们缺）。
api_v1_router = make_router(prefix="/api/v1")

api_v1_router.include_router(auth.router)
api_v1_router.include_router(users.router)
api_v1_router.include_router(animes.router)
api_v1_router.include_router(records.router)
api_v1_router.include_router(recommend.router)
api_v1_router.include_router(chat.router)
api_v1_router.include_router(analysis.router)
api_v1_router.include_router(admin.router)
api_v1_router.include_router(internal.router)


@api_v1_router.get("/health", tags=["运维"], summary="健康检查")
async def health() :
    """进程级健康检查（容器 healthcheck / 前端"系统正常"指示灯）。

    与 `/internal/health` 的区别：这个**对外**，只暴露能力开关，
    不暴露任何数据统计 —— 否则健康检查会变成一个免费的信息泄露接口。
    """
    from server.core.cache import get_cache
    from server.core.config import settings

    cache = get_cache()
    db_ok = False
    try:
        db_ok = bool(get_gw().ping())
    except Exception:
        db_ok = False

    st = runtime_state()
    llm_ok = bool(settings.LLM_API_KEY) and not settings.LLM_API_KEY.startswith("sk-xxxx")
    return Enveloped(data={
        "status": "ok" if db_ok else "degraded",
        "app_env": settings.APP_ENV,
        "db": "up" if db_ok else "down",
        "cache": cache.backend_name,
        "llm": "configured" if llm_ok else "not_configured",
        "agents_ready": bool(st.get("ready")),
        "degraded": list(st.get("degraded") or []),
        "warmup": dict(st.get("warmup") or {}),
    })


@api_v1_router.get("/health/deep", tags=["运维"], summary="深度健康检查")
async def health_deep(user: dict | None = Depends(optional_user)) :
    """带数据的连通性自检：跑一遍最小链路，报告每段真实可达性。

    为什么需要它：`/health` 全绿但推荐实际不可用的情况完全可能存在 ——
    模型文件缺失时 A2 会静默降级到热门，`/health` 是看不出来的。
    """
    from agents.common.envelope import Envelope
    from agents.common.registry import AgentRegistry
    from agents.recall import adapter as recall_adapter
    from agents.recall.config import RecallConfig
    from server.core.response import new_trace_id

    out: dict = {"agents": {}, "db_counts": {}}
    try:
        out["db_counts"] = get_gw().counts()
    except Exception as exc:
        out["db_counts"] = {"error": f"{type(exc).__name__}: {exc}"}

    uid = int(user["id"]) if user else 1
    try:
        agent = AgentRegistry.get("A1")
        env = Envelope.request(from_agent="API", to_agent="A1", action="profile.get",
                               payload={"user_id": uid, "scope": "both"},
                               trace_id=new_trace_id(), timeout_ms=8000)
        r = await agent.handle(env)
        out["agents"]["A1"] = {"ready": True, "ok": not r.is_error,
                               "elapsed_ms": r.meta.elapsed_ms}
    except Exception as exc:
        out["agents"]["A1"] = {"ready": False,
                               "reason": f"{type(exc).__name__}: {exc}"}

    try:
        nm = recall_adapter.register_recall_model(RecallConfig())
        out["recall_model"] = {"name": nm, "ok": True}
    except Exception as exc:
        out["recall_model"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    return Enveloped(data=out)


__all__ = ["api_v1_router"]
