# -*- coding: utf-8 -*-
"""AniRec 应用入口。

    python -m uvicorn server.main:app --reload --port 8000

启动后：
* Swagger UI  → http://127.0.0.1:8000/docs
* 演示页面    → http://127.0.0.1:8000/          （`web/index.html`）

三件必须在启动期做完的事（顺序见 `server/runtime.py` 的文档）
-----------------------------------------------------------
1. 装配运行时（建表 / 绑网关 / 注册 Agent / **预热模型**）
2. 注册统一异常处理器（否则 FastAPI 默认返回 `{"detail": ...}`，与协议不符）
3. 把 `EnvelopeRoute` 设为默认路由类（响应包裹只有一处实现）

⚠️ 预热不是可选项：不预热时首次推荐会因模型冷加载（~2.3s）触发 A2/A3 超时降级，
用户第一次看到的是热门兜底 —— 且日志里不会报错。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles

from server.api.v1 import api_v1_router
from server.core.config import PROJECT_ROOT, settings
from server.core.exceptions import not_found, register_exception_handlers
from server.core.logging import setup_logging
from server.core.response import EnvelopeRoute, mark_request_start, new_trace_id
from server.core.logging import set_trace_id

logger = logging.getLogger(__name__)

WEB_DIR = PROJECT_ROOT / "web"
# 生产前端的构建产物（`cd frontend && npm run build`）。
# 存在时优先托管它，`web/index.html` 退为无构建环境下的演示页。
DIST_DIR = PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):     # noqa: ARG001
    setup_logging()
    logger.info("=" * 62)
    logger.info("AniRec 启动中 | env=%s | db=%s",
                settings.APP_ENV, settings.database_url.split("@")[-1])
    from server.runtime import abootstrap, ashutdown

    st = await abootstrap()
    logger.info("运行时：ready=%s degraded=%s warmup=%s",
                st.get("ready"), st.get("degraded"), st.get("warmup"))
    logger.info("接口文档：http://127.0.0.1:%s/docs", settings.APP_PORT)
    logger.info("=" * 62)
    try:
        yield
    finally:
        await ashutdown()
        logger.info("AniRec 已停止")


app = FastAPI(
    title="AniRec · 多智能体动漫追番推荐系统",
    version="3.0.0",
    description=(
        "**分层架构**：`server/`（应用）→ `agents/`（A0–A9 智能体）→ `models/`（纯模型）。\n\n"
        "**硬约束**：LLM 不参与召回与排序；指标唯一实现于 `models/eval/metrics.py`。\n\n"
        "所有接口返回统一响应体 `{code, message, data, meta, trace_id, elapsed_ms}`。"
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# 所有 JSON 路由自动包统一响应体（协议 §1.1）。
# 逐条路由手写包裹一定会漏，而漏掉的那条只有前端联调时才发现。
app.router.route_class = EnvelopeRoute

register_exception_handlers(app)

# CORS：演示前端可能跑在 5173（Vite）。生产要收敛成白名单。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173",
                   "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Trace-Id"],
)


@app.middleware("http")
async def trace_middleware(request, call_next):
    """给每个请求一个 `trace_id`，贯穿日志与响应体。"""
    tid = request.headers.get("X-Trace-Id") or new_trace_id()
    set_trace_id(tid)
    mark_request_start()
    response = await call_next(request)
    response.headers["X-Trace-Id"] = tid
    return response


app.include_router(api_v1_router)


def _spa_entry() -> FileResponse | JSONResponse:
    """首页/SPA 回退共用的入口选择：正式前端 → 演示页 → JSON 提示。"""
    for page in (DIST_DIR / "index.html", WEB_DIR / "index.html"):
        if page.exists():
            return FileResponse(str(page))
    return JSONResponse({
        "name": "AniRec API",
        "docs": "/docs",
        "health": "/api/v1/health",
        "hint": "前端页面不存在（frontend/dist 与 web/ 均为空）；接口可直接用 /docs 调试",
    })


@app.get("/", include_in_schema=False)
async def index():
    """站点首页：优先正式前端（dist），否则演示页。"""
    return _spa_entry()


# 前端构建产物挂载。⚠️ 静态目录挂在 catch-all（spa_fallback）**之前**更稳：
# 即使顺序反了，catch-all 里的文件直供逻辑也能兜住（变异检验确认过两种
# 顺序行为一致），但 mount 在前可以让 /assets/* 少走一层 Python 分发。
# check_dir=False 的理由见下方 /static 注释（import 时目录可以尚不存在）。
app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets"), check_dir=False),
          name="assets")


# SPA 路由回退（catch-all 必须是**最后一个**注册的 GET 路由 —— FastAPI 按
# 注册顺序匹配，前面的 API/文档路由优先命中）。作用：浏览器刷新 /login、
# /interest 这类前端深层路由时不 404，统一回落到 index.html。
# ⚠️ `/api/*`、`/docs`、`/openapi.json`、`/static/*` 未匹配到的路径**必须
# 重新抛信封 404**，不能回退成 index.html —— 否则 API 的"路径不存在"
# 会变成 200 + 一页 HTML，前端拦截器和测试都难以察觉。
@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    if full_path.startswith(("api/", "docs", "openapi.json", "redoc", "static/")):
        raise not_found(f"路径 /{full_path} 不存在")
    # dist 里真实存在的文件（favicon 等）直接给，其余一律回 SPA 入口
    candidate = (DIST_DIR / full_path).resolve()
    if candidate.is_file() and str(candidate).startswith(str(DIST_DIR.resolve())):
        return FileResponse(str(candidate))
    return _spa_entry()


# 静态资源挂载。
# ⚠️ 之前写成 `if WEB_DIR.exists(): app.mount(...)` —— mount 发生在 **import 时**，
# 所以"服务先起、`web/` 后建"会导致 `/static/*` 静默 404（路由根本不存在，
# 不是 404 找不到文件），而 `/` 却正常。改成 `check_dir=False` 后路由恒存在，
# 目录是启动后才创建的也能访问到 —— 目录真缺失时 Starlette 自己返回 404，
# 不会把整站启动搞挂。
app.mount("/static", StaticFiles(directory=str(WEB_DIR), check_dir=False),
          name="static")


__all__ = ["app"]
