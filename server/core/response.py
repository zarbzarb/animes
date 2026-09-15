# -*- coding: utf-8 -*-
"""统一响应体：**所有**接口（含错误）都返回同一层信封。

见 `docs/api-specification.md` §1.1::

    {"code": 0, "message": "ok", "data": {...}, "trace_id": "tr_xxx", "elapsed_ms": 265}

为什么用「路由统一包裹」而不是每个 handler 手写
-----------------------------------------------
文档的接口检查清单明确写着「响应包裹统一响应体（由 `core/response.py` 自动完成，
不要手写）」。手写的问题不是繁琐，而是**总有人漏**：一旦某条路由直接 `return dict`，
前端拿到的结构就与其它接口不同，而单元测试通常只测自己那条路 —— 不报错的错最贵。

实现方式：`EnvelopeRoute`（见下）在路由出口处包一层，handler 只返回**业务载荷**。
需要特殊 code/message 时返回 `Enveloped(...)`，而不是自己拼 JSON。
"""

from __future__ import annotations

import json
import secrets
import time
from contextvars import ContextVar
from typing import Any, Callable, Mapping, Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

# ---------------------------------------------------------------- trace_id

_request_start: ContextVar[float] = ContextVar("request_start", default=0.0)


def new_trace_id() -> str:
    """`tr_` + 12 位 hex（协议 §3.2 规定）。"""
    return "tr_" + secrets.token_hex(6)


def mark_request_start() -> float:
    t0 = time.perf_counter()
    _request_start.set(t0)
    return t0


def elapsed_ms() -> int:
    t0 = _request_start.get()
    if not t0:
        return 0
    return int(round((time.perf_counter() - t0) * 1000))


# ---------------------------------------------------------------- 信封构造


#: 哨兵键名。`Enveloped` 序列化后会带上它，`EnvelopeRoute` 据此识别并剥掉。
#: 为什么需要它 —— 见 `EnvelopeRoute` 的注释（出口拿不到原始对象）。
ENVELOPE_MARK = "_envelope"


class Enveloped(dict):
    """handler 想自定义 `code/message/meta` 时返回它，而不是自己拼 dict。

    ⚠️ 它是 `dict` 的子类而**不是** dataclass，有两个必须的原因：

    1. 路由出口拿到的是**已经序列化过的 JSON**，无法再对原始对象做
       `isinstance(..., Enveloped)` 判断（早期版本的 bug：那个判断恒为 False，
       于是 `degraded()` 与所有自定义 `code/message` 被静默吞成 `code=0/ok`）。
       因此改为在序列化结果里留哨兵键，由 `EnvelopeRoute` 识别。
    2. 路由函数**不能**写 `-> Enveloped` 返回注解：FastAPI 会把返回注解当作
       `response_model`，Pydantic 只按 dataclass 字段裁剪输出 ——
       实测症状是响应体恰好只剩 `{data, code, message, meta}`，
       少了 `trace_id` 与 `elapsed_ms`，而接口本身不报任何错。
    """

    def __init__(self, data: Any = None, code: int = 0, message: str = "ok",
                 meta: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(data=data, code=int(code), message=message,
                         meta=dict(meta or {}))
        self[ENVELOPE_MARK] = True


def envelope(
    data: Any = None,
    *,
    code: int = 0,
    message: str = "ok",
    meta: Optional[Mapping[str, Any]] = None,
    trace_id: Optional[str] = None,
    elapsed: Optional[int] = None,
) -> dict:
    from server.core.logging import get_trace_id

    return {
        "code": int(code),
        "message": message,
        "data": data,
        "meta": dict(meta or {}),
        "trace_id": trace_id or get_trace_id() or "-",
        "elapsed_ms": elapsed_ms() if elapsed is None else int(elapsed),
    }


def ok(data: Any = None, message: str = "ok", meta: Optional[Mapping[str, Any]] = None) -> dict:
    return envelope(data, code=0, message=message, meta=meta)


def degraded(data: Any, reason: str, message: str = "部分能力已降级，结果仍可用") -> Enveloped:
    """降级**不是错误**：HTTP 仍 200，用 `code=60401` + `meta.degraded` 表达。

    见 `docs/api-specification.md` §6 的「重要」提示：前端不应把
    「新番候选没取到」当成错误弹窗。
    """
    return Enveloped(
        data=data,
        code=60401,
        message=message,
        meta={"degraded": True, "degraded_reason": reason},
    )


def paginate(items: list, page: int, size: int, total: int) -> dict:
    """`data` 的分页结构（§1.2）。`total` 必须是**过滤后**的总数。"""
    pages = (total + size - 1) // size if size else 0
    return {
        "list": list(items),
        "pagination": {"page": page, "size": size, "total": total, "pages": pages},
    }


# ---------------------------------------------------------------- 路由包裹


class EnvelopeRoute(APIRoute):
    """自动把 handler 的返回值包成统一响应体。

    只在 `application/json` 且状态码 < 400 时包裹；SSE（`/chat/message`）
    与文件响应原样透传，否则流式会被整体读进内存。

    ⚠️ 两个坑（都真实踩过，别改回去）：

    1. **必须在每个子路由上显式设 `route_class`**，不能只写
       `app.router.route_class = EnvelopeRoute`。子路由的 `APIRoute` 是在
       **装饰器执行的那一刻**由该路由器自己的 `route_class` 创建并固定的，
       父层（`make_router()` 出的父路由、乃至 `app.router`）事后改设置都追不回去。
       所以统一走 `make_router()`。
       实测复核（2026-09-14，fastapi 0.141.1）：把朴素 `APIRouter()` 的子路由
       挂到已设好 `route_class` 的父路由下，响应仍是裸 `{"pong": true}`；
       换成 `make_router()` 立刻变成完整信封。**规则不变。**
    2. **handler 不能标注 `-> Enveloped`**。那个注解会被 FastAPI 当成
       `response_model`，Pydantic 按 dataclass 字段裁剪，响应体就只剩
       `{data, code, message, meta}`，`trace_id`/`elapsed_ms` 消失。
       `Enveloped` 也因此改成了带哨兵键的 `dict` 子类。
    """

    def get_route_handler(self) -> Callable:
        original = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            response = await original(request)
            media = response.headers.get("content-type", "")
            if not media.startswith("application/json") or response.status_code >= 400:
                return response

            raw = getattr(response, "body", b"") or b""
            try:
                payload = json.loads(raw) if raw else None
            except (ValueError, UnicodeDecodeError):
                return response                       # 不是 JSON 就别动它

            if isinstance(payload, dict) and payload.pop(ENVELOPE_MARK, False):
                body = envelope(payload.get("data"),
                                code=payload.get("code", 0),
                                message=payload.get("message", "ok"),
                                meta=payload.get("meta"))
            else:
                body = ok(payload)

            headers = {k: v for k, v in response.headers.items()
                       if k.lower() != "content-length"}
            return Response(
                content=json.dumps(body, ensure_ascii=False),
                status_code=response.status_code,
                media_type="application/json",
                headers=headers,
                # ⚠️ 必须把 FastAPI 挂在原响应上的 BackgroundTasks 带过去，
                # 否则所有 background.add_task（如 POST /records 的增量重排）
                # 都会被静默丢弃：接口返回 recompute_scheduled=true，任务永不执行。
                # 实测症状：加番后 user_profile 永不更新（2026-09-15）。
                background=getattr(response, "background", None),
            )

        return custom_route_handler


def json_error(status_code: int, code: int, message: str) -> JSONResponse:
    """给异常处理器用：错误也要带 trace_id 与 elapsed_ms（§1.4）。"""
    return JSONResponse(
        status_code=status_code,
        content=envelope(None, code=code, message=message),
    )


def make_router(**kwargs) -> APIRouter:
    """创建**带统一响应体包裹**的 `APIRouter`。

    ⚠️ 必须显式给每个子路由传 `route_class`，不能用
    `app.router.route_class = EnvelopeRoute` 一劳永逸：子路由的 `APIRoute`
    是在装饰器执行时由该路由器自己的 `route_class` 创建并固定的。
    实测症状很隐蔽 —— 接口不报错，只是响应里少了 `trace_id` 与 `elapsed_ms`，
    因为 FastAPI 把 handler 返回的 `Enveloped` 当普通 dataclass 序列化成了
    `{code, data, message, meta}`（正好是除了那两个字段之外的四个）。

    回归覆盖：`tests/test_api/test_envelope.py` 会遍历**全部 44 个端点**
    验证信封完整性。注意那份测试不能用 `app.routes` 递归找子路由 ——
    fastapi 0.141.1 起 `include_router` 不再摊平，见该文件 `api_routes()`。
    """
    kwargs.setdefault("route_class", EnvelopeRoute)
    return APIRouter(**kwargs)


__all__ = [
    "ENVELOPE_MARK",
    "Enveloped",
    "EnvelopeRoute",
    "degraded",
    "elapsed_ms",
    "envelope",
    "json_error",
    "make_router",
    "mark_request_start",
    "new_trace_id",
    "ok",
    "paginate",
]
