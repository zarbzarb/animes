# -*- coding: utf-8 -*-
"""业务错误码与全局异常处理器。

错误码**不是自己编的**，全部来自 `docs/agent-interaction-protocol.md` §4.2 的清单。
编码规则::

      A 0 4 0 1
      │ │ └──┬──┘
      │ │    └── 具体错误序号（3 位）
      │ └────── 错误类别（0=通用 1=入参 2=依赖 3=超时 4=降级 5=LLM）
      └──────── 级别：4=客户端可修复  5=服务端错误  6=Agent 链路错误

新增错误码**必须同时更新协议文档**（`docs/dev-conventions.md` 的「代码与文档同一 PR」），
否则前端拿到一个没人认识的码。
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any, Mapping, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from server.core.response import envelope

logger = logging.getLogger(__name__)


class ErrCode(IntEnum):
    """协议 §4.2 错误码清单（值是业务码，不是 HTTP 状态码）。"""

    OK = 0
    INVALID_PARAM = 40001
    UNAUTHORIZED = 40101
    FORBIDDEN = 40301
    RESOURCE_NOT_FOUND = 40401
    DUPLICATE_REQUEST = 40901
    RATE_LIMITED = 42901
    INTERNAL_ERROR = 50001
    MODEL_UNAVAILABLE = 50201
    DB_UNAVAILABLE = 50202
    CACHE_UNAVAILABLE = 50203
    INDEX_MISSING = 50204
    AGENT_TIMEOUT = 50301
    AGENT_CHAIN_BROKEN = 60301
    AGENT_DEGRADED = 60401
    DEADLOCK_DETECTED = 60501
    TOOL_CALL_LIMIT = 60502
    CONTEXT_OVERFLOW = 60503
    LLM_TIMEOUT = 60601
    LLM_RATE_LIMITED = 60602
    LLM_INVALID_JSON = 60603
    LLM_HALLUCINATION = 60604


# 业务码 -> HTTP 状态码（§6 的映射表）
_HTTP_STATUS: Mapping[int, int] = {
    ErrCode.INVALID_PARAM: 400,
    ErrCode.UNAUTHORIZED: 401,
    ErrCode.FORBIDDEN: 403,
    ErrCode.RESOURCE_NOT_FOUND: 404,
    ErrCode.DUPLICATE_REQUEST: 409,
    ErrCode.RATE_LIMITED: 429,
    ErrCode.INTERNAL_ERROR: 500,
    ErrCode.MODEL_UNAVAILABLE: 503,
    ErrCode.DB_UNAVAILABLE: 503,
    ErrCode.CACHE_UNAVAILABLE: 200,      # 缓存不可用要**绕过缓存直算**，不是错误
    ErrCode.INDEX_MISSING: 503,
    ErrCode.AGENT_TIMEOUT: 504,          # §7：超时返回 504 + 50301
    ErrCode.AGENT_CHAIN_BROKEN: 503,
    ErrCode.AGENT_DEGRADED: 200,         # 降级一律 200
    ErrCode.DEADLOCK_DETECTED: 500,
    ErrCode.TOOL_CALL_LIMIT: 200,
    ErrCode.CONTEXT_OVERFLOW: 200,
    ErrCode.LLM_TIMEOUT: 200,            # LLM 超时降级到模板文案
    ErrCode.LLM_RATE_LIMITED: 200,
    ErrCode.LLM_INVALID_JSON: 200,
    ErrCode.LLM_HALLUCINATION: 200,
}


class BizError(Exception):
    """业务异常。`detail` 只用于日志与 `meta`，不外泄到 message。"""

    def __init__(
        self,
        code: int,
        message: str = "",
        *,
        http_status: Optional[int] = None,
        detail: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ):
        self.code = int(code)
        try:
            self.name = ErrCode(self.code).name
        except ValueError:
            self.name = "UNKNOWN"
        self.message = message or self.name
        self.http_status = http_status or _HTTP_STATUS.get(self.code, 500)
        self.detail: dict = dict(detail or {})
        # 限流要带 Retry-After（§7 的「超时响应」约定同源）
        self.headers: dict = dict(headers or {})
        super().__init__(f"[{self.code} {self.name}] {self.message}")


# ---------------- 常用快捷构造（让调用点读起来是业务语言）----------------
def invalid_param(msg: str = "参数错误", **kw) -> BizError:
    return BizError(ErrCode.INVALID_PARAM, msg, **kw)


def unauthorized(msg: str = "登录状态已失效，请重新登录", **kw) -> BizError:
    return BizError(ErrCode.UNAUTHORIZED, msg, **kw)


def forbidden(msg: str = "无权限访问", **kw) -> BizError:
    return BizError(ErrCode.FORBIDDEN, msg, **kw)


def not_found(msg: str = "资源不存在", **kw) -> BizError:
    return BizError(ErrCode.RESOURCE_NOT_FOUND, msg, **kw)


def model_unavailable(msg: str = "模型不可用", **kw) -> BizError:
    return BizError(ErrCode.MODEL_UNAVAILABLE, msg, **kw)


def agent_timeout(msg: str = "处理超时，请稍后重试", **kw) -> BizError:
    return BizError(ErrCode.AGENT_TIMEOUT, msg, **kw)


def register_exception_handlers(app: FastAPI) -> None:
    """把异常统一转成响应体（否则 FastAPI 默认返回 `{"detail": ...}`）。"""

    @app.exception_handler(BizError)
    async def _biz(request: Request, exc: BizError) -> JSONResponse:  # noqa: ARG001
        if exc.http_status >= 500:
            logger.warning("业务异常 %s %s detail=%s", exc.code, exc.message, exc.detail)
        return JSONResponse(
            status_code=exc.http_status,
            headers=exc.headers or None,
            content=envelope(None, code=exc.code, message=exc.message,
                             meta={"error_name": exc.name, **exc.detail}),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:  # noqa: ARG001
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", ()) if x != "body")
        msg = f"参数错误：{loc} {first.get('msg', '')}".strip()
        return JSONResponse(
            status_code=400,
            content=envelope(None, code=int(ErrCode.INVALID_PARAM), message=msg,
                             meta={"error_name": "INVALID_PARAM"}),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:  # noqa: ARG001
        code = {400: ErrCode.INVALID_PARAM, 401: ErrCode.UNAUTHORIZED,
                403: ErrCode.FORBIDDEN, 404: ErrCode.RESOURCE_NOT_FOUND,
                409: ErrCode.DUPLICATE_REQUEST,
                429: ErrCode.RATE_LIMITED}.get(exc.status_code, ErrCode.INTERNAL_ERROR)
        return JSONResponse(
            status_code=exc.status_code,
            content=envelope(None, code=int(code), message=str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
        # 未捕获异常必须打完整栈，且把 trace_id 回给前端 —— 否则用户报障时无从查起
        logger.exception("未捕获异常：%s", exc)
        return JSONResponse(
            status_code=500,
            content=envelope(None, code=int(ErrCode.INTERNAL_ERROR),
                             message="服务端异常，请稍后重试"),
        )


__all__ = [
    "BizError",
    "ErrCode",
    "agent_timeout",
    "forbidden",
    "invalid_param",
    "model_unavailable",
    "not_found",
    "register_exception_handlers",
    "unauthorized",
]
