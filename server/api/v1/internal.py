# -*- coding: utf-8 -*-
"""Agent 内部接口（api-specification.md §4.1）。

**这不是业务接口**，而是「进程内调用 → 跨进程调用」的传输适配层：
现在 A0 直接 `call_agent` 走内存；将来若要把某个 Agent 拆成独立服务，
A0 只需把传输实现换成打这个端点，**业务代码一行不改**。

安全上它要求：内网来源 + `X-Internal-Token`（协议 §1.3 的 🛡️）。
两道都要，因为企业内网并不是可信边界（同一 VPC 里任何被攻陷的机器都能访问）。
"""

from __future__ import annotations

import logging
import ipaddress

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field

from server.core.exceptions import forbidden, invalid_param
from server.core.response import make_router, Enveloped
from server.core.security import check_internal_token
from server.deps import get_gw
from server.services.agent_bridge import call_agent_direct

logger = logging.getLogger(__name__)
router = make_router(prefix="/internal", tags=["内部"])

_NAME_TO_ID = {"orchestrator": "A0", "profile": "A1", "recall": "A2",
               "coldstart": "A3", "fusion": "A4", "explain": "A5",
               "drift": "A6", "chat": "A7", "dataops": "A8", "eval": "A9"}

# 允许的内网来源。`127.0.0.1` / `::1` 恒允许（本机调试与同机 Agent 容器）。
_ALLOWED_NETS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
)


class EnvelopeIn(BaseModel):
    """协议 §3.1 的消息信封（这里只做透传，不校验内部字段）。"""

    header: dict = Field(default_factory=dict)
    context: dict = Field(default_factory=dict)
    payload: dict = Field(default_factory=dict)
    meta: dict = Field(default_factory=dict)


def _require_internal(request: Request,
                      token: str | None = Header(default=None, alias="X-Internal-Token")) -> None:
    check_internal_token(token)
    host = request.client.host if request.client else ""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise forbidden(f"来源地址不可解析：{host}")
    if not any(ip in net for net in _ALLOWED_NETS):
        raise forbidden(f"内部接口仅限内网访问，当前来源 {host}")


@router.post("/agents/{name}", summary="Agent 内部调用（内网）",
             dependencies=[Depends(_require_internal)])
async def invoke_agent(name: str, body: EnvelopeIn) :
    aid = name.upper()
    if aid not in {f"A{i}" for i in range(10)}:
        aid = _NAME_TO_ID.get(name.lower(), "")
    if not aid:
        raise invalid_param(f"未知 Agent：{name}")

    header = dict(body.header or {})
    action = str(header.get("action") or body.payload.get("_action") or "")
    if not action:
        raise invalid_param("信封 header 缺少 action")

    r = await call_agent_direct(aid, action, dict(body.payload or {}),
                                timeout_ms=int(header.get("timeout_ms") or 8000),
                                user_id=header.get("user_id"))
    return Enveloped(
        data={"agent": aid, "action": action, "ok": r.ok, "payload": r.payload,
              "meta": r.meta, "chain": list(r.chain or [])},
        code=0 if r.ok else int(r.error_code or 50001),
        message="ok" if r.ok else (r.message or r.error_name),
    )


@router.get("/health", summary="内网健康检查")
async def internal_health() :
    gw = get_gw()
    try:
        ok = bool(gw.ping())
    except Exception as exc:
        logger.warning("DB ping 失败：%s", exc)
        ok = False
    return Enveloped(data={"db": ok, "counts": gw.counts() if ok else {}})
