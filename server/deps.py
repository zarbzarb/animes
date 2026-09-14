# -*- coding: utf-8 -*-
"""FastAPI 依赖：鉴权、网关、限流。

为什么单独一个模块而不是塞进 `api/v1/*`
--------------------------------------
同一份「取当前用户」的逻辑会被 20+ 个 handler 用到。放在这里之后，
handler 只要写 `user: dict = Depends(current_user)`，鉴权口径只有一处 ——
否则总有一条路由忘了校验 token，而这种漏洞不会报错、只会静默放行。

`gateway` 用进程内单例：它本身无状态（每次调用自己开 `session_scope`），
每个请求新建一个只是浪费对象分配。
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, Header, Request

from server.core.exceptions import forbidden, not_found, unauthorized
from server.core.security import decode_access_token
from server.db.gateway import SqlGateway

logger = logging.getLogger(__name__)

__all__ = [
    "current_admin",
    "current_user",
    "get_gw",
    "is_visible_anime",
    "optional_user",
    "rate_limit",
]

_gateway: Optional[SqlGateway] = None


def get_gw() -> SqlGateway:
    """进程内单例网关（Agent 层通过 `bind_gateway` 拿同一个实例）。"""
    global _gateway
    if _gateway is None:
        _gateway = SqlGateway()
    return _gateway


def reset_gateway() -> None:
    """测试用。"""
    global _gateway
    _gateway = None


# ---------------------------------------------------------------- 鉴权


def _bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


async def optional_user(authorization: Optional[str] = Header(default=None)) -> Optional[dict]:
    """匿名可访问的接口用（如 `/animes/*`）。token 无效**不报错**，视作匿名。"""
    token = _bearer(authorization)
    if not token:
        return None
    try:
        payload = decode_access_token(token)
    except Exception:      # 匿名接口不该因为带了个坏 token 就 401
        return None
    uid = payload.get("sub")
    if uid is None:
        return None
    user = get_gw().get_user(int(uid))
    return user if (user and int(user.get("status", 1)) == 1) else None


async def current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """需要登录的接口用。任何失败都是 `40101`。"""
    token = _bearer(authorization)
    if not token:
        raise unauthorized("缺少 Authorization: Bearer <token>")
    payload = decode_access_token(token)          # 失败时抛 40101
    uid = payload.get("sub")
    if uid is None:
        raise unauthorized("token 载荷缺少 sub")
    user = get_gw().get_user(int(uid))
    if user is None:
        raise unauthorized("用户不存在或已注销")
    if int(user.get("status", 1)) != 1:
        raise forbidden("账号已被禁用")
    return user


async def current_admin(user: dict = Depends(current_user)) -> dict:
    """管理后台接口用。"""
    if int(user.get("role", 0)) != 1:
        raise forbidden("需要管理员权限")
    return user


def is_visible_anime(a: Optional[dict]) -> bool:
    """用户侧的番剧**可见性**判定：存在、未禁用、且在线。

    ⚠️ 为什么要同时看两个标志（2026-09-14 修的真实不一致）：
    `set_anime_offline()` 置的是 `is_online=0`，列表/计数/热门/批量四条读路
    （`list_anime` / `count_anime` / `popular_anime_ids` / `get_animes`）都过滤它，
    **唯独单查 `get_anime` 不过滤** —— 因为 admin 编辑下架番也要能查到它。
    结果是：管理员点"下架"，番从列表消失，但详情页/相似接口/加追番的
    存在性检查照样放行，且不报任何错。

    所以可见性规则收敛在这里这一份。用户侧读门（详情 / 相似 / 加追番 /
    反馈）统一调用；admin 路径继续直接用 `get_anime`。

    ⚠️ 历史备注：本模块曾有 `require_anime` 依赖封装同一件事，但它**从未被
    任何路由引用**（各接口都在内联写 `get_anime(...) is None`）—— 死代码
    制造了"有统一依赖"的假象。已删除；新增需要番剧门的接口请直接用本函数。
    """
    if not a:
        return False
    return (int(a.get("is_forbidden", 0)) != 1
            and int(a.get("is_online", 1)) == 1)


def rate_limit(spec: str):
    """按协议 §7 的限额表限流。用法：`dependencies=[Depends(rate_limit("login"))]`。

    `spec` 是 `core/rate_limit.py` 里的限额常量名（`LOGIN` / `RECOMMEND_FEED` / …）。
    返回的是**依赖工厂**：FastAPI 要求依赖可调用且签名可解析，
    所以这里返回内部协程，而不是直接返回 `enforce` 的结果。

    身份键：**有 token 用 sub，没 token 退回 IP** —— 登录接口按 IP 防撞库，
    其余按用户算（NAT 后面一个 IP 可能是几百人，按 IP 会误伤无关用户）。
    """
    from server.core import rate_limit as _rl

    chosen = getattr(_rl, spec.upper(), None)
    if chosen is None:
        raise RuntimeError(f"未定义的限流档位：{spec}（见 core/rate_limit.py）")

    async def _dep(request: Request,
                   authorization: Optional[str] = Header(default=None)) -> None:
        uid: Optional[int] = None
        token = _bearer(authorization)
        if token:
            try:
                uid = int(decode_access_token(token).get("sub"))
            except Exception:
                uid = None
        await _rl.enforce(chosen, request, uid)

    _dep.__name__ = f"rate_limit_{spec}"
    return _dep
