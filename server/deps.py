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
    "optional_user",
    "rate_limit",
    "require_anime",
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


async def require_anime(anime_id: int) -> dict:
    """按**数据集 animeID** 取番剧，不存在直接 404。"""
    a = get_gw().get_anime(int(anime_id))
    if a is None or int(a.get("is_forbidden", 0)) == 1:
        raise not_found(f"动漫 {anime_id} 不存在或已下架")
    return a


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
