# -*- coding: utf-8 -*-
"""认证模块（api-specification.md §3.1）。

4 个接口：注册 / 登录 / 刷新 / 登出。除了 `register` / `login` 都要求已登录。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from server.core.exceptions import BizError, ErrCode, unauthorized
from server.core.response import make_router, Enveloped, degraded
from server.core.security import create_access_token, hash_password, verify_password
from server.deps import current_user, get_gw, rate_limit
from server.core.config import settings

logger = logging.getLogger(__name__)
router = make_router(prefix="/auth", tags=["认证"])


class RegisterIn(BaseModel):
    username: str = Field(..., min_length=4, max_length=50)
    password: str = Field(..., min_length=8, max_length=64)
    email: str | None = None
    nickname: str | None = Field(None, max_length=50)

    @field_validator("username")
    @classmethod
    def _username_charset(cls, v: str) -> str:
        import re
        if not re.fullmatch(r"[A-Za-z0-9_]+", v):
            raise ValueError("只允许字母、数字、下划线")
        return v

    @field_validator("password")
    @classmethod
    def _password_strength(cls, v: str) -> str:
        if not any(c.isalpha() for c in v) or not any(c.isdigit() for c in v):
            raise ValueError("至少包含字母与数字")
        return v

    @field_validator("email")
    @classmethod
    def _email_shape(cls, v: str | None) -> str | None:
        import re
        if v in (None, ""):
            return None
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", v):
            raise ValueError("邮箱格式不正确")
        return v


class LoginIn(BaseModel):
    username: str
    password: str


def _token_bundle(user: dict) -> dict:
    role = "admin" if int(user.get("role", 0)) == 1 else "user"
    token = create_access_token(int(user["id"]), role=role)
    return {
        "user": {"id": int(user["id"]), "username": user["username"],
                 "nickname": user.get("nickname"), "role": int(user.get("role", 0)),
                 "avatar_url": user.get("avatar_url")},
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": int(settings.JWT_EXPIRE_MINUTES) * 60,
    }


@router.post("/register", summary="注册", dependencies=[Depends(rate_limit("login"))])
async def register(body: RegisterIn) :
    gw = get_gw()
    if gw.get_user_by_username(body.username):
        raise BizError(ErrCode.DUPLICATE_REQUEST, "用户名已存在",
                       detail={"field": "username"})
    if body.email:
        # 邮箱唯一性：网关没有按邮箱查的方法，用列表扫一遍即可（量小）
        for u in gw.list_users(limit=1000):
            if (u.get("email") or "").lower() == body.email.lower():
                raise BizError(ErrCode.DUPLICATE_REQUEST, "邮箱已被注册",
                               detail={"field": "email"})
    user = gw.create_user(username=body.username,
                          password_hash=hash_password(body.password),
                          nickname=body.nickname or body.username,
                          email=body.email, role=0)
    logger.info("新用户注册：%s(id=%s)", user["username"], user["id"])
    return Enveloped(data=_token_bundle(user), code=0, message="注册成功")


@router.post("/login", summary="登录", dependencies=[Depends(rate_limit("login"))])
async def login(body: LoginIn) :
    gw = get_gw()
    user = gw.get_user_by_username(body.username)
    # 用户名不存在与密码错误返回**同一个**错误码：否则接口变成账号枚举器
    if user is None or not verify_password(body.password, user.get("password_hash") or ""):
        raise unauthorized("用户名或密码错误")
    if int(user.get("status", 1)) != 1:
        raise BizError(ErrCode.FORBIDDEN, "账号已被禁用")
    gw.touch_login(int(user["id"]))
    return Enveloped(data=_token_bundle(user), code=0, message="登录成功")


@router.post("/refresh", summary="刷新 token")
async def refresh(user: dict = Depends(current_user)) :
    return Enveloped(data=_token_bundle(user), code=0, message="已刷新")


@router.post("/logout", summary="登出")
async def logout(user: dict = Depends(current_user)) :
    """无状态 JWT + 无 Redis 时的登出。

    ⚠️ 这是**诚实降级**：真正的登出需要把 token 加入黑名单（Redis `jwt:blacklist:{jti}`），
    Redis 不可用时无法实现服务端吊销。此时返回 `degraded`，前端应当
    **依旧清除本地 token**（这才是主要防线），并提示"本地已登出"。
    """
    from server.core.cache import get_cache

    cache = get_cache()
    if cache.backend_name == "redis":
        await cache.set_json(f"jwt:revoked:{user['id']}", {"at": user.get("id")},
                             ttl=int(settings.JWT_EXPIRE_MINUTES) * 60)
        return Enveloped(data={"revoked": True}, code=0, message="已登出")

    return degraded({"revoked": False, "local_only": True},
                    reason="cache_unavailable",
                    message="本地 token 已失效；服务端吊销需要 Redis")
