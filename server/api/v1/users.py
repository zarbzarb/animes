# -*- coding: utf-8 -*-
"""用户自助接口（api-specification.md 总览 #5–#8）。

`DELETE /users/me/memory` 是**隐私要求**的落地：清空个人记忆数据
（`agent_memory_short` 的会话记忆 + `user_profile` 的画像摘要 + 兴趣胶囊）。
不清 `watch_record`（那是用户自己的追番数据，不属于"记忆"）；
但会作废已生成的推荐与解释，因为它们是从记忆里推出来的。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel, Field

from server.core.exceptions import BizError, ErrCode, invalid_param
from server.core.response import make_router, Enveloped, new_trace_id
from server.core.security import hash_password, verify_password
from server.deps import current_user, get_gw

logger = logging.getLogger(__name__)
router = make_router(prefix="/users", tags=["用户"])


class ProfilePatch(BaseModel):
    nickname: str | None = Field(None, min_length=1, max_length=50)
    email: str | None = None
    avatar_url: str | None = Field(None, max_length=512)


class PasswordIn(BaseModel):
    old_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=64)


def _me(u: dict) -> dict:
    return {"id": int(u["id"]), "username": u["username"],
            "nickname": u.get("nickname"), "email": u.get("email"),
            "avatar_url": u.get("avatar_url"), "role": int(u.get("role", 0)),
            "status": int(u.get("status", 1)),
            "last_login_at": u.get("last_login_at")}


@router.get("/me", summary="当前用户信息")
async def me(user: dict = Depends(current_user)) :
    return Enveloped(data=_me(user))


@router.put("/me", summary="修改资料")
async def update_me(body: ProfilePatch,
                    user: dict = Depends(current_user)) :
    gw = get_gw()
    uid = int(user["id"])
    data = body.model_dump(exclude_none=True)
    if not data:
        raise invalid_param("没有需要更新的字段")

    if "email" in data and data["email"]:
        for u in gw.list_users(limit=1000):
            if (int(u["id"]) != uid
                    and (u.get("email") or "").lower() == data["email"].lower()):
                raise BizError(ErrCode.DUPLICATE_REQUEST, "邮箱已被占用",
                               detail={"field": "email"})
    gw.update_user(uid, data)
    return Enveloped(data=_me(gw.get_user(uid) or user), code=0, message="已更新")


@router.post("/me/avatar", summary="上传头像")
async def upload_avatar(file: UploadFile = File(...),
                        user: dict = Depends(current_user)) :
    """头像上传：校验类型/大小 → 存 `data/uploads/` → 更新 `user.avatar_url`。

    返回相对路径（/static/uploads/...），前端拼域名即可；旧头像文件不删
    （同名覆盖前先换文件名，避免浏览器缓存看到旧图）。
    """
    ext_map = {"image/jpeg": ".jpg", "image/png": ".png",
               "image/webp": ".webp", "image/gif": ".gif"}
    ext = ext_map.get(file.content_type or "")
    if ext is None:
        raise invalid_param("只支持 jpg / png / webp / gif 格式")
    blob = await file.read()
    if len(blob) > 2 * 1024 * 1024:
        raise invalid_param("头像不能超过 2MB")
    if not blob:
        raise invalid_param("文件为空")

    from server.core.config import PROJECT_ROOT
    updir = PROJECT_ROOT / "data" / "uploads"
    updir.mkdir(parents=True, exist_ok=True)
    name = f"avatar_{int(user['id'])}_{new_trace_id()[:8]}{ext}"
    (updir / name).write_bytes(blob)

    url = f"/static/uploads/{name}"
    get_gw().update_user(int(user["id"]), {"avatar_url": url})
    return Enveloped(data={"avatar_url": url}, code=0, message="已上传")


@router.put("/me/password", summary="改密码")
async def change_password(body: PasswordIn,
                          user: dict = Depends(current_user)) :
    gw = get_gw()
    full = gw.get_user_by_username(user["username"]) or {}
    if not verify_password(body.old_password, full.get("password_hash") or ""):
        raise BizError(ErrCode.UNAUTHORIZED, "原密码不正确")
    if body.old_password == body.new_password:
        raise invalid_param("新密码不能与原密码相同")
    if not (any(c.isalpha() for c in body.new_password)
            and any(c.isdigit() for c in body.new_password)):
        raise invalid_param("新密码至少包含字母与数字")
    gw.update_password(int(user["id"]), hash_password(body.new_password))
    return Enveloped(data={"changed": True}, code=0, message="密码已修改")


@router.delete("/me/memory", summary="清空个人记忆数据")
async def clear_memory(user: dict = Depends(current_user)) :
    """清空**记忆类**数据（画像 / 会话记忆 / 兴趣胶囊），保留追番记录。

    清完必须作废推荐与解释：它们是从被清掉的记忆推出来的，
    留着会出现"记忆已清空，但首页还是按老口味推荐"的明显矛盾。
    """
    gw = get_gw()
    uid = int(user["id"])
    stats = gw.delete_user_memory(uid)
    try:
        gw.invalidate_recommendations(uid)
    except Exception as exc:
        logger.warning("清空记忆后作废推荐失败：%s", exc)

    from server.core.cache import get_cache, profile_key, rec_scope_prefix

    try:
        cache = get_cache()
        # 画像是一个键；推荐结果是 `rec:{uid}:{scene}[:{gid}]` 一整片，
        # 必须按前缀删（理由见 `Cache.delete_prefix`）。
        await cache.delete(profile_key(uid))
        await cache.delete_prefix(rec_scope_prefix(uid))
    except Exception as exc:
        logger.debug("清缓存失败（忽略）：%s", exc)

    return Enveloped(data=stats, code=0, message="个人记忆已清空（追番记录保留）")
