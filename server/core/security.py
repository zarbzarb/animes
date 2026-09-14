# -*- coding: utf-8 -*-
"""认证：密码哈希 + JWT 签发/校验。

两个刻意的工程决定
------------------
1. **JWT 只放 `sub`（user_id）与 `role`，不放任何 PII**。协议文档要求
   LLM 不接触用户明文隐私字段；把邮箱/手机号写进 token 等于把它们散播到
   前端存储、日志与第三方链路里。
2. **密码哈希与 JWT 都有纯标准库回退**。`passlib` / `python-jose` 在某些
   bcrypt / cryptography 版本组合下会 import 失败（`passlib` 读 `bcrypt.__about__`
   在 bcrypt 4.x 被删掉后就会报错）。认证是**主链路的第一道门**，
   不能因为一个可选依赖的版本差异就让整个服务起不来 —— 所以这里用
   `bcrypt` 原生库优先、`pbkdf2_hmac` 兜底，两条路径的哈希串都自带算法前缀，
   可共存互认。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any, Optional

from server.core.config import settings
from server.core.exceptions import BizError, forbidden, unauthorized

logger = logging.getLogger(__name__)

_PBKDF2_ROUNDS = 260_000     # OWASP 2023 对 pbkdf2-sha256 的建议下限
_PBKDF2_PREFIX = "pbkdf2_sha256"


# ------------------------------------------------------------ 密码哈希


def _bcrypt_module():
    try:
        import bcrypt  # type: ignore
        return bcrypt
    except Exception:  # pragma: no cover - 取决于本机依赖
        return None


def hash_password(plain: str) -> str:
    if not plain:
        raise ValueError("密码不能为空")
    bc = _bcrypt_module()
    if bc is not None:
        return bc.hashpw(plain.encode("utf-8"), bc.gensalt()).decode("utf-8")

    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"),
                                 salt.encode("utf-8"), _PBKDF2_ROUNDS)
    return f"{_PBKDF2_PREFIX}${_PBKDF2_ROUNDS}${salt}${digest.hex()}"


def verify_password(plain: str, hashed: str) -> bool:
    """恒定时间比较；算法由哈希串前缀决定，两种实现可共存。"""
    if not plain or not hashed:
        return False
    if hashed.startswith(_PBKDF2_PREFIX + "$"):
        try:
            _, rounds, salt, expected = hashed.split("$", 3)
            digest = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"),
                                         salt.encode("utf-8"), int(rounds))
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(digest.hex(), expected)

    bc = _bcrypt_module()
    if bc is None:
        # 库里存的是 bcrypt 串但本机没有 bcrypt：明确报错，别静默返回 False
        # （静默 False 会让"所有老用户都登不上"，排查时极难定位）
        logger.error("检测到 bcrypt 哈希但本机缺少 bcrypt 库，无法校验密码")
        return False
    try:
        return bc.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


# ------------------------------------------------------------ JWT


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def _jose():
    try:
        from jose import jwt  # type: ignore
        return jwt
    except Exception:  # pragma: no cover
        return None


def create_access_token(user_id: int, *, role: str = "user",
                        expires_minutes: Optional[int] = None,
                        extra: Optional[dict] = None) -> str:
    """签发 access token。载荷只放 sub / role / iat / exp（无 PII）。"""
    now = int(time.time())
    ttl = int(expires_minutes or settings.JWT_EXPIRE_MINUTES)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "iat": now,
        "exp": now + ttl * 60,
    }
    if extra:
        payload.update(extra)

    jwt = _jose()
    if jwt is not None:
        return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

    # 回退：标准库实现的 HS256（只支持 HS256/HS384/HS512）
    alg = settings.JWT_ALGORITHM.upper()
    if alg.startswith("HS"):
        bits = int(alg[2:])
        header = {"alg": alg, "typ": "JWT"}
        seg = (_b64url_encode(json.dumps(header, separators=(",", ":")).encode())
               + "." + _b64url_encode(json.dumps(payload, separators=(",", ":")).encode()))
        mac = hmac.new(settings.JWT_SECRET.encode("utf-8"), seg.encode("ascii"),
                       getattr(hashlib, f"sha{bits}")).digest()
        return seg + "." + _b64url_encode(mac)
    raise RuntimeError(f"缺少 python-jose，无法签发 {settings.JWT_ALGORITHM}")


def decode_access_token(token: str) -> dict:
    """校验并解出载荷；任何失败都抛 `40101`（不区分"过期"与"伪造"）。"""
    jwt = _jose()
    if jwt is not None:
        try:
            return jwt.decode(token, settings.JWT_SECRET,
                              algorithms=[settings.JWT_ALGORITHM])
        except Exception as exc:  # ExpiredSignatureError / JWTError
            raise unauthorized(f"token 无效：{type(exc).__name__}") from exc

    try:
        seg, sig = token.rsplit(".", 1)
        header = json.loads(_b64url_decode(seg.split(".")[0]))
        alg = str(header.get("alg", "HS256")).upper()
        bits = int(alg[2:]) if alg.startswith("HS") else 256
        expected = hmac.new(settings.JWT_SECRET.encode("utf-8"), seg.encode("ascii"),
                            getattr(hashlib, f"sha{bits}")).digest()
        if not hmac.compare_digest(_b64url_decode(sig), expected):
            raise BizErrorUnauthorized("token 签名不匹配")
        payload = json.loads(_b64url_decode(seg.split(".")[1]))
    except BizErrorUnauthorized as exc:
        # 自己抛的鉴权错要原样冒泡，别被下面的宽 except 改写成"解析失败" ——
        # 两者对排查的意义完全不同（一个是有人在改 token，一个是格式不对）
        raise unauthorized(exc.reason) from exc
    except Exception as exc:
        raise unauthorized("token 解析失败") from exc

    if int(payload.get("exp", 0)) < int(time.time()):
        raise unauthorized("token 已过期")
    return payload


class BizErrorUnauthorized(Exception):
    """仅供 `decode_access_token` 内部区分「签名不匹配」与「格式错误」。"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def check_internal_token(token: Optional[str]) -> None:
    """`/internal/agents/*` 的内网令牌校验（协议 §1.3 的 🛡️ 约定）。"""
    if not token or not hmac.compare_digest(token, settings.INTERNAL_TOKEN):
        raise forbidden("内部接口令牌无效")


__all__ = [
    "BizError",
    "check_internal_token",
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "verify_password",
]
