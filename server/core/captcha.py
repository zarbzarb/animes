# -*- coding: utf-8 -*-
"""登录验证码（纯 Python 生成 SVG，零第三方依赖）。

设计说明
--------
* 验证码是一次性的：校验成功**或**失败都立即作废（防重放暴力试错）。
* 存储复用 `server.core.cache`（内存/Redis 同一套门面），TTL 5 分钟。
* 字符集刻意去掉 `0O1lI` 这类易混字形，减少用户输入挫败感。
* 只做**前端强制 + 服务端可选强制**（`settings.AUTH_CAPTCHA_STRICT`）：
  - 前端登录页永远要求验证码；
  - 服务端默认宽松（带 captcha_id 就校验），测试/冒烟脚本不传也能登录；
  - 生产环境在 .env 里把 `AUTH_CAPTCHA_STRICT=true` 打开后，不带验证码的
    登录请求直接 422。
"""
from __future__ import annotations

import random
import secrets
import uuid

from server.core.config import settings
from server.core.cache import get_cache

# 去掉易混字符后的安全字符集
_ALPHABET = "23456789abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ"
_TTL = 300          # 5 分钟
_LENGTH = 4

_PALETTE = ("#7c4a64", "#4a6b7c", "#5a6e46", "#6b5a8a", "#8a5a4a", "#3f6f6a")


def _svg(code: str) -> str:
    """4 个字符画成扭曲、着色、带干扰线的 SVG（浏览器原生渲染，无需 Pillow）。"""
    rng = random.Random(secrets.randbits(64))
    w, h = 108, 40
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="#f7f4f6" rx="6"/>',
    ]
    # 干扰线
    for _ in range(3):
        y = rng.randint(6, h - 6)
        c = rng.choice(_PALETTE)
        parts.append(
            f'<path d="M0 {y} Q {w // 3} {rng.randint(4, h - 4)}, {w} {rng.randint(6, h - 6)}" '
            f'stroke="{c}" stroke-width="1" fill="none" opacity="0.35"/>')
    # 字符：随机旋转 + 基线抖动 + 颜色
    for i, ch in enumerate(code):
        x = 14 + i * (w - 28) // max(1, _LENGTH - 1) - 6
        y = h // 2 + rng.randint(-4, 4)
        rot = rng.randint(-24, 24)
        fs = rng.randint(22, 27)
        parts.append(
            f'<text x="{x}" y="{y}" font-family="Georgia, serif" font-size="{fs}" '
            f'font-weight="bold" fill="{rng.choice(_PALETTE)}" '
            f'text-anchor="middle" dominant-baseline="central" '
            f'transform="rotate({rot} {x} {y})">{ch}</text>')
    # 干扰点
    for _ in range(24):
        parts.append(
            f'<circle cx="{rng.randint(2, w - 2)}" cy="{rng.randint(2, h - 2)}" '
            f'r="1" fill="{rng.choice(_PALETTE)}" opacity="0.4"/>')
    parts.append("</svg>")
    return "".join(parts)


async def new_captcha() -> dict:
    """生成一条验证码：`{captcha_id, svg}`，答案落缓存（TTL 5 分钟）。"""
    code = "".join(secrets.choice(_ALPHABET) for _ in range(_LENGTH))
    captcha_id = uuid.uuid4().hex
    cache = get_cache()
    await cache.set_json(f"captcha:{captcha_id}", code, ttl=_TTL)
    return {"captcha_id": captcha_id, "svg": _svg(code), "expires_in": _TTL}


async def verify_captcha(captcha_id: str, code: str) -> bool:
    """一次性校验：无论对错都作废（防同一条验证码反复试错）。

    `captcha_id` 为空直接 False；大小写不敏感。
    """
    if not captcha_id or not code:
        return False
    cache = get_cache()
    key = f"captcha:{captcha_id}"
    stored = await cache.get_json(key)
    if stored is None:
        return False
    await cache.delete(key)                      # 一次性
    return str(stored).strip().lower() == str(code).strip().lower()
