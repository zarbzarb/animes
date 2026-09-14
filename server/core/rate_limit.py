# -*- coding: utf-8 -*-
"""限流：固定窗口计数器，桶放在 `core/cache.py` 的后端里。

限额不是自己定的，逐条来自 `docs/api-specification.md` §7：

| 接口 | 限流 |
|---|---|
| `/auth/login` | 10 次/分钟/IP |
| `/recommend/feed` | 60 次/分钟/用户 |
| `/recommend/*/explain` | 30 次/分钟/用户 |
| `/chat/message` | 20 次/分钟/用户 |
| `/analysis/*` | 30 次/分钟/用户 |
| `/admin/*` | 120 次/分钟/用户 |

两个刻意的取舍
--------------
1. **缓存不可用时放行**，不拦截。限流是保护措施，不是业务规则：
   Redis 挂了就把正常用户挡在门外，属于「保护措施造成的事故」。
2. **按用户而不是按 IP**（登录接口除外）。校园网/公司出口 NAT 下，
   同一 IP 可能是几百人，按 IP 限流会误伤。
"""

from __future__ import annotations

from typing import Optional

from fastapi import Request

from server.core.cache import get_cache, rate_key
from server.core.exceptions import BizError, ErrCode

# 文档 §7 的限额表，集中在此，避免每个路由各写一个数字
LOGIN = ("login", 10, 60, "ip")
RECOMMEND_FEED = ("recommend_feed", 60, 60, "user")
RECOMMEND_EXPLAIN = ("recommend_explain", 30, 60, "user")
CHAT_MESSAGE = ("chat_message", 20, 60, "user")
ANALYSIS = ("analysis", 30, 60, "user")
ADMIN = ("admin", 120, 60, "user")
INTERNAL = ("internal", 0, 60, "user")        # 0 = 不限（内网）


async def enforce(spec: tuple, request: Request,
                  user_id: Optional[int] = None) -> None:
    """执行一次限流检查。超限抛 `42901`（带 `Retry-After`）。"""
    scope, limit, window, by = spec
    if limit <= 0:
        return

    if by == "ip":
        ident = (request.client.host if request.client else "unknown")
    else:
        if user_id is None:
            ident = (request.client.host if request.client else "unknown")
        else:
            ident = str(user_id)

    count = await get_cache().incr(rate_key(scope, ident), window)
    if count is None:          # 缓存不可用 → 放行
        return
    if count > limit:
        raise BizError(
            ErrCode.RATE_LIMITED,
            f"请求过于频繁（{scope} 上限 {limit} 次/{window}s），请稍后重试",
            headers={"Retry-After": str(window)},
        )


__all__ = [
    "ADMIN",
    "ANALYSIS",
    "CHAT_MESSAGE",
    "INTERNAL",
    "LOGIN",
    "RECOMMEND_EXPLAIN",
    "RECOMMEND_FEED",
    "enforce",
]
