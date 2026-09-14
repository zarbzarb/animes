# -*- coding: utf-8 -*-
"""缓存与限流：**Redis 不可用时必须降级，而不是报错**。

设计依据
--------
`docs/architecture.md` 的两条硬要求：
1. 「可降级：任一 Agent 失败不得导致整条链路 500」；
2. 「缓存前置是 P95 < 300ms 的主要手段」。

所以这一层的语义是：**有 Redis 就用 Redis，没有 Redis 就用进程内 TTL 字典**。
两种情况接口完全一致，调用方（A4 融合排序 Agent / 推荐 service）不需要知道
背后是什么。`code=50203 CACHE_UNAVAILABLE` 的处理策略也是「绕过缓存直算」，
不是返回错误。

⚠️ 进程内字典在 uvicorn 多 worker 下**不共享**，只适合开发与单机答辩环境；
生产必须起 Redis（拓扑见 `docs/architecture.md` 第七节）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from server.core.config import settings

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 0.5      # 秒。Redis 探测必须快，否则拖慢首个请求


class MemoryCache:
    """进程内 TTL 字典（降级用）。"""

    backend = "memory"

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at and expires_at < time.time():
                self._data.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: str, ttl: int = 0) -> None:
        async with self._lock:
            self._data[key] = ((time.time() + ttl) if ttl else 0.0, value)
            # 防止长期运行把内存吃满：超过 10 万键就清掉已过期的
            if len(self._data) > 100_000:
                now = time.time()
                self._data = {k: v for k, v in self._data.items()
                              if not v[0] or v[0] > now}

    async def delete(self, *keys: str) -> int:
        async with self._lock:
            return sum(1 for k in keys if self._data.pop(k, None) is not None)

    async def delete_prefix(self, prefix: str) -> int:
        async with self._lock:
            hit = [k for k in self._data if k.startswith(prefix)]
            return sum(1 for k in hit if self._data.pop(k, None) is not None)

    async def incr(self, key: str, ttl: int) -> int:
        """计数并设定过期（限流用）。返回自增后的值。"""
        async with self._lock:
            item = self._data.get(key)
            now = time.time()
            if item is None or (item[0] and item[0] < now):
                self._data[key] = (now + ttl, "1")
                return 1
            expires_at, value = item
            nxt = int(value) + 1
            self._data[key] = (expires_at, str(nxt))
            return nxt

    async def close(self) -> None:
        self._data.clear()

    async def ping(self) -> bool:
        return True


class RedisCache:
    """`redis.asyncio` 客户端包装。所有方法都可能抛异常，由 `Cache` 兜住。"""

    backend = "redis"

    def __init__(self, url: str) -> None:  # pragma: no cover - 需要真实 Redis
        from redis import asyncio as aioredis  # 延迟导入
        self._client = aioredis.from_url(
            url, encoding="utf-8", decode_responses=True,
            socket_connect_timeout=_CONNECT_TIMEOUT,
            socket_timeout=_CONNECT_TIMEOUT,
        )

    async def get(self, key: str) -> Optional[str]:  # pragma: no cover
        return await self._client.get(key)

    async def set(self, key: str, value: str, ttl: int = 0) -> None:  # pragma: no cover
        if ttl:
            await self._client.set(key, value, ex=ttl)
        else:
            await self._client.set(key, value)

    async def delete(self, *keys: str) -> int:  # pragma: no cover
        return int(await self._client.delete(*keys))

    async def delete_prefix(self, prefix: str) -> int:  # pragma: no cover
        """按前缀删（`SCAN` + 分批 `DEL`）。

        ⚠️ **不要用 `KEYS pattern`**：它是 O(N) 且**阻塞整个 Redis 实例**，
        在几十万键的库上会卡住所有请求，是生产事故的经典来源。
        `SCAN` 游标式遍历，每次只返回少量键，不阻塞。
        """
        n, batch = 0, []
        async for k in self._client.scan_iter(match=prefix + "*", count=200):
            batch.append(k)
            if len(batch) >= 200:
                n += int(await self._client.delete(*batch))
                batch.clear()
        if batch:
            n += int(await self._client.delete(*batch))
        return n

    async def incr(self, key: str, ttl: int) -> int:  # pragma: no cover
        pipe = self._client.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl)
        result = await pipe.execute()
        return int(result[0])

    async def close(self) -> None:  # pragma: no cover
        await self._client.aclose()

    async def ping(self) -> bool:  # pragma: no cover
        return bool(await self._client.ping())


class Cache:
    """门面：自动选择后端 + 全部操作 `try/except` 降级。

    降级策略见协议 §4.2：`50203 CACHE_UNAVAILABLE` → **绕过缓存直算**。
    因此这里读失败返回 `None`、写失败静默忽略，绝不把异常抛给主链路。
    """

    def __init__(self) -> None:
        self._backend: Any = MemoryCache()
        self._probed = False
        self._degraded_reason: Optional[str] = None

    async def _ensure_backend(self) -> None:
        if self._probed:
            return
        self._probed = True
        if not settings.CACHE_ENABLED:
            self._degraded_reason = "CACHE_ENABLED=false"
            return
        try:
            candidate = RedisCache(settings.redis_url)
            ok = await asyncio.wait_for(candidate.ping(), timeout=_CONNECT_TIMEOUT * 3)
            if ok:
                self._backend = candidate
                logger.info("缓存后端：Redis（%s）", settings.redis_url)
                return
            self._degraded_reason = "redis ping 返回假"
        except Exception as exc:
            self._degraded_reason = f"{type(exc).__name__}: {exc}"
        logger.warning("Redis 不可用（%s），缓存降级为进程内 TTL 字典："
                       "多 worker 下不共享，生产环境请起 Redis", self._degraded_reason)

    @property
    def backend_name(self) -> str:
        return self._backend.backend

    @property
    def degraded_reason(self) -> Optional[str]:
        return self._degraded_reason

    async def get_json(self, key: str) -> Optional[Any]:
        await self._ensure_backend()
        try:
            raw = await self._backend.get(key)
        except Exception as exc:
            logger.warning("缓存读失败（忽略，直算）：%s", exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            logger.warning("缓存值不是合法 JSON，已丢弃：%s", key)
            return None

    async def set_json(self, key: str, value: Any, ttl: int = 0) -> bool:
        await self._ensure_backend()
        try:
            await self._backend.set(key, json.dumps(value, ensure_ascii=False),
                                    ttl or 0)
            return True
        except Exception as exc:
            logger.warning("缓存写失败（忽略）：%s", exc)
            return False

    async def delete(self, *keys: str) -> int:
        await self._ensure_backend()
        try:
            return await self._backend.delete(*keys)
        except Exception as exc:
            logger.warning("缓存删除失败（忽略）：%s", exc)
            return 0

    async def delete_prefix(self, prefix: str) -> int:
        """按前缀整片删。返回删掉的键数（后端不可用时 0）。

        为什么需要它（2026-09-14 修的真实 bug）
        ------------------------------------
        推荐结果缓存不是"一个用户一个键"，而是 `rec:{uid}:{scene}[:{gid}]`
        —— 同一个用户在**综合 / 分题材 / 新番 / 对话**四个场景各有一份，
        分题材还要再按 12 个题材各一份。行为发生变化（新增追番、清空记忆）时，
        只知道 `user_id`，不可能枚举出所有 scene/genre 组合。
        早期代码用 `delete(rec_key(uid))` 即 `delete("rec:1")` 去删，
        而真实键是 `rec:1:0` —— **一个都删不中，且不报任何错**，
        表现为"用户加了追番，推荐 24 小时内毫无变化"（TTL 就是 24h）。
        """
        await self._ensure_backend()
        try:
            return await self._backend.delete_prefix(prefix)
        except Exception as exc:
            logger.warning("缓存按前缀删除失败（忽略）：%s", exc)
            return 0

    async def incr(self, key: str, ttl: int) -> Optional[int]:
        """限流计数。缓存不可用时返回 `None` —— 调用方应**放行**而不是拦截。"""
        await self._ensure_backend()
        try:
            return await self._backend.incr(key, ttl)
        except Exception as exc:
            logger.warning("缓存计数失败（限流放行）：%s", exc)
            return None

    async def close(self) -> None:
        try:
            await self._backend.close()
        except Exception:  # pragma: no cover
            pass

    async def health(self) -> dict:
        await self._ensure_backend()
        return {
            "backend": self.backend_name,
            "ok": True,
            "degraded": self.backend_name != "redis",
            "degraded_reason": self._degraded_reason,
        }


_cache: Optional[Cache] = None


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache


# --------------------------------------------------------------- Key 规范
# 与 docs/architecture.md §5.3 的表一致，禁止在别处手写字符串拼接。
# 例外：`agents/` 侧为避免反向依赖会自己拼同样的串（A1 的 `profile:{uid}`、
# A4 的 `rec:` 系列）；那些串由 `tests/test_agents/test_cache_keys.py` 锁住，
# 两侧不一致时测试直接红 —— 靠"注释里写一句同构"是锁不住的（本文件原来的
# `rec_key` 就与 A4 实际写入的键**不一致**，而且没有任何测试发现）。
def rec_scope_prefix(user_id: int) -> str:
    """某个用户的**全部推荐结果缓存键**的公共前缀 —— 失效操作的唯一入口。

    A4 写入的是 `rec:{uid}:{scene}` / `rec:{uid}:{scene}:{gid}`
    （见 `agents/fusion/adapter.py::cache_key`）。失效必须按前缀整片清，
    因为调用方只知道 `user_id`，枚举不出所有 scene/genre 组合。
    详见 `Cache.delete_prefix` 的 docstring（那里记录了踩过的坑）。
    """
    return f"rec:{int(user_id)}:"


def rec_new_key(user_id: int) -> str:
    return f"rec:new:{user_id}"


def profile_key(user_id: int) -> str:
    return f"profile:{user_id}"


def short_memory_key(user_id: int, session_id: str) -> str:
    return f"mem:short:{user_id}:{session_id}"


def rate_key(scope: str, ident: str) -> str:
    return f"rate:{scope}:{ident}"


BEHAVIOR_STREAM = "stream:user_behavior"


__all__ = [
    "BEHAVIOR_STREAM",
    "Cache",
    "MemoryCache",
    "RedisCache",
    "get_cache",
    "profile_key",
    "rate_key",
    "rec_new_key",
    "rec_scope_prefix",
    "short_memory_key",
]
