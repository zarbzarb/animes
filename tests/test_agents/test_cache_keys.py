# -*- coding: utf-8 -*-
"""缓存键的**跨层一致性**守卫（`agents/` ↔ `server/`）。

为什么必须有这个文件
------------------
`agents/` 不许反向依赖 `server/`（R3，见 `scripts/check_imports.py`），
所以两边**各自拼串**。拼串这件事没有任何编译器能帮你对齐 ——
而注释里写一句"与 server 侧同构"是锁不住的：本文件就是为这个洞而生的。

被它抓到过的真实 bug（2026-09-14）
--------------------------------
`server/core/cache.py::rec_key(uid)` 返回 `rec:{uid}`，
而 A4 实际写入的是 `rec:{uid}:{scene}`（分题材还带 `:{gid}`）。
失效逻辑拿前者去 `delete` —— **一个真实键都删不中，且不报任何错**。
用户表现："刚加了追番，推荐 24 小时内毫无变化"（TTL 就是 24h）。
当时两条 docstring 都写着"两侧同构、有测试断言一致"，而那个测试**并不存在**。

所以这里断言的不是"两个函数长得一样"，而是**失效能生效的充分条件**：
A4 生成的每一个键，都必须落在 server 侧 `rec_scope_prefix` 的前缀之下。
"""

import asyncio

import pytest

from agents.common.memory import memory_key
from agents.fusion.adapter import cache_key as a4_cache_key
from agents.profile.agent import ProfileAgent
from server.core.cache import (
    MemoryCache,
    profile_key,
    rec_new_key,
    rec_scope_prefix,
    short_memory_key,
)

USERS = [1, 7, 12, 999]
SCENES = [0, 1, 2, 3]          # 综合 / 分题材 / 新番 / 对话
GENRES = [None, 1, 7, 12]


# =====================================================================
# 单键：两侧必须逐字符相同
# =====================================================================
class TestSingleKeyParity:
    @pytest.mark.parametrize("uid", USERS)
    def test_profile_key(self, uid):
        assert ProfileAgent._key(uid) == profile_key(uid) == f"profile:{uid}"

    @pytest.mark.parametrize("uid", USERS)
    def test_short_memory_key(self, uid):
        sid = f"sess_{uid}"
        assert memory_key(uid, sid) == short_memory_key(uid, sid)
        assert memory_key(uid, sid) == f"mem:short:{uid}:{sid}"

    def test_profile_key_is_int_normalized(self):
        """两侧都对入参做 `int()`，否则 `ProfileAgent._key("7")` 与
        `profile_key(7)` 会算出同一个键但类型不同，缓存命中判定就会飘。"""
        assert ProfileAgent._key("7") == profile_key(7)


# =====================================================================
# 推荐结果键：前缀覆盖关系（失效逻辑的正确性依赖它）
# =====================================================================
class TestRecKeyScope:
    @pytest.mark.parametrize("uid", USERS)
    @pytest.mark.parametrize("scene", SCENES)
    def test_scene_key_falls_under_scope(self, uid, scene):
        k = a4_cache_key(uid, scene)
        assert k == f"rec:{uid}:{scene}"
        assert k.startswith(rec_scope_prefix(uid)), (
            f"键 {k} 不在失效前缀 {rec_scope_prefix(uid)} 下 —— "
            "前缀删除会漏掉它，表现是刷新后推荐不变")

    @pytest.mark.parametrize("uid", USERS)
    @pytest.mark.parametrize("scene", SCENES)
    @pytest.mark.parametrize("gid", GENRES)
    def test_genre_key_falls_under_scope(self, uid, scene, gid):
        k = a4_cache_key(uid, scene, gid)
        expect = f"rec:{uid}:{scene}" if gid is None else f"rec:{uid}:{scene}:{gid}"
        assert k == expect
        assert k.startswith(rec_scope_prefix(uid))

    def test_scope_does_not_leak_across_users(self):
        """`rec:1:` 不能覆盖 `rec:12:...` —— 全靠前缀末尾那个冒号。

        这是最容易改坏的一处：有人把 `rec:{uid}` 写成不带尾冒号，
        uid=1 的失效就会顺手清掉 uid=12、uid=1xx 的缓存，
        症状是"别人刷新一次，我的推荐就重算了"（性能问题且极难定位）。
        """
        assert not a4_cache_key(12, 0).startswith(rec_scope_prefix(1))
        assert not a4_cache_key(123, 0).startswith(rec_scope_prefix(12))
        assert a4_cache_key(1, 0).startswith(rec_scope_prefix(1))

    def test_new_anime_key_is_not_under_scope(self):
        """`rec:new:{uid}`（新番专区）**刻意**不在失效前缀下。

        当前 `rec:new:` 没有写入方（A3 未实现该缓存），所以这条只是把
        语义锁住：哪天 A3 真的写它了，失效逻辑需要**显式**把它加进来，
        而不是指望前缀顺手覆盖（`rec:new:1` 不以 `rec:1:` 开头）。
        """
        assert not rec_new_key(1).startswith(rec_scope_prefix(1))

    def test_prefix_has_trailing_colon(self):
        assert rec_scope_prefix(42) == "rec:42:"


# =====================================================================
# delete_prefix 本身：MemoryCache（真实可跑，不依赖 Redis）
# =====================================================================
class TestDeletePrefix:
    def _cache_with(self, keys):
        c = MemoryCache()
        for k in keys:
            asyncio.run(c.set(k, "v", ttl=60))
        return c

    @staticmethod
    def _alive(cache, keys):
        """用公开 API 判断哪些键还活着 —— 不去戳 `_data` 内部结构。"""
        return {k for k in keys if asyncio.run(cache.get(k)) is not None}

    def test_deletes_exactly_the_scope(self):
        """这是修 bug 的那条断言的镜像：只清目标用户、且真的清到了。"""
        keys = []
        for uid in (1, 12):
            for scene in SCENES:
                keys.append(a4_cache_key(uid, scene))
                keys.append(a4_cache_key(uid, scene, 3))
        c = self._cache_with(keys)

        n = asyncio.run(c.delete_prefix(rec_scope_prefix(1)))
        assert n == len(SCENES) * 2, f"应删掉 uid=1 的全部 {len(SCENES)*2} 个键，实际 {n}"
        assert self._alive(c, keys) == {a4_cache_key(12, s, g)
                                        for s in SCENES for g in (None, 3)}, \
            "误删了 uid=12 的键"

    def test_exact_key_delete_would_miss(self):
        """把旧实现钉在耻辱柱上：精确删 `rec:{uid}` 一个键都删不掉。

        这条测试的价值是"如果哪天有人把失效逻辑改回精确键删除，它会红"。
        """
        keys = [a4_cache_key(1, s) for s in SCENES]
        c = self._cache_with(keys)
        assert asyncio.run(c.delete("rec:1")) == 0
        assert self._alive(c, keys) == set(keys), "不该删掉任何键"

    def test_delete_prefix_on_empty_is_zero(self):
        c = MemoryCache()
        assert asyncio.run(c.delete_prefix("rec:404:")) == 0

    def test_delete_prefix_does_not_touch_other_prefixes(self):
        keys = [profile_key(1), "mem:short:1:s", rec_new_key(1)]
        c = self._cache_with(keys)
        assert asyncio.run(c.delete_prefix(rec_scope_prefix(1))) == 0
        assert self._alive(c, keys) == set(keys)


# =====================================================================
# 缓存接口契约：Agent 会调用的方法，两个实现都得有
# =====================================================================
# 这是 Agent 层（`Runtime.cache`）实际会调用的方法集。
# 漏一个不会有任何导入期报错，只在真走到那条路径时抛 AttributeError ——
# 而"未注入缓存"正是**降级路径**，出错时最难看懂。
# 真实案例：新增 `delete_prefix` 后 `_NoCache` 没跟上。
AGENT_CACHE_API = ("get_json", "set_json", "delete", "delete_prefix", "incr")


class TestCacheInterfaceParity:
    @pytest.mark.parametrize("name", AGENT_CACHE_API)
    def test_no_cache_provides_method(self, name):
        from agents.common.ports import _NoCache
        assert callable(getattr(_NoCache(), name, None)), (
            f"`_NoCache` 缺少 {name} —— 未注入缓存时会 AttributeError")

    @pytest.mark.parametrize("name", AGENT_CACHE_API)
    def test_real_cache_provides_method(self, name):
        from server.core.cache import Cache
        assert callable(getattr(Cache(), name, None))

    def test_no_cache_is_async_and_harmless(self):
        """空实现必须**真的什么都不做**且不抛异常（降级路径不能二次失败）。"""
        from agents.common.ports import _NoCache
        c = _NoCache()
        assert asyncio.run(c.get_json("k")) is None
        assert asyncio.run(c.set_json("k", {"a": 1})) is False
        assert asyncio.run(c.delete("k")) == 0
        assert asyncio.run(c.delete_prefix("rec:1:")) == 0
        assert asyncio.run(c.incr("k", 60)) is None
