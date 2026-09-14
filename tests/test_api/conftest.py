# -*- coding: utf-8 -*-
"""`tests/test_api/` 的夹具：在**真应用**上跑 HTTP 层测试。

三个设计决策，每个都有非它不可的理由
------------------------------------

**1. 用真 SQLite 库，不用 mock 网关。**

`SqlGateway` 有 60 个方法，手写 fake 等于把网关的实现抄一遍 —— 抄错了测试
照样绿，而且测的是"我理解的网关"而不是网关。`server/db/session.py` 本来就
为测试态留好了通路（`reset_engine` / `init_db` / `settings.is_test` /
`is_sqlite`），且 `SqlGateway` 全程只走 `session_scope()` —— 所以只要换掉
`_engine` 单例，44 条路由的数据访问就全部落到临时内存库上。

**2. 用真 `server.main.app`，不用另搭一个 `FastAPI()`。**

自行装配的测试应用会漏掉 CORS、trace 中间件、异常处理器、统一响应体路由类、
静态挂载 —— 而 `test_envelope.py` 要验的恰恰就是这些。测试应用与线上应用
一旦分叉，"测试全绿但线上没有 trace_id" 这种事就会永久隐身。

**3. 不跑 lifespan（不预热模型）。**

`TestClient(app)` 只有在 `with` 块里才执行 lifespan；直接调 `.get()` 不会触发。
于是模型预热（~2.3s、依赖 checkpoint 与 GPU）被跳过，测试不需要 GPU 也能跑。
代价是 Agent 运行时未装配，需要模型的接口会走降级路径 —— 这不是缺陷，
正好顺带把降级行为也测了。
"""

from __future__ import annotations

import functools

import pytest
from fastapi.testclient import TestClient

from server.core.config import settings
from server.core.security import create_access_token, hash_password

#: 测试账号统一密码。跑 bcrypt 一次约 0.1~0.3s，跨用例复用同一个哈希串。
TEST_PASSWORD = "anirec_test_2026"

#: (src_anime_id, 标题, 年份, 评分, 题材 id, 交互数)
#: 前三部是普通番；9003 的交互数低于 `COLD_START_THRESHOLD=10`，
#: 用来区分"新番/冷启"分支（`_anime_dict` 的 `is_cold` 由阈值算出）。
ANIME_SEED: tuple[tuple[int, str, int, float, list[int], int], ...] = (
    (9001, "测试番·热血机战", 2024, 8.2, [1, 6], 120),
    (9002, "测试番·日常治愈", 2023, 7.6, [3], 80),
    (9003, "测试番·悬疑新作", 2026, 0.0, [5], 2),
)


@functools.lru_cache(maxsize=1)
def _password_hash() -> str:
    return hash_password(TEST_PASSWORD)


# ---------------------------------------------------------------- 库


@pytest.fixture()
def api_engine(monkeypatch):
    """一块干净的 SQLite 内存库（含题材/Agent 种子行）+ 强制内存缓存。

    ⚠️ 缓存开关必须放在这里（所有用例的公共祖先），而不是单独的夹具：
    不挂它的话，用例里第一次 `get_cache()` 会去探真实 Redis ——
    本机 Redis 没起时每次都要吃一个连接超时。关掉 `CACHE_ENABLED` 走的是
    `Cache._ensure_backend` 的文档化降级路径，直接停在 MemoryCache。

    ⚠️ 必须用 `monkeypatch.setattr` 改 `_engine` / `_SessionLocal` 而不是调
    `reset_engine()`：monkeypatch 会在用例结束后**自动还原原值**，
    而 `reset_engine()` 会把单例永久留在 sqlite 上，污染后续用例。
    """
    from server.core import cache as cache_mod
    from server.db import session as db_session

    monkeypatch.setattr(settings, "CACHE_ENABLED", False)
    # 每例一份全新空缓存：否则限流计数（LOGIN 10 次/60s）会跨用例累积，
    # 且推荐缓存键会从上一个用例泄漏过来 —— "单独跑绿、全量跑红"的偶发失败。
    monkeypatch.setattr(cache_mod, "_cache", None)

    engine = db_session.make_engine("sqlite:///:memory:")
    monkeypatch.setattr(db_session, "_engine", engine)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    # with_seed：12 类题材 + 10 行 Agent 状态。少了题材，`/animes?genre_id=` 会查空
    db_session.init_db(engine=engine, with_seed=True)
    return engine


@pytest.fixture()
def gw(api_engine, monkeypatch):
    """网关单例，指向上面那块内存库。"""
    from server import deps

    monkeypatch.setattr(deps, "_gateway", None)
    return deps.get_gw()


@pytest.fixture()
def seeded(gw):
    """两个账号（普通 / 管理员）+ 三部番。"""
    u = gw.create_user(username="demo", password_hash=_password_hash(),
                       nickname="演示用户")
    a = gw.create_user(username="root", password_hash=_password_hash(),
                       nickname="管理员", role=1)
    for src, title, year, score, gids, n in ANIME_SEED:
        gw.upsert_anime({"src_anime_id": src, "title": title, "year": year,
                         "score": score, "genre_ids": gids, "type": "TV",
                         "episodes": 12, "n_interactions": n,
                         "is_online": 1})
    return {"user": u, "admin": a}


# ---------------------------------------------------------------- 缓存


@pytest.fixture()
def cache(api_engine):
    """缓存对象（已在 `api_engine` 里强制为空 MemoryCache）。

    给缓存失效类测试用：可以直接 `set_json` 预置键，再断言被业务动作清掉。
    """
    from server.core.cache import get_cache

    return get_cache()


# ---------------------------------------------------------------- 客户端


@pytest.fixture()
def password() -> str:
    """`seeded` 用户的明文密码。经 fixture 注入而不是让测试文件
    重复声明常量 —— 两处声明总有一天会漂移。"""
    return TEST_PASSWORD


@pytest.fixture()
def client(api_engine):
    """真应用 + 不跑 lifespan。见模块 docstring 第 2、3 点。"""
    from server.main import app

    return TestClient(app)


@pytest.fixture()
def user_token(seeded):
    return create_access_token(int(seeded["user"]["id"]), role="user")


@pytest.fixture()
def admin_token(seeded):
    return create_access_token(int(seeded["admin"]["id"]), role="admin")


@pytest.fixture()
def auth(user_token):
    return {"Authorization": f"Bearer {user_token}"}


@pytest.fixture()
def admin_auth(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}
