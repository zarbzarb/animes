# -*- coding: utf-8 -*-
"""Agent 运行时的**注入点**：数据库会话、缓存、时钟。

为什么用「注入」而不是直接 `from server.db.session import session_scope`
------------------------------------------------------------------------
依赖方向 **`server/ → agents/ → models/`** 是单向的，`scripts/check_imports.py`
会静态拦下反向 import。而 Agent 确实需要读 `watch_record`、写 `agent_trace`
（`docs/architecture.md` §5.2 明确列了"由谁写/由谁读"）。

解法是**端口-适配器**：Agent 只声明"我需要一个能给会话的作用域"，
具体是不是 SQLAlchemy、是不是 MySQL，由应用层在启动时注入：

    # server/main.py 启动时
    from agents.common.ports import bind_runtime, Runtime
    bind_runtime(Runtime(session_scope=session_scope, cache=get_cache()))

好处有三个，都不是理论上的：
1. **CI 依赖检查通过**，且不是靠 `# noqa` 绕过去的；
2. **测试可以注入内存 SQLite**，不必起 MySQL；
3. 将来把某个 Agent 拆成独立服务（协议 M1 的"生产可切 HTTP/gRPC"），
   只需换一个 Runtime 实现，Agent 代码一行不改。

未注入时的行为：**能降级就降级**（返回空数据 + `degraded`），不能降级才抛
`DB_UNAVAILABLE` —— 而不是在 import 期就炸掉。
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Generator, Optional

from agents.common.errors import AgentError, make

logger = logging.getLogger(__name__)


class _NullSession:
    """未注入 DB 时的占位：任何操作都失败，但**失败得体面**。"""

    def __getattr__(self, item: str):
        raise make("DB_UNAVAILABLE", f"未注入数据库会话，无法执行 {item}",
                   hint="server 启动时请调用 agents.common.ports.bind_runtime")


@contextmanager
def _null_session_scope() -> Generator[_NullSession, None, None]:
    yield _NullSession()


class _NoCache:
    """未注入缓存时的空实现：读永远 miss、写忽略（等价于"绕过缓存直算"）。

    ⚠️ 方法集必须与 `server/core/cache.py::Cache` 的**公开方法**保持同步。
    漏一个方法不会有任何导入期报错，只在真用到它的那条路径上炸
    `AttributeError` —— 而这条路径（未注入缓存）恰恰是降级路径，
    出问题的时候最难看懂。
    """

    backend = "none"

    async def get_json(self, key: str) -> Optional[Any]:
        return None

    async def set_json(self, key: str, value: Any, ttl: int = 0) -> bool:
        return False

    async def delete(self, *keys: str) -> int:
        return 0

    async def delete_prefix(self, prefix: str) -> int:
        """与 `Cache.delete_prefix` 对应（缓存失效按前缀清）。"""
        return 0

    async def incr(self, key: str, ttl: int) -> Optional[int]:
        return None


@dataclass
class Runtime:
    """应用层注入给 Agent 层的全部外部能力。"""

    session_scope: Callable[[], ContextManager[Any]] = _null_session_scope
    cache: Any = field(default_factory=_NoCache)
    # 业务数据访问（`agents/common/data.py::DataGateway` 的实例，
    # 实现见 `server/db/gateway.py`）。**Agent 读业务表一律走它**，不 import ORM。
    gateway: Any = None
    # 供 A2 的 adapter 取模型实例（由 registry 提供）
    model_registry: Optional[Any] = None
    clock: Callable[[], float] = time.perf_counter
    # 调用链落库出口。**为什么是函数而不是让 tracing 自己 import ORM**：
    # `agent_trace` 的 ORM 类住在 `server/`，而 `agents/` 不允许 import `server/`
    # （`scripts/check_imports.py` 静态拦截，否则"Agent 可拆成独立服务"这条退路就没了）。
    # 于是把"怎么写这张表"作为能力注入进来：server 启动时传一个闭包。
    trace_sink: Optional[Callable[[dict], bool]] = None
    # 一些形态标记，供 /admin/agents/health 展示
    db_ready: bool = False
    cache_backend: str = "none"

    @contextmanager
    def db(self) -> Generator[Any, None, None]:
        """`with get_runtime().db() as session:` —— 事务边界由注入者决定。"""
        with self.session_scope() as session:  # type: ignore[misc]
            yield session


_default_runtime = Runtime()


def bind_runtime(runtime: Runtime) -> None:
    """应用层启动时调用（或在测试的 fixture 里）。

    顺带把 `runtime.gateway` 绑定到 `agents.common.data` 的全局端口 ——
    这样应用层只需注入一次，Agent 侧用 `require_gateway()` 就地取用。
    """
    global _default_runtime
    _default_runtime = runtime
    if runtime.gateway is not None:
        from agents.common.data import bind_gateway
        bind_gateway(runtime.gateway)
    logger.info("Agent 运行时已注入：db_ready=%s cache=%s",
                runtime.db_ready, runtime.cache_backend)


def get_runtime() -> Runtime:
    return _default_runtime


def reset_runtime() -> None:
    """测试用：回到"未注入"状态。"""
    global _default_runtime
    _default_runtime = Runtime()
    from agents.common.data import reset_gateway
    reset_gateway()


def require_db(runtime: Optional[Runtime] = None) -> Runtime:
    """需要 DB 的 Agent 开头调一次，明确失败原因（而不是让 None 在深层炸掉）。"""
    rt = runtime or get_runtime()
    if not rt.db_ready:
        raise AgentError(code=make("DB_UNAVAILABLE").code,
                         message="数据库不可用：运行时未注入会话",
                         detail={"hint": "agents.common.ports.bind_runtime"})
    return rt


__all__ = [
    "Runtime",
    "bind_runtime",
    "get_runtime",
    "require_db",
    "reset_runtime",
]
