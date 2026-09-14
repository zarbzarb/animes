# -*- coding: utf-8 -*-
"""数据访问端口（DataGateway）—— Agent 与「业务表」之间的唯一契约。

问题
----
Agent 必须读 `watch_record`、写 `user_profile` / `recommend_result` / `agent_trace`
（`docs/architecture.md` §5.2 列了谁读谁写）。最直接的写法是

    from server.db.models.record import WatchRecord      # ← 不行

但依赖方向 `server/ → agents/ → models/` 是单向的，`scripts/check_imports.py`
会静态拦下它。而且这条约束有实质理由，不是洁癖：一旦 Agent 直接 import 应用层的
ORM，Agent 就绑死在「同一个进程 + SQLAlchemy + 这份表结构」上，
「同一套 Agent 也能拆成独立服务」（协议 §二 M1）这条退路就没有了。

解法：契约用 `Protocol` 声明，实现由应用层注入
--------------------------------------------
`typing.Protocol` 的关键好处 —— **它是纯静态的，运行时不 import 任何东西**：
本文件不认识 SQLAlchemy，也不认识 `server/`。`server/db/gateway.py` 的
`SqlGateway` 在结构上满足这个 Protocol（静态鸭子类型），于是

* `agents/` 的静态依赖检查通过，且不是靠 `# noqa` 绕过；
* 单元测试可以注入一个内存假实现（`tests/agents/fakes.py::FakeGateway`），
  不必起 MySQL；
* 将来换存储（比如换成 gRPC 服务）只改 `server/` 侧的一个类。

⚠️ 契约纪律
-----------
1. **返回普通 Python 类型**（`dict` / `list` / `float`），**不返回 ORM 对象**。
   返回 ORM 对象会让 Agent 隐式依赖会话生命周期（懒加载在 `session` 关闭后
   会抛 `DetachedInstanceError`），这类 bug 只在生产并发下暴露。
2. **方法粒度按「Agent 需要什么」定，不按「表有什么」定**。
   比如 A1 要的是"按题材聚合的观看统计"，不是"把整张表给我"。
3. 所有方法**不做打分排序**（那是 `models/` 的事），只做取数与落库。
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from agents.common.errors import AgentError, make

__all__ = [
    "DataGateway",
    "GatewayNotBound",
    "bind_gateway",
    "get_gateway",
    "require_gateway",
    "reset_gateway",
]


class GatewayNotBound(AgentError):
    """未注入数据网关。**可降级**：调用方应返回空数据，而不是 500。"""

    def __init__(self, detail_msg: str = "") -> None:
        super().__init__(code=make("DB_UNAVAILABLE").code,
                         message="数据网关未注入：" + (detail_msg or "未绑定"),
                         detail={"hint": "server 启动时调用 "
                                         "agents.common.data.bind_gateway(SqlGateway(...))"})


@runtime_checkable
class DataGateway(Protocol):
    """Agent 需要的全部数据能力。**实现见 `server/db/gateway.py::SqlGateway`**。

    实现者只需"有这些方法"，不需要继承本类（结构化子类型）。
    方法名分组：用户 / 动漫 / 行为 / 画像 / 胶囊 / 推荐 / 解释 / 指标 / 配置。
    """

    # ---------------- 用户 ----------------
    def get_user(self, user_id: int) -> Optional[dict]:
        """`{id, username, nickname, role, status, src_user_id}` 或 None。"""

    def get_user_by_username(self, username: str) -> Optional[dict]:
        """含 `password_hash`（仅供 server 鉴权用，Agent 不应使用）。"""

    def user_exists(self, user_id: int) -> bool: ...

    # ---------------- 动漫 ----------------
    def get_anime(self, anime_id: int) -> Optional[dict]:
        """单条动漫元数据（含 `genres` 题材 id 列表、`is_cold_start`）。"""

    def get_animes(self, anime_ids: Sequence[int]) -> dict[int, dict]:
        """批量取，**返回 dict 而不是 list**，调用方按 id 取用避免再排序。"""

    def search_anime(self, keyword: str, limit: int = 10) -> list[dict]:
        """标题/别名模糊匹配（MySQL 走 FULLTEXT，SQLite 走 LIKE）。"""

    def anime_genres(self, anime_ids: Sequence[int]) -> dict[int, list[int]]:
        """`{anime_id: [genre_id, ...]}`；不区分主次（主标签用 `get_anime`）。"""

    def genre_map(self) -> dict[int, str]:
        """`{genre_id: name_cn}`（12 类）。"""

    def list_anime(self, *, limit: int = 100, offset: int = 0,
                   genre_id: Optional[int] = None, min_year: Optional[int] = None,
                   cold_only: bool = False,
                   order_by: str = "n_interactions") -> list[dict]:
        """列表查询。`cold_only=True` 只取新番（A3 用）。"""

    def popular_anime_ids(self, limit: int = 200,
                          genre_id: Optional[int] = None) -> list[int]:
        """按 `n_interactions` 倒序的热门 id（**降级兜底**的唯一来源）。"""

    def similar_anime_ids(self, anime_id: int, top_k: int = 20) -> list[int]:
        """共现近邻。A3A7 的 `get_similar` 工具用；无预计算表时返回空。"""

    # ---------------- 行为（watch_record）----------------
    def get_watch_records(self, user_id: int, *, statuses: Optional[Sequence[int]] = None,
                          limit: Optional[int] = None,
                          order: str = "asc") -> list[dict]:
        """`[{anime_id, status, rating, progress, watched_at, updated_at}]`。

        `order="asc"` 按 `updated_at` **升序** —— 这是构造序列输入的口径
        （`docs/database-design.md` §3.6：线上按 `updated_at`，离线按 animeID 原始顺序，
        两者的对齐靠 `src_*_id`）。
        """

    def get_genre_affinity(self, user_id: int) -> list[dict]:
        """按题材聚合的观看统计（A1 的唯一数据源）：

        `[{genre_id, count, avg_rating, recent_count, weighted}]`，
        `weighted` 已按"近期观看权重更高"折算，A1 不再自己算权重。
        """

    def get_user_activity(self, user_id: int) -> dict:
        """`{total, done, watching, dropped, plan, last_active_at, first_active_at,
        n_recent_30d}`。一次查询取全，避免 A1 发 4 条 SQL。"""

    def user_genre_time_series(self, user_id: int, *, granularity: str = "quarter",
                               start: Optional[str] = None,
                               end: Optional[str] = None) -> list[dict]:
        """`[{period: "2025Q2", genre_id, count}]` —— A6 漂移分析的时间序列。"""

    # ---------------- 画像（user_profile）----------------
    def get_profile(self, user_id: int) -> Optional[dict]: ...

    def upsert_profile(self, user_id: int, data: dict) -> None:
        """存在则更新（`version` 自增），不存在则插入。"""

    # ---------------- 兴趣胶囊（user_interest_capsule）----------------
    def get_capsules(self, user_id: int) -> list[dict]:
        """`[{capsule_id, label, strength, vec_ref, model_ver}]`。"""

    def replace_capsules(self, user_id: int, capsules: Sequence[dict]) -> None: ...

    # ---------------- 推荐结果（recommend_result）----------------
    def save_recommendations(self, user_id: int, scene: int,
                             items: Sequence[dict], *, batch_id: str,
                             ttl_minutes: int = 1440) -> int:
        """**先删同 (user, scene, batch_id) 再插**，保证重跑幂等。返回写入行数。"""

    def get_recommendations(self, user_id: int, scene: int,
                            limit: int = 20) -> list[dict]: ...

    # ---------------- 解释（recommend_explain）----------------
    def save_explain(self, user_id: int, anime_id: int, data: dict) -> None: ...

    def get_explain(self, user_id: int, anime_id: int,
                    style: str = "concise") -> Optional[dict]: ...

    # ---------------- 冷启动池（cold_start_pool）----------------
    def list_cold_start_pool(self, *, limit: int = 100,
                             active_only: bool = True) -> list[dict]: ...

    def upsert_cold_start_pool(self, anime_id: int, data: dict) -> None: ...

    # ---------------- 反馈与指标 ----------------
    def feedback_counts(self, *, start: Optional[str] = None,
                        end: Optional[str] = None,
                        scene: Optional[int] = None) -> dict:
        """`{action: count}` —— A9 在线指标（CTR/CVR）的分子分母。"""

    def save_metrics(self, rows: Sequence[dict]) -> int:
        """写 `metric_snapshot`（`uk_date_scene_metric` 冲突则更新）。"""

    def query_metrics(self, *, metric_type: str, days: int = 30,
                      scene: Optional[int] = None) -> list[dict]: ...

    def save_data_quality(self, rows: Sequence[dict]) -> int: ...

    def get_config(self, key: str) -> Optional[Any]:
        """运行时配置（`system_config`，`is_hot_reload=1` 的项可被覆盖）。"""

    def set_config(self, key: str, value: Any, *, value_type: str = "string",
                   updated_by: Optional[int] = None) -> None: ...

    # ---------------- 会话记忆（agent_memory_short，持久摘要）----------------
    def append_memory_turn(self, user_id: int, session_id: str, turn_no: int,
                           role: str, content: str, *, tokens: int = 0,
                           shown_anime_ids: Optional[Sequence[int]] = None,
                           extracted: Optional[dict] = None) -> None:
        """落一条**已脱敏**的会话摘要（主记忆在 Redis，这里只做备份/复盘）。"""

    def get_memory_turns(self, user_id: int, session_id: str,
                         limit: int = 20) -> list[dict]: ...

    def latest_session_id(self, user_id: int) -> Optional[str]: ...

    # ---------------- Agent 状态（agent_state）----------------
    def list_agent_states(self) -> list[dict]: ...

    def upsert_agent_state(self, agent_id: str, data: dict) -> None: ...

    def trace_stats(self, *, since: Optional[str] = None) -> list[dict]:
        """A9 用：按 Agent 聚合 trace 的耗时/错误率。"""


# ---------------------------------------------------------------- 绑定与取用
_gateway: Optional[Any] = None


def bind_gateway(gateway: Any) -> None:
    """应用层启动时注入。测试注入假实现。"""
    global _gateway
    _gateway = gateway


def get_gateway() -> Optional[Any]:
    return _gateway


def require_gateway() -> Any:
    """需要数据库的 Agent 在 `invoke()` 开头调一次，失败信息明确。"""
    if _gateway is None:
        raise GatewayNotBound()
    return _gateway


def reset_gateway() -> None:
    """测试用。"""
    global _gateway
    _gateway = None
