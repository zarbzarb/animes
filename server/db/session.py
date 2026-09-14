# -*- coding: utf-8 -*-
"""数据库会话、引擎与建表/初始化。

三种运行形态都要能跑起来，这是本文件的主要复杂度来源：

| 场景 | 库 | 约束 |
|---|---|---|
| 单元测试 | SQLite 内存 | 无外部依赖、**不分区**、不需要 MySQL 特有语法 |
| 本机答辩/开发 | 本机 MySQL 8 | 完整 DDL（分区、FULLTEXT、utf8mb4） |
| 生产 | 独立 MySQL | 同上 + 连接池参数 |

因此：**ORM 层只用跨库的列类型**，MySQL 专属特性（`PARTITION BY`、
`FULLTEXT KEY`）在 `init_db()` 里用原生 SQL 补 —— 而不是塞进 ORM 让测试也依赖 MySQL。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from server.core.config import settings
from server.db.base import Base

# ⚠️ 这一行**不是**可有可无的：`Base.metadata` 里有哪些表，
# 完全取决于"模型类有没有被 import 过"。少 import 一个模块 → 建表少一张，
# 而且**不报错**（只是那张表不存在，等运行时才炸）。
# 本文件刚写完时就踩过一次：init_db 里 seed 数据才发现 genre 表没建出来。
from server.db import models as _models  # noqa: F401  (导入以注册元数据)

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def make_engine(url: Optional[str] = None, *, echo: bool = False) -> Engine:
    url = url or settings.database_url
    kwargs: dict = {"echo": echo, "future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # 内存库必须允许跨线程（FastAPI 的 TestClient 与异步端点不在同一线程）
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            from sqlalchemy.pool import StaticPool
            kwargs["poolclass"] = StaticPool
    else:
        kwargs.update(pool_size=int(settings.MYSQL_POOL_SIZE), max_overflow=10,
                      pool_recycle=3600)
    engine = create_engine(url, **kwargs)

    if url.startswith("sqlite"):
        # SQLite 默认不校验外键，测试里"删父表留下孤儿行"就不会报错 ——
        # 与本项目 MySQL 侧行为不一致，必须打开
        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _rec):  # pragma: no cover
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autoflush=False, autocommit=False,
            expire_on_commit=False, future=True)
    return _SessionLocal


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """脚本/任务用的上下文管理器（自动 commit / rollback / close）。"""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Generator[Session, None, None]:
    """FastAPI 依赖注入用（见 `server/api/deps.py`）。"""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def reset_engine(url: Optional[str] = None, **kw) -> Engine:
    """测试用：换库并清空单例。"""
    global _engine, _SessionLocal
    _engine, _SessionLocal = None, None
    _engine = make_engine(url, **kw)
    return _engine


# ---------------------------------------------------------------- 建表

# MySQL 专属补充：ORM 表达不了（或表达出来会污染 SQLite 测试）的部分。
# 缺了它们不会导致数据错，但会：分区缺失 → agent_trace 无限膨胀；
# FULLTEXT 缺失 → A7 的 search_anime 退化成 LIKE 全表扫。
#
# ⚠️ 分区必须先改主键：MySQL 规定「分区键必须是主键的一部分」，而 ORM 侧
# `agent_trace` 用自增 `id` 单列主键（SQLite 不支持复合主键 + 分区）。
# 所以这里先 `DROP PRIMARY KEY` 再建 `(id, created_at)` 复合主键 ——
# `id` 仍是 AUTO_INCREMENT，MySQL 只要求它位于某个索引的首列，复合主键满足。
# 实测：不先改主键直接分区会报 1503
#   "A PRIMARY KEY must include all columns in the table's partitioning function"。
_MYSQL_AGENT_TRACE_PARTITION = (
    "ALTER TABLE `agent_trace` DROP PRIMARY KEY, "
    "ADD PRIMARY KEY (`id`, `created_at`)",
    """ALTER TABLE `agent_trace` PARTITION BY RANGE COLUMNS(`created_at`) (
         PARTITION p202609 VALUES LESS THAN ('2026-10-01'),
         PARTITION p202610 VALUES LESS THAN ('2026-11-01'),
         PARTITION p202611 VALUES LESS THAN ('2026-12-01'),
         PARTITION pmax VALUES LESS THAN (MAXVALUE))""",
)

# 全文索引：A7 的 search_anime 走 MATCH ... AGAINST（否则退化成 LIKE %kw%）
_MYSQL_ANIME_FULLTEXT = (
    "ALTER TABLE `anime` ADD FULLTEXT KEY `ft_title` (`title`, `alt_title`)",
)

# 毫秒精度：trace 的耗时分析需要亚秒分辨率
_MYSQL_TRACE_PRECISION = (
    "ALTER TABLE `agent_trace` MODIFY COLUMN `created_at` DATETIME(3) NOT NULL "
    "DEFAULT CURRENT_TIMESTAMP(3)",
)

# 兼容旧引用（历史上是一个扁平元组）
_MYSQL_POST_DDL = (_MYSQL_AGENT_TRACE_PARTITION[1], _MYSQL_ANIME_FULLTEXT[0],
                   _MYSQL_TRACE_PRECISION[0])

# MySQL 建表时希望使用 InnoDB + utf8mb4（否则日文标题与 emoji 会乱码）。
# 连接串里已带 `?charset=utf8mb4`，库级别的默认字符集由 `scripts/init_db.py`
# 的 `CREATE DATABASE ... CHARACTER SET utf8mb4` 保证 —— 不在这里塞 per-table kwargs，
# 那会让 SQLite 测试也带上 MySQL 方言选项。


def init_db(*, engine: Optional[Engine] = None, with_seed: bool = True,
            apply_mysql_extras: bool = True) -> Engine:
    """建表（幂等）+ 初始化基础数据。**不删除任何已有表**。"""
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    if not engine.url.drivername.startswith("sqlite") and apply_mysql_extras:
        _apply_mysql_extras(engine)

    if with_seed:
        with session_scope() as session:
            seed_genres(session)
            seed_agents(session)
    return engine


def _apply_mysql_extras(engine: Engine) -> None:
    """补充 ORM 表达不了的 MySQL 特性。**已经存在就跳过**（幂等）。"""
    probes = {
        "agent_trace": ("SELECT COUNT(*) FROM information_schema.partitions "
                        "WHERE table_name='agent_trace' AND partition_name IS NOT NULL"),
        "anime": ("SELECT COUNT(*) FROM information_schema.statistics "
                  "WHERE table_name='anime' AND index_name='ft_title'"),
    }
    with engine.begin() as conn:
        existing = {}
        for table, probe in probes.items():
            try:
                existing[table] = int(conn.execute(text(probe)).scalar() or 0) > 0
            except Exception as exc:  # pragma: no cover
                logger.warning("探测 MySQL 特性失败（跳过）：%s", exc)
                existing[table] = True

        if existing.get("agent_trace"):
            logger.info("agent_trace 已有分区，跳过")
        else:
            for ddl in _MYSQL_AGENT_TRACE_PARTITION:
                try:
                    conn.execute(text(ddl))
                except Exception as exc:  # pragma: no cover
                    logger.warning(
                        "agent_trace 分区未建立（不影响功能，但表会持续膨胀）：%s\n"
                        "  修复：手工执行 ALTER TABLE agent_trace "
                        "DROP PRIMARY KEY, ADD PRIMARY KEY (id, created_at); "
                        "再 ALTER TABLE agent_trace PARTITION BY RANGE COLUMNS(created_at) ...",
                        exc)
                    break
            else:
                logger.info("已为 agent_trace 建立按月分区")

        if existing.get("anime"):
            logger.info("anime 已有 ft_title 全文索引，跳过")
        else:
            for ddl in _MYSQL_ANIME_FULLTEXT:
                try:
                    conn.execute(text(ddl))
                    logger.info("已为 anime.title/alt_title 建立全文索引")
                except Exception as exc:  # pragma: no cover
                    logger.warning("建全文索引失败：%s", exc)

        for ddl in _MYSQL_TRACE_PRECISION:
            try:
                conn.execute(text(ddl))
            except Exception as exc:  # pragma: no cover
                logger.warning("agent_trace.created_at 毫秒精度设置失败：%s", exc)


# ---------------------------------------------------------------- 初始化数据

# 12 类题材（database-design.md §3.3 的表），id 与 configs/genre_taxonomy.yaml 对齐
GENRES: tuple[tuple[int, str, str, list, str], ...] = (
    (1, "热血战斗", "Action", ["Action"], "高强度对抗与成长"),
    (2, "冒险奇幻", "Adventure-Fantasy", ["Adventure", "Fantasy"], "世界观探索"),
    (3, "日常治愈", "Slice-of-Life", ["Slice of Life", "Gourmet"], "生活流"),
    (4, "恋爱校园", "Romance", ["Romance"], "情感线为主"),
    (5, "悬疑推理", "Mystery", ["Mystery", "Suspense"], "解谜与反转"),
    (6, "科幻机战", "Sci-Fi", ["Sci-Fi"], "科技设定"),
    (7, "喜剧搞笑", "Comedy", ["Comedy"], "以笑点驱动"),
    (8, "运动竞技", "Sports", ["Sports"], "竞技成长"),
    (9, "超自然灵异", "Supernatural", ["Supernatural", "Horror"], "灵异设定"),
    (10, "剧情文艺", "Drama", ["Drama", "Award Winning", "Avant Garde"], "作者性表达"),
    (11, "青春音乐", "Music-Idol", [], "音乐/偶像（由细分标签判定）"),
    (12, "后宫福利", "Ecchi-Harem", ["Ecchi"], "多角关系"),
)

# 10 个 Agent（database-design.md §4.1 的「初始化 10 行」）
AGENTS: tuple[tuple[str, str], ...] = (
    ("A0", "orchestrator"), ("A1", "profile"), ("A2", "recall"),
    ("A3", "coldstart"), ("A4", "fusion"), ("A5", "explain"),
    ("A6", "drift"), ("A7", "chat"), ("A8", "dataops"), ("A9", "eval"),
)


def seed_genres(session: Session) -> int:
    from server.db.models.anime import Genre
    if session.query(Genre).count():
        return 0
    for gid, cn, en, mal, desc in GENRES:
        session.add(Genre(id=gid, name_cn=cn, name_en=en, mal_genres=list(mal),
                          description=desc, sort_order=gid))
    session.flush()
    logger.info("已初始化 %d 类题材", len(GENRES))
    return len(GENRES)


def seed_agents(session: Session) -> int:
    from server.db.models.agent_state import AgentState
    if session.query(AgentState).count():
        return 0
    for aid, name in AGENTS:
        session.add(AgentState(agent_id=aid, agent_name=name, status=1,
                               health_score=1.0))
    session.flush()
    logger.info("已初始化 %d 个 Agent 状态行", len(AGENTS))
    return len(AGENTS)


def drop_all(engine: Optional[Engine] = None) -> None:
    """⚠️ 仅供测试与本地重置使用。**不要**在任何在线代码路径调用。"""
    Base.metadata.drop_all(engine or get_engine())


__all__ = [
    "AGENTS",
    "GENRES",
    "drop_all",
    "get_engine",
    "get_session",
    "get_session_factory",
    "init_db",
    "make_engine",
    "reset_engine",
    "session_scope",
]
