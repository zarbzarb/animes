# -*- coding: utf-8 -*-
"""ORM 模型集合（与 `docs/database-design.md` 的表一一对应）。

**为什么集中 import**：Alembic 的 autogenerate 与 `Base.metadata.create_all()`
都依赖"模型类已经被 import 过"。少 import 一个模块，建表就少一张，
而且**不会报错**（只是那张表不存在，等到运行时才炸）。

对应关系（表 → 文件）：

| 文件 | 表 |
|---|---|
| `user.py` | user |
| `anime.py` | anime / genre / anime_genre / anime_content |
| `record.py` | watch_record |
| `profile.py` | user_profile |
| `capsule.py` | user_interest_capsule |
| `recommend.py` | recommend_result / recommend_explain / cold_start_pool / user_feedback |
| `agent_state.py` | agent_state / agent_trace |
| `memory.py` | agent_memory_short |
| `metric.py` | metric_snapshot / system_config / data_quality_log |
"""

from server.db.models.agent_state import AgentState, AgentTrace
from server.db.models.anime import Anime, AnimeContent, AnimeGenre, Genre
from server.db.models.capsule import UserInterestCapsule
from server.db.models.memory import SessionMemory
from server.db.models.metric import DataQualityLog, MetricSnapshot, SystemConfig
from server.db.models.profile import UserProfile
from server.db.models.recommend import (
    ColdStartPool, RecommendExplain, RecommendResult, UserFeedback,
)
from server.db.models.record import WatchRecord
from server.db.models.user import User

__all__ = [
    "AgentState",
    "AgentTrace",
    "Anime",
    "AnimeContent",
    "AnimeGenre",
    "ColdStartPool",
    "DataQualityLog",
    "Genre",
    "MetricSnapshot",
    "RecommendExplain",
    "RecommendResult",
    "SessionMemory",
    "SystemConfig",
    "User",
    "UserFeedback",
    "UserInterestCapsule",
    "UserProfile",
    "WatchRecord",
]
