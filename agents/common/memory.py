# -*- coding: utf-8 -*-
"""长短期记忆读写封装（`docs/agent-memory-design.md`、表 `agent_memory_short`）。

两套记忆，职责不同，**不能互相替代**
------------------------------------
| | 短期（工作记忆） | 长期（持久摘要） |
|---|---|---|
| 载体 | Redis `mem:short:{uid}:{sid}` | MySQL `agent_memory_short` |
| TTL | 30 min（对话 30min 无交互即遗忘） | 90 天（`RETENTION_DAYS`） |
| 内容 | 最近 N 轮逐字对话 + 本轮曝光过的番 id | **脱敏后的摘要**，不是逐字 |
| 用途 | A7 多轮对话的上下文 | session 过期后复盘、A7 训练数据积累 |
| 失败语义 | 读不到 → 空历史（对话退化但不报错） | 写不进 → 只 warning |

⚠️ 三条纪律
-----------
1. **写入前必须脱敏**（`llm.scrub_pii`）。邮箱/手机号一旦落库就是合规问题，
   而 `agent_memory_short.content` 是明文列。
2. **短期记忆是有界队列**：超过 `max_turns` 从最旧开始丢。上下文长度爆掉
   会直接触发 `60503 CONTEXT_OVERFLOW`。
3. **不做向量化**。长期记忆是"给模型读的文字"，不是"给检索用的向量" ——
   本项目的检索走 `models/` 的序列模型，不要在这里引入第二套相似度。

为什么 `shown_anime_ids` 单独存
------------------------------
A7 需要"别推荐我上轮已经推过的" —— 这个信息从对话文本里抽不可靠
（LLM 可能写成"《XX》"也可能写"那部番"），所以曝光列表显式记录。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from agents.common.data import get_gateway
from agents.common.llm import scrub_pii
from agents.common.ports import get_runtime

__all__ = [
    "MEMORY_TTL_SEC",
    "MAX_TURNS",
    "SESSION_MEMORY_TTL_SEC",
    "Turn",
    "append_turn",
    "clear_short",
    "get_history",
    "get_shown_anime_ids",
    "memory_key",
    "persist_summary",
    "record_shown",
]

SESSION_MEMORY_TTL_SEC = 30 * 60        # 短期记忆 30min
MEMORY_TTL_SEC = SESSION_MEMORY_TTL_SEC
MAX_TURNS = 20                          # 短期记忆保留的轮数上限


def memory_key(user_id: int, session_id: str) -> str:
    return f"mem:short:{int(user_id)}:{session_id}"


@dataclass
class Turn:
    role: str                                   # user / assistant
    content: str
    turn_no: int = 0
    tokens: int = 0
    shown_anime_ids: list[int] = field(default_factory=list)
    extracted: dict = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "turn_no": self.turn_no,
                "tokens": self.tokens, "shown_anime_ids": list(self.shown_anime_ids),
                "extracted": dict(self.extracted), "at": self.at}

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(role=str(d.get("role", "user")),
                   content=str(d.get("content", "")),
                   turn_no=int(d.get("turn_no", 0)),
                   tokens=int(d.get("tokens", 0)),
                   shown_anime_ids=[int(x) for x in (d.get("shown_anime_ids") or [])],
                   extracted=dict(d.get("extracted") or {}),
                   at=float(d.get("at", time.time())))


async def get_history(user_id: int, session_id: str,
                      limit: int = 10) -> list[dict]:
    """取最近 `limit` 轮（返回 `[{"role","content"}, ...]` 直接可喂 LLM）。

    缓存不可用 → 返回空列表（对话退化为单轮，但**不报错**）。
    """
    cache = get_runtime().cache
    raw: Optional[Any] = await cache.get_json(memory_key(user_id, session_id))
    turns = [Turn.from_dict(t) for t in (raw or [])]
    return [{"role": t.role, "content": t.content} for t in turns[-limit:]]


async def get_shown_anime_ids(user_id: int, session_id: str) -> list[int]:
    """本会话已曝光过的番 id（去重）。A7 用它做"别再推同一部"。"""
    cache = get_runtime().cache
    raw: Optional[Any] = await cache.get_json(memory_key(user_id, session_id))
    seen: list[int] = []
    for t in (raw or []):
        for x in (t.get("shown_anime_ids") or []):
            i = int(x)
            if i not in seen:
                seen.append(i)
    return seen


async def append_turn(user_id: int, session_id: str, role: str, content: str, *,
                      tokens: int = 0, shown_anime_ids: Optional[Sequence[int]] = None,
                      extracted: Optional[dict] = None) -> None:
    """追加一轮。**内容先脱敏**，再写入有界队列（超出 `MAX_TURNS` 丢最旧）。"""
    cache = get_runtime().cache
    key = memory_key(user_id, session_id)
    raw: Optional[Any] = await cache.get_json(key)
    turns = [Turn.from_dict(t) for t in (raw or [])]

    turns.append(Turn(
        role=role, content=scrub_pii(str(content)),
        turn_no=(turns[-1].turn_no + 1) if turns else 1,
        tokens=int(tokens or 0),
        shown_anime_ids=[int(x) for x in (shown_anime_ids or [])],
        extracted=dict(extracted or {}),
    ))
    turns = turns[-MAX_TURNS:]
    await cache.set_json(key, [t.to_dict() for t in turns], ttl=SESSION_MEMORY_TTL_SEC)


async def record_shown(user_id: int, session_id: str,
                       anime_ids: Sequence[int]) -> None:
    """只记曝光（不产生新的对话轮）：把 id 挂到最近一条 assistant 轮上。

    没有 assistant 轮时，补一条**空内容的 assistant 轮**专门承载曝光列表 ——
    这样"别重复推荐"在只有工具调用没有文本的轮次里也成立。
    """
    if not anime_ids:
        return
    cache = get_runtime().cache
    key = memory_key(user_id, session_id)
    raw: Optional[Any] = await cache.get_json(key)
    turns = [Turn.from_dict(t) for t in (raw or [])]
    if turns and turns[-1].role == "assistant":
        for i in anime_ids:
            if int(i) not in turns[-1].shown_anime_ids:
                turns[-1].shown_anime_ids.append(int(i))
    else:
        turns.append(Turn(role="assistant", content="",
                          turn_no=(turns[-1].turn_no + 1) if turns else 1,
                          shown_anime_ids=[int(i) for i in anime_ids]))
    await cache.set_json(key, [t.to_dict() for t in turns[-MAX_TURNS:]],
                         ttl=SESSION_MEMORY_TTL_SEC)


async def clear_short(user_id: int, session_id: str) -> int:
    cache = get_runtime().cache
    return int(await cache.delete(memory_key(user_id, session_id)) or 0)


# ---------------------------------------------------------------- 长期（持久摘要）
def persist_summary(user_id: int, session_id: str, turn_no: int, role: str,
                    content: str, *, tokens: int = 0,
                    shown_anime_ids: Optional[Sequence[int]] = None,
                    extracted: Optional[dict] = None) -> bool:
    """把一轮摘要写进 `agent_memory_short`。**先脱敏**。

    与短期记忆不同，这里**同步**写 —— 因为它不在主链路的返回路径上，
    调用方（A7 的收尾、或离线任务）可以接受一次 DB 写入。
    网关未注入 → 返回 False，不抛。
    """
    gw = get_gateway()
    if gw is None:
        return False
    try:
        gw.append_memory_turn(
            user_id=user_id, session_id=session_id, turn_no=int(turn_no),
            role=role, content=scrub_pii(str(content)), tokens=int(tokens or 0),
            shown_anime_ids=list(shown_anime_ids or []),
            extracted=dict(extracted or {}),
        )
        return True
    except Exception:       # 摘要落库失败不影响对话
        import logging
        logging.getLogger(__name__).warning("会话摘要落库失败（已忽略）", exc_info=True)
        return False
