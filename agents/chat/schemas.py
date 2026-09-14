# -*- coding: utf-8 -*-
"""A7 输入输出契约（Pydantic v2）。"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 为什么 `session_id` 要容错 `None`（2026-09-14 实测踩到）
# ------------------------------------------------------
# `session_id` 声明为非可选 `str = ""`，**省略**该字段没问题；但
# `server/services/agent_bridge.py::chat_once` 会**显式**传
# `"session_id": session_id`，而它拿到的是 API 层的 `str | None` ——
# 于是 `None` 覆盖了默认值，Pydantic 抛 ValidationError，
# `/chat/message` 对"不带 session_id 的客户端"直接 **500**。
# 冒烟脚本因为显式传了 `""` 而一直没暴露这个洞。
#
# 语义上这不算错误：契约写明"空则由服务端生成"，`None` 与 `""` 一样是"空"。
# 所以在**输入契约这一个点**容错，而不是要求每个调用方都记得写 `or ""`。


class ChatInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # `user_id=0` 表示**匿名对话**：允许闲聊，但没有画像与个性化召回。
    # 刻意用 0 而不是 None —— 与 `recommend_result.user_id` 的 `>0` 约束区分开，
    # 免得把"未登录"写成一条 user_id=0 的假记录。
    user_id: int = Field(0, ge=0, description="0 = 匿名")
    session_id: str = Field("", max_length=64, description="会话 id，空则由服务端生成")
    message: str = Field("", max_length=2000)
    history: Optional[list[dict]] = Field(
        None, description="客户端自带历史（[{role,content}]）；为空则读服务端短期记忆")
    max_tool_calls: Optional[int] = Field(None, ge=0, le=5,
                                          description="覆盖默认的 3 次上限")
    top_k: Optional[int] = Field(None, ge=1, le=20)
    style: Literal["casual", "concise"] = "casual"

    @field_validator("session_id", mode="before")
    @classmethod
    def _session_not_none(cls, v):
        return "" if v is None else v


class RecommendCard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    src_anime_id: Optional[int] = None
    anime_id: Optional[int] = None
    title: str = ""
    year: Optional[int] = None
    score: Optional[float] = None
    reason: Optional[str] = None


class ChatOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str = ""
    reply: str = ""
    recommend_cards: list[RecommendCard] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list,
                                  description="本轮实际执行过的工具名（按序）")
    used_fallback: bool = False
    degraded_reason: Optional[str] = None
    tokens_used: int = 0
    # 事实校验拦下的书名号番名（非空说明 LLM 编了库外作品，已整段替换）
    violations: list[str] = Field(default_factory=list)


class ToolCallInput(BaseModel):
    """`chat.tool_call`：让前端/测试**显式**触发一次工具（无 LLM）。"""

    model_config = ConfigDict(extra="ignore")

    user_id: int = Field(0, ge=0)
    session_id: str = ""
    tool: str = Field(..., min_length=1)
    arguments: dict = Field(default_factory=dict)

    @field_validator("session_id", mode="before")
    @classmethod
    def _session_not_none(cls, v):
        # 同 `ChatInput` 的理由：`/internal/agents/A7` 调试口也能收到 null
        return "" if v is None else v


class ToolCallOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tool: str = ""
    result: dict = Field(default_factory=dict)
    elapsed_ms: int = 0
    error: Optional[str] = None
