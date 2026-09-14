# -*- coding: utf-8 -*-
"""A0 输入输出契约（Pydantic v2）。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class OrchestratorInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    trace_id: str = Field("", description="由 Gateway 生成并透传")
    user_id: int = Field(0, ge=0, description="未登录时为 0（走热门兜底）")
    raw_query: Optional[str] = Field(None, description="用户原始自然语言（可选）")
    intent_hint: Optional[str] = Field(None, description="前端显式意图，最高优先")
    intent: Optional[str] = Field(None, description="内部：已识别的意图（聚合时必填）")
    params: dict = Field(default_factory=dict)
    context: dict = Field(default_factory=dict, description="会话历史等")
    batch_id: Optional[str] = None
    # 内部：已执行的流水线结果（`orchestrate.aggregate` 用）
    pipeline_values: dict = Field(default_factory=dict)


class AgentChainEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent: str
    action: str
    ok: bool = True
    elapsed_ms: int = 0
    degraded: bool = False


class OrchestratorOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: str
    intent_label: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    intent_method: str = Field("rule", description="hint / rule / llm / default")
    result: dict = Field(default_factory=dict, description="业务载荷")
    agent_chain: list[AgentChainEntry] = Field(default_factory=list)
    degraded: list[dict] = Field(default_factory=list)
    elapsed_ms: int = 0
    trace_id: str = ""
    lane: Optional[str] = Field(None, description="call_path 便于前端展示调用树")
