# -*- coding: utf-8 -*-
"""A7 对话推荐配置。

与 `agents/common/settings.py` 的分工：那边是**环境级**配置（LLM 端点、超时表），
这里是**该 Agent 的行为参数**（工具次数、上下文轮数、卡片数）。
两者都不读 `.env` 之外的来源，也不反向依赖 `server/`。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ChatConfig"]


@dataclass
class ChatConfig:
    # ---- 协议 §A7 硬约束 ----
    # "工具调用次数上限 3 次/轮（防止无限循环）"
    max_tool_calls: int = 3
    # "history[]（最近 10 轮）"
    history_limit: int = 10

    # ---- 产出规模 ----
    cards_top_k: int = 5
    # 工具一次性取多少候选（要大于 cards_top_k：多取一些才有重排空间）
    candidates_per_tool: int = 20

    # ---- 超时预算（毫秒）----
    # ⚠️ LLM 超时必须**远小于**信封预算：一轮最多 4 次 LLM 调用
    #（3 次工具往返 + 1 次收束），若单次给到信封预算量级，
    # 第一次调用超时就把整个 A7 拖死，base 的 wait_for 会直接掐掉。
    llm_timeout_ms: int = 6000
    finalize_timeout_ms: int = 3000
    tool_timeout_ms: int = 1500
    # 自己留的保险：可用预算 = 信封预算 − slack。
    # 这样 A7 总能在被 `BaseAgent.handle` 强制超时**之前**返回降级话术，
    # 而不是让前端拿到一个 AGENT_TIMEOUT 错误。
    budget_slack_ms: int = 400
    # 预算低于这个值就不再发起新的 LLM/工具调用，直接收束
    min_budget_ms: int = 500

    # ---- 文本约束 ----
    max_message_len: int = 500
    # 注入 prompt 的"已曝光"列表长度（防重复推荐）
    shown_limit: int = 30

    # ---- LLM 采样 ----
    temperature: float = 0.7
    max_tokens: int = 700

    # ---- 事实校验 ----
    # 回复中出现的《番名》必须能在库里查到，否则整段换成模板话术
    verify_titles: bool = True

    # ---- 缓存 ----
    # 同一 (user, session, message) 的短时缓存，防抖重复点击
    cache_ttl: int = 120
