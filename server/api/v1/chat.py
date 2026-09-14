# -*- coding: utf-8 -*-
"""对话推荐（api-specification.md §3.4）。

`/chat/message` 按协议是 **SSE 流式**。实现说明（重要，别当成 bug）
---------------------------------------------------------------
协议给的帧类型是 `tool_call` / `tool_result` / `chunk` / `cards` / `done`。
本实现是**「先算完、再按帧回放」**，不是 token 级真流式，原因：
A7 的对话循环里「调工具 → 看结果 → 再决定调不调下一个工具」是**串行依赖**的，
只有最后一步的文案生成本可以流。
把它做成真流式，收益是把首字延迟从 ~2.4s 降到 ~0.6s，代价是
A7 的循环逻辑要拆成生成器（当前 600+ 行的收敛/校验逻辑全部要改），
而前端在这一步的体验差别只是"文字逐字出现"。
**取舍：先把帧协议跑通**（前端能正确渲染工具提示与卡片），
真流式作为后续优化 —— 那时只改 `_frames()` 的数据来源，帧协议不变。

所以前端**现在就能正确开发**：帧格式与协议逐字段一致。
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from server.core.response import make_router, new_trace_id
from server.deps import current_user, get_gw, rate_limit
from server.services.agent_bridge import chat_once

logger = logging.getLogger(__name__)
router = make_router(prefix="/chat", tags=["对话"])


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


@router.post("/session", summary="新建会话")
async def new_session(user: dict = Depends(current_user)) -> dict:
    import uuid

    sid = f"sess_{uuid.uuid4().hex[:8]}"
    return {"session_id": sid, "expires_in": 1800}


@router.get("/history", summary="会话历史")
async def history(session_id: str | None = Query(None),
                  limit: int = Query(20, ge=1, le=100),
                  user: dict = Depends(current_user)) -> dict:
    """读 `agent_memory_short`（A7 的短期记忆，按轮次追加）。"""
    gw = get_gw()
    uid = int(user["id"])
    sid = session_id or gw.latest_session_id(uid)
    if not sid:
        return {"session_id": None, "turns": []}
    rows = gw.get_memory_turns(uid, sid, limit=limit) or []
    return {"session_id": sid,
            "turns": [{"turn_no": r.get("turn_no"), "role": r.get("role"),
                       "content": r.get("content"), "created_at": r.get("created_at")}
                      for r in rows]}


@router.post("/message", summary="对话推荐（SSE 流式）",
             dependencies=[Depends(rate_limit("chat_message"))])
async def message(body: ChatIn, user: dict = Depends(current_user)):
    uid = int(user["id"])
    trace_id = new_trace_id()
    payload, meta = await chat_once(uid, body.message, session_id=body.session_id)
    return StreamingResponse(
        _frames(payload, meta, trace_id, uid),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "X-Trace-Id": trace_id},
    )


async def _frames(payload: dict, meta: dict, trace_id: str, uid: int):
    """按协议 §3.4 的帧序列回放。

    帧之间**不加 sleep 伪造打字机效果** —— 前端拿到 `chunk` 自己按字数节流更好，
    服务端 sleep 会白占一个连接（20 次/分钟的限额下很容易被吃满）。
    """
    def _sse(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    # ① 工具调用/结果：A7 的 `tool_calls` 里已含 name 与结果计数
    for call in (payload.get("tool_calls") or []):
        name = call.get("tool") or call.get("name") or "unknown"
        args = call.get("arguments") or call.get("args") or {}
        yield _sse({"type": "tool_call", "tool": name, "args": args})
        if call.get("count") is not None or call.get("error"):
            yield _sse({"type": "tool_result", "tool": name,
                        "count": call.get("count"),
                        "error": call.get("error")})

    # ② 文本
    reply = payload.get("reply") or ""
    if reply:
        yield _sse({"type": "chunk", "content": reply})

    # ③ 卡片
    cards = payload.get("recommend_cards") or []
    if cards:
        yield _sse({"type": "cards", "cards": cards})

    # ④ 降级/违规信息（前端可以显示"回答可能不完整"）
    if payload.get("used_fallback"):
        yield _sse({"type": "notice", "level": "degraded",
                    "message": "回答由兜底策略生成",
                    "reason": payload.get("degraded_reason")})
    if payload.get("violations"):
        yield _sse({"type": "notice", "level": "warning",
                    "message": "部分内容未通过事实校验",
                    "items": payload["violations"]})

    # ⑤ 结束
    yield _sse({"type": "done", "session_id": payload.get("session_id"),
                "tool_calls": len(payload.get("tool_calls") or []),
                "tokens_used": int(payload.get("tokens_used") or 0),
                "elapsed_ms": int(meta.get("elapsed_ms") or 0),
                "trace_id": trace_id, "user_id": uid})
