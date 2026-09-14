# -*- coding: utf-8 -*-
"""A7 输入契约的单测。

存在的理由是一类**只有真实客户端才会触发**的洞：schema 声明得没错，
但调用方多传了一个字段、或者把"可选"写成了显式 `None`，
于是默认值被覆盖、Pydantic 直接抛 ValidationError，接口 500。
单元测试里默认都是 `Model()` / `Model(**完整字段)`，永远碰不到。

实测案例：`server/services/agent_bridge.py::chat_once` 传
`"session_id": session_id`（值来自 API 层的 `str | None`），
客户端不带 `session_id` 时 `/chat/message` 返回 500；
而冒烟脚本因为显式传了 `""` 一直没暴露。
"""

import pytest

from agents.chat.schemas import ChatInput, ToolCallInput


class TestChatInputSessionId:
    """`session_id` 的"空"有三种写法：省略 / `""` / `None`，都必须接受。"""

    def test_omitted(self):
        assert ChatInput(message="hi").session_id == ""

    def test_empty_string(self):
        assert ChatInput(message="hi", session_id="").session_id == ""

    def test_explicit_none_is_normalized(self):
        """回归：`None` 曾被当成非法值，导致 /chat/message 500。"""
        assert ChatInput(message="hi", session_id=None).session_id == ""

    def test_real_value_passes_through(self):
        assert ChatInput(session_id="sess_abc").session_id == "sess_abc"

    def test_too_long_is_rejected(self):
        """上限 64 仍然生效 —— 容错 `None` 不等于放宽长度约束。"""
        with pytest.raises(Exception):
            ChatInput(session_id="x" * 65)

    def test_non_string_still_rejected(self):
        """容错只针对"空"，`123` 这种类型错误必须照样报错。"""
        with pytest.raises(Exception):
            ChatInput(session_id=123)


class TestChatInputDefaults:
    def test_anonymous_default(self):
        """`user_id=0` = 匿名（不是 `None`，避免写出 user_id=0 的假记录）。"""
        assert ChatInput().user_id == 0

    def test_negative_user_id_rejected(self):
        with pytest.raises(Exception):
            ChatInput(user_id=-1)

    def test_extra_fields_ignored(self):
        """`extra="ignore"` —— API 层多传字段不该让整条链路 500。"""
        m = ChatInput(user_id=1, message="hi", nonsense_field="x", scene=3)
        assert m.user_id == 1

    def test_message_length_cap(self):
        with pytest.raises(Exception):
            ChatInput(message="x" * 2001)

    def test_style_enum(self):
        assert ChatInput(style="concise").style == "concise"
        with pytest.raises(Exception):
            ChatInput(style="verbose")


class TestToolCallInput:
    def test_session_id_none_is_normalized(self):
        m = ToolCallInput(tool="search_anime", session_id=None)
        assert m.session_id == ""

    def test_tool_is_required(self):
        with pytest.raises(Exception):
            ToolCallInput()

    def test_arguments_default_to_empty_dict(self):
        assert ToolCallInput(tool="t").arguments == {}

    def test_empty_tool_rejected(self):
        with pytest.raises(Exception):
            ToolCallInput(tool="")
