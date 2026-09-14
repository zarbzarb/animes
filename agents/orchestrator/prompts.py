# -*- coding: utf-8 -*-
"""A0 的意图识别提示词。

本文件只放**提示词常量**（`docs/agent-prompt-design.md` 的约定：
提示词与调用代码分离，改文案不碰逻辑）。调用与解析在 `intent.py`。
"""

from __future__ import annotations

__all__ = ["INTENT_PROMPT_VERSION"]

INTENT_PROMPT_VERSION = "a0.intent.v1"
