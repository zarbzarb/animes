# -*- coding: utf-8 -*-
"""A7 对话推荐子包。

⚠️ `tools` 必须在 `agent` **之前**导入：工具靠模块级 `@tool` 装饰器注册到
`agents/common/tools.py` 的全局表里，`ToolRunner` / `openai_tools` 只会看到
已经 import 过的工具。若顺序反了，运行期表现为"LLM 要求调用 search_anime，
而 ToolRunner 说该工具未注册"—— 报错点与真实原因（少了一次 import）无关，很难查。
"""

from agents.chat import tools  # noqa: F401  （注册工具，必须早于 agent）
from agents.chat.agent import ChatAgent  # noqa: F401

__all__ = ["ChatAgent"]
