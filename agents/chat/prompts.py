# -*- coding: utf-8 -*-
"""A7 提示词（`docs/agent-prompt-design.md` 的分层约定：system / context / task）。

三条纪律写进 system，不是"礼貌提示"而是**硬约束**：
1. **只能谈数据里存在的作品** —— 编造番名是这类系统最典型的事故，
   而且用户很难察觉（听起来很像真的）。所以除了 prompt 约束，
   还有 `strategy.verify_reply_titles` 做机械校验兜底。
2. **不做排序** —— 排序由 A4 给，LLM 只负责"把已定好的列表说人话"（ADR-1）。
3. **不承诺、不臆造用户偏好** —— 只能引用注入的画像事实。
"""

from __future__ import annotations

from typing import Sequence

__all__ = [
    "FINALIZE_PROMPT",
    "SYSTEM_PROMPT",
    "build_messages",
    "fallback_reply",
    "no_llm_reply",
]

SYSTEM_PROMPT = """你是 AniRec 的动漫推荐助手，用中文和用户聊番剧。

# 你拥有的工具
你可以调用工具来查库、找相似、拿推荐、生成理由。**工具返回的数据才是事实**：
- search_anime(keyword)                  按关键词查番剧
- recall_by_genre(genres, top_k)         按题材推荐（题材用中文名）
- get_similar(anime_id, top_k)           找与某部番相似的番
- get_recommendation(user_id, top_k)     拿该用户的个性化推荐
- explain_recommendation(anime_id)       生成某部番的推荐理由

# 铁律（违反即为事故）
1. **只能推荐工具返回过的作品。** 绝对不要凭记忆写任何番名 ——
   你没见过的作品一律不许出现在回复里。
2. **不要自己做排序。** 工具/排序模块给出的顺序就是最终顺序，你只解释、不复排。
3. **不要编造用户的观看历史或偏好。** 只能用我给的事实。
4. **工具调用最多 3 次。** 信息够了就立刻作答，不要反复查。
5. 用户问与动漫无关的事：礼貌说明你只负责动漫推荐，并把话题引回推荐。

# 回复风格
- 口语化、自然，像懂番的朋友，不要罗列参数和分数。
- 2~4 句。先直给答案，再给一句理由。
- 提到作品时用《作品名》格式。
"""

FINALIZE_PROMPT = (
    "工具调用次数已用完。请**立刻**基于上面已经拿到的数据作答，"
    "不要再请求调用工具。如果数据不足，就直说还缺少什么，并给出一句引导建议。"
)

_NO_LLM_GUIDE = (
    "我这边暂时连不上智能对话服务，没法跟你聊着挑番。"
    "你可以先去「推荐」页看看综合榜单，或者告诉我想要什么题材，我按题材帮你挑。"
)


def no_llm_reply(titles: Sequence[str]) -> str:
    """LLM 不可用时的最小话术。

    有真实候选卡片时**要说清是榜单结果**，不要假装是个性化推荐 ——
    这是"降级要可感知"的体现（协议 §6：降级必须写进 `meta.degraded`，
    前端也会展示降级提示；文案口径要一致）。

    ⚠️ 措辞区分「对话服务」与「推荐模型」：实际不可用的是 LLM（A7 的对话与
    工具调用），而**召回排序链路本身是好的** —— 这些卡片正是它算出来的。
    早期文案写成"推荐模型没连上"，用户会以为整个推荐挂了，与事实相反。
    """
    if not titles:
        return _NO_LLM_GUIDE
    head = "、".join(f"《{t}》" for t in list(titles)[:3])
    return (f"智能对话暂时不可用，先按热度给你几部：{head}。"
            "想看更贴你口味的，可以去「推荐」页看看综合榜单。")


def fallback_reply(reason: str = "") -> str:
    """整链兜底（工具与 LLM 都不可用）。`reason` 只写日志，不写进回复文案。"""
    return _NO_LLM_GUIDE


def build_messages(*, message: str, history: Sequence[dict] = (),
                   profile_hint: str = "", shown_titles: Sequence[str] = (),
                   style: str = "casual") -> list[dict]:
    """拼 messages：system → （事实上下文）→ history → user。

    为什么把画像/已曝光放在 **system 之后的第二条 user 消息**，而不是塞进 system：
    system 是缓存的稳定前缀（部分服务端会做 prompt caching），
    每轮都变的用户事实放进去会让缓存全部失效。
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    facts: list[str] = []
    if profile_hint:
        facts.append(f"用户画像事实：{profile_hint}")
    if shown_titles:
        listed = "、".join(f"《{t}》" for t in list(shown_titles)[:8])
        facts.append(f"本会话已经推过这些，不要重复推：{listed}")
    if style == "concise":
        facts.append("回复请简短，1~2 句。")
    if facts:
        messages.append({"role": "system", "content": "\n".join(facts)})

    for h in history:
        role = str(h.get("role") or "user")
        if role not in ("user", "assistant"):
            continue
        content = str(h.get("content") or "").strip()
        if content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})
    return messages
