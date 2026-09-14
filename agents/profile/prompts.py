# -*- coding: utf-8 -*-
"""A1 画像摘要的提示词（`docs/agent-prompt-design.md`）。

定位：LLM 在这里**只做措辞**。题材分布、活跃度、完看率都是统计算出来的，
LLM 拿到的输入就是这些数字，任务是"说人话"，不是"下结论"。

三条写死的约束
-------------
1. **不得引入输入之外的题材名**。LLM 看到的是 `热血战斗/悬疑推理`，
   它不能写成"你喜欢机甲"（哪怕是它自己推断的）。这条靠 `validate_summary` 兜。
2. **不得出现用户标识**（昵称、uid）。摘要会展示在公共看板上。
3. 长度 ≤ 40 字。超过就截断 —— 前端卡片只放一行。
"""

from __future__ import annotations

import re

from agents.common.llm import scrub_pii

__all__ = ["PROMPT_VERSION", "SYSTEM_PROMPT", "build_messages", "fallback_summary",
           "validate_summary"]

PROMPT_VERSION = "a1.summary.v1"

SYSTEM_PROMPT = """你是动漫推荐系统的用户画像文案助手。
根据给定的**统计结果**，用一句中文概括这位用户的看番口味。

硬性要求：
1. 只使用输入中出现的题材名称，不得新增或改写题材名。
2. 不要出现用户名、用户 ID、邮箱、手机号等任何标识信息。
3. 长度不超过 40 个汉字，不要表情符号，不要引号，不要句末括号补充。
4. 句式参考：「偏好{题材A}与{题材B}的{活跃度}用户」。
5. 只输出这一句话本身，不要任何解释或前缀。
"""

_STYLE_HINT = {
    "concise": "简洁，一句话即可。",
    "detailed": "可以带一句观看强度的说明，但仍不超过 40 字。",
    "casual": "语气轻松口语化，像朋友点评，不超过 40 字。",
}


def build_messages(top_genres: list[dict], activity_label: str,
                   preferred_types: list[str], style: str = "concise") -> list[dict]:
    genres = "、".join(str(g.get("genre", "")) for g in top_genres[:3] if g.get("genre"))
    local = {"low": "轻度", "medium": "中度", "high": "重度"}.get(activity_label, "中度")
    types = "/".join(preferred_types) if preferred_types else "无偏好"
    user = (f"题材（按兴趣强度排序，最强在前）：{genres or '暂无'}\n"
            f"活跃度：{local}追番用户\n"
            f"观看类型偏好：{types}\n"
            f"风格要求：{_STYLE_HINT.get(style, _STYLE_HINT['concise'])}")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def fallback_summary(top_genres: list[dict], activity_label: str,
                     preferred_types: list[str]) -> str:
    """**模板兜底**：LLM 不可用/超时/输出不合格时使用。

    协议 §5.2 要求 A1 失败后仍返回可用画像，所以摘要必须有"无 LLM 版本"。
    这个模板与 LLM 版本风格一致，用户分不出哪条是降级产物 —— 这正是降级的目标。
    """
    local = {"low": "轻度", "medium": "中度", "high": "重度"}.get(activity_label, "中度")
    names = [str(g.get("genre", "")) for g in top_genres[:2] if g.get("genre")]
    if not names:
        return f"{local}追番用户，口味尚在形成中"
    if len(names) == 1:
        return f"偏好{names[0]}的{local}追番用户"
    return f"偏好{names[0]}与{names[1]}的{local}追番用户"


_BAD_CHARS = re.compile(r"[<>{}`\n\r]|用户\s*\d+|@|=|\bid\b", re.IGNORECASE)


def validate_summary(text: str, allowed_genres: list[str]) -> tuple[bool, str]:
    """校验 LLM 输出。返回 `(是否通过, 清洗后的文本)`。

    只有**明确违规**才判不通过（引入库外题材、含标识信息、含控制字符）。
    宁可放过也不要误判 —— 误判会让降级率虚高，论文里的"LLM 可用率"就不真了。
    """
    if not text:
        return False, ""
    cleaned = scrub_pii(text).strip().strip('"').strip("“”")
    cleaned = cleaned.split("\n")[0].strip()
    if not cleaned:
        return False, ""
    if len(cleaned) > 60:                       # 40 字要求 + 宽容度
        cleaned = cleaned[:40]
    if _BAD_CHARS.search(cleaned):
        return False, cleaned

    # 库外题材检测：输出里出现的「题材 + 番/类型」字样若不在白名单里 → 可疑
    if allowed_genres:
        for token in re.findall(r"[\u4e00-\u9fff]{2,6}", cleaned):
            if token.endswith("番") or token in ("机甲", "后宫", "异世界", "推理",
                                                "战斗", "日常", "恋爱", "悬疑"):
                if not any(token in g or g in token for g in allowed_genres):
                    return False, cleaned
    return True, cleaned
