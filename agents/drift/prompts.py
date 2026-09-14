# -*- coding: utf-8 -*-
"""A6 的 LLM 解读提示词。

边界（`docs/agents.md` §A6）：**漂移判定由统计方法定，LLM 仅解读**。
所以提示词里给 LLM 的输入是"已经算好的漂移点与分布"，任务是措辞。
提示词明确禁止 LLM 自行判断"是否漂移"—— 否则统计阈值就形同虚设。
"""

from __future__ import annotations

__all__ = ["PROMPT_VERSION", "SYSTEM_PROMPT", "build_messages"]

PROMPT_VERSION = "a6.interpret.v1"

SYSTEM_PROMPT = """你是动漫推荐系统的兴趣变化解读助手。
系统已经用统计方法算出了用户的题材分布变化，你的任务只是**把它说成人话**。

硬性要求：
1. 不得质疑或重新判断"是否发生漂移" —— 结论已经由统计给出。
2. 只能使用输入中出现的题材名称，不得新增题材。
3. 不要出现用户名、ID、邮箱、手机号。
4. 1~2 句中文，不超过 50 个汉字，不要表情、不要引号、不要换行。
5. 只输出解读本身，不要任何前缀。

句式参考：「你在 2025Q2 后从日常治愈转向悬疑推理，口味明显变重了」
"""


def build_messages(points: list[dict], radar_labels: list[str],
                   radar: list[float], style: str = "concise") -> list[dict]:
    def nm(gid) -> str:
        try:
            i = int(gid) - 1
            return radar_labels[i] if 0 <= i < len(radar_labels) else f"题材{gid}"
        except (TypeError, ValueError):
            return "未知"

    lines = []
    for p in points[-2:]:
        lines.append(f"- {p.get('prev_period')} → {p.get('period')}："
                     f"从{nm(p.get('from_genre_id'))}转向{nm(p.get('to_genre_id'))}"
                     f"（JS 差异 {p.get('js')}）")
    top = sorted(zip(radar_labels, radar), key=lambda kv: -kv[1])[:3]
    style_hint = {"concise": "一句话概括变化。",
                  "detailed": "两句，先说变化再说现在集中在哪。",
                  "casual": "口语化，像朋友点评。"}.get(style, "一句话概括变化。")

    user = ("已检测到的漂移点：\n" + ("\n".join(lines) or "- 无（分布稳定）") +
            "\n\n近期的题材分布（按强度降序前 3）：\n" +
            "\n".join(f"- {n}：{round(v * 100, 1)}%" for n, v in top if v > 0) +
            f"\n\n风格要求：{style_hint}")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]
