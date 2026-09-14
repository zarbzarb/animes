# -*- coding: utf-8 -*-
"""A6 · 兴趣漂移分析智能体（`DriftAgent`）。

职责（`docs/agents.md` §A6）：支撑「兴趣漂移分析看板」。

三条边界
-------
* ❌ 不参与推荐排序
* ❌ 不修改画像（**只读**）
* ❌ 漂移判定由统计方法定，LLM 仅解读

降级（协议 §5.2：A6 2000ms、失败只返回数值、不返回解读）
-----------------------------------------------------
LLM 不可用 → 返回 `radar` / `trend` / `drift_points` 数值，
`interpretation` 用**模板**（而不是置空）：
看板上只有雷达图没有一句话，展示效果会明显打折，而模板文案零成本。
"""

from __future__ import annotations

import logging
from typing import Optional

from agents.common.base import BaseAgent
from agents.common.data import GatewayNotBound, get_gateway
from agents.common.envelope import Envelope
from agents.common.errors import AgentError, invalid_param
from agents.common.llm import get_llm
from agents.drift import prompts, strategy
from agents.drift.schemas import DriftInput

logger = logging.getLogger(__name__)

_TIMEOUT_MS = 1500
_MAX_REASON = 50


class DriftAgent(BaseAgent):
    agent_id = "A6"
    name = "drift"
    description = "兴趣漂移：季度分布 + JS 散度/CUSUM 变点 + LLM 解读"
    online = False          # P2 离线，不在主链路上

    async def invoke(self, payload: dict, env: Envelope) -> dict:
        action = env.header.action
        if action == "drift.interpret":
            return await self._interpret_only(payload, env)
        if action != "drift.analyze":
            raise invalid_param(f"未支持的 action：{action}")
        return await self._analyze(payload, env)

    async def _analyze(self, payload: dict, env: Envelope) -> dict:
        inp = DriftInput.model_validate(payload)
        gw = get_gateway()
        if gw is None:
            raise GatewayNotBound("A6 需要网关取时间序列")

        rows = gw.user_genre_time_series(
            inp.user_id, granularity=inp.granularity,
            start=inp.start, end=inp.end) or []
        genre_map = self._genre_map(gw)

        radar = strategy.radar(rows)
        points = strategy.detect_drift_points(rows,
                                             threshold=inp.drift_threshold)
        trend = strategy.trend_series(rows)
        labels = [genre_map.get(i, f"题材{i}") for i in range(1, strategy.N_GENRES + 1)]

        n_periods = len({str(r.get("period")) for r in rows if r.get("period")})
        template = strategy.interpretation_template(radar, points, genre_map)
        text, tokens, degraded = await self._interpret(
            points, labels, radar, template, inp.style, env)

        return {"user_id": inp.user_id, "granularity": inp.granularity,
                "radar": radar, "radar_labels": labels, "trend": trend,
                "drift_points": points, "interpretation": text,
                "is_stable": not points, "n_periods": n_periods,
                "degraded": degraded, "tokens_used": tokens}

    async def _interpret_only(self, payload: dict, env: Envelope) -> dict:
        """只做解读（数值已由调用方算好）。"""
        points = list((payload or {}).get("drift_points") or [])
        labels = list((payload or {}).get("radar_labels") or [])
        radar = [float(x) for x in ((payload or {}).get("radar") or [])]
        style = str((payload or {}).get("style") or "concise")
        template = strategy.interpretation_template(radar, points, {
            i + 1: n for i, n in enumerate(labels)})
        text, tokens, degraded = await self._interpret(
            points, labels, radar, template, style, env)
        return {"interpretation": text, "degraded": degraded,
                "tokens_used": tokens,
                "prompt_ver": prompts.PROMPT_VERSION}

    # ------------------------------------------------------------ LLM
    async def _interpret(self, points, labels, radar, template, style: str,
                         env: Envelope) -> tuple[str, int, bool]:
        llm = get_llm()
        if not llm.available:
            return template, 0, True
        if not points and sum(radar) <= 0:
            return template, 0, True        # 无数据可解读
        res = await llm.complete(
            prompts.build_messages(points, labels, radar, style),
            temperature=0.4, max_tokens=100, timeout_ms=_TIMEOUT_MS)
        if not res.ok:
            return template, 0, True
        text = self._clean(res.text)
        if not text:
            return template, res.tokens_used, True
        return text, res.tokens_used, False

    @staticmethod
    def _clean(text: str) -> str:
        from agents.common.llm import scrub_pii
        out = scrub_pii(str(text)).strip().split("\n")[0].strip().strip('"“”')
        for bad in ("解读：", "理由：", "解读:", "理由:"):
            if out.startswith(bad):
                out = out[len(bad):].strip()
        if len(out) > _MAX_REASON:
            out = out[:_MAX_REASON].rstrip("，、, ") + "。"
        if out and not out.endswith("。"):
            out += "。"
        return out

    @staticmethod
    def _genre_map(gw) -> dict[int, str]:
        try:
            return gw.genre_map() or {}
        except Exception:
            return {}

    async def fallback(self, payload: dict, env: Envelope,
                       error: AgentError) -> Optional[dict]:
        """协议 §5.2：A6 失败 → 数值型看板（这里给空数值 + 说明文案）。"""
        return {"user_id": int((payload or {}).get("user_id")
                               or env.context.user_id or 0),
                "granularity": str((payload or {}).get("granularity") or "quarter"),
                "radar": [0.0] * strategy.N_GENRES, "radar_labels": [],
                "trend": [], "drift_points": [],
                "interpretation": "观看记录不足或数据暂不可用，无法分析兴趣变化",
                "is_stable": True, "n_periods": 0,
                "degraded": True, "tokens_used": 0}


__all__ = ["DriftAgent"]
