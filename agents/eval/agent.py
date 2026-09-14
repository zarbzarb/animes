# -*- coding: utf-8 -*-
"""A9 · 评估监控智能体（`EvalAgent`）—— 系统的"体检医生"。

职责（`docs/agents.md` §A9）
---------------------------
离线指标（复用 `models/eval/metrics.py`）+ 在线指标（聚合 `user_feedback`）
+ 指标异常检测（环比 > 10% 告警）+ 调用链瓶颈定位 + 归因说明。

三条边界
-------
* ❌ 不修改模型
* ❌ 不修改推荐结果
* ❌ **指标公式必须与 `models/eval/` 保持一致**（唯一实现）

降级（协议 §5.2：A9 P2 离线、失败跳过本次快照）
--------------------------------------------
`fallback` 返回空指标 + `degraded`，**不抛**。离线任务的失败不该
在管理后台弹一个红色错误 —— 它只需要"这次没数据"。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from agents.common.base import BaseAgent
from agents.common.data import GatewayNotBound, get_gateway
from agents.common.envelope import Envelope
from agents.common.errors import AgentError, invalid_param
from agents.common.llm import get_llm
from agents.eval import strategy
from agents.eval.schemas import EvalInput

logger = logging.getLogger(__name__)

_TIMEOUT_MS = 2500
_ALERT_PROMPT_VER = "a9.alarm.v1"

_SYSTEM = """你是推荐系统的运维助手。根据给定的指标异常，用一句话说明**可能原因**
与**建议动作**。

硬性要求：
1. 只能基于输入的指标与耗时数据推断，不得编造未给出的数据。
2. 不超过 60 个汉字，不要表情，不要换行。
3. 只输出这句话本身。"""


class EvalAgent(BaseAgent):
    agent_id = "A9"
    name = "eval"
    description = "评估监控：离线/在线指标 + 异常告警 + 瓶颈定位"
    online = False          # P2 离线

    async def invoke(self, payload: dict, env: Envelope) -> dict:
        action = env.header.action
        if action == "eval.online_snapshot":
            return await self._online(payload, env)
        if action == "eval.offline_metrics":
            return await self._offline(payload, env)
        if action == "eval.alarm":
            return await self._alarm(payload, env)
        raise invalid_param(f"未支持的 action：{action}")

    # ------------------------------------------------------------ 在线快照
    async def _online(self, payload: dict, env: Envelope) -> dict:
        inp = EvalInput.model_validate(payload)
        gw = get_gateway()
        if gw is None:
            raise GatewayNotBound("A9 需要网关聚合在线指标")

        start, end = self._range(inp)
        counts = gw.feedback_counts(start=start, end=end, scene=inp.scene) or {}
        rows = strategy.online_metrics(counts)

        metrics = []
        today = (end or date.today().isoformat())
        for r in rows:
            metrics.append({"metric_date": today, "scene": int(inp.scene or 0),
                            "metric_type": r["metric_type"],
                            "metric_value": r["metric_value"],
                            "sample_size": r["sample_size"],
                            "extra": {"source": "user_feedback"}})

        anomalies = self._anomalies(gw, metrics, inp)
        stats = self._agent_stats(gw, inp)
        bottleneck = strategy.spot_bottleneck(stats)

        if inp.persist and metrics:
            try:
                gw.save_metrics(metrics)
            except Exception as exc:
                logger.warning("A9 指标落库失败（已忽略）：%s", exc)

        alert_text, tokens = await self._alert_text(anomalies, bottleneck, env)
        return {"mode": inp.mode, "metrics": metrics,
                "trend": self._trend(gw, inp), "anomalies": anomalies,
                "bottleneck_agent": bottleneck, "agent_stats": stats,
                "alert_text": alert_text, "degraded": False,
                "tokens_used": tokens}

    # ------------------------------------------------------------ 离线指标
    async def _offline(self, payload: dict, env: Envelope) -> dict:
        inp = EvalInput.model_validate(payload)
        rows = strategy.load_experiment_metrics(inp.exp_dir or "") if inp.exp_dir else []
        metrics = []
        today = date.today().isoformat()
        for r in rows:
            metrics.append({"metric_date": today, "scene": r.get("scene", 0),
                            "metric_type": r["metric_type"],
                            "metric_value": r["metric_value"],
                            "sample_size": r.get("sample_size", 0),
                            "model_ver": r.get("model_ver"),
                            "extra": r.get("extra")})
        gw = get_gateway()
        anomalies = self._anomalies(gw, metrics, inp)
        if inp.persist and metrics and gw is not None:
            try:
                gw.save_metrics(metrics)
            except Exception as exc:
                logger.warning("A9 离线指标落库失败：%s", exc)
        return {"mode": "offline", "metrics": metrics,
                "trend": [], "anomalies": anomalies,
                "bottleneck_agent": None, "agent_stats": [],
                "alert_text": None, "degraded": not bool(metrics),
                "tokens_used": 0,
                "note": None if metrics else f"未在 {inp.exp_dir} 找到指标文件"}

    async def _alarm(self, payload: dict, env: Envelope) -> dict:
        """只生成告警文案（异常列表由调用方传入）。"""
        anomalies = list((payload or {}).get("anomalies") or [])
        bottleneck = (payload or {}).get("bottleneck_agent")
        text, tokens = await self._alert_text(anomalies, bottleneck, env)
        return {"anomalies": anomalies, "alert_text": text,
                "bottleneck_agent": bottleneck, "tokens_used": tokens,
                "prompt_ver": _ALERT_PROMPT_VER}

    # ------------------------------------------------------------ 辅助
    @staticmethod
    def _range(inp: EvalInput) -> tuple[Optional[str], Optional[str]]:
        if inp.date_range and len(inp.date_range) == 2:
            return inp.date_range[0], inp.date_range[1]
        end = date.today()
        start = end - timedelta(days=int(inp.days))
        return start.isoformat(), end.isoformat()

    def _anomalies(self, gw, metrics: list[dict], inp: EvalInput) -> list[dict]:
        """与库里的上一期数值做环比。取不到历史 → 只记录不告警。"""
        if gw is None or not metrics:
            return []
        out: list[dict] = []
        for m in metrics:
            try:
                hist = gw.query_metrics(metric_type=m["metric_type"],
                                        days=int(inp.days) + 1,
                                        scene=m.get("scene")) or []
            except Exception:
                hist = []
            prev = None
            for h in hist:
                if h.get("metric_date") != m["metric_date"]:
                    prev = float(h.get("metric_value") or 0.0)
                    break
            row = dict(m)
            row["prev_value"] = prev
            if prev is not None:
                out.extend(strategy.detect_anomalies(
                    [row], drop_threshold=inp.drop_threshold))
        return out

    @staticmethod
    def _agent_stats(gw, inp: EvalInput) -> list[dict]:
        try:
            return list(gw.trace_stats() or [])
        except Exception as exc:
            logger.debug("A9 取调用链统计失败：%s", exc)
            return []

    @staticmethod
    def _trend(gw, inp: EvalInput) -> list[dict]:
        out: list[dict] = []
        try:
            for mt in ("ctr", "cvr", "fav_rate"):
                rows = list(gw.query_metrics(metric_type=mt, days=int(inp.days),
                                             scene=inp.scene) or [])
                for r in rows:
                    out.append({"metric_type": mt, "metric_date": r.get("metric_date"),
                                "metric_value": float(r.get("metric_value") or 0.0),
                                "scene": r.get("scene", 0)})
        except Exception as exc:
            logger.debug("A9 取指标趋势失败：%s", exc)
        out.sort(key=lambda r: (r["metric_type"], str(r["metric_date"])))
        return out

    async def _alert_text(self, anomalies: list[dict], bottleneck: Optional[str],
                          env: Envelope) -> tuple[Optional[str], int]:
        if not anomalies and not bottleneck:
            return None, 0
        template = self._alert_template(anomalies, bottleneck)
        llm = get_llm()
        if not llm.available:
            return template, 0
        lines = [f"- {a['message']}" for a in anomalies[:5]]
        if bottleneck:
            lines.append(f"- 调用链瓶颈 Agent：{bottleneck}")
        res = await llm.complete(
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": "\n".join(lines)}],
            temperature=0.3, max_tokens=120, timeout_ms=_TIMEOUT_MS)
        if not res.ok:
            return template, 0
        text = str(res.text).strip().split("\n")[0].strip()[:60]
        return (text or template), res.tokens_used

    @staticmethod
    def _alert_template(anomalies: list[dict], bottleneck: Optional[str]) -> str:
        if anomalies:
            worst = max(anomalies, key=lambda a: a.get("level", 0))
            head = f"{worst['metric_type']} 异常：{worst['message']}"
        else:
            head = "指标正常"
        if bottleneck:
            return f"{head}；耗时瓶颈在 {bottleneck}，建议检查其依赖的模型/数据库"
        return head

    async def fallback(self, payload: dict, env: Envelope,
                       error: AgentError) -> Optional[dict]:
        """协议 §5.2：A9 失败 → 跳过本次快照（空结构，不报错）。"""
        return {"mode": str((payload or {}).get("mode") or "online"),
                "metrics": [], "trend": [], "anomalies": [],
                "bottleneck_agent": None, "agent_stats": [],
                "alert_text": None, "degraded": True, "tokens_used": 0,
                "degraded_reason": error.name}


__all__ = ["EvalAgent"]
