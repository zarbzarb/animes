# -*- coding: utf-8 -*-
"""A8 · 数据运营智能体（`DataOpsAgent`）。

职责（`docs/agents.md` §A8）：新番入库、向量生成触发、质量校验、修复建议。

三条边界
-------
* ❌ 不修改用户行为数据
* ❌ **不直接删除动漫（只标记下架）**
* ❌ 不参与在线推荐

⚠️ 一个必须说清的实现边界
-----------------------
"触发 DistilBERT 批量编码 / 重建 FAISS 索引"这类重活**不在 Agent 里做**：
它们是 `scripts/build_content_vectors.py` 的职责（需要读 parquet、写 32MB npy、
可能要 GPU）。A8 只做三件事：① 校验输入；② 维护 `cold_start_pool` 表；
③ 产出"需要重建索引"的**工单**（`suggestions`）。

把重活塞进 Agent 会让"Agent 可在 API 进程内运行"这条设计不成立
（一个 3 分钟的编码任务会把 worker 占住），而且 `agents/` 不允许写张量运算。

降级（协议 §5.2：A8 无超时约束、失败记录失败项下次重跑）
---------------------------------------------------
`fallback` 返回"部分成功报告"，**不抛**。
"""

from __future__ import annotations

import logging
from typing import Optional

from agents.common.base import BaseAgent
from agents.common.data import get_gateway
from agents.common.envelope import Envelope
from agents.common.errors import AgentError, invalid_param
from agents.dataops import strategy
from agents.dataops.schemas import DataOpsInput

logger = logging.getLogger(__name__)


class DataOpsAgent(BaseAgent):
    agent_id = "A8"
    name = "dataops"
    description = "数据运营：质检 / 新番入库 / 索引工单"
    online = False          # P2 离线

    async def invoke(self, payload: dict, env: Envelope) -> dict:
        action = env.header.action
        if action == "dataops.quality_check":
            return await self._quality_check(payload, env)
        if action == "dataops.sync_new_anime":
            return await self._sync_new_anime(payload, env)
        if action == "dataops.rebuild_index":
            return await self._rebuild_index(payload, env)
        raise invalid_param(f"未支持的 action：{action}")

    # ------------------------------------------------------------ 质检
    async def _quality_check(self, payload: dict, env: Envelope) -> dict:
        inp = DataOpsInput.model_validate(payload)
        gw = get_gateway()
        if gw is None:
            raise AgentError(strategy.ALERT_NORMAL + 50202,
                             "A8 质检需要网关读取动漫表")
        rows = gw.list_anime(limit=int(inp.params.get("limit") or 5000),
                             offset=int(inp.params.get("offset") or 0),
                             cold_only=False)
        problems = strategy.check_rows(rows, required=("title", "src_anime_id"))
        report = strategy.summarize(problems, target_table="anime",
                                    total_rows=len(rows))
        suggestions = self._suggest(report)
        if inp.persist and not inp.dry_run:
            try:
                gw.save_data_quality(report)
            except Exception as exc:
                logger.warning("A8 质检报告落库失败（已忽略）：%s", exc)
        return {"task": "quality_check", "processed_count": len(rows),
                "failed_items": [p for p in problems if p["check_type"] ==
                                 strategy.CHECK_FORBIDDEN][:50],
                "quality_report": report, "suggestions": suggestions,
                "dry_run": inp.dry_run, "degraded": False, "tokens_used": 0}

    # ------------------------------------------------------------ 新番入库
    async def _sync_new_anime(self, payload: dict, env: Envelope) -> dict:
        """校验 → 题材归类 → （非 dry_run 时）写入冷启动池。

        ⚠️ 本环境**没有外部新番源**（数据集是静态的），所以这里只处理
        `params.items` / `items` 显式传入的记录。这一点必须如实报告：
        返回里带 `note` 说明"未配置外部数据源"，而不是假装"同步了 0 条"。
        """
        inp = DataOpsInput.model_validate(payload)
        gw = get_gateway()
        items = list(inp.items or inp.params.get("items") or [])
        problems = strategy.check_rows(items, required=("title", "src_anime_id"))
        blocked = {p["anime_id"] for p in problems
                   if p["check_type"] == strategy.CHECK_FORBIDDEN}

        ok, failed = [], []
        for it in items:
            aid = int(it.get("src_anime_id") or 0)
            if aid in blocked:
                failed.append({"anime_id": aid,
                               "reason": "forbidden_genre"})
                continue
            mal = [int(x) for x in (it.get("genre_raw") or [])
                   if str(x).strip().lstrip("-").isdigit()]
            mapped, unmapped = strategy.classify_genres(mal)
            if not mapped:
                failed.append({"anime_id": aid, "reason": "no_genre_mapped",
                               "unmapped": unmapped})
                continue
            ok.append({**it, "genre_ids": mapped})

        written = 0
        if gw is not None and not inp.dry_run:
            for it in ok:
                try:
                    gw.upsert_cold_start_pool(int(it["src_anime_id"]), {
                        "season": it.get("season") or "unknown",
                        "n_interactions": 0, "content_vec_ready": 0,
                        "is_active": 1})
                    written += 1
                except Exception as exc:
                    failed.append({"anime_id": it.get("src_anime_id"),
                                   "reason": f"upsert_failed:{type(exc).__name__}"})

        return {"task": "sync_new_anime", "processed_count": len(ok),
                "failed_items": failed,
                "quality_report": strategy.summarize(
                    problems, target_table="anime", total_rows=len(items)),
                "suggestions": [
                    f"已入池 {written} 条，需重建内容向量与索引后才能被 A3 召回"
                    if written else "无写入（dry_run 或无有效记录）",
                    "内容向量重建请执行 scripts/build_content_vectors.py",
                ] + (["存在违规题材，已拦截未入库"] if blocked else []),
                "dry_run": inp.dry_run, "degraded": False, "tokens_used": 0,
                "note": None if items else "未配置外部新番数据源，且未显式传入 items"}

    # ------------------------------------------------------------ 索引工单
    async def _rebuild_index(self, payload: dict, env: Envelope) -> dict:
        """维护 `cold_start_pool`（A3 的候选来源），并产出重建工单。

        **不自己跑编码**：那是 `scripts/` 的事（见模块 docstring）。
        """
        inp = DataOpsInput.model_validate(payload)
        gw = get_gateway()
        if gw is None:
            raise AgentError(strategy.ALERT_NORMAL + 50202,
                             "A8 重建索引需要网关")
        rows = gw.list_anime(limit=int(inp.params.get("limit") or 5000),
                             cold_only=True,
                             min_year=int(inp.params.get("min_year") or 2021))
        active = 0
        if not inp.dry_run:
            for r in rows:
                try:
                    gw.upsert_cold_start_pool(int(r["src_anime_id"]), {
                        "season": r.get("season") or "unknown",
                        "n_interactions": int(r.get("n_interactions") or 0),
                        "content_vec_ready": 0, "is_active": 1})
                    active += 1
                except Exception as exc:
                    logger.warning("A8 池写入失败（%s）：%s", r.get("src_anime_id"), exc)
        return {"task": "rebuild_index", "processed_count": active,
                "failed_items": [],
                "quality_report": [],
                "suggestions": [
                    f"冷启动池目标 {len(rows)} 条，已激活 {active} 条",
                    "下一步：python scripts/build_content_vectors.py 重建内容向量",
                    "随后 A3 会自动加载新矩阵（无缓存则下次调用时重载）",
                ],
                "dry_run": inp.dry_run, "degraded": False, "tokens_used": 0}

    # ------------------------------------------------------------ 建议
    @staticmethod
    def _suggest(report: list[dict]) -> list[str]:
        out: list[str] = []
        for r in report:
            if r["problem_rows"] <= 0:
                continue
            if r["check_type"] == strategy.CHECK_FORBIDDEN:
                out.append(f"合规：{r['problem_rows']} 部含违规题材，"
                           "请置 is_forbidden=1 并下架")
            elif r["check_type"] == strategy.CHECK_DUPLICATE:
                out.append(f"{r['problem_rows']} 条 src_anime_id 重复，"
                           "建议保留交互数最高的一条")
            elif r["check_type"] == strategy.CHECK_GENRE_UNMAPPED:
                out.append(f"{r['problem_rows']} 条的题材无法归并到 12 类，"
                           "会拉低 A3/A4 的题材重合度")
            elif r["check_type"] == strategy.CHECK_MISSING:
                out.append(f"{r['problem_rows']} 条缺必填字段，前端卡片会空白")
            elif r["check_type"] == strategy.CHECK_OUTLIER:
                out.append(f"{r['problem_rows']} 条年份/评分越界，会影响新番判定")
        return out or ["数据质量检查通过，无待处理项"]

    async def fallback(self, payload: dict, env: Envelope,
                       error: AgentError) -> Optional[dict]:
        """协议 §5.2：A8 失败 → 部分成功报告（不抛）。"""
        return {"task": str((payload or {}).get("task") or "quality_check"),
                "processed_count": 0, "failed_items": [], "quality_report": [],
                "suggestions": [f"本次任务失败（{error.name}），已记录，下次重跑"],
                "dry_run": bool((payload or {}).get("dry_run")),
                "degraded": True, "tokens_used": 0,
                "degraded_reason": error.name}


__all__ = ["DataOpsAgent"]
