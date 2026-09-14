# -*- coding: utf-8 -*-
"""离线批量重排（`/admin/tasks/offline-recommend` 与每日定时任务的实现）。

它做的事与线上 `/recommend/feed` **完全一致**（同一套 A0 流水线），
区别只在触发方式：线上是用户点出来的，这里是定时为活跃用户预先算好并落库。

为什么必须复用同一条流水线
------------------------
若离线用另一套简化逻辑（哪怕只是"少调 A5"），线上线下的结果就会不一致：
用户提前算好的推荐与刷新后看到的推荐不同，反馈数据（点击/收藏）会归因到
错误的推荐批次上，直接污染 CTR 与后续的模型迭代依据。
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from server.core.response import new_trace_id
from server.deps import get_gw
from server.services.agent_bridge import feed

logger = logging.getLogger(__name__)

# 每个用户的场景：综合 / 新番（分题材按需，不预生成以免爆量）
_SCENES = (("RECOMMEND_FEED", 0), ("RECOMMEND_NEW_ANIME", 2))

BATCH_ID_PREFIX = "offline"


def _batch_id() -> str:
    return f"{BATCH_ID_PREFIX}_{time.strftime('%Y%m%d_%H%M')}"


async def run_offline_recommend(*, user_id: Optional[int] = None,
                               limit: int = 50) -> dict:
    """为 `user_id`（或前 `limit` 个用户）预生成推荐并落库。

    返回统计，便于管理端展示"跑了多少、失败多少、耗时多久"。
    """
    gw = get_gw()
    if user_id:
        uids = [int(user_id)]
    else:
        uids = [int(u["id"]) for u in gw.list_users(limit=int(limit))]

    batch = _batch_id()
    ok = failed = 0
    total_ms = 0
    errors: list[dict] = []
    n_items = 0

    for uid in uids:
        for intent, scene in _SCENES:
            t0 = time.perf_counter()
            params = {"top_n": 20, "batch_id": batch}
            try:
                data, meta = await feed(uid, size=20, intent_hint=intent,
                                        params=params)
            except Exception as exc:
                failed += 1
                errors.append({"user_id": uid, "scene": scene,
                               "error": f"{type(exc).__name__}: {exc}"})
                logger.warning("离线推荐失败 user=%s scene=%s：%s", uid, scene, exc)
                continue
            items = data.get("items") or []
            n_items += len(items)
            # `feed` 已经让 A4 落库（A4 自己调 gateway.save_recommendations），
            # 这里只做兜底：A4 因降级未落库时补一次，保证下游（A7 卡片、
            # 在线看板）不会读到空。
            if items and not meta.get("batch_id"):
                try:
                    gw.save_recommendations(uid, scene, [
                        {"anime_id": i["anime"].get("src_anime_id"),
                         "src_anime_id": i["anime"].get("src_anime_id"),
                         "rank_no": i["rank"], "final_score": i.get("final_score"),
                         "behavior_score": i.get("behavior_score"),
                         "content_score": i.get("content_score"),
                         "interest_id": i.get("interest_id"),
                         "is_cold_start": i.get("is_cold_start")}
                        for i in items], batch_id=batch)
                except Exception as exc:
                    logger.warning("补写推荐结果失败 user=%s：%s", uid, exc)
            ok += 1
            total_ms += int((time.perf_counter() - t0) * 1000)

    result = {
        "batch_id": batch,
        "trace_id": new_trace_id(),
        "users": len(uids),
        "ok": ok,
        "failed": failed,
        "n_items": n_items,
        "avg_ms": int(total_ms / ok) if ok else 0,
        "sync": True,
        "note": ("当前为同步执行（演示规模 <1s）。生产应改为投递调度器后立即返回，"
                 "避免请求挂起"),
        "errors": errors[:10],
    }
    logger.info("离线推荐完成：%s", {k: v for k, v in result.items() if k != "errors"})
    return result


__all__ = ["BATCH_ID_PREFIX", "run_offline_recommend"]
