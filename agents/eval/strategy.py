# -*- coding: utf-8 -*-
"""A9 的指标计算与异常检测策略 —— **纯函数**。

⚠️ 最重要的一条：离线指标**不许在这里重算**
------------------------------------------
`docs/database-design.md` §5.1 的原话：「离线指标**必须**来自
`models/eval/metrics.py`，不允许在这里重算 —— 数值可以复用，公式不许复用」。

本项目已经为"口径分裂"栽过两次（指标唯一实现、负样本唯一实现），
所以本模块的策略是：

* 需要**从样本算指标**时 → `recompute_offline()` 直接调
  `models/eval/metrics.py`（唯一实现），本文件不写第二份 HR/NDCG 公式；
* 需要**读已有结果**时 → `load_experiment_metrics()` 从实验结果 JSON 读，
  那些数值本来就是 `models/eval/metrics.py` 产出的。

在线指标（CTR/CVR）不一样：它们不是排序指标，没有"唯一实现"问题，
`user_feedback` 聚合口径就是定义本身。
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Sequence

__all__ = [
    "ALERT_CRITICAL",
    "ALERT_NORMAL",
    "ALERT_WARN",
    "detect_anomalies",
    "load_experiment_metrics",
    "online_metrics",
    "recompute_offline",
    "spot_bottleneck",
]

ALERT_NORMAL, ALERT_WARN, ALERT_CRITICAL = 0, 1, 2

# 反馈动作码（与 server/db/models/recommend.py 对齐；AGENTS 层不 import server）
FB_EXPOSE, FB_CLICK, FB_FAV, FB_DISLIKE, FB_LIKE = 0, 1, 2, 3, 4


def online_metrics(counts: dict, *, sample_size: int = 0) -> list[dict]:
    """从 `user_feedback` 的按动作计数算在线指标。

    `counts` 形如 `{"0": 12000, "1": 900, "2": 210, ...}`（键是动作码字符串）。
    返回 `[{metric_type, metric_value, sample_size}]`。

    ⚠️ 分母为 0 → 返回**空列表**而不是 0。写 0 会在看板上画出"点击率跌到 0"
    的断崖，而真相是"当天没有曝光数据"。这种图会误导人做错误的决策。
    """
    def n(action: int) -> float:
        for k in (action, str(action)):
            if k in counts:
                return float(counts[k])
        return 0.0

    expose = n(FB_EXPOSE)
    if expose <= 0:
        return []
    out = [
        {"metric_type": "ctr", "metric_value": round(n(FB_CLICK) / expose, 5),
         "sample_size": int(expose)},
        {"metric_type": "fav_rate", "metric_value": round(n(FB_FAV) / expose, 5),
         "sample_size": int(expose)},
    ]
    clicks = n(FB_CLICK)
    if clicks > 0:
        out.append({"metric_type": "cvr",
                    "metric_value": round(n(FB_FAV) / clicks, 5),
                    "sample_size": int(clicks)})
    if sample_size:
        for row in out:
            row["sample_size"] = int(sample_size)
    return out


# 「越大越好」的指标。用来确定告警方向 —— 把 CTR 的跌幅与
# 「降级率」的涨幅混在一个方向判断里，会漏报掉一半异常。
HIGHER_IS_BETTER = {"hr5": True, "hr10": True, "ndcg5": True, "ndcg10": True,
                    "mrr": True, "recall10": True, "ctr": True, "cvr": True,
                    "fav_rate": True, "degrade_rate": False, "p95_ms": False}


def detect_anomalies(rows: Sequence[dict], *, drop_threshold: float = 0.10,
                     min_samples: int = 30) -> list[dict]:
    """环比异常检测：`|变化| > drop_threshold` 且方向为"变差" → 告警。

    `min_samples` 的意义：曝光只有 5 次时 CTR 可能从 0.20 掉到 0.00
    （−100%），但那是噪声。样本不足直接跳过 —— 比"配置一个更复杂的
    显著性检验"更简单可靠，也不会因为误报让人忽略真正的告警。
    """
    out: list[dict] = []
    for r in rows:
        cur = float(r.get("metric_value") or 0.0)
        prev = r.get("prev_value")
        if prev is None:
            continue
        prev = float(prev)
        if prev == 0:
            continue
        if int(r.get("sample_size") or 0) < int(min_samples):
            continue
        change = (cur - prev) / abs(prev)
        worse = (change < 0) if HIGHER_IS_BETTER.get(r["metric_type"], True) \
            else (change > 0)
        if not worse or abs(change) < float(drop_threshold):
            continue
        level = ALERT_CRITICAL if abs(change) >= float(drop_threshold) * 2 \
            else ALERT_WARN
        arrow = "下跌" if change < 0 else "上涨"
        out.append({
            "metric_type": r["metric_type"], "scene": int(r.get("scene") or 0),
            "current": cur, "previous": prev, "change": round(change, 4),
            "level": level,
            "message": (f"{r['metric_type']} 环比{arrow} {abs(change):.1%}"
                        f"（{prev:.4f} → {cur:.4f}）"),
        })
    return out


def spot_bottleneck(stats: Sequence[dict]) -> Optional[str]:
    """从 `agent_trace` 的按 Agent 聚合里找瓶颈。

    判据是「平均耗时 × 调用量占比」而不是单看平均耗时：
    一个 200ms 但每天只调 3 次的 Agent 不是瓶颈；
    一个 30ms 但占全部调用 40% 的 Agent 才是。
    """
    usable = [s for s in stats if int(s.get("calls") or 0) > 0]
    if not usable:
        return None
    total_ms = sum(float(s.get("total_ms") or 0) for s in usable)
    if total_ms <= 0:
        return None
    ranked = sorted(usable,
                    key=lambda s: -(float(s.get("total_ms") or 0)))
    top = ranked[0]
    share = float(top.get("total_ms") or 0) / total_ms
    return str(top.get("agent_id") or top.get("to_agent")) if share >= 0.25 else None


def load_experiment_metrics(exp_dir: str) -> list[dict]:
    """读实验结果目录下的指标文件（数值由 `models/eval/metrics.py` 产出）。

    支持两种布局（本项目的实验脚本产出的是第二种）：
    * `{exp_dir}/metrics.json`          → `{"hr@10": 0.5, ...}`
    * `{exp_dir}/result.json`           → `{"metrics": {...}, "model_ver": ...}`
    """
    if not exp_dir or not os.path.isdir(exp_dir):
        return []
    out: list[dict] = []
    for name in ("metrics.json", "result.json", "baselines.json", "summary.json"):
        path = os.path.join(exp_dir, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        block = data.get("metrics") if isinstance(data, dict) else None
        block = block or data
        if not isinstance(block, dict):
            continue
        for k, v in block.items():
            if not isinstance(v, (int, float)):
                continue
            key = str(k).lower().replace("@", "")
            if key in ("hr5", "hr10", "ndcg5", "ndcg10", "mrr", "recall10"):
                out.append({"metric_type": key, "metric_value": float(v),
                            "scene": 0, "sample_size": int(data.get("n_samples") or 0)
                            if isinstance(data, dict) else 0,
                            "model_ver": (data.get("model_ver")
                                          or data.get("experiment_name")
                                          if isinstance(data, dict) else None),
                            "extra": {"source": path}})
        if out:
            break
    return out


def recompute_offline(scores, positive_index: int = 0, ks: Sequence[int] = (5, 10)
                      ) -> dict[str, float]:
    """**从打分张量算离线指标 —— 唯一实现是 `models/eval/metrics.py`。**

    本函数只做"取排名 → 调指标"，公式一行都不在这里。
    任何在这里手写 HR/NDCG 的改动都是回归。
    """
    import torch

    from models.eval.metrics import hr_at_k, ndcg_at_k, positive_rank

    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
    ranks = positive_rank(scores, pos_index=int(positive_index))
    out: dict[str, float] = {}
    for k in ks:
        out[f"hr{k}"] = float(hr_at_k(ranks, int(k)))
        out[f"ndcg{k}"] = float(ndcg_at_k(ranks, int(k)))
    return out
