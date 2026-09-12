# -*- coding: utf-8 -*-
"""评估子包：指标与评估循环。

- `metrics.py`   HR@K / NDCG@K / Recall@K / MRR 的**唯一实现**，纯函数、无 IO。
- `evaluator.py` 评估循环（全量 / 分题材 / 冷启动子集），负责组装候选集、
  剔除无效样本、调用 metrics 并落 `metrics.json`（M2.2，尚未实现）。

约定：**任何地方都不要再写一遍 HR/NDCG 公式**，import 本包即可。
"""

from models.eval.metrics import (  # noqa: F401
    hr_at_k,
    mrr,
    ndcg_at_k,
    positive_rank,
    ranking_metrics,
    recall_at_k,
    stable_order,
    topk_indices,
)

__all__ = [
    "hr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "mrr",
    "ranking_metrics",
    "positive_rank",
    "stable_order",
    "topk_indices",
]
