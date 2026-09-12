# -*- coding: utf-8 -*-
"""评估子包：指标与评估循环。

- `metrics.py`   HR@K / NDCG@K / Recall@K / MRR 的**唯一实现**，纯函数、无 IO。
- `evaluator.py` 评估循环：组装 1 正 + C 负候选、分批前向、剔除无效样本、
  调用 metrics 并产出可直接落盘的 `metrics.json` 结构。

职责不重叠（改任何一处都会影响实验可比性）：

| 关注点 | 唯一实现位置 |
|---|---|
| 抽哪些负样本 | `models/data/negatives.py` |
| 历史序列怎么拼 | `models/sasrec/dataset.py` |
| 指标公式 | `models/eval/metrics.py` |
| 把上面三块串起来跑前向 | `models/eval/evaluator.py` |

约定：**任何地方都不要再写一遍 HR/NDCG 公式**，import 本包即可。
"""

from models.eval.evaluator import (  # noqa: F401
    EvalData,
    build_eval_data,
    collect_ranks,
    drop_leaked_samples,
    evaluate,
    evaluate_cold_start,
    evaluate_grouped,
    find_leaked_mask,
)
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
    # 指标
    "hr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "mrr",
    "ranking_metrics",
    "positive_rank",
    "stable_order",
    "topk_indices",
    # 评估循环
    "EvalData",
    "build_eval_data",
    "collect_ranks",
    "evaluate",
    "evaluate_grouped",
    "evaluate_cold_start",
    "find_leaked_mask",
    "drop_leaked_samples",
]
