# -*- coding: utf-8 -*-
"""对比基线 —— `models/baselines/`

本包提供 E1（整体效果对比）所需的三个基线。所有基线都遵守同一条硬约束：

    **与本文模型共用同一划分 / 同一负样本池 / 同一个评估器 / 同一早停策略**

具体落在三件事上（`docs/evaluation-plan.md` §4.1）：

1. **打分接口统一** —— 三个基线都实现
   `score(input_ids [B,L], candidate_ids [B,1+C]) -> [B,1+C]`，
   与 `models/eval/evaluator.py` 的 `ScoreFn` 契约逐字一致，
   于是评估代码里**没有**「如果是基线就走另一条分支」这种东西。
2. **负样本不由基线自己采** —— 训练侧走 `models/data/negatives.py`
   （`fit()` 内部调用），评估侧的 1 正 100 负由 `evaluator.build_eval_data`
   统一生成，且缓存后跨模型复用。
3. **早停不由基线自己实现** —— 需要训练的 GRU4Rec 直接调
   `models/sasrec/train.py::fit()`，连 patience 的单位换算都共用。

| 模块 | 类型 | 需要训练 | 参数量 |
|---|---|---|---|
| `popularity.py` | 热度（零信息） | 否 | 0 |
| `itemcf.py` | 传统协同过滤 | 否 | 0 |
| `gru4rec.py` | 经典序列模型（RNN） | 是 | ≈1.03M（H=64, L=1） |

⚠️ `models/` 层的纪律是**纯函数、无 IO**：本包只提供"给定数据 → 打分/模型"，
读 pkl、选档位、写报告都在 `scripts/run_baselines.py`。
"""

from __future__ import annotations

from models.baselines.gru4rec import GRU4Rec, GRU4RecConfig
from models.baselines.itemcf import (
    ItemCFScorer,
    build_cooccurrence,
    build_itemcf_similarity,
)
from models.baselines.popularity import (
    PopularityScorer,
    count_train_frequency,
    normalize_frequency,
)

__all__ = [
    "PopularityScorer",
    "count_train_frequency",
    "normalize_frequency",
    "ItemCFScorer",
    "build_cooccurrence",
    "build_itemcf_similarity",
    "GRU4Rec",
    "GRU4RecConfig",
]
