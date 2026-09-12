# -*- coding: utf-8 -*-
"""数据侧子包：训练集的样本切分、子集选择与负采样。

- `user_subset.py` 确定性嵌套用户抽样，纯函数、无 IO（档位制的实现基础）。
- `negatives.py`   负采样唯一实现，纯函数、无 IO（评估公平性的实现基础）。

两条「唯一实现」约定（不要在别处另写一套）：

1. **"选哪些用户"只在这里实现一次**。`models/sasrec/dataset.py` 只负责按掩码
   组装序列，不要在里面另写随机抽样，否则档位之间会失去嵌套关系，
   "小档调参、大档直接用"的策略随之失效（见 `configs/scale.yaml`）。
2. **"抽哪些负样本"只在这里实现一次**。所有模型、所有实验组必须共用同一批
   负样本，否则指标差异里会混进"谁的负样本更容易区分"这一无关变量，
   对比直接失效（见 `docs/evaluation-plan.md` 4.1）。
"""

from models.data.negatives import (  # noqa: F401
    DEFAULT_NEG_SEED,
    NegativeSampleResult,
    sample_negatives,
)
from models.data.user_subset import (  # noqa: F401
    is_nested,
    kept_user_ids,
    mask_from_order,
    nested_user_subset,
    subset_order,
    user_hash_keys,
)

__all__ = [
    # 用户抽样
    "nested_user_subset",
    "kept_user_ids",
    "mask_from_order",
    "subset_order",
    "user_hash_keys",
    "is_nested",
    # 负采样
    "sample_negatives",
    "NegativeSampleResult",
    "DEFAULT_NEG_SEED",
]
