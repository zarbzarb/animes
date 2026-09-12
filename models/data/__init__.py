# -*- coding: utf-8 -*-
"""数据侧子包：训练集的样本切分与子集选择。

- `user_subset.py` 确定性嵌套用户抽样，纯函数、无 IO（档位制的实现基础）。

约定：**"选哪些用户"的逻辑只在这里实现一次**。
`models/sasrec/dataset.py` 只负责按掩码组装序列，不要在里面另写一套随机抽样，
否则档位之间会失去嵌套关系，"小档调参、大档直接用"的策略随之失效
（见 `configs/scale.yaml`）。
"""

from models.data.user_subset import (  # noqa: F401
    is_nested,
    kept_user_ids,
    mask_from_order,
    nested_user_subset,
    subset_order,
    user_hash_keys,
)

__all__ = [
    "nested_user_subset",
    "kept_user_ids",
    "mask_from_order",
    "subset_order",
    "user_hash_keys",
    "is_nested",
]
