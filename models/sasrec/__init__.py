# -*- coding: utf-8 -*-
"""序列推荐模型子包。

- `dataset.py`  滑动窗口训练集与评估输入构造（纯函数 + Dataset，无 IO）。

分层纪律（`docs/dev-conventions.md`）：
`models/` 内所有代码必须**纯函数、无 IO** —— 不查库、不读配置文件、不调 LLM。
需要读 `seq_dataset.pkl` / `configs/*.yaml` 的调用由 `scripts/` 层完成，
把加载好的普通 Python 对象传进来即可。
"""

from models.sasrec.dataset import (  # noqa: F401
    PAD_ITEM,
    TARGET_SLOTS,
    SlidingWindowDataset,
    build_eval_inputs,
    count_windows,
    make_padded_id_array,
)

__all__ = [
    "PAD_ITEM",
    "TARGET_SLOTS",
    "SlidingWindowDataset",
    "make_padded_id_array",
    "build_eval_inputs",
    "count_windows",
]
