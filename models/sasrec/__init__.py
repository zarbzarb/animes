# -*- coding: utf-8 -*-
"""序列推荐子包：SASRec 基座 + 滑动窗口数据集 + 训练循环。

- `dataset.py` 滑动窗口训练集与评估输入构造（Dataset + 纯函数，无 IO）。
- `config.py`  超参 dataclass（对齐 `configs/model.yaml`），含合法性校验。
- `model.py`   SASRec 网络（因果自注意力 + FFN），对外 `encode()` / `score()`。
- `train.py`   训练循环（BCE + AMP + warmup + 早停），通过回调与外部解耦。

分层纪律（`docs/dev-conventions.md`）：
`models/` 内所有代码必须**纯函数、无 IO** —— 不查库、不读配置文件、不调 LLM。
需要读 `seq_dataset.pkl` / `configs/*.yaml` 的调用由 `scripts/` 层完成，
把加载好的普通 Python 对象传进来即可。

⚠️ 训练循环里**没有**任何写盘动作：最优权重以 `state_dict` 形式返回，
由 `scripts/train.py` 决定存到哪。这样 `models/` 层可以被 Agent 进程
直接 import 而不带任何副作用。
"""

from models.sasrec.config import (  # noqa: F401
    OptimConfig,
    SASRecConfig,
    TrainConfig,
    cosine_lambda,
)
from models.sasrec.dataset import (  # noqa: F401
    PAD_ITEM,
    TARGET_SLOTS,
    SlidingWindowDataset,
    WindowBatchIterator,
    build_eval_inputs,
    count_windows,
    make_padded_id_array,
)
from models.sasrec.model import (  # noqa: F401
    SASRec,
    SASRecBlock,
    causal_pad_allow_mask,
)
from models.sasrec.train import (  # noqa: F401
    EpochRecord,
    FitResult,
    build_optimizer,
    build_scheduler,
    fit,
    set_seed,
    train_one_epoch,
)

__all__ = [
    # 数据
    "PAD_ITEM",
    "TARGET_SLOTS",
    "SlidingWindowDataset",
    "WindowBatchIterator",
    "make_padded_id_array",
    "build_eval_inputs",
    "count_windows",
    # 配置
    "SASRecConfig",
    "OptimConfig",
    "TrainConfig",
    "cosine_lambda",
    # 模型
    "SASRec",
    "SASRecBlock",
    "causal_pad_allow_mask",
    # 训练
    "fit",
    "train_one_epoch",
    "build_optimizer",
    "build_scheduler",
    "set_seed",
    "FitResult",
    "EpochRecord",
]
