# -*- coding: utf-8 -*-
"""checkpoint 子包 —— `models/` 层里**唯一被允许做文件 IO 的地方**。

为什么这里可以破例
------------------
`models/` 的纪律是「无 IO」（`docs/project-structure.md` 三）。但"训练好的
权重存到哪"这件事必须有人做，而它又不能放在 `scripts/` —— 因为
`scripts/run_experiments.py`（M2.8）、`agents/recall/adapter.py`（阶段三）
都要读同一份权重，若保存格式由脚本各写各的，很快就会出现
「A 脚本存的 B 脚本读不了」。

因此本包承担「权重落盘的**格式唯一定义**」，并严格限定能力边界：

* **只做** `torch.save` / `torch.load` 与目录创建；
* **不做**：读 `.env`、查数据库、调 LLM、解析 YAML、决定实验参数。
  也就是说它接收已经准备好的普通对象，不自己去外面找东西。

格式（改则所有已存权重作废）
----------------------------
::

    {
      "format": "anirec.checkpoint.v1",   # 版本号，加载时校验
      "meta":   {...},                    # 实验名/档位/种子/epoch/指标/结构快照
      "state_dict": {...},                # 纯 CPU tensor，无引用
    }

`meta` 里冗余记录模型结构（`hidden_size` / `num_layers` / `n_items` 等），
所以**只看 checkpoint 就能重建模型**，不必回头翻当时的 YAML ——
这是"实验可复现"的最低要求：三个月后要复现某个数字时，
实验目录里的 YAML 很可能已被改动过。
"""

from models.checkpoint.io import (  # noqa: F401
    CHECKPOINT_FORMAT,
    checkpoint_name,
    load_checkpoint,
    load_model_from_checkpoint,
    load_state_into,
    save_checkpoint,
)

__all__ = [
    "CHECKPOINT_FORMAT",
    "checkpoint_name",
    "save_checkpoint",
    "load_checkpoint",
    "load_state_into",
    "load_model_from_checkpoint",
]
