# -*- coding: utf-8 -*-
"""多兴趣子包：MIND 式动态路由胶囊 + 复用 SASRec 主干的多兴趣模型。

- `config.py`  `MultiInterestConfig`（= SASRecConfig + K / 路由迭代数）。
- `capsule.py` squash / 动态路由 / 兴趣分化度诊断（纯计算层）。
- `model.py`   `MultiInterestSASRec`，对外 `encode_interests()` / `score()`，
               打分契约与基座逐字一致（评估器零改动）。

分层纪律与基座相同：本包内**纯函数、无 IO**，读数据与配置是 `scripts/` 的事。
"""

from models.multi_interest.capsule import (  # noqa: F401
    InterestRouting,
    interest_diversity,
    squash,
)
from models.multi_interest.config import MultiInterestConfig  # noqa: F401
from models.multi_interest.model import MultiInterestSASRec  # noqa: F401

__all__ = [
    "InterestRouting",
    "MultiInterestConfig",
    "MultiInterestSASRec",
    "interest_diversity",
    "squash",
]
