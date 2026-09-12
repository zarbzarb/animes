# -*- coding: utf-8 -*-
"""多兴趣胶囊配置 —— `models/multi_interest/config.py`

为什么单独一个 dataclass 而不往 `SASRecConfig` 里塞字段
----------------------------------------------------
M2.4 的 `SASRecConfig` docstring 明确说过：胶囊开关不属于它。
分层关系是：

    MultiInterestConfig = SASRecConfig（复用基座的全部结构超参）
                        + num_interests / routing_iters（M2.5 新增）

这样 E2 消融里「纯 SASRec vs +多兴趣」的其余超参**天然一致**
（同一个 SASRecConfig 实例传下去），不存在"两边各配一份然后配飘了"的可能。

yaml 口径
--------
`configs/model.yaml` 的 `model:` 段是**扁平**的：基座字段与胶囊字段并列，
`from_dict` 负责按键拆分（`num_interests` / `routing_iters` 归本配置，
其余归 `SASRecConfig.from_dict`）。模型层不读文件，拆分只处理内存 dict。
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from models.sasrec.config import SASRecConfig

__all__ = ["MultiInterestConfig"]

# 这两个键属于本配置，其余 `model:` 段的键全部交给 SASRecConfig
_MI_KEYS = {"num_interests", "routing_iters"}


@dataclass(frozen=True)
class MultiInterestConfig:
    """多兴趣模型的结构超参。

    参数
    ----
    sasrec        基座结构（hidden/layers/heads/dropout 等全部沿用）。
    num_interests 兴趣胶囊数 K。项目口径 **K=4**（`configs/genre_taxonomy.yaml`
                  的 12 类题材是它的经验依据：用户兴趣通常横跨 2~4 类）。
                  HPO 网格 {2,4,6,8}（evaluation-plan 4.2），改这里即可扫。
    routing_iters 动态路由迭代次数。MIND 原论文默认 3；再多收益递减且
                  每次 forward 多一遍 O(B·L·K·H) 的 agreement 计算。
    """

    sasrec: SASRecConfig
    num_interests: int = 4
    routing_iters: int = 3

    # ---------------- 校验 ----------------
    def __post_init__(self) -> None:
        if int(self.num_interests) < 2:
            raise ValueError(
                f"num_interests 必须 >= 2（K=1 退化成单向量 SASRec，"
                f"应直接用基座模型），收到 {self.num_interests}")
        if int(self.routing_iters) < 1:
            raise ValueError(
                f"routing_iters 必须 >= 1（0 次迭代等于只做加权平均，"
                f"不成胶囊），收到 {self.routing_iters}")
        if not isinstance(self.sasrec, SASRecConfig):
            raise TypeError(
                f"sasrec 必须是 SASRecConfig 实例，收到 {type(self.sasrec).__name__}")

    # ---------------- 派生属性 ----------------
    @property
    def n_items(self) -> int:
        """透传基座的物品池规模，便于上层不用再 `.sasrec.n_items`。"""
        return int(self.sasrec.n_items)

    # ---------------- 构造 ----------------
    @classmethod
    def from_dict(cls, model_cfg: dict, n_items: int) -> "MultiInterestConfig":
        """从 `configs/model.yaml` 的 `model` 段构造（键按 `_MI_KEYS` 拆分）。

        与 `SASRecConfig.from_dict` 同一条规则：不认识的键被忽略
        （checkpoint 快照里多出来的诊断字段不会让重建失败）。
        """
        cfg = dict(model_cfg or {})
        mi_kwargs = {k: cfg[k] for k in _MI_KEYS if k in cfg}
        base_kwargs = {k: v for k, v in cfg.items() if k not in _MI_KEYS}
        return cls(
            sasrec=SASRecConfig.from_dict(base_kwargs, n_items=int(n_items)),
            **mi_kwargs,
        )
