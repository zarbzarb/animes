# -*- coding: utf-8 -*-
"""多兴趣序列推荐模型 —— `models/multi_interest/model.py`

结构（M2.5，与 MIND 同族但训练协议与本项目 SASRec 基线完全一致）
----------------------------------------------------------------
    SASRec 编码主干（与 M2.4 逐字相同，含因果 mask 规则）
      -> 逐位置隐状态 [B, L, H]
      -> 动态路由（capsule.py）-> K 个兴趣胶囊 [B, K, H]
      -> 候选打分：s(c) = max_j ( e_j · E[c] )，E 与输入嵌入**共享权重**

刻意保持的两条可比性约束（E2 消融成立的前提）
----------------------------------------------
1. **编码主干不改**：多兴趣模型复用 `SASRec` 整个 encode 路径
   （嵌入、post-LN 块、对角恒开的因果 mask），新增的只有路由层
   （4,096 个参数，hidden=64 时）。因此 E2 里"+多兴趣"的增益
   不能被"主干也变了"污染。
2. **训练协议不改**：`fit()` 只调 `model.score(x, cand)`，
   损失、1 正 1 负候选、BCE、早停全部照旧。打分从单向量点积换成
   max-over-interests 是**唯一**的行为差异。

为什么打分用 max 而不是 attention 加权
--------------------------------------
MIND 在 serving 端用 max（或 target-attention）。这里选 max 的理由：
* 无需引入额外参数，E2 的参数增量保持可核算；
* max 对"用户有多个兴趣、候选只命中一个"的场景是语义正确的匹配方式；
* 加权聚合会把不相关兴趣的分数也混进来，1 正 1 负的 BCE 下反而模糊边界。
MIND 训练用 target-attention 属于另一协议，不适合直接搬过来当消融组。

⚠️ 必须做的量级对齐（score_scale = sqrt(H)）
--------------------------------------------
squash 把兴趣向量范数压进 (0,1]，而基座的 user_repr 是 LayerNorm 的
输出（范数 ≈ sqrt(H)）；本项目 item_emb 按 SASRec 论文口径 N(0,0.02²)
初始化（H=64 时范数仅约 0.16）。不缩放的话：

    多兴趣 logits |score| ≤ 0.008   vs   基线 logits ~ 0.3 量级

BCE 在线性区学习极慢（玩具任务实测 10 epoch 才从 0.693 降到 0.566），
同样 lr / epochs 下多兴趣模型会系统性欠收敛 —— E2 会把"训练预算不足"
误判成"多兴趣没用"。乘 sqrt(H) 与基座 `scale_emb` 是同一条惯例
（MIND 原实现无此步是因为它的嵌入初始化范数≈1，天然匹配 squash 口径）。

注意：正数缩放**不改变评估排序**（max 对正数缩放保序），只影响
训练时的 loss 动态 —— 这是拿它做对齐而不用担心可比性的原因。

接口契约（与 M2.4 完全相同，评估器零改动）
------------------------------------------
    score(input_ids [B, L], candidate_ids [B, 1+C]) -> scores [B, 1+C]

第 0 列恒为正样本、分越大越相关 —— 评估器照旧直接吃 `model.score`。
另提供 `encode_interests()` 供诊断（兴趣分化度）与阶段三 Agent 使用。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from models.multi_interest.capsule import InterestRouting, interest_diversity
from models.multi_interest.config import MultiInterestConfig
from models.sasrec.model import PAD_ITEM, SASRec

__all__ = ["MultiInterestSASRec"]


class MultiInterestSASRec(nn.Module):
    """SASRec 主干 + 动态路由的多兴趣模型。

    用法
    ----
    >>> cfg = MultiInterestConfig.from_dict({"hidden_size": 64, "num_interests": 4}, n_items=1000)
    >>> model = MultiInterestSASRec(cfg)
    >>> x = torch.zeros(4, 48, dtype=torch.long)
    >>> cand = torch.randint(1, 1001, (4, 101))
    >>> model.score(x, cand).shape
    torch.Size([4, 101])
    """

    def __init__(self, cfg: MultiInterestConfig, repr_provider: nn.Module = None):
        super().__init__()
        self.mi_cfg = cfg
        self.backbone = SASRec(cfg.sasrec, repr_provider=repr_provider)
        self.routing = InterestRouting(
            hidden_size=int(cfg.sasrec.hidden_size),
            num_interests=int(cfg.num_interests),
            routing_iters=int(cfg.routing_iters),
        )
        # 量级对齐常数（见模块 docstring「必须做的量级对齐」）：
        # 让 squashed 兴趣向量与基座 user_repr（LayerNorm 输出，范数≈sqrt(H)）同量级
        self.score_scale = float(cfg.sasrec.hidden_size) ** 0.5

    # ------------------------------------------------------------------
    # 便捷属性：让 fit() / train_one_epoch 里既有的
    # `model.cfg.n_items` 取值链路不用改
    # ------------------------------------------------------------------
    @property
    def cfg(self) -> MultiInterestConfig:
        """主配置。注意 `cfg.n_items` 走 MultiInterestConfig 的透传。"""
        return self.mi_cfg

    @property
    def num_interests(self) -> int:
        return int(self.mi_cfg.num_interests)

    @property
    def repr_provider(self):
        """物品表示提供者（M2.6b 候选侧内容融合）。

        统一暴露在**模型顶层**：基座与多兴趣模型的 provider 都在
        `SASRec` 上，但调用方（训练脚本、报告、诊断）不该各自记住
        "多兴趣要多走一层 .backbone" —— 这种路径分歧正是 `--content-fusion`
        在多兴趣分支第一次就崩掉的原因（AttributeError）。
        """
        return self.backbone.repr_provider

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def encode_interests(
        self,
        seq: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码 + 路由。

        参数
        ----
        seq  `[B, L]` long，PAD=0（左填充，与基座同一口径）。
        mask `[B, L]` bool，None 时按 `seq != 0` 推导。

        返回
        ----
        hidden    `[B, L, H]`（透传自基座，诊断/内容融合 M2.6 还要用）
        interests `[B, K, H]` 已 squash 的兴趣胶囊
        """
        hidden, _ = self.backbone.encode(seq, mask)
        valid = (seq != PAD_ITEM) if mask is None else mask.to(torch.bool)
        interests = self.routing(hidden, valid)                  # [B, K, H]
        return hidden, interests

    def forward(self, seq: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """习惯入口：直接返回兴趣胶囊 `[B, K, H]`。"""
        _, interests = self.encode_interests(seq, mask)
        return interests

    # ------------------------------------------------------------------
    # 打分（与 evaluator 的 score_fn 契约一致）
    # ------------------------------------------------------------------
    def score(
        self,
        input_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """多兴趣打分：`[B, L] x [B, C] -> [B, C]`，max over K。

        与 `models/eval/evaluator.py` 的 `score_fn` 契约**逐字一致**
        （第 0 列正样本、分越大越相关），评估器零改动。
        """
        _, interests = self.encode_interests(input_ids)          # [B, K, H]

        # 与基座同一条入口：内容融合（M2.6b）时这里是融合后的物品表示
        emb_weight = self.backbone.item_table()                  # [V, H]
        if emb_weight.device != candidate_ids.device:
            candidate_ids = candidate_ids.to(emb_weight.device)

        cand = candidate_ids.clamp(min=PAD_ITEM,
                                   max=self.mi_cfg.sasrec.n_items)
        item_vec = emb_weight[cand]                              # [B, C, H]

        if self.backbone.out_proj is not None:
            # 基座关闭了权重绑定时，兴趣向量同样要先投到输出空间
            interests = self.backbone.out_proj(interests)

        # s_jc = (sqrt(H) · e_j) · E[c] -> [B, K, C]，再对兴趣维取 max。
        # sqrt(H) 是量级对齐（见模块 docstring），不改评估排序
        scores = torch.einsum("bkh,bch->bkc",
                              interests * self.score_scale, item_vec)
        return scores.max(dim=1).values                          # [B, C]

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    def interest_diversity(self, seq: torch.Tensor) -> float:
        """这批用户 K 个兴趣的平均两两余弦相似度（eval 模式下调用）。"""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                _, interests = self.encode_interests(seq)
        finally:
            if was_training:
                self.train()
        return interest_diversity(interests)

    @property
    def n_params(self) -> int:
        """可训练参数量。与基座对比的增量应只有路由层。"""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def config_snapshot(self) -> dict:
        """结构摘要，供 checkpoint 与训练日志落盘。

        ⚠️ `arch` 字段是 `load_model_from_checkpoint` 重建时的分发依据，
        M2.4 存的 SASRec checkpoint 没有 `arch`，加载时按 "sasrec" 兜底。
        """
        base = self.backbone.config_snapshot()
        return {
            "arch": "multi_interest",
            "num_interests": int(self.mi_cfg.num_interests),
            "routing_iters": int(self.mi_cfg.routing_iters),
            "n_interest_params": int(self.routing.proj.weight.numel()),
            **base,
        }
