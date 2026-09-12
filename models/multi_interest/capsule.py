# -*- coding: utf-8 -*-
"""动态路由胶囊层 —— `models/multi_interest/capsule.py`

MIND 口径（Li et al., 2019「Multi-Interest Network with Dynamic Routing」）
--------------------------------------------------------------------------
对每个用户，把 SASRec 编码出的逐位置隐状态 `hidden [B, L, H]` 路由成
K 个兴趣胶囊 `interests [B, K, H]`。一次迭代做三件事：

    votes:  u_i = W · hidden_i              （低维胶囊，每位置一个，W 共享）
    agree:  b_ij = b_ij + u_i · e_j         （与上次的高维胶囊求内积并累加）
    route:  c_ij = softmax_j(b_ij)          （**softmax 在胶囊维 j**，MIND 式）
            e_j = squash( Σ_i c_ij · u_i )  （对位置 i 求和得到高维胶囊）

初值 b=0、e=squash(Σ c·u)；迭代 `routing_iters` 次。首次迭代等价于
「按位置均匀聚合后 squash」，之后逐步分化出 K 个兴趣方向。

三个必须写清的实现细节（改则与 MIND 论文口径不可比）
----------------------------------------------------
1. **softmax 在 j（胶囊维）而不是 i（位置维）**。这是 MIND 与部分开源
   实现的分歧点：softmax 在 j 意味着每个位置的票在 K 个兴趣间分配，
   长序列里的高范数位置天然话语权更大（这正 MIND 的设计意图）。
2. **PAD 位置在投票前置零**（`valid` 掩码乘在 u 上），而不是在路由
   系数上加 -inf。原因：softmax 在 j 维，PAD 行的 c_ij 依然合法
   （非 NaN），把它的票 u_i 置零后对 Σ_i 无任何贡献 —— 比 mask logits
   更简单且数值上严格等价。⚠️ 前提是 hidden 在 PAD 位置的值**不能**
   参与路由：SASRec 的 `encode()` 文档已声明 PAD 位置的输出无意义，
   这里是唯一把它挡在门外的关口。
3. **squash 带 eps**：fp16 下 ‖x‖ 可能为 0（全 PAD 用户的票全为零），
   `‖x‖²/(1+‖x‖²) · x/‖x‖` 分母为零会出 NaN，加 eps 压住。

路由是**纯前向计算、无随机性、无持久状态**：b 每次 forward 从 0 重来。
因此 eval 模式下同一输入永远得到同一组兴趣向量（测试锁定）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["squash", "InterestRouting", "interest_diversity"]


# =====================================================================
# squash
# =====================================================================
def squash(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """向量 squash：把范数压进 (0, 1]，方向不变。

        squash(x) = (‖x‖² / (1 + ‖x‖²)) · x / ‖x‖

    eps 的作用见模块 docstring 第 3 条 —— 全零输入必须返回全零而不是
    NaN（全 PAD 用户在路由首迭代就会走到这一步）。
    """
    sq_norm = x.pow(2).sum(dim=dim, keepdim=True)              # [.., 1]
    scale = sq_norm / (1.0 + sq_norm)
    norm = torch.sqrt(sq_norm + eps)                            # [.., 1]
    return scale * x / norm


# =====================================================================
# 路由层
# =====================================================================
class InterestRouting(nn.Module):
    """把逐位置隐状态路由成 K 个兴趣胶囊。

    唯一可训练参数是低维投影 `W`（H×H，无 bias）—— 路由本身没有参数，
    这对 E2 消融很重要：**多兴趣模型的参数增量只有 W 的 4,096 个**
    （hidden=64 时），指标差异不能归因于容量增长。
    """

    def __init__(self, hidden_size: int, num_interests: int,
                 routing_iters: int, dropout: float = 0.0):
        super().__init__()
        if int(hidden_size) <= 0:
            raise ValueError(f"hidden_size 必须为正，收到 {hidden_size}")
        if int(num_interests) < 2:
            raise ValueError(
                f"num_interests 必须 >= 2，收到 {num_interests}（K=1 请用基座）")
        if int(routing_iters) < 1:
            raise ValueError(
                f"routing_iters 必须 >= 1，收到 {routing_iters}")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout 应在 [0, 1)，收到 {dropout}")

        self.hidden_size = int(hidden_size)
        self.num_interests = int(num_interests)
        self.routing_iters = int(routing_iters)

        # 低维胶囊投影：每位置 hidden -> K 份票（一份 per 兴趣）。
        # MIND 的 votes 是"低维胶囊到每个高维胶囊各一张票"，
        # 这里用一次 Linear 展开 K·H 再 view，等价且最快。
        self.proj = nn.Linear(self.hidden_size,
                              self.hidden_size * self.num_interests,
                              bias=False)
        self.vote_dropout = nn.Dropout(float(dropout)) if dropout > 0 else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 与 SASRec 基座同口径的 N(0, 0.02²)：不做 Xavier，避免与
        # 基座初始化量级不一致导致路由 logits 初始就失衡
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """路由：`[B, L, H] x [B, L] -> [B, K, H]`。

        参数
        ----
        hidden `[B, L, H]` 逐位置隐状态（含 PAD 位置，将被置零）。
        valid  `[B, L]` bool，True = 真实物品位置。

        返回
        ----
        `[B, K, H]` 兴趣胶囊（已 squash，范数 ≤ 1）。
        """
        if hidden.dim() != 3:
            raise ValueError(
                f"hidden 应为 (B, L, H)，实际 {tuple(hidden.shape)}")
        if valid.shape != hidden.shape[:2]:
            raise ValueError(
                f"valid 形状 {tuple(valid.shape)} 与 hidden "
                f"{tuple(hidden.shape[:2])} 不一致")
        if hidden.size(-1) != self.hidden_size:
            raise ValueError(
                f"hidden 最后一维 {hidden.size(-1)} 与路由层期望的 "
                f"{self.hidden_size} 不一致")

        b, length, _ = hidden.shape
        k, h = self.num_interests, self.hidden_size
        valid = valid.to(torch.bool)

        # ---------- votes：[B, L, K, H]，PAD 位置整票置零 ----------
        u = self.proj(hidden).view(b, length, k, h)
        if self.vote_dropout is not None:
            u = self.vote_dropout(u)
        u = u * valid.unsqueeze(-1).unsqueeze(-1).to(u.dtype)

        # ---------- 路由迭代（纯计算，无参数）----------
        # b_logits 每次调用从 0 开始：路由状态不跨 batch 存续
        b_logits = hidden.new_zeros(b, length, k)                # [B, L, K]
        interests = None

        for _ in range(self.routing_iters):
            # softmax 在 j（胶囊维）：每个位置的票分给 K 个兴趣
            coupling = torch.softmax(b_logits, dim=-1)           # [B, L, K]
            # e_j = squash(Σ_i c_ij · u_i)
            agg = torch.einsum("blk,blkh->bkh", coupling, u)     # [B, K, H]
            interests = squash(agg, dim=-1)

            # agreement：b_ij += u_i · e_j（MIND Algorithm 1 的累加式更新）
            b_logits = b_logits + torch.einsum(
                "blkh,bkh->blk", u, interests)

        return interests                                         # [B, K, H]

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"hidden={self.hidden_size}, interests={self.num_interests}, "
                f"iters={self.routing_iters}")


# =====================================================================
# 诊断
# =====================================================================
def interest_diversity(interests: torch.Tensor, eps: float = 1e-8) -> float:
    """K 个兴趣的平均两两余弦相似度（越小 = 兴趣越分化）。

    这是 M2.5 最重要的诊断量：路由塌缩（所有位置挤进同一胶囊）会表现为
    该值趋近 1；健康状态通常在 0.2~0.6（兴趣之间有区分但同属一个用户）。
    只在 eval 模式下调用才有意义（dropout 会扰动范数）。

    返回标量 float；K < 2 时返回 0.0（无对可算）。
    """
    if interests.dim() != 3 or interests.size(1) < 2:
        return 0.0
    normed = F.normalize(interests, dim=-1, eps=eps)             # [B, K, H]
    sim = torch.einsum("bkh,bjh->bkj", normed, normed)           # [B, K, K]
    k = sim.size(1)
    # 去掉对角线（自身相似恒为 1），只留 K(K-1)/2 对
    off_diag = sim.sum(dim=(1, 2)) - k
    denom = k * (k - 1)
    return float((off_diag / denom).mean())
