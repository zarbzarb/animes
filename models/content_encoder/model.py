# -*- coding: utf-8 -*-
"""候选侧内容融合（M2.6b）：内容向量作为**输入特征**参与端到端训练。

与 `fusion.py` 的本质区别（这就是 M2.6 之后要补的那条路径）
--------------------------------------------------------
`fusion.py` 是**后验**融合：行为模型照旧训练，只在打分时把 `w_c · 内容分`
加到行为分上。M2.6 实测（debug 档）表明它无增益：7:3 → 0.7874 对纯行为
0.7878，冷启动 5:5 → 0.6350 对 7:3 0.6414。

本模块是**结构级**融合：把内容向量拼进物品表示，让模型自己在训练中学
「什么时候该信任内容、信任到什么程度」：

    table = Linear( [ item_emb[v] , content[v] ] )      # (H + D) -> H

为什么这样该有增益（H4 假设）
------------------------------
新番（year>=2021）在训练集中不可见时，它的**行为嵌入**几乎无梯度信号；
但它的**内容向量**是离线算好的、训练时就存在。拼进输入特征后，模型
在冷启动物品上仍有一份可用的表示 —— 这正是「内容融合缓解新番冷启动」
这条假设要验证的机制，而后验加权做不到这一点（它只是在分数上补，
补不动「embedding 没学过」这件事）。

⚠️ 可比性约束（E2 消融成立的前提）
----------------------------------
1. **内容向量是 buffer，不是参数**：不参与梯度（离线 PCA 产物，
   训练它既无意义也会破坏「内容通路固定」的可解释性）。
2. **新增参数恰好是投影层**：concat 模式 `(H + D) · H + H`，
   hidden=64 / D=512 时为 36,928（占基线 1,107,328 的 3.3%）。
   E2 报告须同时给出这个数字。
3. **PAD 行恒零**：`SASRec.score/encode` 依赖 `table[0] == 0`
   （PAD 位置不产生语义、也不该被候选打分命中）。投影层有 bias，
   所以必须显式把第 0 行清零。
4. **训练协议不变**：`fit()` 照旧只调 `model.score()`，1 正 1 负 BCE、
   早停、评估器全部不动 —— 与 M2.4 基线的唯一差异就是物品表示。

投影层初始化沿用 SASRec 论文口径 N(0, 0.02²)，bias 置零：这样训练
起点处「内容那一半」的初始贡献为零，模型是从「等价于纯行为基座」
出发再学出来的 —— 若训练后指标没变，说明模型选择忽略内容，
这本身就是 E2 要报告的结论，而不是初始化造成的假象。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.sasrec.model import PAD_ITEM

__all__ = ["ItemContentFusion", "build_model"]

INIT_STD = 0.02          # 与 SASRec 论文 / models/sasrec/model.py 同一口径
SUPPORTED_MODES = ("concat", "add")


class ItemContentFusion(nn.Module):
    """物品表示提供者：`item_weight [V, H] -> table [V, H]`。

    用法
    ----
    >>> import torch
    >>> D, H, V = 8, 16, 20
    >>> content = torch.randn(V, D); content[0] = 0.0
    >>> provider = ItemContentFusion(content, hidden_size=H)
    >>> w = torch.randn(V, H)
    >>> table = provider(w)
    >>> table.shape
    torch.Size([20, 16])
    >>> float(table[0].abs().sum())      # PAD 行恒零
    0.0
    """

    def __init__(
        self,
        content_matrix: torch.Tensor,
        hidden_size: int,
        mode: str = "concat",
        dropout: float = 0.0,
    ):
        """
        参数
        ----
        content_matrix `[V, D]` 内容向量矩阵，**第 0 行必须是 PAD（全零）**，
                       行序约定「第 v 行 = 物品 idx v」。构造时会做
                       防御性 L2 归一（与 `fusion.ContentScorer` 同一约定）。
        hidden_size    基座 hidden 维 H。
        mode           `"concat"`（拼接后降维，默认，对应项目文档口径）
                       或 `"add"`（**残差**：`item_emb + proj(content)`；
                        参数更少，且训练起点等价于基线，适合回答
                        "内容有没有增量"这个问题）。
        dropout        作用在融合后的物品表示上。默认 0.0 —— 与基线保持
                        一致的自由度，避免把「正则强度不同」混进消融差异。
        """
        super().__init__()
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"mode 只能是 {SUPPORTED_MODES}，收到 {mode!r}")
        if content_matrix.dim() != 2:
            raise ValueError(f"content_matrix 应为二维，收到 {tuple(content_matrix.shape)}")
        if int(content_matrix.shape[0]) <= PAD_ITEM:
            raise ValueError("content_matrix 至少要包含 PAD 行与一个物品行")

        h = int(hidden_size)
        d = int(content_matrix.shape[1])
        self.hidden_size = h
        self.content_dim = d
        self.mode = mode

        # ---------- 内容向量：固定特征，不是参数 ----------
        mat = content_matrix.float()
        norms = mat.norm(dim=-1, keepdim=True)
        mat = torch.where(norms > 1e-12, mat / norms.clamp(min=1e-12),
                          torch.zeros_like(mat))
        mat[PAD_ITEM] = 0.0
        self.register_buffer("content", mat)

        # ---------- 融合投影 ----------
        in_dim = h + d if mode == "concat" else d
        self.proj = nn.Linear(in_dim, h)

        # PAD 行清零用的乘子（registered buffer → 自动跟随 .to(device)）
        keep = torch.ones(int(mat.shape[0]), 1)
        keep[PAD_ITEM] = 0.0
        self.register_buffer("_pad_mask", keep)

        self.emb_dropout = nn.Dropout(float(dropout))
        self.reset_parameters()

    # ------------------------------------------------------------------
    def reset_parameters(self) -> None:
        """concat：N(0, 0.02²)；add（残差）：**零初始化**。

        为什么 add 模式必须零初始化
        ---------------------------
        内容向量是单位向量（‖c‖=1），因此 `proj` 输出的量级 ≈ σ_w·√H：
        σ_w=0.02 时约 0.16，与 `item_emb` 的初始化范数（0.02·√64）**同量级**。
        也就是说，非零初始化下残差增量与嵌入本身一样大，"起点等价基线"
        这个前提就不成立了 —— 那样测出来的差异无法归因于内容。

        零初始化（零权重 + 零 bias）让训练**严格从基线出发**：
        第 0 步的 `table` 与纯基座逐位相同，之后只有在内容确实能降低损失时
        参数才会离开 0。这是适配器/残差微调的常规做法（LoRA 一族同理），
        也是"内容有没有增量"这个消融问题唯一干净的起点。

        梯度不会因此消失：dL/dW = δᵀ · contentᵀ，其中 δ 是上游误差，
        只要 content ≠ 0 且 δ ≠ 0，第一步就有非零梯度。
        """
        if self.mode == "add":
            nn.init.zeros_(self.proj.weight)
        else:
            nn.init.normal_(self.proj.weight, mean=0.0, std=INIT_STD)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    # ------------------------------------------------------------------
    def forward(self, item_weight: torch.Tensor) -> torch.Tensor:
        """`item_weight [V, H]` → 融合后的物品表示 `[V, H]`（PAD 行恒零）。"""
        if item_weight.dim() != 2 or int(item_weight.shape[1]) != self.hidden_size:
            raise ValueError(
                f"item_weight 应为 (V, {self.hidden_size})，"
                f"收到 {tuple(item_weight.shape)}")
        if int(item_weight.shape[0]) != int(self.content.shape[0]):
            raise ValueError(
                f"物品嵌入行数 {int(item_weight.shape[0])} 与内容矩阵 "
                f"{int(self.content.shape[0])} 不一致 —— 行序必须同为物品 idx")

        x = self.content if self.mode == "add" else torch.cat(
            [item_weight, self.content], dim=-1)
        out = self.proj(x)
        if self.mode == "add":
            # 残差口径：table = item_emb + proj(content)。
            # ⚠️ 这是与 concat 的**关键差异**：concat 的 Linear 可以自由重学
            # 整个物品空间（实测训练后融合表与原嵌入余弦仅 0.0097，等于
            # 把 tie-embedding 已有的表示又学了一遍，白付优化代价）；
            # 残差口径下 proj 初始化很小（N(0,0.02²)），训练**起点就等价于
            # 基线**，模型只需要学"内容带来的增量"。要做"内容到底有没有用"
            # 这个判断，残差口径更干净；concat 更接近项目文档的原始设计。
            out = out + item_weight
        out = self.emb_dropout(out)
        # PAD 行乘 0：既保证语义正确（PAD 不该有表示），也保证「全 PAD 序列」
        # 与基线一致（基线的 PAD 行是 padding_idx 且被显式清零）
        return out * self._pad_mask

    # ------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        """本 provider 引入的可训练参数量（E2 报告要用的那个数字）。"""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def config_snapshot(self) -> dict:
        return {
            "mode": self.mode,
            "content_dim": int(self.content_dim),
            "hidden_size": int(self.hidden_size),
            "dropout": float(self.emb_dropout.p),
            "n_params": self.n_params,
        }


# =====================================================================
# 工厂：训练脚本与 checkpoint 重建共用同一条装配路径
# =====================================================================
def build_model(
    arch: str,
    sasrec_cfg,
    n_items: int,
    content_matrix: Optional[torch.Tensor] = None,
    content_mode: str = "concat",
    content_dropout: float = 0.0,
    mi_cfg=None,
):
    """按 `arch` 构造模型，可选注入内容融合 provider。

    为什么要有这个工厂：`scripts/train.py`（训练）与
    `models/checkpoint/io.py`（按 meta 重建）必须走**同一条装配路径**，
    否则「训练时是 concat、加载时建成 add」这类错配会静默发生，
    权重能加载（形状一样）但语义变了。

    参数
    ----
    arch            `"sasrec"` / `"multi_interest"`。
    sasrec_cfg      `SASRecConfig`（multi_interest 时为 None，用 mi_cfg.sasrec）。
    n_items         物品池规模（校验内容矩阵行数用）。
    content_matrix  `[n_items+1, D]`；为 None 时不加融合。
    content_mode    `"concat"` / `"add"`。
    mi_cfg          `MultiInterestConfig`，arch="multi_interest" 时必填。
    """
    from models.sasrec.model import SASRec

    if content_matrix is not None:
        if int(content_matrix.shape[0]) != int(n_items) + 1:
            raise ValueError(
                f"内容矩阵行数 {int(content_matrix.shape[0])} != n_items+1 "
                f"({int(n_items) + 1})；行序约定是「第 v 行 = 物品 idx v」")

    if arch == "multi_interest":
        from models.multi_interest.model import MultiInterestSASRec
        if mi_cfg is None:
            raise ValueError("arch='multi_interest' 必须传 mi_cfg")
        provider = None
        if content_matrix is not None:
            provider = ItemContentFusion(
                content_matrix, hidden_size=int(mi_cfg.sasrec.hidden_size),
                mode=content_mode, dropout=content_dropout)
        return MultiInterestSASRec(mi_cfg, repr_provider=provider)

    if arch == "sasrec":
        provider = None
        if content_matrix is not None:
            provider = ItemContentFusion(
                content_matrix, hidden_size=int(sasrec_cfg.hidden_size),
                mode=content_mode, dropout=content_dropout)
        return SASRec(sasrec_cfg, repr_provider=provider)

    raise ValueError(f"未知 arch {arch!r}（可选：sasrec / multi_interest）")


def load_content_matrix_for_fusion(path: str, n_items: int) -> torch.Tensor:
    """按融合口径装载内容矩阵（复用 `fusion.load_content_matrix` 的行序约定）。

    单独包一层是为了让 checkpoint 重建路径与训练路径**共用同一份**
    装载逻辑（行序错位是本项目最容易静默出错的地方）。
    """
    from models.content_encoder.fusion import load_content_matrix
    return load_content_matrix(path, n_items)
