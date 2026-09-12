# -*- coding: utf-8 -*-
"""内容融合（M2.6）：排序侧加权 + 内容相似度打分。

定位与边界
----------
本模块只做「打分」这一层的事：

* `load_content_matrix`  把 `data/features/content_vec_512.npy` 装载成
  以**模型物品 idx** 为下标的矩阵（第 0 行 = PAD 恒零）；
* `ContentScorer`        用用户历史的内容向量均值当「用户内容画像」，
  与候选内容向量算余弦相似度，得到**内容分**；
* `fusion_score`         行为分与内容分按 7:3（普通）/ 5:5（冷启动）加权，
  签名与 `docs/project-structure.md` 的契约一致；
* `ContentFusedModel`    把任一具备 `score(input_ids, candidate_ids)` 契约的
  行为模型（SASRec / 多兴趣）包一层，对外仍是同一 score 契约。

⚠️ 训练侧约定：**融合只发生在打分/评估时，训练仍只训行为基座**
（BCE 只看行为分）。理由：
1. checkpoint 完全复用 —— 同一份行为权重可以在不同融合权重下复评，
   E2 消融不需要为每个融合权重重训一次；
2. 内容通路不参与训练梯度，行为基座的收敛动态与 E2_1 纯基线完全一致，
   消融差异可以完全归因于融合本身；
3. 内容向量是离线 PCA 产物（已归一化），没有可学参数，训练它没有意义。

冷启动口径
----------
`n_inter`（用户有效交互数）低于 `COLD_START_THRESHOLD=10` 时视为冷启动，
权重切 5:5（`configs/model.yaml: content_fusion.w_content_cold`）。
注意评估口径的冷启动实验（E3，year>=2021 新番 holdout）是**显式**传
0.5:0.5 跑的，与本函数的自动切换是两条路径 —— 后者主要服务在线服务里
交互数少的新用户。
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

# 数据口径硬约束（与 configs/data.yaml、阶段一定稿一致）：
# 有效交互数低于该阈值的用户按冷启动处理。
COLD_START_THRESHOLD = 10

DEFAULT_W_BEHAVIOR = 0.7
DEFAULT_W_CONTENT = 0.3
DEFAULT_W_CONTENT_COLD = 0.5


def anchor_features_dir() -> str:
    """内容特征目录（data/features/），与工作目录无关。"""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "data", "features")


def load_content_matrix(
    path: str,
    n_items: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    """装载内容向量矩阵，对齐模型物品 idx。

    `content_vec_512.npy` 的行序约定（阶段一验收定稿）：
    **第 i-1 行 = 模型物品 idx i**（idx 从 1 开始，0 是 PAD）。

    参数
    ----
    path    npy 文件路径，形状须为 `(n_items, D)`。
    n_items 物品池规模（15,687）。
    device  目标设备；None 则留在 CPU。

    返回
    ----
    `float32` 张量 `(n_items + 1, D)`：第 0 行全零（PAD 的内容向量），
    其余行重新做 L2 归一化（防御性 —— 阶段一产物本已归一，但本模块
    的余弦计算依赖这一点，不能靠「上游应该没错」）。
    """
    vec = np.load(path)
    if vec.ndim != 2:
        raise ValueError(f"内容矩阵应为 2 维，收到 shape={vec.shape}")
    if vec.shape[0] != int(n_items):
        raise ValueError(
            f"内容矩阵行数 {vec.shape[0]} 与物品池 {n_items} 不一致 —— "
            "行序约定是「第 i-1 行 = 物品 idx i」，错位会让全部内容分错乱")
    vec = vec.astype(np.float32)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vec = vec / norms

    mat = np.zeros((int(n_items) + 1, vec.shape[1]), dtype=np.float32)
    mat[1:] = vec
    t = torch.from_numpy(mat)
    if device is not None:
        t = t.to(device)
    return t


class ContentScorer(nn.Module):
    """内容分打分器：用户内容画像（历史向量均值）与候选的余弦相似度。

    score 契约与 evaluator 完全一致：
        score(input_ids[B, L], candidate_ids[B, 1+C]) -> [B, 1+C]
    第 0 列恒为正样本 —— 这一条由调用方（build_eval_data）保证，
    本模块对每一列一视同仁地打分。

    空历史（全 PAD）用户画像为零向量，与任何候选的点积都是 0，
    即「没有任何内容证据时内容分中性」，不会 NaN 也不会乱猜。
    """

    def __init__(self, content_matrix: Tensor):
        super().__init__()
        if content_matrix.dim() != 2:
            raise ValueError("content_matrix 必须是 (n_items+1, D) 二维张量")
        # 防御性归一：余弦计算要求候选向量也是单位向量，不能依赖上游
        mat = content_matrix.float()
        norms = mat.norm(dim=-1, keepdim=True)
        mat = mat / norms.clamp(min=1e-12)
        # 零行（PAD、空向量）保持为零，不能被 1e-12 放大成噪声
        mat = torch.where(norms > 1e-12, mat, torch.zeros_like(mat))
        self.register_buffer("content", mat)

    @property
    def dim(self) -> int:
        return int(self.content.shape[1])

    def user_profile(self, input_ids: Tensor) -> Tensor:
        """用户内容画像 `(B, D)`：有效历史物品内容向量的均值（L2 归一）。"""
        mask = (input_ids > 0).unsqueeze(-1).float()          # [B, L, 1]
        hist = self.content[input_ids.clamp(min=0)]            # [B, L, D]
        denom = mask.sum(dim=1).clamp(min=1.0)                 # [B, 1]
        profile = (hist * mask).sum(dim=1) / denom             # [B, D]
        norms = profile.norm(dim=-1, keepdim=True)
        # 空历史（norm=0）不归一，保持零向量：与任何候选点积恒 0
        return profile / norms.clamp(min=1e-12)

    def score(self, input_ids: Tensor, candidate_ids: Tensor) -> Tensor:
        """内容分 `(B, 1+C)`：画像与候选内容向量的余弦相似度。"""
        profile = self.user_profile(input_ids)                 # [B, D]
        cand = self.content[candidate_ids]                     # [B, 1+C, D]
        return torch.einsum("bd,bcd->bc", profile, cand)


def fusion_score(
    behavior: Tensor,
    content: Tensor,
    n_inter: Tensor,
    w_behavior: float = DEFAULT_W_BEHAVIOR,
    w_content: float = DEFAULT_W_CONTENT,
    w_content_cold: float = DEFAULT_W_CONTENT_COLD,
    cold_threshold: int = COLD_START_THRESHOLD,
) -> Tensor:
    """按交互数自动切权的排序侧融合（契约签名，勿改语义）。

    参数
    ----
    behavior  `(B, C)` 行为模型分数。
    content   `(B, C)` 内容分（同列序 —— 第 0 列都是正样本）。
    n_inter   `(B,)`  每用户有效交互数（长整型）。
    w_*       普通用户权重 7:3；交互数 < cold_threshold 的用户切 5:5。

    返回
    ----
    `(B, C)` 最终分数。逐样本独立选权，同一个 batch 里冷热用户可以混排。

    注意：纯函数、无 IO、无可学参数 —— `models/` 层的分层约束（勿破坏）。
    """
    if behavior.shape != content.shape:
        raise ValueError(
            f"behavior {tuple(behavior.shape)} 与 content {tuple(content.shape)} 形状不一致")
    if not (0.0 <= float(w_behavior) <= 1.0):
        raise ValueError(f"w_behavior 必须在 [0,1]，收到 {w_behavior}")

    wb = torch.as_tensor(float(w_behavior), dtype=behavior.dtype, device=behavior.device)
    wc = torch.as_tensor(float(w_content), dtype=behavior.dtype, device=behavior.device)
    wc_cold = torch.as_tensor(
        float(w_content_cold), dtype=behavior.dtype, device=behavior.device)
    if float(w_behavior) + float(w_content) != 1.0:
        raise ValueError("普通用户权重 w_behavior + w_content 必须为 1")
    if not (0.0 <= float(w_content_cold) <= 1.0):
        raise ValueError(f"w_content_cold 必须在 [0,1]，收到 {w_content_cold}")

    cold = (torch.as_tensor(n_inter, device=behavior.device) < int(cold_threshold))
    wb_row = torch.where(cold, 1.0 - wc_cold, wb)
    wc_row = torch.where(cold, wc_cold, wc)
    return wb_row.unsqueeze(-1) * behavior + wc_row.unsqueeze(-1) * content


class ContentFusedModel(nn.Module):
    """行为模型 + 内容打分器的融合包装。

    对外仍是 evaluator 契约：`score(input_ids, candidate_ids) -> [B, 1+C]`。
    `backbone` 必须实现同一契约（SASRec / MultiInterestSASRec 均可）。

    ⚠️ 本包装**不参与训练**：`fit()` 应直接训 `backbone`（融合只在
    评估/打分时发生，见模块 docstring 的三条理由）。
    """

    def __init__(self, backbone: nn.Module, scorer: ContentScorer,
                 w_behavior: float = DEFAULT_W_BEHAVIOR,
                 w_content: float = DEFAULT_W_CONTENT,
                 w_content_cold: float = DEFAULT_W_CONTENT_COLD,
                 cold_threshold: int = COLD_START_THRESHOLD):
        super().__init__()
        self.backbone = backbone
        self.scorer = scorer
        self.w_behavior = float(w_behavior)
        self.w_content = float(w_content)
        self.w_content_cold = float(w_content_cold)
        self.cold_threshold = int(cold_threshold)

    @property
    def n_params(self) -> int:
        # 融合本身零参数；只统计 backbone（scorer 的内容矩阵是 buffer）
        return int(sum(p.numel() for p in self.backbone.parameters()))

    def score(self, input_ids: Tensor, candidate_ids: Tensor) -> Tensor:
        behavior = self.backbone.score(input_ids, candidate_ids)
        content = self.scorer.score(input_ids, candidate_ids)
        n_inter = (input_ids > 0).sum(dim=1)                   # [B] 可见历史长度
        return fusion_score(
            behavior, content, n_inter,
            w_behavior=self.w_behavior, w_content=self.w_content,
            w_content_cold=self.w_content_cold, cold_threshold=self.cold_threshold)
