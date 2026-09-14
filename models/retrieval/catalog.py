# -*- coding: utf-8 -*-
"""在线召回用的**纯计算**层：序列构造、全库打分、逐兴趣 TopK。

为什么单独开一个子包，而不是让 A2 的 adapter 自己算
--------------------------------------------------
`docs/project-structure.md` 的边界表写得很硬：`agents/` **不可以写张量运算**。
理由不是洁癖 —— 张量运算一旦散落到 Agent 里，就会出现两套口径：
训练/评估用 `model.score()`，线上召回用手写的 `user_repr @ table.T`。
两者在「要不要乘 sqrt(H)」「要不要过 out_proj」「candidate 要不要 clamp」
上稍有出入，**分数排序就变了，而且没有任何报错**。

所以本模块只做一件事：**把 `model.score()` 的内部口径逐字复用一遍**，
只是把候选集从「评估时的 101 个」放大成「全库 15,687 个」。

与 `model.score()` 的关系
------------------------
| | `model.score()` | 本模块 |
|---|---|---|
| 候选 | 评估给的 `[B, 1+C]` | 全库 `[1, V]` |
| 用途 | 算 HR/NDCG | 产出召回候选 |
| 口径 | 第 0 列正样本 | 无正样本，纯 top-k |
| 缩放 | 基座不乘 / 多兴趣乘 `sqrt(H)` | **完全相同**（照抄） |

⚠️ 一处刻意的差异：本模块**不在 `candidate_ids` 里放正样本**，
所以返回的 `(index, score)` 里 `index` 就是池内物品索引（1..n_items），
调用方按 `inv_smap` 反查即可。评估路径不经过这里，两条路互不干扰。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

__all__ = [
    "PAD_ITEM",
    "build_input_ids",
    "catalog_scores",
    "n_items_of",
    "resolve_item_table",
    "topk_per_interest",
    "user_vectors",
]

# 池内索引 0 保留给 PAD（与 models/sasrec/model.py 同一口径）
PAD_ITEM = 0


def resolve_item_table(model: torch.nn.Module) -> torch.Tensor:
    """取物品表示表 `[V, H]`，兼容基座与多兴趣两种层级。

    `item_table()` 定义在 `SASRec` 上；`MultiInterestSASRec` 是把基座包在
    `self.backbone` 里（M2.6b 的 `repr_provider` 也刻意只在顶层转发），
    所以这里要沿着 `backbone` 找一层。**不要**写成 `model.item_emb.weight` ——
    那会绕过内容融合 provider，拿到未融合的表示，排序与离线指标不一致。
    """
    if hasattr(model, "item_table"):
        return model.item_table()                       # type: ignore[no-any-return]
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "item_table"):
        return backbone.item_table()                    # type: ignore[no-any-return]
    raise AttributeError(
        f"{type(model).__name__} 上找不到 item_table()（也没有可用的 .backbone）")


def n_items_of(model: torch.nn.Module) -> int:
    """从模型里取出「物品数」（不含 PAD），兼容三种架构。

    为什么不用 `model.cfg.n_items`：多兴趣模型的 cfg 是 `MultiInterestConfig`，
    它的 `n_items` 在 `cfg.sasrec` 里；直接取会 AttributeError。
    """
    for attr in ("cfg", "mi_cfg"):
        cfg = getattr(model, attr, None)
        if cfg is None:
            continue
        inner = getattr(cfg, "sasrec", cfg)
        val = getattr(inner, "n_items", None)
        if val is not None:
            return int(val)
    raise AttributeError("无法从模型上取到 n_items（既没有 cfg.n_items，"
                         "也没有 cfg.sasrec.n_items / mi_cfg）")


def build_input_ids(seq_indices: Sequence[int], cap: int,
                    device: Optional[torch.device] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """把池内索引序列变成**左填充**的 `[1, L]` 输入 + 有效掩码。

    ⚠️ 左填充是硬口径（`models/sasrec/dataset.py`）：模型的
    `causal_pad_allow_mask` 会把 PAD 位置的历史屏蔽掉，右填充会让
    真实物品 attend 到一堆 PAD。返回掩码是为了让调用方能显式校验
    「有效位置是每行的后缀」。
    """
    ids = [int(x) for x in seq_indices if int(x) != PAD_ITEM]
    ids = ids[-int(cap):]                       # 只保留最近 cap 条
    length = int(cap)
    out = torch.zeros(1, length, dtype=torch.long, device=device)
    valid = torch.zeros(1, length, dtype=torch.bool, device=device)
    if ids:
        out[0, length - len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)
        valid[0, length - len(ids):] = True
    return out, valid


def user_vectors(model: torch.nn.Module,
                 input_ids: torch.Tensor) -> torch.Tensor:
    """返回 `[B, K, H]` 用户兴趣向量。

    * 单向量模型（SASRec / GRU4Rec）→ `K = 1`；
    * 多兴趣模型 → `K = num_interests`。

    **缩放与投影照抄 `model.score()`**：多兴趣乘 `sqrt(H)`（量级对齐），
    关闭权重绑定时过 `out_proj`。少做任何一步都会让线上排序与离线指标不一致。
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if hasattr(model, "encode_interests"):
                _, interests = model.encode_interests(input_ids)     # [B, K, H]
                vecs = interests
                backbone = getattr(model, "backbone", None)
                scale = float(getattr(model, "score_scale", 1.0))
                if backbone is not None and getattr(backbone, "out_proj", None) is not None:
                    vecs = backbone.out_proj(vecs)
                return vecs * scale
            _, user_repr = model.encode(input_ids)                   # [B, H]
            if getattr(model, "out_proj", None) is not None:
                user_repr = model.out_proj(user_repr)
            return user_repr.unsqueeze(1)                            # [B, 1, H]
    finally:
        if was_training:
            model.train()


def catalog_scores(model: torch.nn.Module, input_ids: torch.Tensor,
                   *, forbid_mask: Optional[torch.Tensor] = None,
                   item_table: Optional[torch.Tensor] = None,
                   chunk: int = 4096) -> torch.Tensor:
    """对**全库**打分，返回 `[B, K, V]`（V = n_items + 1，含 PAD 列）。

    `forbid_mask` 是 `[V]` bool，True 的位置分数置 `-inf`
    （离线侧 `preprocess.py` 已导出 `forbidden_item_indices`，线上直接复用）。
    `item_table` 允许调用方传入**已缓存**的物品表示表：内容融合时这张表是
    `(V, H+D) @ (H+D, H)` 的一次矩阵乘，每次请求重算纯属浪费
    （推理期 `item_emb.weight` 冻结，表是常量）。
    `chunk` 控制分块大小：单用户全库内积在 6GB 卡上完全放得下，
    但分块能让它在 CPU 上也不炸，且显存占用与物品数解耦。
    """
    with torch.no_grad():
        vecs = user_vectors(model, input_ids)                       # [B, K, H]
        table = resolve_item_table(model) if item_table is None else item_table
        # 表可能被 repr_provider（内容融合）变换过；与 score() 走同一入口
        table = table.to(device=vecs.device, dtype=vecs.dtype)
        v = table.shape[0]

        out = torch.empty(vecs.shape[0], vecs.shape[1], v,
                          dtype=vecs.dtype, device=vecs.device)
        for start in range(0, v, int(chunk)):
            stop = min(v, start + int(chunk))
            # [B, K, H] x [c, H]^T -> [B, K, c]
            out[:, :, start:stop] = torch.einsum(
                "bkh,ch->bkc", vecs, table[start:stop])

        out[:, :, PAD_ITEM] = float("-inf")        # 永不出 PAD
        if forbid_mask is not None:
            fm = forbid_mask.to(device=out.device, dtype=torch.bool)
            out = out.masked_fill(fm.view(1, 1, -1), float("-inf"))
        return out


def topk_per_interest(scores: torch.Tensor, k: int, *,
                      exclude: Optional[Sequence[Sequence[int]]] = None
                      ) -> list[list[list[tuple[int, float]]]]:
    """`[B, K, V]` → `[b][k] = [(index, score), ...]` 每个用户每个兴趣的 TopK。

    `exclude[b]` 给第 b 个用户要排除的物品（既看过的）。
    排除方式是把对应位分数置 `-inf` 而不是删除候选 —— 保持张量形状固定，
    避免"某个兴趣被排空了导致下标错位"。被排除的位置在返回时会被过滤掉，
    所以某个兴趣返回的条数可能少于 `k`。
    """
    b, kk, v = scores.shape
    work = scores
    if exclude:
        work = scores.clone()
        for bi, ids in enumerate(exclude[:b]):
            if not ids:
                continue
            idx = torch.tensor([int(i) for i in ids if 0 <= int(i) < v],
                               dtype=torch.long, device=work.device)
            if idx.numel():
                work[bi, :, idx] = float("-inf")

    k_eff = max(1, min(int(k), v))
    vals, idxs = torch.topk(work, k_eff, dim=-1)            # [B, K, k]
    vals = vals.detach().cpu()
    idxs = idxs.detach().cpu()

    out: list[list[tuple[int, float]]] = []
    for bi in range(b):
        per_interest: list[tuple[int, float]] = []
        for ki in range(kk):
            pairs = [(int(idxs[bi, ki, j]), float(vals[bi, ki, j]))
                     for j in range(k_eff)]
            # 过滤掉被 -inf 排除的位置
            pairs = [(i, s) for i, s in pairs if s != float("-inf")]
            per_interest.append(pairs)
        out.append(per_interest)
    return out
