# -*- coding: utf-8 -*-
"""GRU4Rec 基线 —— `models/baselines/gru4rec.py`

论文：Hidasi et al., *Session-based Recommendations with Recurrent Neural
Networks* (ICLR 2016)。这里实现的是**与本项目评估协议对齐**的版本，
不是原论文的 session-parallel mini-batch 训练（那是训练技巧，不是模型）。

结构（`docs/evaluation-plan.md` §4：隐藏层 64、1 层 GRU、同嵌入维度）
------------------------------------------------------------------
::

    item_emb [V, H]  ->  Dropout  ->  GRU(H -> H, num_layers)  ->  取末位
      ->  dense: Linear(H, H, bias=False)  ->  与 item_emb 点积打分（权重绑定）

`dense` + 权重绑定这一层是 RecBole `GRU4Rec.full_sort_predict` 的口径：
输出层复用输入嵌入，与本文模型（SASRec 的 tie-embedding）一致，
这样"输出层参数量"不会变成一个说不清的额外变量。

⚠️ 为什么**不能**用 `pack_padded_sequence`（本机实测踩到的坑）
---------------------------------------------------------------
`torch.nn.utils.rnn.pack_padded_sequence` 的实现前提是**右填充**：
它按 `lengths` 取每行**开头**的 L 个位置（`data[:, :length]`）。
本项目的序列是**左填充**（有效物品在行的**末尾**），所以直接调用它
取到的是每行最前面的若干 PAD —— 结果既不等于"只喂有效位置"，
又会随"这一行垫了多少 PAD"而变化。

实测（H=16、1 层 GRU、同一段 3 个物品的历史）：

| 实现 | 「3 个有效位置」手算 | pack_padded_sequence |
|---|---|---|
| L=4（垫 1 个 PAD） | 基准 | 偏离 1.5e-2 |
| L=6（垫 3 个 PAD） | 与基准**逐位相同** | 与基准偏离 1.7e-2，且两次互相不同 |

CPU 与 CUDA 结果一致，说明这不是内核问题，是**用法错误**。
更危险的是它不报错、指标也算得出来。

本实现改为「滚动对齐 + 掩码取末位」：

    pad = L - length
    dense[i] = roll(seq[i], -pad[i])        # 有效物品滚到行首，PAD 挤到行尾
    out = GRU(dense[:, :T])                 # T = 本 batch 的最大有效长度
    user_repr[i] = out[i, length[i] - 1]    # 取"最后一个有效位置"，尾部 PAD 的输出不读

于是 `dense` 的形状只取决于「本 batch 的有效长度」，与调用方垫了多少
PAD **完全无关**（同一段历史 ⇒ 同一个张量形状 ⇒ 逐位相同的表示）。
`tests/test_baselines/test_baselines.py::test_repr_invariant_to_left_padding`
就是把这条钉住的。

`roll` 的正确性依赖一条明确前提：**每行的有效位置是它的后缀**。
本项目的两条上游都满足（`SlidingWindowDataset` 左填充、
`build_eval_inputs` 物理剥离新番后仍是后缀），但这里仍显式校验一次 ——
不满足时立刻报错，而不是算出一个看起来正常的错数字（E3 冷启动若改成
"中间挖空"式剥离就会触发这条）。

训练协议
--------
与本文模型**完全共用** `models/sasrec/train.py::fit()`：
1 正 1 负 BCE、同一负采样实现、同一早停口径（验证集 `ndcg@10`、patience=10）。
本文件不含任何训练循环 —— 那是 `fit()` 的职责（`models/` 层无 IO、无重复实现）。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch
import torch.nn as nn

__all__ = ["PAD_ITEM", "GRU4RecConfig", "GRU4Rec"]

# 池内索引 0 保留给 PAD，物品从 1 开始（与 models/data/negatives.py 同一口径）
PAD_ITEM = 0

# 与 models/sasrec/config.py 的 _TARGET_SLOTS 同源：留一法下 val/test 各占 1 位。
# 两处都要改的口径，所以两处都写断言/注释（见该文件的说明）。
_TARGET_SLOTS = 2


# =====================================================================
# 配置
# =====================================================================
@dataclass(frozen=True)
class GRU4RecConfig:
    """GRU4Rec 结构超参。

    字段刻意与 `SASRecConfig` 的**公共子集**同名（hidden_size / dropout /
    max_seq_len / n_items），这样 `scripts/run_baselines.py` 能用同一份
    `configs/model.yaml` 的 `model` 段构造基线，"同嵌入维度、同 dropout"
    这条公平性约束由代码保证，而不是靠人肉对齐两份配置文件。

    参数
    ----
    n_items     物品池规模（不含 PAD）；嵌入表开 `n_items + 1` 行。
    max_seq_len 完整序列长度上限（含 val/test 两位），只用于校验输入长度。
    hidden_size 隐层维度 H，同时是嵌入维度（= 64，与本文模型一致）。
    num_layers  GRU 层数，默认 1（`docs/evaluation-plan.md` §4 口径）。
    dropout     嵌入与（多层时）层间的 dropout 比例。
    """

    n_items: int
    max_seq_len: int = 50
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2

    # ---------------- 派生属性 ----------------
    @property
    def input_cap(self) -> int:
        """单次前向的输入长度上限 = `max_seq_len - 2`（val/test 各占 1 位）。"""
        return int(self.max_seq_len) - _TARGET_SLOTS

    @property
    def vocab_size(self) -> int:
        """嵌入表行数（含 PAD）。"""
        return int(self.n_items) + 1

    # ---------------- 校验 ----------------
    def __post_init__(self) -> None:
        if int(self.n_items) <= 0:
            raise ValueError(f"n_items 必须为正，收到 {self.n_items}")
        if int(self.max_seq_len) < 4:
            raise ValueError(f"max_seq_len={self.max_seq_len} 太小")
        if int(self.hidden_size) <= 0:
            raise ValueError(f"hidden_size 必须为正，收到 {self.hidden_size}")
        if int(self.num_layers) <= 0:
            raise ValueError(f"num_layers 必须为正，收到 {self.num_layers}")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"dropout 应在 [0, 1)，收到 {self.dropout}")

    # ---------------- 构造 ----------------
    @classmethod
    def from_dict(cls, model_cfg: dict, n_items: int) -> "GRU4RecConfig":
        """从配置字典构造；只取本 dataclass 认得的键（多余键忽略）。

        与 `SASRecConfig.from_dict` 同一条约定：`configs/model.yaml` 的
        `model` 段同时喂给基座与基线，各自取自己认得的字段。
        """
        known = {f.name for f in fields(cls)} - {"n_items"}
        kwargs = {k: v for k, v in (model_cfg or {}).items() if k in known}
        return cls(n_items=int(n_items), **kwargs)


# =====================================================================
# 模型
# =====================================================================
class GRU4Rec(nn.Module):
    """GRU4Rec 基线。接口与 `SASRec` 对齐（`encode` / `score` / `n_params`）。

    用法
    ----
    >>> cfg = GRU4RecConfig(n_items=15687, hidden_size=64, num_layers=1)
    >>> model = GRU4Rec(cfg)
    >>> x = torch.zeros(4, 48, dtype=torch.long)
    >>> cand = torch.randint(1, 15688, (4, 101))
    >>> model.score(x, cand).shape
    torch.Size([4, 101])
    """

    def __init__(self, cfg: GRU4RecConfig):
        super().__init__()
        self.cfg = cfg
        h = int(cfg.hidden_size)

        # padding_idx=0：该行恒为 0 且**不接收梯度**。两件事都重要 ——
        # 恒为 0 让"零输入"这个前提成立（配合 pack_padded_sequence 用），
        # 不接收梯度保证 PAD 行不会漂移出零。
        self.item_emb = nn.Embedding(cfg.vocab_size, h, padding_idx=PAD_ITEM)
        self.emb_dropout = nn.Dropout(float(cfg.dropout))
        self.gru = nn.GRU(
            h, h, num_layers=int(cfg.num_layers), batch_first=True,
            # nn.GRU 的层间 dropout 只在 num_layers > 1 时生效；
            # 单层时 PyTorch 会告警，所以这里显式置 0。
            dropout=(float(cfg.dropout) if int(cfg.num_layers) > 1 else 0.0),
        )
        # 输出投影：与输入嵌入绑定（RecBole full_sort_predict 口径）
        self.dense = nn.Linear(h, h, bias=False)

        self.reset_parameters()

    # ------------------------------------------------------------------
    def reset_parameters(self) -> None:
        """物品嵌入与输出投影按项目统一口径 N(0, 0.02²) 初始化。

        `nn.Embedding(padding_idx=...)` 构造时会把 PAD 行置 0，而
        `normal_` 会把它覆盖掉 —— 所以必须在初始化**之后再置一次 0**
        （与 `SASRec.reset_parameters` 同一个坑）。

        GRU 自身的参数保留 PyTorch 默认初始化（U(-1/√H, 1/√H)），
        与原论文/RecBole 一致；换成 N(0, 0.02²) 会让门控在小 H 下
        几乎不激活，训练早期明显更慢。
        """
        std = 0.02
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=std)
        nn.init.normal_(self.dense.weight, mean=0.0, std=std)
        with torch.no_grad():
            self.item_emb.weight[PAD_ITEM].zero_()

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def encode(
        self,
        seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """把历史序列编码成用户表示 `[B, H]`。

        `mask` 参数为**接口对齐**保留（`SASRec.encode(seq, mask)` 同形参），
        GRU4Rec 不参与注意力，有效位置由左填充约定 + `mask` 共同确定。

        实现见模块 docstring：**滚动对齐后取末位**，而不是
        `pack_padded_sequence`（后者假设右填充，在本项目会静默算错）。
        """
        if seq.dim() != 2:
            raise ValueError(f"seq 应为 (B, L)，实际 {tuple(seq.shape)}")
        b, length = seq.shape
        if length > int(self.cfg.input_cap):
            raise ValueError(
                f"输入长度 {length} 超过上限 {self.cfg.input_cap}"
                f"（= max_seq_len {self.cfg.max_seq_len} 减去 {_TARGET_SLOTS} 个留出位）")
        if b == 0:
            return torch.zeros((0, int(self.cfg.hidden_size)),
                               dtype=self.item_emb.weight.dtype,
                               device=self.item_emb.weight.device)
        if length == 0:
            return torch.zeros((b, int(self.cfg.hidden_size)),
                               dtype=self.item_emb.weight.dtype,
                               device=self.item_emb.weight.device)

        valid = (seq != PAD_ITEM) if mask is None else \
            (mask.to(torch.bool) & (seq != PAD_ITEM))
        lengths = valid.sum(dim=1).to(torch.int64)          # [B]
        pad_count = length - lengths                        # [B]

        idx = torch.arange(length, device=seq.device).unsqueeze(0)        # [1, L]
        # 前提校验：有效位置必须是每行的后缀（滚动对齐的正确性依赖它）。
        # 不满足时报错而不是继续 —— 否则会算出一个"看起来正常"的错表示。
        if not torch.equal(valid, idx >= pad_count.unsqueeze(1)):
            raise ValueError(
                "GRU4Rec 要求每行的有效位置是它的**后缀**（左填充约定），"
                "但收到的 mask 在行内不连续；请检查上游的填充/剥离逻辑")
        if bool((lengths > 0).all()):
            t_max = int(lengths.max())
        else:
            # 存在整行 PAD 时，至少要留 1 步，否则 GRU 收到空张量。
            t_max = max(1, int(lengths.max()))

        # 把有效物品滚到行首（PAD 挤到行尾），尾部 PAD 的输出之后不会被读取。
        gather_idx = (idx + pad_count.unsqueeze(1)) % length
        dense = torch.gather(seq, 1, gather_idx)[:, :t_max]               # [B, T]

        emb = self.emb_dropout(self.item_emb(dense))                      # [B, T, H]
        out, _ = self.gru(emb)

        last = (lengths.clamp(min=1) - 1)
        user_repr = out[torch.arange(b, device=out.device), last]         # [B, H]
        return torch.where(lengths.unsqueeze(1) > 0, user_repr,
                           torch.zeros_like(user_repr))

    def forward(self, seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`nn.Module` 的习惯入口：返回用户表示 `[B, H]`。"""
        return self.encode(seq, mask)

    # ------------------------------------------------------------------
    # 打分（与 evaluator 的 score_fn 契约一致）
    # ------------------------------------------------------------------
    def score(
        self,
        input_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """对候选物品打分：`[B, L] x [B, C] -> [B, C]`。

        与 `models/eval/evaluator.py` 的 `score_fn` **逐字一致**：
        第 0 列是正样本、分越大越相关。因此 E1 里 GRU4Rec 与本文模型
        走的是**同一套**评估代码（`docs/evaluation-plan.md` 4.1）。
        """
        emb_weight = self.item_emb.weight                   # [V, H]（绑定输出层）
        user_repr = self.encode(input_ids)                  # [B, H]
        user_repr = self.dense(user_repr)

        if emb_weight.device != candidate_ids.device:
            candidate_ids = candidate_ids.to(emb_weight.device)
        cand = candidate_ids.clamp(min=PAD_ITEM, max=int(self.cfg.n_items))
        item_vec = emb_weight[cand]                         # [B, C, H]
        return (item_vec * user_repr.unsqueeze(1)).sum(dim=-1)      # [B, C]

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        """可训练参数量（E1 表里"参数量"一列要用，证明对比是等预算的）。"""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def config_snapshot(self) -> dict:
        """结构摘要，供 checkpoint 与实验报告落盘。

        必须带 `arch`：`models/checkpoint/io.py::load_model_from_checkpoint`
        靠它决定用哪个类重建 —— 缺了它 GRU4Rec 的权重会被当成 SASRec 加载，
        然后报一堆形状不匹配（比静默错好，但仍然是不必要的返工）。
        """
        return {
            "arch": "gru4rec",
            "n_items": int(self.cfg.n_items),
            "max_seq_len": int(self.cfg.max_seq_len),
            "input_cap": int(self.cfg.input_cap),
            "hidden_size": int(self.cfg.hidden_size),
            "num_layers": int(self.cfg.num_layers),
            "dropout": float(self.cfg.dropout),
            "share_item_emb": True,
            "n_params": self.n_params,
        }

    def expected_n_params(self) -> int:
        """按公式算出的参数量，用于单测对拍（防止"参数量对齐"只是个说法）。

        = 嵌入 V·H  +  GRU 单层 6H² + 6H  +  输出投影 H²
        """
        h = int(self.cfg.hidden_size)
        L = int(self.cfg.num_layers)
        emb = int(self.cfg.vocab_size) * h
        gru = L * (3 * h * h + 3 * h + 3 * h * h + 3 * h)
        dense = h * h
        return int(emb + gru + dense)

    def __repr__(self) -> str:                       # pragma: no cover - 诊断用
        return (f"GRU4Rec(n_items={self.cfg.n_items}, hidden={self.cfg.hidden_size}, "
                f"layers={self.cfg.num_layers}, n_params={self.n_params:,})")
