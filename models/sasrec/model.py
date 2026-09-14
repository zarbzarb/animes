# -*- coding: utf-8 -*-
"""SASRec 基座网络 —— `models/sasrec/model.py`

结构（与原论文「Self-Attentive Sequential Recommendation」一致）
----------------------------------------------------------------
    嵌入(item + position) -> Dropout
      -> N x [因果自注意力 → Add&LN → FFN → Add&LN]
      -> 取「最后一个有效位置」的隐状态作为用户表示

刻意保留的三个与原论文一致的细节（改则与公开实现不可比）：

* 嵌入乘 `sqrt(H)`（`scale_emb`）；
* FFN 输出乘 `1/sqrt(2)`（`scale_ffn`）；
* **post-LN**：`x = LN(x + Dropout(sublayer(x)))`，而不是 pre-LN。

**输入/输出嵌入共享权重**（tie embedding）。这既省一半参数，也让
「输出层没见过某物品」与「输入层没见过某物品」不会出现两种语义。
消融时可用 `share_item_emb=false` 单独关掉。

物品表示注入（M2.6b 内容融合的挂载点）
-------------------------------------
构造时可传 `repr_provider`，它把行为嵌入变换成最终使用的物品表示：

    table = repr_provider(item_emb.weight)      # [V, H]

`encode()` 与 `score()` **都经 `item_table()` 取表**，因此输入侧与输出侧
永远一致。不需要融合时传 None，行为与 M2.4 **逐位相同**（有回归测试）。
这条设计让「候选侧内容融合」不必改动 SASRec 主体，也让 M2.5 的多兴趣模型
能原样复用（`MultiInterestSASRec(cfg, repr_provider=...)`）。

⚠️ 取表后用 `F.embedding(..., padding_idx=0)` 而不是裸张量索引：前者保留
「PAD 行不接收梯度」的保障，否则 PAD 行会随训练漂移出零。

接口契约（Agent 与评估器都依赖）
--------------------------------
::

    encode(seq, mask=None) -> (hidden [B, L, H], user_repr [B, H])
    score(input_ids, candidate_ids) -> scores [B, 1 + C]

`score` 的签名与 `models/eval/evaluator.py` 要求的 `score_fn` **完全一致**
（第 0 列恒为正样本、分越大越相关），因此评估器可以直接吃这个方法，
不需要任何适配层 —— 这正是「所有模型共用同一套评估代码」这条公平性
约束能成立的前提（`docs/evaluation-plan.md` 4.1）。

关于 mask：两个必须写清楚的坑
-----------------------------
**坑 1：全 PAD 的行会算出 NaN。**
`SlidingWindowDataset` 展开滑动窗口时，第 0 个窗口的输入是空序列
（`t=0` 时没有任何历史），整行都是 PAD；推而广之，左填充下**任何 PAD
位置**作为 query 时，它能看到的历史（`k <= q`）也全是 PAD。
若此时把所有 key 都 mask 成 `-inf`，softmax 的分母为 0 → NaN。
NaN 会沿着残差继续往下传，最终污染整个 batch 的 hidden。

因此本模块的 mask 规则是：

    allow[q][k] = (k <= q) and (seq[k] != PAD or k == q)

即**对角永远打开**（query 至少能看自己），从而保证 softmax 恒有非 -inf 项。
这条规则同时解决了两件事：
* 不产生 NaN；
* 真实物品位置**不会** attend 到左侧的 PAD（否则填充会污染语义）。

**坑 2：PAD 位置的输出没有意义，不能拿来当用户表示。**
用户表示取「最后一个**有效**位置」的隐状态。由于 `dataset.py` 采用左填充，
这在数值上等价于 `hidden[:, -1]`（最后一个位置恒为最新物品），
但这里仍显式按有效位置取，这样即使将来改成右填充也不会静默算错。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sasrec.config import SASRecConfig

__all__ = ["SASRec", "SASRecBlock", "causal_pad_allow_mask"]

# 池内索引 0 保留给 PAD，物品从 1 开始（与 models/data/negatives.py 同一口径）
PAD_ITEM = 0


# =====================================================================
# mask
# =====================================================================
def causal_pad_allow_mask(seq: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """返回 `[B, L, L]` 的布尔 mask，True 表示「query 可以看到这个 key」。

    规则见模块 docstring：`(k <= q) and (seq[k] 有效 or k == q)`。

    为什么要同时用 `seq` 和 `valid` 两个输入
    ----------------------------------------
    `valid` 允许调用方显式声明有效位置（例如将来对序列做物理剥离时，
    有效位置可能不连续）；不传时上层会用 `seq != PAD_ITEM` 推导。
    两者都传进来可以避免在注意力里再做一次比较。

    参数
    ----
    seq    `[B, L]` long，物品池内索引（PAD = 0）。仅用于取设备与形状。
    valid  `[B, L]` bool，True = 该位置是真实物品。
    """
    if seq.dim() != 2:
        raise ValueError(f"seq 应为 (B, L)，实际 {tuple(seq.shape)}")
    if valid.shape != seq.shape:
        raise ValueError(
            f"valid 形状 {tuple(valid.shape)} 与 seq {tuple(seq.shape)} 不一致")

    b, length = seq.shape
    device = seq.device

    # 因果：下三角为 True（含对角线）——允许看自己和过去，不看未来
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))

    # key 有效：True 表示这个 key 是真实物品（PAD 的 key 不该被 attend）
    key_ok = valid.unsqueeze(1)                      # [B, 1, L]
    # 对角线豁免：即便自己是 PAD，也允许"看自己"，保证 softmax 恒有可见项
    eye = torch.eye(length, dtype=torch.bool, device=device)

    return causal.unsqueeze(0) & (key_ok | eye.unsqueeze(0))


# =====================================================================
# 注意力块
# =====================================================================
class SASRecBlock(nn.Module):
    """一个 Transformer 块（因果自注意力 + FFN），post-LN。

    只接受已经算好的 `[B, L, L]` allow mask —— mask 与层数无关，
    在 `SASRec.encode()` 里算一次复用 N 层，避免每层重复构造 `L x L` 矩阵。
    """

    def __init__(self, cfg: SASRecConfig):
        super().__init__()
        h = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_heads)
        self.head_dim = h // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(h, h)
        self.k_proj = nn.Linear(h, h)
        self.v_proj = nn.Linear(h, h)
        self.out_proj = nn.Linear(h, h)

        self.attn_dropout = nn.Dropout(float(cfg.dropout))
        self.resid_dropout = nn.Dropout(float(cfg.dropout))

        inner = h * int(cfg.ffn_ratio)
        self.ffn_in = nn.Linear(h, inner)
        self.ffn_out = nn.Linear(inner, h)
        self.act = nn.GELU()          # 原论文用 GELU（不是 ReLU）
        self.ffn_dropout = nn.Dropout(float(cfg.dropout))

        self.norm1 = nn.LayerNorm(h, eps=1e-8)
        self.norm2 = nn.LayerNorm(h, eps=1e-8)

        # 原论文的 FFN 缩放：输出乘 1/sqrt(2)
        self.ffn_scale = (1.0 / math.sqrt(2.0)) if cfg.scale_ffn else 1.0

    # ------------------------------------------------------------------
    def _attention(self, x: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
        """多头因果自注意力。`x` 为 `[B, L, H]`，`allow` 为 `[B, L, L]` bool。"""
        b, length, h = x.shape
        nh, hd = self.num_heads, self.head_dim

        # [B, L, H] -> [B, heads, L, head_dim]
        q = self.q_proj(x).view(b, length, nh, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, length, nh, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, length, nh, hd).transpose(1, 2)

        # 缩放点积：先乘 scale 再 matmul，与原论文/RecBole 的实现一致
        scores = torch.matmul(q * self.scale, k.transpose(-2, -1))   # [B, nh, L, L]

        # 屏蔽：allow 是 [B, L, L]，补一个 head 维广播到 [B, 1, L, L]
        scores = scores.masked_fill(
            ~allow.unsqueeze(1), torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)                                  # [B, nh, L, hd]
        out = out.transpose(1, 2).contiguous().view(b, length, h)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
        # post-LN：x = LN(x + Dropout(sublayer(x)))
        x = self.norm1(x + self.resid_dropout(self._attention(x, allow)))

        ffn = self.ffn_out(self.ffn_dropout(self.act(self.ffn_in(x))))
        if self.ffn_scale != 1.0:
            ffn = ffn * self.ffn_scale
        x = self.norm2(x + self.resid_dropout(ffn))
        return x


# =====================================================================
# SASRec
# =====================================================================
class SASRec(nn.Module):
    """SASRec 基座。构造参数只用 `SASRecConfig`，不散落超参。

    用法
    ----
    >>> cfg = SASRecConfig(n_items=15687, hidden_size=64, num_layers=2, num_heads=2)
    >>> model = SASRec(cfg)
    >>> x = torch.zeros(4, 48, dtype=torch.long)     # 左填充的历史
    >>> hidden, user_repr = model.encode(x)
    >>> hidden.shape, user_repr.shape
    (torch.Size([4, 48, 64]), torch.Size([4, 64]))
    >>> cand = torch.randint(1, 15688, (4, 101))     # 第 0 列应为正样本
    >>> model.score(x, cand).shape
    torch.Size([4, 101])
    """

    def __init__(
        self,
        cfg: SASRecConfig,
        repr_provider: Optional[nn.Module] = None,
    ):
        """参数
        ----
        cfg            结构配置。
        repr_provider  可选的「物品表示提供者」（M2.6b 内容融合用）。
                       传入时，**输入嵌入与输出打分都改用它的输出**：
                           table = repr_provider(item_emb.weight)   # [V, H]
                       即物品表示 = f(可学的行为嵌入, 固定的内容向量)。
                       provider 必须保证第 0 行（PAD）为零向量。
                       为 None 时行为与不加融合**逐位一致**（回归保护）。
        """
        super().__init__()
        self.cfg = cfg
        h = int(cfg.hidden_size)
        self.repr_provider = repr_provider

        # ---------- 嵌入 ----------
        # padding_idx=0：该行初始化为全 0，且训练时不更新梯度。
        # 位置嵌入表按 max_seq_len 开（而非 input_cap），这样即使将来
        # 调整 TARGET_SLOTS 也只动 config，不用动权重形状。
        self.item_emb = nn.Embedding(cfg.vocab_size, h, padding_idx=PAD_ITEM)
        self.pos_emb = nn.Embedding(int(cfg.max_seq_len), h)
        self.emb_dropout = nn.Dropout(float(cfg.dropout))
        self.emb_norm = nn.LayerNorm(h, eps=1e-8)

        self.emb_scale = math.sqrt(h) if cfg.scale_emb else 1.0

        # ---------- 主体 ----------
        self.blocks = nn.ModuleList(
            [SASRecBlock(cfg) for _ in range(int(cfg.num_layers))])

        # ---------- 输出侧 ----------
        # share_item_emb=True 时不新建输出矩阵，直接用 item_emb.weight 做点积
        # （权重绑定）。取反时需要一个把 user_repr 投到嵌入空间的线性层。
        self.out_proj = None if cfg.share_item_emb else nn.Linear(h, h, bias=False)

        # 位置嵌入表的后半段（>= input_cap）永远不会被用到，
        # 显式注册成 buffer 便于事后核对，不参与训练。
        self.register_buffer(
            "_unused_pos_slots",
            torch.tensor(int(cfg.max_seq_len) - int(cfg.input_cap), dtype=torch.long),
            persistent=False,
        )

        self.reset_parameters()

    # ------------------------------------------------------------------
    def reset_parameters(self) -> None:
        """按原论文初始化：截断正态 N(0, 0.02²)。

        注意顺序：`nn.Embedding(padding_idx=...)` 会在构造时把 PAD 行置 0，
        这里的 `normal_` 会把它覆盖掉，所以必须**在初始化之后再置一次 0**，
        否则 PAD 得到一个随机向量，会让「PAD 与真实物品」在嵌入空间里
        产生一个固定的偏置方向。
        """
        std = 0.02
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=std)
        with torch.no_grad():
            self.item_emb.weight[PAD_ITEM].zero_()
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=std)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # 重新置零：上面的 Linear 循环不影响 embedding，但 `out_proj` 为 None
        # 时无需处理；这里再确认一次 PAD 行，防御将来有人调整上面语句的顺序。
        with torch.no_grad():
            self.item_emb.weight[PAD_ITEM].zero_()

    # ------------------------------------------------------------------
    # 物品表示表（输入嵌入与输出打分共用同一条入口）
    # ------------------------------------------------------------------
    def item_table(self, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """返回打分/嵌入用的物品表示表 `[V, H]`。

        无 provider 时就是 `item_emb.weight`（权重绑定，与 M2.4 一致）；
        有 provider 时（M2.6b 内容融合）是它作用后的表。**输入侧与输出侧
        必须走同一个方法** —— 否则会出现"输入用融合表示、输出用纯嵌入"
        的隐性不一致，指标看着还行但语义是错的。
        """
        if self.repr_provider is None:
            return self.item_emb.weight if weight is None else weight
        return self.repr_provider(self.item_emb.weight if weight is None else weight)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def encode(
        self,
        seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        item_table: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把历史序列编码成逐位置隐状态与用户表示。

        参数
        ----
        seq         `[B, L]` long，池内索引，PAD = 0（左填充）。
        mask        `[B, L]` bool，True = 有效位置。None 时按 `seq != 0` 推导。
                    契约来自 `docs/project-structure.md` 的模型对外接口。
        item_table  可选的预算好的物品表示表。`score()` 会先算一次表再传进来，
                    避免同一步里把 provider 重复算两遍（内容融合时表是
                    (V, H+D) -> (V, H) 的一次矩阵乘，重复算不划算）。

        返回
        ----
        hidden     `[B, L, H]` 逐位置隐状态（PAD 位置的值无意义，仅用于对齐）
        user_repr  `[B, H]` 序列表示，取「最后一个有效位置」的隐状态
        """
        if seq.dim() != 2:
            raise ValueError(f"seq 应为 (B, L)，实际 {tuple(seq.shape)}")
        b, length = seq.shape
        if length > int(self.cfg.input_cap):
            raise ValueError(
                f"输入长度 {length} 超过上限 {self.cfg.input_cap}"
                f"（= max_seq_len {self.cfg.max_seq_len} 减去 "
                f"{self.cfg.max_seq_len - self.cfg.input_cap} 个留出位）；"
                "位置嵌入会越界，请检查 Dataset 的填充长度")

        valid = (seq != PAD_ITEM) if mask is None else mask.to(torch.bool)
        if valid.shape != seq.shape:
            raise ValueError(
                f"mask 形状 {tuple(valid.shape)} 与 seq {tuple(seq.shape)} 不一致")

        # ---------- 嵌入 ----------
        positions = torch.arange(length, device=seq.device).unsqueeze(0)  # [1, L]
        table = self.item_table() if item_table is None else item_table
        # ⚠️ 必须用 F.embedding(padding_idx=...) 而不是 table[seq]：
        # nn.Embedding 的 padding_idx 语义**包含「第 0 行不接收梯度」**，
        # 裸张量索引会把这个保障丢掉 —— PAD 行会随训练漂移出零，
        # 于是「PAD 与真实物品」在嵌入空间里长出一个固定偏置方向。
        # 回归测试 tests/test_sasrec/test_model.py::test_pad_row_receives_no_gradient
        # 就是抓这个的（M2.6b 改造时被抓到过一次）。
        x = F.embedding(seq, table, padding_idx=PAD_ITEM) + self.pos_emb(positions)
        if self.emb_scale != 1.0:
            x = x * self.emb_scale
        x = self.emb_norm(x)
        x = self.emb_dropout(x)

        # ---------- 因果掩码（算一次，N 层复用）----------
        allow = causal_pad_allow_mask(seq, valid)      # [B, L, L]

        for block in self.blocks:
            x = block(x, allow)

        # ---------- 取用户表示 ----------
        # 「最后一个有效位置」：用 (valid * 位置下标).max() 取，
        # 与填充方向无关（左填充下等价于 hidden[:, -1]，但不依赖这个假设）。
        pos_idx = torch.arange(length, device=seq.device).unsqueeze(0)   # [1, L]
        last_idx = (valid.long() * pos_idx).max(dim=1).values            # [B]
        # 极防御：整行无有效位置时（正常不会出现），回退到第 0 个位置，
        # 保证索引合法；这种样本的表示本身无意义，但不会让训练崩掉。
        last_idx = last_idx.clamp(min=0)
        user_repr = x[torch.arange(b, device=seq.device), last_idx]      # [B, H]

        return x, user_repr

    def forward(
        self,
        seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """`nn.Module` 的习惯入口：返回逐位置隐状态 `[B, L, H]`。"""
        hidden, _ = self.encode(seq, mask)
        return hidden

    # ------------------------------------------------------------------
    # 打分（与 evaluator 的 score_fn 契约一致）
    # ------------------------------------------------------------------
    def score(
        self,
        input_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """对候选物品打分：`[B, L] x [B, C] -> [B, C]`。

        与 `models/eval/evaluator.py` 的 `score_fn` 契约**逐字一致**：
        第 0 列是正样本、分越大越相关。因此评估器可以直接
        `evaluate(model.score, data)`，不需要任何包装。

        ⚠️ 这是 M2.4 的「单向量」打分：用整个序列压成一个 `user_repr`。
        M2.5 的多兴趣模型会把这里换成 K 个胶囊分别打分再合并，
        但**签名与候选列序保持不变** —— 这样评估器与所有实验脚本都不用改。
        """
        # 表只算一次，输入与输出共用（内容融合时尤其重要：provider 是一次
        # (V, H+D)->(V, H) 的矩阵乘，重复算会白白多花一份算力）
        emb_weight = self.item_table()
        _, user_repr = self.encode(input_ids, item_table=emb_weight)      # [B, H]
        if emb_weight.device != candidate_ids.device:
            # 候选索引在 CPU 上时（例如手写测试）不静默失败
            candidate_ids = candidate_ids.to(emb_weight.device)

        cand = candidate_ids.clamp(min=PAD_ITEM, max=self.cfg.n_items)
        item_vec = emb_weight[cand]                           # [B, C, H]

        if self.out_proj is not None:
            # 不共享权重时，把用户表示投到同一空间再点积
            user_repr = self.out_proj(user_repr)

        return (item_vec * user_repr.unsqueeze(1)).sum(dim=-1)  # [B, C]

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        """可训练参数量（报告 E2 消融时要用，证明增益不来自参数增长）。"""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def config_snapshot(self) -> dict:
        """结构摘要，供 checkpoint 与训练日志落盘（复现实验时比对用）。"""
        return {
            "n_items": int(self.cfg.n_items),
            "max_seq_len": int(self.cfg.max_seq_len),
            "input_cap": int(self.cfg.input_cap),
            "hidden_size": int(self.cfg.hidden_size),
            "num_layers": int(self.cfg.num_layers),
            "num_heads": int(self.cfg.num_heads),
            "dropout": float(self.cfg.dropout),
            "ffn_ratio": int(self.cfg.ffn_ratio),
            "share_item_emb": bool(self.cfg.share_item_emb),
            "n_params": self.n_params,
        }
