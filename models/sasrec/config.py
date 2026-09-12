# -*- coding: utf-8 -*-
"""SASRec 超参的 dataclass 定义 —— 对齐 `configs/model.yaml`。

为什么要有这一层
----------------
`configs/model.yaml` 是给人读的（有注释、有分组），而模型内部需要的是
「带类型、有校验、改了会立刻报错」的对象。若模型直接吃 dict：

* 拼错一个键（`hidden_size` 写成 `hidden_dim`）不会有任何提示，
  只会静默用上 dataclass 的默认值 —— 实验记录里写着 128，实际跑的是 64；
* `num_heads=3` 配 `hidden_size=64` 这种非法组合要等 forward 时才崩，
  而且崩在 `view()` 的形状错误上，看不出是配置问题。

所以本模块是配置进入模型层之后的**唯一入口**：校验在这里，
过了这道门就保证组合合法。

分层纪律
--------
本模块属于 `models/`，因此 **不读文件、不读 .env、不 import yaml**。
构造实例所需的 dict 由 `scripts/train.py` 读好之后传进来
（见 `from_dict()`）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional

__all__ = [
    "SASRecConfig",
    "TrainConfig",
    "OptimConfig",
]

# val 与 test 在完整序列末尾各占 1 个位置（留一法），
# 与 models/sasrec/dataset.py 的 TARGET_SLOTS 是同一个口径。
# 此处不复用那个常量是为了让 config 模块零依赖，改口径时两处都要动，
# 因此加这条断言把这个隐患摆在明面上（见 tests/test_sasrec/test_model.py）。
_TARGET_SLOTS = 2


# =====================================================================
# 模型结构
# =====================================================================
@dataclass(frozen=True)
class SASRecConfig:
    """SASRec 网络结构超参。

    字段与 `configs/model.yaml` 的 `model` 段一一对应；
    多兴趣胶囊（K）与内容融合的开关属于 M2.5 / M2.6，
    不在本 dataclass 里，避免 M2.4 阶段引入无用分支。

    参数
    ----
    n_items     物品池规模（= `len(smap)` = 15,687）。**不含 PAD**：
                嵌入表会开 `n_items + 1` 行，PAD 占下标 0。
    max_seq_len 完整序列长度上限（含 val / test 两位），来自 data.yaml 的
                `sequence.max_len`。位置嵌入表按它开，但实际只会用到
                `input_cap` 以内。
    hidden_size 隐层维度 H。
    num_layers  因果注意力块层数。
    num_heads   注意力头数，必须整除 hidden_size。
    dropout     embedding / attention / FFN 的 dropout 比例。
    ffn_ratio   FFN 中间层相对 hidden 的倍率。SASRec 原论文为 4。
    share_item_emb
                是否让输入嵌入与输出打分共用同一个物品嵌入矩阵。
                原论文共享；共享能省一半参数，也让「输出层没见过的物品」
                与「输入层没见过的物品」保持一致。默认 True。
    scale_emb / scale_ffn
                原论文的两处缩放技巧（嵌入乘 sqrt(H)、FFN 输出乘 1/sqrt(2)）。
                保持与原论文一致，否则与公开实现的对数不可比。
    """

    n_items: int
    max_seq_len: int = 50
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.2
    ffn_ratio: int = 4
    share_item_emb: bool = True
    scale_emb: bool = True
    scale_ffn: bool = True

    # ---------------- 派生属性 ----------------
    @property
    def input_cap(self) -> int:
        """单次前向的输入长度上限 = `max_seq_len - 2`（val/test 各占 1 位）。

        与 `models/sasrec/dataset.py` 的 `input_cap` 必须一致：
        位置嵌入表、因果 mask、Dataset 的滑动窗口都以它为准。
        """
        return int(self.max_seq_len) - _TARGET_SLOTS

    @property
    def vocab_size(self) -> int:
        """嵌入表行数（含 PAD）。"""
        return int(self.n_items) + 1

    @property
    def head_dim(self) -> int:
        return int(self.hidden_size) // int(self.num_heads)

    # ---------------- 校验 ----------------
    def __post_init__(self) -> None:
        if int(self.n_items) <= 0:
            raise ValueError(f"n_items 必须为正，收到 {self.n_items}")
        if int(self.max_seq_len) < 4:
            raise ValueError(
                f"max_seq_len={self.max_seq_len} 太小：至少要能容纳 "
                f"1 个历史 + {_TARGET_SLOTS} 个留出位")
        if int(self.hidden_size) <= 0 or int(self.num_layers) <= 0:
            raise ValueError("hidden_size / num_layers 必须为正")
        if int(self.num_heads) <= 0:
            raise ValueError(f"num_heads 必须为正，收到 {self.num_heads}")
        if int(self.hidden_size) % int(self.num_heads) != 0:
            # 这条最容易在调 HPO 时踩到：hidden=64 配 heads=3 无法均分
            raise ValueError(
                f"hidden_size({self.hidden_size}) 必须能被 "
                f"num_heads({self.num_heads}) 整除，否则多头无法均分")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"dropout 应在 [0, 1)，收到 {self.dropout}")
        if int(self.ffn_ratio) <= 0:
            raise ValueError(f"ffn_ratio 必须为正，收到 {self.ffn_ratio}")

    # ---------------- 构造 ----------------
    @classmethod
    def from_dict(cls, model_cfg: dict, n_items: int) -> "SASRecConfig":
        """从 `configs/model.yaml` 的 `model` 段构造。

        只取本 dataclass 认得的字段，多余键（例如 `use_multi_interest`、
        `num_interests`、`routing_iters` 这些属于 M2.5 的开关）被忽略，
        这样 M2.5 加字段时不需要回来改这里。

        `n_items` 不在 yaml 里（它由数据集决定），必须显式传入。
        """
        known = {f.name for f in fields(cls)} - {"n_items"}
        kwargs = {k: v for k, v in (model_cfg or {}).items() if k in known}
        return cls(n_items=int(n_items), **kwargs)


# =====================================================================
# 优化器 / 学习率
# =====================================================================
@dataclass(frozen=True)
class OptimConfig:
    """优化器与学习率调度。

    为什么把 warmup 做成独立字段
    ---------------------------
    M2.0 把 `batch_size` 从 256 提到 1024 之后，`lr=0.001` 这个"经典值"
    并没有跟着重标定（大 batch 下梯度噪声更小，通常要更大 lr 或更长 warmup）。
    所以 M2.4 需要能在 dev 档快速试 `{0.001, 0.002, 0.004}` × warmup 开关，
    见 `scripts/train.py --lr ... --warmup-ratio ...`。

    参数
    ----
    lr            峰值学习率。
    weight_decay  Adam 的 L2 权重衰减。
    warmup_ratio  线性升温步数占总步数的比例，0 表示关闭。
    lr_decay      `"constant"`（升温后保持）或 `"cosine"`（余弦退火到 min_lr_ratio）。
    min_lr_ratio  余弦退火的下限（相对峰值），仅 `lr_decay="cosine"` 时生效。
    betas         Adam 的动量项，保持 PyTorch 默认，显式列出便于消融。
    eps           数值稳定项。
    """

    lr: float = 0.001
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    lr_decay: str = "constant"
    min_lr_ratio: float = 0.0
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if float(self.lr) <= 0:
            raise ValueError(f"lr 必须为正，收到 {self.lr}")
        if float(self.weight_decay) < 0:
            raise ValueError(f"weight_decay 不能为负，收到 {self.weight_decay}")
        if not 0.0 <= float(self.warmup_ratio) < 1.0:
            raise ValueError(f"warmup_ratio 应在 [0, 1)，收到 {self.warmup_ratio}")
        if self.lr_decay not in ("constant", "cosine"):
            raise ValueError(
                f"lr_decay 只能是 'constant' / 'cosine'，收到 {self.lr_decay!r}")

    @classmethod
    def from_dict(cls, train_cfg: dict) -> "OptimConfig":
        """从 `configs/model.yaml` 的 `train` 段取优化器相关字段。"""
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for k, v in (train_cfg or {}).items():
            if k in known:
                # YAML 里的 betas 是 list，转成 tuple 以保持哈希/不可变语义
                kwargs[k] = tuple(v) if k == "betas" and isinstance(v, list) else v
        return cls(**kwargs)


# =====================================================================
# 训练控制
# =====================================================================
@dataclass(frozen=True)
class TrainConfig:
    """训练循环的控制参数（不含模型结构与优化器细节）。

    参数
    ----
    batch_size      批大小。M2.0 实测甜点 1024（显存 1.19GB / 6GB）。
    epochs          训练轮数上限；实际由 scale 档位的 max_epochs 收紧。
    early_stop_patience  连续多少个「评估点」无提升就停。
    early_stop_metric    早停监控的指标名，必须是 `metrics` 认识的键。
    grad_clip_norm  梯度 L2 范数裁剪阈值，0 表示不裁剪。
    amp             是否启用混合精度（autocast + GradScaler）。
    amp_dtype       `"fp16"` 或 `"bfloat16"`。RTX 2060（sm_75）只有 fp16
                    Tensor Core，bf16 在 sm_75 上会退化为无加速，故默认 fp16。
    amp_min_batch   低于该 batch 时自动关闭 AMP。
                    ⚠️ 实测依据：bs=128/256 时 AMP 反而更慢（0.80x / 0.91x，
                    kernel 启动开销盖过收益），bs>=512 才转正。
                    自动降级比"记得手动关"更可靠。
    num_workers    DataLoader 进程数。**默认 0**，表示训练走
                   `dataset.WindowBatchIterator`（单进程批量取数，无 collate 开销）。
                   改成 >0 会退回 `torch.utils.data.DataLoader`，
                   在本机（Windows）实测慢 15 倍，理由见该类的 docstring。
    device          `"cuda"` / `"cpu"`。
    eval_every_n_epochs  每多少个 epoch 评估一次验证集。
    log_every_n_steps    每多少 step 打印一次训练进度。
    seed           单次训练的随机种子。
    """

    batch_size: int = 1024
    epochs: int = 200
    early_stop_patience: int = 10
    early_stop_metric: str = "ndcg@10"
    grad_clip_norm: float = 5.0
    amp: bool = True
    amp_dtype: str = "fp16"
    amp_min_batch: int = 512
    num_workers: int = 0
    device: str = "cuda"
    eval_every_n_epochs: int = 5
    log_every_n_steps: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0:
            raise ValueError(f"batch_size 必须为正，收到 {self.batch_size}")
        if int(self.epochs) <= 0:
            raise ValueError(f"epochs 必须为正，收到 {self.epochs}")
        if int(self.early_stop_patience) < 0:
            raise ValueError("early_stop_patience 不能为负（0 表示只按 epochs 停）")
        if float(self.grad_clip_norm) < 0:
            raise ValueError("grad_clip_norm 不能为负（0 表示不裁剪）")
        if self.amp_dtype not in ("fp16", "bfloat16"):
            raise ValueError(
                f"amp_dtype 只能是 'fp16' / 'bfloat16'，收到 {self.amp_dtype!r}")
        if int(self.eval_every_n_epochs) <= 0:
            raise ValueError("eval_every_n_epochs 必须为正")

    # ---------------- 派生属性 ----------------
    def amp_enabled(self) -> bool:
        """实际是否启用 AMP：配置开启 **且** batch 足够大 **且** 在 CUDA 上。

        三条缺一不可。把判断集中在这里，训练循环里就不用再写 if
        （写两处很容易出现"日志说开了 AMP、实际没开"这种不一致）。
        """
        if not self.amp:
            return False
        if str(self.device) != "cuda":
            return False
        return int(self.batch_size) >= int(self.amp_min_batch)

    def should_eval(self, epoch: int, total_epochs: int) -> bool:
        """第 `epoch`（1-based）是否评估。**最后一个 epoch 必定评估** ——
        否则当总轮数不是 `eval_every_n_epochs` 的整数倍时，
        最后一轮永远不会被评估，早停也就看不到它。
        """
        e = int(epoch)
        if e >= int(total_epochs):
            return True
        return e % int(self.eval_every_n_epochs) == 0

    @classmethod
    def from_dict(cls, train_cfg: dict, eval_cfg: Optional[dict] = None) -> "TrainConfig":
        """从 `configs/model.yaml` 的 `train` + `eval` 段构造。

        `eval.every_n_epochs` 会覆盖 train 段的同名默认值，
        以便配置里只写一处。
        """
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in (train_cfg or {}).items() if k in known}
        if eval_cfg:
            n = eval_cfg.get("every_n_epochs")
            if n is not None:
                kwargs["eval_every_n_epochs"] = int(n)
        return cls(**kwargs)


def cosine_lambda(step: int, total_steps: int, warmup_steps: int,
                  min_ratio: float = 0.0, lr_decay: str = "constant") -> float:
    """返回第 `step` 步的**学习率倍率**（相对峰值），供 `LambdaLR` 使用。

    形状：`0 → warmup 线性升到 1 → 之后恒定（constant）或余弦降到 min_ratio`。

    ⚠️ 采用「按 step 计」而不是「按 epoch 计」：本数据集一个 epoch 有
    ~26.7 万 step（main 档 1,092 万个样本 / batch 1024），按 epoch 升温
    意味着前几万步都在极小 lr 上跑，等于白费时间；按 step 升温才能
    在头一个 epoch 内就爬到峰值。

    纯函数、无副作用，因此可以直接单元测试数值（见 test_train.py）。
    """
    s = int(step)
    total = max(1, int(total_steps))
    warm = max(0, int(warmup_steps))

    if warm > 0 and s < warm:
        return float(s + 1) / float(warm)      # +1：第 0 步不为 0，避免整步无梯度

    if lr_decay != "cosine":
        return 1.0

    progress = float(max(0, s - warm)) / float(max(1, total - warm))
    progress = min(1.0, progress)
    low = float(min_ratio)
    return low + (1.0 - low) * 0.5 * (1.0 + math.cos(math.pi * progress))
