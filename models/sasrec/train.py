# -*- coding: utf-8 -*-
"""SASRec 训练循环 —— `models/sasrec/train.py`

职责边界
--------
本文件只做「用一批批样本更新参数」和「按早停规则挑出最好的一轮」。
所有 IO（读 pkl、写 checkpoint、写日志文件）都不在这里 ——
`models/` 层的纪律是无 IO（见 `docs/project-structure.md` 三）。
因此：

* 数据从调用方给的 `train_loader` 拿；
* 验证通过 `eval_fn(model) -> dict` 回调，本文件不 import 评估器；
* 最好的权重以 `state_dict`（CPU、已 detach）形式**返回**给调用方，
  由 `scripts/train.py` 决定存到哪。

损失与候选构造
--------------
每个样本的候选是 `[正样本, 1 个负样本]`，用 `BCEWithLogitsLoss`：

    candidates[:, 0] = target           （标签 1）
    candidates[:, 1] = 训练侧负样本      （标签 0）

负样本由 `models/data/negatives.py::sample_train_negatives()` 采（唯一实现）。
⚠️ **训练侧每位置 1 个负样本**，与评估侧的 1 正 100 负是两件事，
详见该模块 docstring 的对照表。

三个必须写清楚的实现约定
------------------------
**1. `patience` 的单位是「epoch」，不是「评估次数」。**
`configs/model.yaml` 里 `early_stop_patience: 10` 与 `epochs: 200` 并列，
读起来是「连续 10 轮没进步就停」。但指标只在评估轮才有，所以内部换算：

    patience_eval_points = max(1, ceil(patience / eval_every_n_epochs))

按当前配置（patience=10、eval_every=5）即「连续 2 次评估无提升 → 停」。
若按「评估次数」直接理解，10 次评估 × 5 epoch = 50 轮，
超过 `max_epochs=30`，早停**永远不会触发**，配置形同虚设。
两种理解都常见（RecBole 用后者），本项目选前者并在此写明，避免歧义。

**2. AMP 只在 batch 足够大时开启，且由 `TrainConfig.amp_enabled()` 集中判断。**
M2.0 实测：bs=128/256 时 AMP 反而慢（0.80x / 0.91x），bs>=512 才转正。
判断只写一处，避免"日志说开了 AMP、实际没开"。

**3. 早停判断用的是「越大越好」的指标。**
`early_stop_metric` 默认 `ndcg@10`，取值高于历史最优才算提升。
若将来换成 loss 之类的「越小越好」指标，必须同时改这里的比较方向 ——
配置里没有方向的字段，这是个已知的粗糙点。
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from models.data.negatives import sample_train_negatives
from models.sasrec.config import OptimConfig, TrainConfig

__all__ = [
    "EpochRecord",
    "FitResult",
    "fit",
    "train_one_epoch",
    "set_seed",
    "build_optimizer",
    "build_scheduler",
]

logger = logging.getLogger(__name__)

# 验证回调：接收模型，返回指标字典（键名与 evaluator 的输出一致，如 "ndcg@10"）
EvalFn = Callable[[nn.Module], Dict[str, float]]


# =====================================================================
# 可复现性
# =====================================================================
def set_seed(seed: int, deterministic: bool = False) -> None:
    """固定 Python / NumPy / PyTorch 的随机源。

    `deterministic=True` 会额外开启 cudnn 的确定性算法。
    **默认关闭**，因为：

    * 本模型的算子是 matmul + softmax，不用卷积，cudnn 确定性开关
      对它几乎没有速度影响，但 M2.7 的 GRU4Rec 基线会用 cudnn 的 RNN；
    * 某些 RNN 实现没有确定性内核，开启后会直接抛错或静默回退到慢速路径。

    所以把这个选择交给调用方，而不是在这里替所有模型做主。
    不开 `deterministic` 时，单卡训练仍可复现（算子本身确定），
    但跨设备/跨版本的逐位复现不保证 —— 论文里不要声称后者。
    """
    s = int(seed)
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =====================================================================
# 优化器 / 调度器
# =====================================================================
def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    """Adam。权重衰减只施加在**矩阵参数**上，Norm 与 bias 不衰减。

    为什么不给所有参数一视同仁地加 weight_decay：对 LayerNorm 的
    `weight`（初始为 1）和 `bias` 施加 L2 会把它们往 0 拉，
    等价于削弱归一化层与偏置的表达能力。这是 AdamW 论文的常见做法，
    在序列推荐的小模型上影响不小。
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [{"params": decay, "weight_decay": float(cfg.weight_decay)}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})

    return torch.optim.Adam(
        groups, lr=float(cfg.lr), betas=tuple(cfg.betas), eps=float(cfg.eps))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: OptimConfig,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    """warmup（+ 可选余弦退火）的 LambdaLR。

    `warmup_ratio=0` 且 `lr_decay="constant"` 时返回 None（不挂调度器），
    让训练循环里少一个空转的分支。
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = int(round(total_steps * float(cfg.warmup_ratio)))
    if warmup_steps <= 0 and cfg.lr_decay == "constant":
        return None

    from models.sasrec.config import cosine_lambda

    def lr_lambda(step: int) -> float:
        return cosine_lambda(
            step=step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_ratio=float(cfg.min_lr_ratio),
            lr_decay=str(cfg.lr_decay),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =====================================================================
# 结果容器
# =====================================================================
@dataclass
class EpochRecord:
    """一轮训练的统计。`metrics` 只在评估轮非空。"""

    epoch: int
    train_loss: float
    lr: float
    elapsed_sec: float
    n_steps: int = 0
    metrics: dict = field(default_factory=dict)

    def as_report(self) -> dict:
        def _num(v):
            # NaN / inf 不是合法 JSON，落盘时会让 json.dump 直接失败；
            # 统一转成 None，让"这一轮没算出来"和"算出来是 0"可区分。
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, (float, np.floating)):
                return round(float(v), 6) if math.isfinite(float(v)) else None
            return v

        return {
            "epoch": int(self.epoch),
            "train_loss": _num(self.train_loss),
            "lr": _num(self.lr),
            "elapsed_sec": _num(self.elapsed_sec),
            "n_steps": int(self.n_steps),
            "metrics": {k: _num(v) for k, v in self.metrics.items()},
        }


@dataclass
class FitResult:
    """`fit()` 的返回值：训练历史 + 最好的那一轮。

    `best_state_dict` 是 **CPU 上的 detached 副本**（不是引用）——
    如果直接存原模型的引用，训练继续更新参数后，这个"最好的权重"
    会跟着变，最后存下来的其实是最后一轮的权重。
    这是早停实现里最隐蔽的一类 bug，故在 `_snapshot_state_dict()` 里
    统一做深拷贝并注明。
    """

    records: List[EpochRecord]
    metric_name: str
    best_epoch: Optional[int] = None
    best_metric: Optional[float] = None
    best_state_dict: Optional[dict] = None
    stopped_early: bool = False
    n_epochs_run: int = 0
    total_elapsed_sec: float = 0.0
    amp_enabled: bool = False

    @property
    def last_loss(self) -> Optional[float]:
        return self.records[-1].train_loss if self.records else None

    def as_report(self) -> dict:
        def _num(v):
            if v is None:
                return None
            f = float(v)
            return round(f, 6) if math.isfinite(f) else None

        return {
            "metric_name": self.metric_name,
            "best_epoch": self.best_epoch,
            "best_metric": _num(self.best_metric),
            "n_epochs_run": int(self.n_epochs_run),
            "stopped_early": bool(self.stopped_early),
            "total_elapsed_sec": _num(self.total_elapsed_sec),
            "amp_enabled": bool(self.amp_enabled),
            "first_loss": None if not self.records else _num(self.records[0].train_loss),
            "last_loss": _num(self.last_loss),
            "history": [r.as_report() for r in self.records],
        }


# =====================================================================
# 内部工具
# =====================================================================
def _snapshot_state_dict(model: nn.Module) -> dict:
    """把 `state_dict` 深拷贝到 CPU，切断与训练中参数的引用关系。"""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _make_candidate_builder(n_items: int, device, seed: int):
    """返回 `targets -> candidates[B, 2]` 的闭包，复用同一个负采样 rng。

    `seed` 是**必须**的：训练侧负采样虽然不必"跨模型一致"，但必须
    "同种子可复现" —— 论文里 `main` 档的 3 个种子要报均值±标准差，
    若负样本流每次都不同，那 3 次跑出来的差异里就混进了随机噪声，
    标准差失去意义。`fit()` 用 `seed + epoch` 让每轮的负样本不同、
    但每轮本身可复现。

    代价是 rng 在闭包里长期存活、按 batch 顺序消耗 —— 这是训练侧
    **有意**的取舍（不需要评估侧那套逐行独立随机流，那样 80μs × 1024 行
    会比前向还慢），理由见 `models/data/negatives.py` 的模块 docstring。
    """
    rng = np.random.default_rng(int(seed))

    def build(targets: torch.Tensor) -> torch.Tensor:
        t = targets.detach().cpu().numpy()
        neg = sample_train_negatives(t, n_items=n_items, rng=rng)
        cand = torch.empty((t.shape[0], 2), dtype=torch.long, device=device)
        cand[:, 0] = targets.to(device)
        cand[:, 1] = torch.as_tensor(neg, dtype=torch.long, device=device)
        return cand

    return build


# =====================================================================
# 单轮训练
# =====================================================================
def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    criterion: nn.Module,
    device,
    scaler=None,
    scheduler=None,
    n_items: Optional[int] = None,
    grad_accum_steps: int = 1,
    log_every_n_steps: int = 0,
    epoch: int = 1,
    neg_seed: int = 0,
) -> tuple:
    """跑一轮训练，返回 `(平均损失, 步数, 最后的 lr)`。

    平均损失按 **样本数** 加权（不是按 step 平均），因为 DataLoader 的
    最后一个 batch 通常是残缺的；按 step 平均会让它被过度放大。

    `neg_seed` 决定本轮的负样本流（见 `_make_candidate_builder`）；
    传 `seed + epoch` 可做到「同种子可复现、不同轮次不重复」。
    """
    model.train()
    if n_items is None:
        n_items = int(getattr(getattr(model, "cfg", None), "n_items", 0))
    if n_items <= 0:
        raise ValueError("必须能确定物品池大小 n_items（从 model.cfg 或显式传入）")

    build_candidates = _make_candidate_builder(n_items, device, seed=neg_seed)
    label_tpl = torch.tensor([1.0, 0.0], device=device)     # [C=2]

    accum = max(1, int(grad_accum_steps))
    total_loss = 0.0
    total_samples = 0
    n_steps = 0
    last_lr = float(optimizer.param_groups[0]["lr"])

    optimizer.zero_grad(set_to_none=True)

    # step 在循环外先初始化：empty loader（例如 user_rows 为空）时
    # 下面的"收尾补一次更新"分支会用到它，不初始化会 NameError。
    step = 0
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        cand = build_candidates(y)                          # [B, 2]，第 0 列是正样本
        labels = label_tpl.unsqueeze(0).expand(cand.size(0), -1)

        with torch.amp.autocast(
            device_type=str(cfg.device).split(":")[0],
            dtype=(torch.float16 if cfg.amp_dtype == "fp16" else torch.bfloat16),
            enabled=bool(scaler is not None and scaler.is_enabled()),
        ):
            scores = model.score(x, cand)
            loss = criterion(scores, labels)

        # 梯度累积：除以 accum 使等效 batch 的梯度量级与 accum=1 一致
        loss_to_back = loss / accum
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss_to_back).backward()
        else:
            loss_to_back.backward()

        if step % accum == 0:
            if scaler is not None and scaler.is_enabled():
                # 先 unscale 再裁剪，否则裁的是被放大过的梯度，阈值失去意义
                scaler.unscale_(optimizer)
            if float(cfg.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg.grad_clip_norm))
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            n_steps += 1
            last_lr = float(optimizer.param_groups[0]["lr"])

            if log_every_n_steps and n_steps % int(log_every_n_steps) == 0:
                logger.info(
                    "  epoch %d | step %d | loss %.4f | lr %.2e",
                    epoch, n_steps, total_loss / max(1, total_samples), last_lr)

        bs = int(x.size(0))
        total_loss += float(loss.detach()) * bs
        total_samples += bs

    # 收尾：最后一个不完整的累积窗口也要更新一次，否则末尾样本被白跑。
    # step == 0 表示 loader 为空，此时不需要也不应该更新。
    if step and step % accum != 0:
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
        if float(cfg.grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg.grad_clip_norm))
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        n_steps += 1
        last_lr = float(optimizer.param_groups[0]["lr"])

    mean_loss = total_loss / total_samples if total_samples else float("nan")
    return mean_loss, n_steps, last_lr


# =====================================================================
# 主训练
# =====================================================================
def fit(
    model: nn.Module,
    train_loader,
    cfg: TrainConfig,
    optim_cfg: OptimConfig,
    eval_fn: Optional[EvalFn] = None,
    n_items: Optional[int] = None,
    grad_accum_steps: int = 1,
    on_epoch_end: Optional[Callable[[EpochRecord, FitResult], None]] = None,
) -> FitResult:
    """训练模型并在验证指标上早停。

    参数
    ----
    model        待训练模型（会被原地更新参数）。
    train_loader 产出 `(input_ids [B, L], targets [B])` 的 DataLoader。
    cfg / optim_cfg  训练控制与优化器配置。
    eval_fn      验证回调 `(model) -> dict`。为 None 时不评估、不早停，
                 只跑满 `cfg.epochs`（用于冒烟测试）。
    n_items      物品池大小；None 时从 `model.cfg.n_items` 取。
    grad_accum_steps  梯度累积步数（1 = 关闭）。
    on_epoch_end 每轮结束后的回调，用于外部记录（例如写 TensorBoard）。
                 回调抛出的异常会向上传播 —— 不吞异常。

    返回
    ----
    `FitResult`，其中 `best_state_dict` 是**最优轮**的 CPU 权重副本。

    ⚠️ 调用方拿到 `best_state_dict` 后应当 `model.load_state_dict()` 再评估，
    不要直接用训练结束时的参数去报指标 —— 最后一轮不一定是最好的一轮。
    """
    device = torch.device(str(cfg.device) if torch.cuda.is_available()
                          or str(cfg.device) == "cpu" else "cpu")
    model.to(device)

    if n_items is None:
        n_items = int(getattr(getattr(model, "cfg", None), "n_items", 0))
    if n_items <= 0:
        raise ValueError("无法确定 n_items，请显式传入")

    try:
        steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, grad_accum_steps)))
    except TypeError:
        # 迭代器没有 __len__：无法预知总步数，也就无法做按比例 warmup
        steps_per_epoch = 0
    total_steps = int(cfg.epochs) * steps_per_epoch

    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, optim_cfg, total_steps) if total_steps else None

    amp_on = bool(cfg.amp_enabled())
    if cfg.amp and not amp_on:
        logger.info(
            "AMP 已配置为开启，但因 batch_size=%d < %d 或 device=%s 而自动关闭"
            "（实测小 batch 下 AMP 反而更慢）",
            cfg.batch_size, cfg.amp_min_batch, cfg.device)
    scaler = torch.amp.GradScaler(
        device=str(cfg.device).split(":")[0], enabled=amp_on)

    criterion = nn.BCEWithLogitsLoss()

    # 早停 patience 的单位换算：epoch -> 评估点（见模块 docstring 第 1 条）
    if eval_fn is not None and int(cfg.early_stop_patience) > 0:
        patience_points = max(
            1,
            math.ceil(int(cfg.early_stop_patience) / int(cfg.eval_every_n_epochs)),
        )
    else:
        patience_points = 0      # 0 = 不早停

    logger.info(
        "训练开始：epochs=%d batch=%d lr=%g warmup=%.0f%% amp=%s "
        "eval_every=%d patience=%d(评估点=%d) steps/epoch=%d device=%s",
        cfg.epochs, cfg.batch_size, optim_cfg.lr, optim_cfg.warmup_ratio * 100,
        amp_on, cfg.eval_every_n_epochs, cfg.early_stop_patience,
        patience_points, steps_per_epoch, device)

    result = FitResult(records=[], metric_name=str(cfg.early_stop_metric),
                       amp_enabled=amp_on)
    bad_points = 0
    t_start = time.time()

    for epoch in range(1, int(cfg.epochs) + 1):
        t0 = time.time()
        mean_loss, n_steps, last_lr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            cfg=cfg,
            criterion=criterion,
            device=device,
            scaler=scaler,
            scheduler=scheduler,
            n_items=n_items,
            grad_accum_steps=grad_accum_steps,
            log_every_n_steps=int(cfg.log_every_n_steps),
            epoch=epoch,
            neg_seed=int(cfg.seed) + int(epoch),
        )
        record = EpochRecord(
            epoch=epoch, train_loss=mean_loss, lr=last_lr,
            elapsed_sec=time.time() - t0, n_steps=n_steps)

        # ---------- 评估 ----------
        if eval_fn is not None and cfg.should_eval(epoch, cfg.epochs):
            model.eval()
            metrics = eval_fn(model)
            model.train()
            record.metrics = dict(metrics)

            metric = float(metrics.get(result.metric_name, float("nan")))
            if math.isnan(metric):
                raise KeyError(
                    f"评估结果里没有指标 {result.metric_name!r}；"
                    f"实际返回的键：{sorted(metrics)[:12]}")

            improved = (result.best_metric is None) or (metric > result.best_metric)
            if improved:
                result.best_metric = metric
                result.best_epoch = epoch
                result.best_state_dict = _snapshot_state_dict(model)
                bad_points = 0
                logger.info(
                    "  epoch %d | loss %.4f | %s %.4f  ← 新的最优",
                    epoch, mean_loss, result.metric_name, metric)
            else:
                bad_points += 1
                logger.info(
                    "  epoch %d | loss %.4f | %s %.4f（最优 %.4f @ epoch %d，"
                    "已连续 %d/%d 次评估无提升）",
                    epoch, mean_loss, result.metric_name, metric,
                    result.best_metric, result.best_epoch, bad_points,
                    patience_points)
        else:
            logger.info("  epoch %d | loss %.4f | lr %.2e | %.1fs",
                        epoch, mean_loss, last_lr, record.elapsed_sec)

        result.records.append(record)
        if on_epoch_end is not None:
            on_epoch_end(record, result)

        # ---------- 早停 ----------
        if patience_points and bad_points >= patience_points:
            result.stopped_early = True
            logger.info(
                "早停：连续 %d 次评估无提升（≈ %d 个 epoch），"
                "最优为 epoch %d 的 %s=%.4f",
                bad_points, bad_points * int(cfg.eval_every_n_epochs),
                result.best_epoch, result.metric_name, result.best_metric)
            break

    result.n_epochs_run = len(result.records)
    result.total_elapsed_sec = time.time() - t_start

    # 一次都没评估过（eval_fn=None）时，把最后一轮的权重当作结果，
    # 这样调用方拿到的 `best_state_dict` 永远不是 None。
    # best_metric 保持 None 而不填 NaN：NaN 不是合法 JSON，
    # 会让训练日志落盘在 json.dump 处失败。
    if result.best_state_dict is None:
        result.best_state_dict = _snapshot_state_dict(model)
        result.best_epoch = result.n_epochs_run
        result.best_metric = None

    logger.info(
        "训练结束：跑了 %d 轮，耗时 %.1fs，最优 %s=%s @ epoch %s%s",
        result.n_epochs_run, result.total_elapsed_sec, result.metric_name,
        ("" if result.best_metric is None else f"{result.best_metric:.4f}"),
        result.best_epoch,
        "（早停）" if result.stopped_early else "")
    return result
