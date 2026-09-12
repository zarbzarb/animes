# -*- coding: utf-8 -*-
"""`models/sasrec/train.py` 的单元测试。

训练循环的 bug 有两类特别值得防：

1. **早停不生效或提前停** —— 不生效会让训练白跑；提前停会截断收敛。
   两条都不会报错，只是数字变差，很容易被归因成"模型不行"。
2. **best_state_dict 是引用而不是副本** —— 存下来的"最优权重"
   会跟着后续训练一起变，最后得到的是最后一轮的权重。
   这是最隐蔽的一类，故单独用测试锁死。

测试全部在 CPU 上跑（`device="cpu"`、`amp=False`），不依赖显卡，
CI 里也能执行。
"""

from __future__ import annotations

import math

import pytest
import torch

from models.sasrec.config import OptimConfig, SASRecConfig, TrainConfig, cosine_lambda
from models.sasrec.model import SASRec
from models.sasrec.train import (
    build_optimizer,
    build_scheduler,
    fit,
    set_seed,
    train_one_epoch,
)

# toy 配置：input_cap = 6 - 2 = 4
CAP = 4
N_ITEMS = 20


def _model(seed: int = 0) -> SASRec:
    torch.manual_seed(seed)
    return SASRec(SASRecConfig(
        n_items=N_ITEMS, max_seq_len=6, hidden_size=16,
        num_layers=1, num_heads=2, dropout=0.0, ffn_ratio=2))


class ToyLoader:
    """最小可用训练集：`target` 是「最后一个物品」的确定性函数。

    这样模型真的**学得动**（loss 会降），才能区分
    "训练循环坏了" 与 "任务本身不可学"。
    """

    def __init__(self, n_batch: int = 6, bs: int = 16, seed: int = 0):
        self.n_batch = int(n_batch)
        self.bs = int(bs)
        self.g = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            x = torch.randint(1, N_ITEMS + 1, (self.bs, CAP), generator=self.g)
            y = (x[:, -1] % 5) + 1          # 由最后一位决定的确定性标签
            yield x, y


def _train_cfg(**kw) -> TrainConfig:
    base = dict(batch_size=16, epochs=3, device="cpu", amp=False,
                num_workers=0, log_every_n_steps=0, eval_every_n_epochs=1,
                early_stop_patience=0, seed=42)
    base.update(kw)
    return TrainConfig(**base)


# =====================================================================
# 1. 学习率调度
# =====================================================================
def test_cosine_lambda_warmup_is_linear_and_then_flat():
    """warmup 段从 1/warm 线性升到 1，之后保持 1（constant）。"""
    kw = dict(total_steps=100, warmup_steps=10, lr_decay="constant")
    assert cosine_lambda(0, **kw) == pytest.approx(1 / 10)
    assert cosine_lambda(4, **kw) == pytest.approx(0.5)
    assert cosine_lambda(9, **kw) == pytest.approx(1.0)
    # warmup 之后恒定
    assert cosine_lambda(10, **kw) == pytest.approx(1.0)
    assert cosine_lambda(99, **kw) == pytest.approx(1.0)


def test_cosine_lambda_without_warmup_is_constant_one():
    """不设 warmup 且 constant 时恒为 1 —— 即"不做调度"，等价于关闭。"""
    for s in (0, 1, 50, 999):
        assert cosine_lambda(s, total_steps=1000, warmup_steps=0,
                             lr_decay="constant") == 1.0


def test_cosine_lambda_decays_to_min_ratio():
    kw = dict(total_steps=100, warmup_steps=10, lr_decay="cosine", min_ratio=0.1)
    assert cosine_lambda(10, **kw) == pytest.approx(1.0)          # 起点
    assert cosine_lambda(100, **kw) == pytest.approx(0.1)         # 终点
    mid = cosine_lambda(55, **kw)                                  # 中点
    assert 0.1 < mid < 1.0
    # 单调不增
    vals = [cosine_lambda(s, **kw) for s in range(10, 101)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))


def test_cosine_lambda_clamps_beyond_total():
    """步数超出总步数时应停在下限，而不是继续震荡或变负。"""
    kw = dict(total_steps=50, warmup_steps=5, lr_decay="cosine", min_ratio=0.0)
    assert cosine_lambda(50, **kw) == pytest.approx(0.0)
    assert cosine_lambda(500, **kw) == pytest.approx(0.0)


def test_scheduler_is_none_when_no_warmup_and_constant():
    """`warmup_ratio=0` + constant 时不挂调度器（少一个空转分支）。"""
    opt = build_optimizer(_model(), OptimConfig(lr=0.01))
    assert build_scheduler(opt, OptimConfig(lr=0.01), total_steps=100) is None


def test_scheduler_steps_lr_down_to_min():
    opt = build_optimizer(_model(), OptimConfig(lr=0.1))
    sch = build_scheduler(
        opt, OptimConfig(lr=0.1, warmup_ratio=0.1, lr_decay="cosine",
                         min_lr_ratio=0.0),
        total_steps=100)
    assert sch is not None
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1 * 0.1)   # 第 0 步
    for _ in range(100):
        opt.step()
        sch.step()
    assert opt.param_groups[0]["lr"] < 1e-3


# =====================================================================
# 2. 优化器
# =====================================================================
def test_optimizer_skips_weight_decay_on_norm_and_bias():
    """LayerNorm 权重与所有 bias 不应被 weight decay 拉向 0。"""
    opt = build_optimizer(_model(), OptimConfig(weight_decay=0.1))
    assert len(opt.param_groups) == 2
    decays = sorted(g["weight_decay"] for g in opt.param_groups)
    assert decays == [0.0, 0.1]

    no_decay = [g for g in opt.param_groups if g["weight_decay"] == 0.0][0]
    # 一维参数（LayerNorm weight/bias、所有 bias）都应落在 no_decay 组
    for p in no_decay["params"]:
        assert p.ndim <= 1


# =====================================================================
# 3. AMP 开关判断
# =====================================================================
def test_amp_enabled_requires_large_batch_and_cuda():
    assert _train_cfg(amp=True, device="cuda", batch_size=1024).amp_enabled()
    # 小 batch 下 AMP 反而更慢（实测），必须自动关闭
    assert not _train_cfg(amp=True, device="cuda", batch_size=128).amp_enabled()
    # CPU 上不开
    assert not _train_cfg(amp=True, device="cpu", batch_size=1024).amp_enabled()
    # 显式关掉
    assert not _train_cfg(amp=False, device="cuda", batch_size=1024).amp_enabled()


def test_amp_min_batch_boundary():
    assert _train_cfg(amp=True, device="cuda", batch_size=511).amp_enabled() is False
    assert _train_cfg(amp=True, device="cuda", batch_size=512).amp_enabled() is True


# =====================================================================
# 4. should_eval：最后一轮必须评估
# =====================================================================
def test_last_epoch_is_always_evaluated():
    cfg = _train_cfg(epochs=30, eval_every_n_epochs=5)
    assert cfg.should_eval(30, 30), "最后一个 epoch 必须评估，否则早停看不到它"
    assert cfg.should_eval(5, 30) and cfg.should_eval(10, 30)
    assert not cfg.should_eval(3, 30)


# =====================================================================
# 5. 单轮训练：loss 必须能下降
# =====================================================================
def test_train_one_epoch_reduces_loss():
    """在可学任务上跑两轮，第二轮的平均 loss 必须显著低于第一轮。

    这条测试的价值在于：它同时校验了候选构造、BCE 方向、
    梯度流向和 optimizer.step() 是否真的被调用过 ——
    任何一环接错，loss 都不会降（或直接 NaN）。
    """
    import torch.nn as nn

    set_seed(0)
    model = _model()
    opt = build_optimizer(model, OptimConfig(lr=0.01))
    cfg = _train_cfg(epochs=2)
    crit = nn.BCEWithLogitsLoss()

    loader = ToyLoader(n_batch=8, seed=1)
    l1, n1, _ = train_one_epoch(model, loader, opt, cfg, crit,
                                torch.device("cpu"), n_items=N_ITEMS, epoch=1)
    l2, n2, _ = train_one_epoch(model, loader, opt, cfg, crit,
                                torch.device("cpu"), n_items=N_ITEMS, epoch=2)
    assert math.isfinite(l1) and math.isfinite(l2)
    assert n1 == n2 == 8
    assert l2 < l1, f"loss 没有下降：{l1:.4f} -> {l2:.4f}"


def test_train_one_epoch_updates_parameters():
    """参数确实被更新了（防止"忘了 step"这类低级错误）。"""
    import torch.nn as nn

    set_seed(0)
    model = _model()
    before = model.item_emb.weight.detach().clone()
    opt = build_optimizer(model, OptimConfig(lr=0.01))
    train_one_epoch(model, ToyLoader(n_batch=2), opt, _train_cfg(),
                    nn.BCEWithLogitsLoss(), torch.device("cpu"),
                    n_items=N_ITEMS, epoch=1)
    assert not torch.equal(before, model.item_emb.weight.detach())


def test_grad_clip_limits_gradient_norm():
    """开启梯度裁剪后，参数更新幅度应明显小于不裁剪时。

    构造一个"梯度很大"的场景（很大的 lr + 很尖的 loss），
    比较两种配置下的参数变化量。
    """
    import torch.nn as nn

    def run(clip: float):
        set_seed(0)
        m = _model()
        opt = build_optimizer(m, OptimConfig(lr=1.0))
        cfg = _train_cfg(grad_clip_norm=clip)
        before = m.item_emb.weight.detach().clone()
        train_one_epoch(m, ToyLoader(n_batch=1), opt, cfg,
                        nn.BCEWithLogitsLoss(), torch.device("cpu"),
                        n_items=N_ITEMS, epoch=1)
        return (m.item_emb.weight.detach() - before).abs().max().item()

    assert run(0.0) > run(0.01)


# =====================================================================
# 6. fit：跑满 / 早停 / 快照副本
# =====================================================================
def _constant_eval(value: float):
    def _fn(model):
        return {"ndcg@10": value, "hr@10": value}
    return _fn


def test_fit_runs_all_epochs_without_eval_fn():
    """不给 eval_fn 时不评估、不早停，老老实实跑满 epochs。"""
    model = _model()
    r = fit(model=model, train_loader=ToyLoader(), cfg=_train_cfg(epochs=4),
            optim_cfg=OptimConfig(lr=0.01), eval_fn=None, n_items=N_ITEMS)
    assert r.n_epochs_run == 4
    assert r.stopped_early is False
    assert len(r.records) == 4
    assert r.best_state_dict is not None, "没有评估时也要给出可用的权重快照"
    assert all(rec.metrics == {} for rec in r.records)


def test_fit_early_stops_on_plateau():
    """指标不再提升时应早停，且最优轮记录正确。

    patience=2、eval_every=1 → 连续 2 次评估无提升即停。
    指标序列：0.1 / 0.2 / 0.3（峰） / 0.29 / 0.28 → 应在第 5 轮停。
    """
    seq = [0.1, 0.2, 0.3, 0.29, 0.28, 0.27, 0.26]
    state = {"i": 0}

    def eval_fn(model):
        i = state["i"]
        state["i"] = i + 1
        v = seq[i] if i < len(seq) else 0.0
        return {"ndcg@10": v}

    model = _model()
    r = fit(model=model, train_loader=ToyLoader(), cfg=_train_cfg(
        epochs=20, eval_every_n_epochs=1, early_stop_patience=2),
        optim_cfg=OptimConfig(lr=0.01), eval_fn=eval_fn, n_items=N_ITEMS)

    assert r.stopped_early is True
    assert r.n_epochs_run == 5, f"应在第 5 轮停，实际 {r.n_epochs_run}"
    assert r.best_epoch == 3
    assert r.best_metric == pytest.approx(0.3)


def test_fit_patience_is_in_epochs_not_eval_points():
    """patience 的单位是 epoch：`patience=4, eval_every=2` → 连续 2 次评估无提升。

    这是本项目容易读错的一处约定（见 train.py 模块 docstring 第 1 条）。
    若按"评估次数"理解，patience=4 会变成连续 4 次评估 = 8 个 epoch，
    训练白跑一倍时间。
    """
    seq = [0.5, 0.5, 0.4, 0.3, 0.2, 0.1] + [0.0] * 20
    state = {"i": 0}

    def eval_fn(model):
        i = state["i"]
        state["i"] = i + 1
        return {"ndcg@10": seq[min(i, len(seq) - 1)]}

    r = fit(model=_model(), train_loader=ToyLoader(), cfg=_train_cfg(
        epochs=30, eval_every_n_epochs=2, early_stop_patience=4),
        optim_cfg=OptimConfig(lr=0.01), eval_fn=eval_fn, n_items=N_ITEMS)
    # 评估点：epoch2(0.5,最优) → 4(0.5,平) → 6(0.4,平) → 停在 epoch 6
    assert r.stopped_early is True
    assert r.best_epoch == 2
    assert r.n_epochs_run == 6, f"期望 6，实际 {r.n_epochs_run}"


def test_best_state_dict_is_a_detached_copy():
    """`best_state_dict` 必须是副本 —— 之后继续训练不能改变它。

    如果它只是 `model.state_dict()` 的引用（浅层未拷贝），
    那么训练继续推进后，"最优权重"会悄悄变成最后一轮的权重，
    而报告的 best_metric 还是旧的那个 —— 两者对不上且极难发现。
    """
    model = _model()
    r = fit(model=model, train_loader=ToyLoader(), cfg=_train_cfg(epochs=3),
            optim_cfg=OptimConfig(lr=0.01), eval_fn=_constant_eval(0.1),
            n_items=N_ITEMS)

    snapshot = r.best_state_dict["item_emb.weight"].clone()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)

    assert torch.equal(r.best_state_dict["item_emb.weight"], snapshot), \
        "best_state_dict 被训练更新污染了"
    assert not torch.equal(r.best_state_dict["item_emb.weight"],
                           model.item_emb.weight)


def test_best_state_dict_is_on_cpu():
    """快照必须在 CPU 上：否则留在显存里，多组实验跑完会 OOM。"""
    model = _model()
    r = fit(model=model, train_loader=ToyLoader(), cfg=_train_cfg(epochs=2),
            optim_cfg=OptimConfig(lr=0.01), eval_fn=_constant_eval(0.1),
            n_items=N_ITEMS)
    assert all(v.device.type == "cpu" for v in r.best_state_dict.values())


def test_best_state_dict_can_be_loaded_back_and_matches_metric():
    """把快照 load 回模型后，指标应与记录的最优值一致。

    这一步证明"保存最优权重"这条链路是通的 —— 否则 M2.8 跑实验时
    会拿一个和报告数字对不上的权重去测测试集。
    """
    calls = {"n": 0}

    def eval_fn(model):
        calls["n"] += 1
        # 用参数范数伪装成一个确定的"指标"，保证 load 回来能复现
        v = float(model.item_emb.weight.detach().abs().sum())
        return {"ndcg@10": v}

    model = _model()
    r = fit(model=model, train_loader=ToyLoader(), cfg=_train_cfg(epochs=3),
            optim_cfg=OptimConfig(lr=0.01), eval_fn=eval_fn, n_items=N_ITEMS)

    model.load_state_dict(r.best_state_dict)
    v_after = float(model.item_emb.weight.detach().abs().sum())
    assert v_after == pytest.approx(r.best_metric, rel=1e-5)


def test_fit_report_is_json_serializable():
    """`as_report()` 要能直接 json.dump —— 训练日志靠它落盘。"""
    import json

    model = _model()
    r = fit(model=model, train_loader=ToyLoader(), cfg=_train_cfg(epochs=2),
            optim_cfg=OptimConfig(lr=0.01), eval_fn=_constant_eval(0.7),
            n_items=N_ITEMS)
    json.dumps(r.as_report(), allow_nan=False)


def test_on_epoch_end_callback_is_called_per_epoch():
    seen = []
    fit(model=_model(), train_loader=ToyLoader(), cfg=_train_cfg(epochs=3),
        optim_cfg=OptimConfig(lr=0.01), eval_fn=None, n_items=N_ITEMS,
        on_epoch_end=lambda rec, res: seen.append(rec.epoch))
    assert seen == [1, 2, 3]


def test_fit_rejects_unknown_early_stop_metric():
    """早停指标名写错时必须立刻报错，而不是静默地用 NaN 比较（永远不提升）。"""
    with pytest.raises(KeyError, match="ndcg@99"):
        fit(model=_model(), train_loader=ToyLoader(),
            cfg=_train_cfg(epochs=2, early_stop_metric="ndcg@99"),
            optim_cfg=OptimConfig(lr=0.01),
            eval_fn=lambda m: {"ndcg@10": 0.5}, n_items=N_ITEMS)


def test_fit_rejects_missing_n_items():
    with pytest.raises(ValueError, match="n_items"):
        fit(model=torch.nn.Linear(2, 2), train_loader=ToyLoader(),
            cfg=_train_cfg(epochs=1), optim_cfg=OptimConfig(lr=0.01),
            eval_fn=None, n_items=0)


# =====================================================================
# 7. 可复现性
# =====================================================================
def test_same_seed_yields_same_loss_trajectory():
    """同种子两次训练，逐步 loss 必须逐位相同。

    这条是 `main` 档"3 个种子报均值±标准差"的前提：
    若同种子都不可复现，标准差里就混进了随机噪声。
    """
    import torch.nn as nn

    def run():
        set_seed(123)
        m = _model(seed=0)
        opt = build_optimizer(m, OptimConfig(lr=0.01))
        crit = nn.BCEWithLogitsLoss()
        losses = []
        for e in (1, 2):
            l, _, _ = train_one_epoch(m, ToyLoader(seed=7), opt, _train_cfg(),
                                      crit, torch.device("cpu"),
                                      n_items=N_ITEMS, epoch=e,
                                      neg_seed=123 + e)
            losses.append(l)
        return losses

    assert run() == run()
