# -*- coding: utf-8 -*-
"""多兴趣模型的单元测试 —— `models/multi_interest/model.py`"""

import pytest
import torch

from models.checkpoint.io import load_model_from_checkpoint, save_checkpoint
from models.multi_interest.config import MultiInterestConfig
from models.multi_interest.model import MultiInterestSASRec
from models.sasrec.config import SASRecConfig
from models.sasrec.model import SASRec
from models.sasrec.train import fit

N_ITEMS, SEQ_LEN = 97, 12          # 用小配置跑 CPU，几秒内出结果
MI_KWARGS = {"num_interests": 4, "routing_iters": 3}


def make_cfg(**overrides) -> MultiInterestConfig:
    base = dict(
        max_seq_len=SEQ_LEN + 2,   # input_cap = SEQ_LEN
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,               # 测试里关 dropout，保证 eval 前向确定
        ffn_ratio=2,
    )
    base.update(overrides)
    return MultiInterestConfig.from_dict(base, n_items=N_ITEMS)


def random_batch(batch_size=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(1, N_ITEMS + 1, (batch_size, SEQ_LEN), generator=g)
    # 造一些部分 PAD 的行（左填充）
    x[0, :4] = 0
    x[3, :SEQ_LEN] = 0             # 整行 PAD
    cand = torch.randint(1, N_ITEMS + 1, (batch_size, 11), generator=g)
    return x, cand


# =====================================================================
# 配置
# =====================================================================
class TestConfig:
    def test_yaml_keys_are_split(self):
        cfg = MultiInterestConfig.from_dict(
            {"hidden_size": 16, "num_layers": 1, **MI_KWARGS}, n_items=N_ITEMS)
        assert cfg.num_interests == 4
        assert cfg.routing_iters == 3
        assert cfg.sasrec.hidden_size == 16
        assert cfg.n_items == N_ITEMS

    def test_num_interests_validated(self):
        with pytest.raises(ValueError, match="num_interests"):
            MultiInterestConfig.from_dict(
                {"hidden_size": 16, "num_interests": 1}, n_items=N_ITEMS)

    def test_snapshot_roundtrip(self):
        cfg = make_cfg()
        model = MultiInterestSASRec(cfg)
        snap = model.config_snapshot()
        assert snap["arch"] == "multi_interest"
        # 快照含派生字段（input_cap / n_params），from_dict 必须能忽略它们
        rebuilt = MultiInterestConfig.from_dict(snap, n_items=snap["n_items"])
        assert rebuilt.num_interests == cfg.num_interests
        assert rebuilt.routing_iters == cfg.routing_iters
        assert rebuilt.sasrec.hidden_size == cfg.sasrec.hidden_size


# =====================================================================
# 模型结构
# =====================================================================
class TestStructure:
    def test_param_increment_is_routing_only(self):
        """E2 消融的公平性前提：参数增量只有路由投影 W。"""
        cfg = make_cfg()
        mi = MultiInterestSASRec(cfg)
        base = SASRec(cfg.sasrec)
        h = cfg.sasrec.hidden_size
        assert mi.n_params - base.n_params == h * cfg.num_interests * h

    def test_backbone_uses_same_config(self):
        cfg = make_cfg()
        mi = MultiInterestSASRec(cfg)
        assert mi.backbone.cfg is cfg.sasrec

    def test_cfg_n_items_chain(self):
        """fit() 里 `model.cfg.n_items` 的取值链路必须通。"""
        cfg = make_cfg()
        mi = MultiInterestSASRec(cfg)
        assert mi.cfg.n_items == N_ITEMS


# =====================================================================
# 前向与打分契约
# =====================================================================
class TestForward:
    @pytest.fixture()
    def model(self):
        torch.manual_seed(42)
        return MultiInterestSASRec(make_cfg()).eval()

    def test_encode_interests_shape(self, model):
        x, _ = random_batch()
        hidden, interests = model.encode_interests(x)
        assert hidden.shape == (6, SEQ_LEN, 16)
        assert interests.shape == (6, 4, 16)

    def test_mask_none_equals_explicit(self, model):
        x, _ = random_batch()
        h1, i1 = model.encode_interests(x)
        h2, i2 = model.encode_interests(x, mask=(x != 0))
        assert torch.equal(h1, h2)
        assert torch.equal(i1, i2)

    def test_all_pad_row_no_nan(self, model):
        x, _ = random_batch()
        hidden, interests = model.encode_interests(x)
        assert torch.isfinite(hidden).all()
        assert torch.isfinite(interests).all()

    def test_score_shape_and_finite(self, model):
        x, cand = random_batch()
        scores = model.score(x, cand)
        assert scores.shape == (6, 11)
        assert torch.isfinite(scores).all()

    def test_score_deterministic_in_eval(self, model):
        x, cand = random_batch()
        s1 = model.score(x, cand)
        s2 = model.score(x, cand)
        assert torch.equal(s1, s2)

    def test_score_on_cpu_candidates(self, model):
        """候选在 CPU、模型权重也在 CPU（手写测试的常见路径）。"""
        x, cand = random_batch()
        assert model.score(x, cand).shape == (6, 11)

    def test_score_fp16_cpu_autocast_no_nan(self, model):
        """bf16 autocast 下不产生 NaN（GPU fp16 的数值行为与之同族）。"""
        x, cand = random_batch()
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            scores = model.score(x, cand)
        assert torch.isfinite(scores.float()).all()

    def test_interest_diversity_method(self, model):
        x, _ = random_batch()
        v = model.interest_diversity(x)
        assert isinstance(v, float) and -1.0 <= v <= 1.0


# =====================================================================
# 训练冒烟（复用 models/sasrec/train.py 的 fit，验证接线）
# =====================================================================
class TestTrainingSmoke:
    def test_fit_reduces_loss_on_toy_task(self):
        """可学习的玩具任务：fit() 两轮后 loss 应显著下降。

        玩具任务设计：目标恒等于序列最后一个物品（模型只要学会
        「抄最近一题」就能把 loss 压到 0），保证 loss 下降反映的是
        学习能力而不是随机波动。
        """
        torch.manual_seed(0)
        cfg = make_cfg()
        model = MultiInterestSASRec(cfg)

        g = torch.Generator().manual_seed(1)
        n, batch = 256, 64
        x = torch.randint(1, N_ITEMS + 1, (n, SEQ_LEN), generator=g)
        y = x[:, -1].clone()           # 目标 = 最后一个物品

        from models.sasrec.config import OptimConfig, TrainConfig
        tc = TrainConfig(batch_size=batch, epochs=5, device="cpu", amp=False,
                         eval_every_n_epochs=1, early_stop_patience=0)
        loader = list(zip(x.split(batch), y.split(batch)))

        result = fit(model=model, train_loader=loader, cfg=tc,
                     optim_cfg=OptimConfig(lr=0.01), n_items=N_ITEMS,
                     eval_fn=None)
        assert result.n_epochs_run == 5
        first_loss = result.records[0].train_loss
        assert result.last_loss < first_loss

    def test_fit_reports_n_items_from_cfg(self):
        """不显式传 n_items 时，fit 应能从 model.cfg.n_items 拿到。"""
        torch.manual_seed(0)
        model = MultiInterestSASRec(make_cfg())
        x = torch.randint(1, N_ITEMS + 1, (32, SEQ_LEN))
        y = x[:, -1].clone()

        from models.sasrec.config import OptimConfig, TrainConfig
        tc = TrainConfig(batch_size=16, epochs=1, device="cpu", amp=False,
                         eval_every_n_epochs=1, early_stop_patience=0)
        result = fit(model=model, train_loader=list(zip(x.split(16), y.split(16))),
                     cfg=tc, optim_cfg=OptimConfig(lr=0.01), n_items=None,
                     eval_fn=None)
        assert result.n_epochs_run == 1


# =====================================================================
# checkpoint 往返
# =====================================================================
class TestCheckpoint:
    def test_save_and_auto_reload(self, tmp_path):
        torch.manual_seed(0)
        model = MultiInterestSASRec(make_cfg()).eval()
        x, cand = random_batch()
        scores_before = model.score(x, cand)

        path = str(tmp_path / "mi_test_best.pt")
        save_checkpoint(path, model, meta={
            "arch": "multi_interest",
            "model_config": model.config_snapshot(),
        })

        # 不传 model：加载器应按 meta["arch"] 自动重建多兴趣模型
        loaded, meta = load_model_from_checkpoint(path)
        assert isinstance(loaded, MultiInterestSASRec)
        assert meta["arch"] == "multi_interest"
        loaded.eval()
        assert torch.equal(loaded.score(x, cand), scores_before)

    def test_snapshot_without_n_items_fails(self, tmp_path):
        torch.manual_seed(0)
        model = MultiInterestSASRec(make_cfg())
        path = str(tmp_path / "mi_bad.pt")
        bad_snapshot = {k: v for k, v in model.config_snapshot().items()
                        if k != "n_items"}
        save_checkpoint(path, model, meta={
            "arch": "multi_interest", "model_config": bad_snapshot})
        with pytest.raises(ValueError, match="n_items"):
            load_model_from_checkpoint(path)
