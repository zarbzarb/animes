# -*- coding: utf-8 -*-
"""候选侧内容融合的单测（M2.6b）。

覆盖：provider 属性（PAD 恒零 / 参数增量 / 内容不可学 / 行序校验）、
mode 分支、SASRec 注入后的契约与回归一致性、工厂装配、端到端可训练性。
"""

import pytest
import torch

from models.content_encoder.model import (
    ItemContentFusion,
    build_model,
    load_content_matrix_for_fusion,
)
from models.sasrec.config import SASRecConfig, TrainConfig, OptimConfig
from models.sasrec.model import PAD_ITEM, SASRec
from models.sasrec.train import fit

N_ITEMS, DIM, H, SEQ = 40, 6, 16, 10


def _content(seed=0, n_items=N_ITEMS, dim=DIM):
    g = torch.Generator().manual_seed(seed)
    mat = torch.randn(n_items + 1, dim, generator=g)
    mat[PAD_ITEM] = 0.0
    return mat


# =====================================================================
# ItemContentFusion
# =====================================================================
class TestItemContentFusion:
    def test_table_shape_and_pad_zero(self):
        p = ItemContentFusion(_content(), hidden_size=H)
        w = torch.randn(N_ITEMS + 1, H)
        t = p(w)
        assert t.shape == (N_ITEMS + 1, H)
        assert float(t[PAD_ITEM].detach().abs().sum()) == 0.0

    def test_content_is_buffer_not_parameter(self):
        p = ItemContentFusion(_content(), hidden_size=H)
        names = [n for n, _ in p.named_parameters()]
        assert names == ["proj.weight", "proj.bias"]

    @pytest.mark.parametrize("mode", ["concat", "add"])
    def test_param_count_matches_formula(self, mode):
        p = ItemContentFusion(_content(), hidden_size=H, mode=mode)
        expect = (H + DIM) * H + H if mode == "concat" else DIM * H + H
        assert p.n_params == expect

    def test_add_mode_starts_exactly_at_baseline(self):
        """add 模式零初始化：第 0 步的 table 与纯基座逐位相同。

        这是它相对 concat 的核心价值 —— 起点严格等价基线，之后只有在
        内容确实降损失时参数才离开 0，"内容有没有增量"因此可归因
        （concat 的 Linear 会自由重学整个物品空间，实测融合表与原嵌入
        余弦仅 0.0097，等于把 tie-embedding 已学到的表示又学一遍）。
        """
        torch.manual_seed(11)
        w = torch.randn(N_ITEMS + 1, H)
        w[PAD_ITEM] = 0.0
        p = ItemContentFusion(_content(), hidden_size=H, mode="add")
        assert torch.allclose(p(w), w, atol=0.0)
        assert float(p(w)[PAD_ITEM].detach().abs().sum()) == 0.0

    def test_add_mode_learns_increment(self):
        """零初始化不等于学不动：一步梯度后残差必须离开 0。"""
        p = ItemContentFusion(_content(), hidden_size=H, mode="add")
        w = torch.randn(N_ITEMS + 1, H, requires_grad=True)
        p(w).sum().backward()
        assert p.proj.weight.grad is not None
        assert float(p.proj.weight.grad.abs().sum()) > 0.0
        with torch.no_grad():
            p.proj.weight.add_(-0.1 * p.proj.weight.grad)
        assert not torch.allclose(p(w.detach()), w.detach(), atol=1e-6)

    def test_pad_row_gets_no_gradient(self):
        p = ItemContentFusion(_content(), hidden_size=H)
        w = torch.randn(N_ITEMS + 1, H, requires_grad=True)
        p(w).sum().backward()
        assert p.proj.weight.grad is not None
        assert torch.isfinite(p.proj.weight.grad).all()

    def test_content_normalized_defensively(self):
        raw = _content() * 5.0
        p = ItemContentFusion(raw, hidden_size=H)
        norms = p.content[1:].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_bias_starts_at_zero(self):
        p = ItemContentFusion(_content(), hidden_size=H)
        assert torch.all(p.proj.bias == 0.0)

    def test_row_count_mismatch_raises(self):
        p = ItemContentFusion(_content(), hidden_size=H)
        with pytest.raises(ValueError, match="不一致"):
            p(torch.randn(N_ITEMS + 5, H))

    def test_bad_hidden_size_raises(self):
        p = ItemContentFusion(_content(), hidden_size=H)
        with pytest.raises(ValueError, match="应为"):
            p(torch.randn(N_ITEMS + 1, H + 1))

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            ItemContentFusion(_content(), hidden_size=H, mode="multiply")

    @pytest.mark.parametrize("mode", ["concat", "add"])
    def test_config_snapshot(self, mode):
        p = ItemContentFusion(_content(), hidden_size=H, mode=mode)
        snap = p.config_snapshot()
        assert snap["mode"] == mode
        assert snap["content_dim"] == DIM
        assert snap["n_params"] == p.n_params

    def test_default_mode_is_add(self):
        """默认口径锁定为 add（零初始化残差）。

        回归保护：默认值一旦漂回 concat，所有实验组的口径都会被悄悄改掉
        （concat 在 SASRec 上整模型 −1.1pp），而权重形状一致、加载不报错、
        指标也能算出来 —— 属于最难发现的错。见 progress.md §4.1 M2.6b。
        """
        assert ItemContentFusion(_content(), hidden_size=H).mode == "add"
        assert build_model("sasrec", _sasrec_cfg(), N_ITEMS,
                           _content()).repr_provider.mode == "add"


# =====================================================================
# SASRec 注入
# =====================================================================
def _sasrec_cfg():
    return SASRecConfig(n_items=N_ITEMS, max_seq_len=SEQ + 2, hidden_size=H,
                        num_layers=1, num_heads=2, dropout=0.0, ffn_ratio=2)


class TestSASRecInjection:
    def test_no_provider_is_identity(self):
        m = SASRec(_sasrec_cfg())
        assert m.item_table() is m.item_emb.weight

    def test_injected_table_used_for_both_sides(self):
        m = SASRec(_sasrec_cfg(), repr_provider=ItemContentFusion(
            _content(), hidden_size=H))
        table = m.item_table()
        assert table.shape == (N_ITEMS + 1, H)
        assert float(table[PAD_ITEM].detach().abs().sum()) == 0.0
        # 输入侧与打分侧必须看到同一张表
        x = torch.randint(1, N_ITEMS + 1, (3, SEQ))
        cand = torch.randint(1, N_ITEMS + 1, (3, 5))
        out = m.score(x, cand)
        assert out.shape == (3, 5) and torch.isfinite(out).all()

    def test_contract_unchanged(self):
        m = SASRec(_sasrec_cfg(), repr_provider=ItemContentFusion(
            _content(), hidden_size=H))
        x = torch.randint(1, N_ITEMS + 1, (4, SEQ))
        cand = torch.randint(1, N_ITEMS + 1, (4, 11))
        hidden, user_repr = m.encode(x)
        assert hidden.shape == (4, SEQ, H) and user_repr.shape == (4, H)
        assert m.score(x, cand).shape == (4, 11)

    def test_pad_row_still_gets_no_gradient(self):
        m = SASRec(_sasrec_cfg(), repr_provider=ItemContentFusion(
            _content(), hidden_size=H))
        x = torch.zeros(2, SEQ, dtype=torch.long)     # 全 PAD 序列
        cand = torch.randint(1, N_ITEMS + 1, (2, 3))
        m.score(x, cand).sum().backward()
        assert float(m.item_emb.weight.grad[PAD_ITEM].detach().abs().sum()) == 0.0

    def test_matches_plain_sasrec_when_provider_learns_nothing(self):
        """provider 用「忽略内容」的权重时，输出应与裸 SASRec 一致。

        这是回归保护的核心用例：证明注入本身不改变数值路径。
        """
        torch.manual_seed(3)
        cfg = _sasrec_cfg()
        plain = SASRec(cfg)
        # ⚠️ 显式指定 concat：本用例用 W = [I | 0] 构造恒等，只有 concat 的
        # 输入维是 (H + D) 才写得出来；add 口径的等价用例是
        # test_add_mode_starts_exactly_at_baseline（零初始化即恒等）。
        fused = SASRec(cfg, repr_provider=ItemContentFusion(
            torch.zeros(N_ITEMS + 1, DIM), hidden_size=H, mode="concat"))
        # 让 provider 退化成恒等：W = [I | 0]，b = 0；
        # ⚠️ 必须整体复制 state_dict —— transformer 块的随机初始化在两个实例
        # 之间是不同的，只复制嵌入会让差异被误判成"注入改变了数值路径"
        with torch.no_grad():
            fused.load_state_dict(plain.state_dict(), strict=False)
            fused.repr_provider.proj.weight.zero_()
            fused.repr_provider.proj.weight[:, :H] = torch.eye(H)
            fused.repr_provider.proj.bias.zero_()
        assert torch.allclose(plain.item_table(), fused.item_table(), atol=1e-7)
        x = torch.randint(1, N_ITEMS + 1, (4, SEQ))
        cand = torch.randint(1, N_ITEMS + 1, (4, 7))
        assert torch.allclose(plain.score(x, cand), fused.score(x, cand), atol=1e-6)

    def test_all_pad_sequence_no_nan(self):
        m = SASRec(_sasrec_cfg(), repr_provider=ItemContentFusion(
            _content(), hidden_size=H))
        x = torch.zeros(2, SEQ, dtype=torch.long)
        cand = torch.randint(1, N_ITEMS + 1, (2, 4))
        assert torch.isfinite(m.score(x, cand)).all()


# =====================================================================
# 工厂
# =====================================================================
class TestBuildModel:
    def test_sasrec_without_content(self):
        m = build_model("sasrec", _sasrec_cfg(), N_ITEMS)
        assert m.repr_provider is None

    @pytest.mark.parametrize("mode,extra", [
        ("concat", (H + DIM) * H + H),
        ("add", DIM * H + H),
    ])
    def test_sasrec_with_content(self, mode, extra):
        m = build_model("sasrec", _sasrec_cfg(), N_ITEMS, _content(),
                        content_mode=mode)
        assert isinstance(m.repr_provider, ItemContentFusion)
        assert m.repr_provider.mode == mode
        base = SASRec(_sasrec_cfg()).n_params
        assert m.n_params == base + extra

    def test_multi_interest_with_content(self):
        from models.multi_interest.config import MultiInterestConfig
        mi = MultiInterestConfig.from_dict(
            dict(hidden_size=H, num_layers=1, num_heads=2, dropout=0.0,
                 max_seq_len=SEQ + 2, num_interests=3, routing_iters=2),
            n_items=N_ITEMS)
        m = build_model("multi_interest", None, N_ITEMS, _content(), mi_cfg=mi)
        assert isinstance(m.backbone.repr_provider, ItemContentFusion)
        x = torch.randint(1, N_ITEMS + 1, (2, SEQ))
        cand = torch.randint(1, N_ITEMS + 1, (2, 5))
        assert m.score(x, cand).shape == (2, 5)

    @pytest.mark.parametrize("mode,extra", [
        ("add", DIM * H + H),           # 默认口径
        ("concat", (H + DIM) * H + H),
    ])
    def test_provider_reachable_at_top_level_for_all_archs(self, mode, extra):
        """`model.repr_provider` 必须在**两种 arch、两种 mode 上都可用**。

        回归保护：多兴趣模型的 provider 实际挂在 `backbone` 上，训练脚本
        按 `model.repr_provider` 取参数量时 AttributeError 崩过一次
        （2026-09-14，--model multi_interest --content-fusion 首次运行）。
        """
        from models.multi_interest.config import MultiInterestConfig
        mi = MultiInterestConfig.from_dict(
            dict(hidden_size=H, num_layers=1, num_heads=2, dropout=0.0,
                 max_seq_len=SEQ + 2, num_interests=3, routing_iters=2),
            n_items=N_ITEMS)
        m2 = build_model("multi_interest", None, N_ITEMS, _content(),
                         content_mode=mode, mi_cfg=mi)
        assert m2.repr_provider is m2.backbone.repr_provider
        assert m2.repr_provider.n_params == extra
        m1 = build_model("sasrec", _sasrec_cfg(), N_ITEMS, _content(),
                         content_mode=mode)
        assert m1.repr_provider.n_params == extra

    def test_row_count_validation(self):
        with pytest.raises(ValueError, match="行数"):
            build_model("sasrec", _sasrec_cfg(), N_ITEMS,
                        torch.zeros(N_ITEMS, DIM))

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError, match="未知 arch"):
            build_model("transformer", _sasrec_cfg(), N_ITEMS)

    def test_loader_matches_pipeline_convention(self, tmp_path):
        import numpy as np
        p = tmp_path / "vec.npy"
        np.save(p, np.random.randn(N_ITEMS, DIM).astype("float32"))
        mat = load_content_matrix_for_fusion(str(p), N_ITEMS)
        assert mat.shape == (N_ITEMS + 1, DIM)
        assert float(mat[PAD_ITEM].abs().sum()) == 0.0


# =====================================================================
# 默认口径的配置一致性（四处不同步 = 静默错配）
# =====================================================================
class TestDefaultModeConfigConsistency:
    """`configs/` 与代码默认必须同为 add。

    M2.6b 起默认口径 = add（零初始化残差）。口径分散在四处：类默认、
    工厂默认、`configs/model.yaml`、`configs/experiment.yaml` 的 E2 组。
    任何一处漂回 concat，实验就会在「权重形状一致、指标也算得出来」的
    情况下换掉口径，所以配置也一并锁住。
    """

    @staticmethod
    def _load(name):
        import yaml
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        with open(root / "configs" / name, encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def test_model_yaml_default_is_add(self):
        assert self._load("model.yaml")["content_fusion"]["mode"] == "add"

    def test_experiment_yaml_e2_uses_add(self):
        groups = self._load("experiment.yaml")["E2_ablation"]["groups"]
        for key in ("E2_3_content_fusion", "E2_4_full"):
            assert groups[key]["content_mode"] == "add", key


# =====================================================================
# 端到端可训练性（冒烟）
# =====================================================================
class TestTrainable:
    def test_loss_decreases(self):
        torch.manual_seed(0)
        cfg = _sasrec_cfg()
        model = build_model("sasrec", cfg, N_ITEMS, _content())
        g = torch.Generator().manual_seed(1)
        x = torch.randint(1, N_ITEMS + 1, (256, SEQ), generator=g)
        y = x[:, -1].clone()
        tc = TrainConfig(batch_size=64, epochs=5, device="cpu", amp=False,
                         eval_every_n_epochs=1, early_stop_patience=0)
        res = fit(model=model, train_loader=list(zip(x.split(64), y.split(64))),
                  cfg=tc, optim_cfg=OptimConfig(lr=0.01), n_items=N_ITEMS,
                  eval_fn=None)
        assert res.n_epochs_run == 5
        # 玩具任务的唯一目的是验证"接线通、梯度能流"，不是收敛质量：
        # 随机序列预测最后一位本身不可学，只要求 5 轮内严格下降
        assert res.records[-1].train_loss < res.records[0].train_loss * 0.8

    def test_only_projection_receives_content_gradient(self):
        """内容矩阵是 buffer：反向后不应出现在任何参数的梯度来源里。"""
        model = build_model("sasrec", _sasrec_cfg(), N_ITEMS, _content())
        provider = model.repr_provider
        assert provider.content.requires_grad is False
        x = torch.randint(1, N_ITEMS + 1, (2, SEQ))
        cand = torch.randint(1, N_ITEMS + 1, (2, 3))
        model.score(x, cand).sum().backward()
        assert provider.proj.weight.grad is not None


# =====================================================================
# checkpoint 往返（防「权重加载成功但语义不同」的静默错误）
# =====================================================================
class TestCheckpointRoundTrip:
    def _save(self, tmp_path, model, extra_meta=None):
        import numpy as np
        from models.checkpoint.io import save_checkpoint

        mat = tmp_path / "content.npy"
        np.save(mat, np.random.RandomState(0).randn(N_ITEMS, DIM) * 0.3)
        meta = {
            "model_config": model.config_snapshot(),
            "content_fusion": {**model.repr_provider.config_snapshot(),
                               "file": str(mat)},
        }
        if extra_meta:
            meta.update(extra_meta)
        return save_checkpoint(str(tmp_path / "ckpt.pt"), model, meta=meta)

    @pytest.mark.parametrize("mode", ["add", "concat"])
    def test_roundtrip_is_bitwise_identical(self, tmp_path, mode):
        from models.checkpoint.io import load_model_from_checkpoint

        torch.manual_seed(5)
        model = build_model("sasrec", _sasrec_cfg(), N_ITEMS, _content(),
                            content_mode=mode)
        model.eval()
        path = self._save(tmp_path, model)
        loaded, meta = load_model_from_checkpoint(path)
        loaded.eval()
        assert isinstance(loaded.repr_provider, ItemContentFusion)
        assert loaded.repr_provider.mode == mode
        # 内容矩阵必须跟着一起回来（buffer 也在 state_dict 里）
        assert torch.allclose(loaded.repr_provider.content,
                              model.repr_provider.content)
        x = torch.randint(1, N_ITEMS + 1, (4, SEQ))
        cand = torch.randint(1, N_ITEMS + 1, (4, 6))
        assert torch.allclose(model.score(x, cand), loaded.score(x, cand), atol=1e-6)

    def test_no_fusion_meta_rebuilds_plain(self, tmp_path):
        from models.checkpoint.io import load_model_from_checkpoint, save_checkpoint

        plain = SASRec(_sasrec_cfg())
        p = save_checkpoint(str(tmp_path / "plain.pt"), plain,
                            meta={"model_config": plain.config_snapshot()})
        loaded, meta = load_model_from_checkpoint(p)
        assert loaded.repr_provider is None

    def test_missing_content_file_raises(self, tmp_path):
        from models.checkpoint.io import load_model_from_checkpoint, save_checkpoint

        model = build_model("sasrec", _sasrec_cfg(), N_ITEMS, _content())
        p = save_checkpoint(str(tmp_path / "ckpt.pt"), model, meta={
            "model_config": model.config_snapshot(),
            "content_fusion": {**model.repr_provider.config_snapshot(),
                               "file": "data/features/__not_exist__.npy"},
        })
        with pytest.raises(FileNotFoundError, match="内容矩阵"):
            load_model_from_checkpoint(p)

    def test_content_fusion_meta_without_file_raises(self, tmp_path):
        from models.checkpoint.io import load_model_from_checkpoint, save_checkpoint

        model = build_model("sasrec", _sasrec_cfg(), N_ITEMS, _content())
        p = save_checkpoint(str(tmp_path / "ckpt.pt"), model, meta={
            "model_config": model.config_snapshot(),
            "content_fusion": {"mode": "concat"},      # 缺 file
        })
        with pytest.raises(ValueError, match="file"):
            load_model_from_checkpoint(p)
