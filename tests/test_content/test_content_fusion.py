# -*- coding: utf-8 -*-
"""内容融合模块的单测（M2.6）。

覆盖：矩阵装载与行序对齐、画像构造（含空历史）、score 契约、
fusion_score 自动切权、ContentFusedModel 端到端契约与零参数性。
"""

import math

import pytest
import torch

from models.content_encoder.fusion import (
    COLD_START_THRESHOLD,
    ContentFusedModel,
    ContentScorer,
    fusion_score,
    load_content_matrix,
)

N_ITEMS, DIM, SEQ = 50, 8, 12


# =====================================================================
# load_content_matrix
# =====================================================================
class TestLoadContentMatrix:
    def test_shape_and_pad_row(self, tmp_path):
        import numpy as np
        p = tmp_path / "vec.npy"
        vec = torch.randn(N_ITEMS, DIM).numpy().astype("float32")
        np.save(p, vec)
        mat = load_content_matrix(str(p), N_ITEMS)
        assert mat.shape == (N_ITEMS + 1, DIM)
        assert torch.all(mat[0] == 0.0)                       # PAD 行恒零
        expect = torch.from_numpy(vec[0])
        expect = expect / expect.norm()                        # 第 i-1 行 = idx i（归一后）
        assert torch.allclose(mat[1], expect, atol=1e-5)

    def test_rows_renormalized(self, tmp_path):
        import numpy as np
        p = tmp_path / "vec.npy"
        np.save(p, (torch.randn(N_ITEMS, DIM) * 7.0).numpy().astype("float32"))
        mat = load_content_matrix(str(p), N_ITEMS)
        norms = mat[1:].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_row_count_mismatch_raises(self, tmp_path):
        import numpy as np
        p = tmp_path / "vec.npy"
        np.save(p, np.zeros((N_ITEMS + 3, DIM), dtype="float32"))
        with pytest.raises(ValueError, match="不一致"):
            load_content_matrix(str(p), N_ITEMS)


# =====================================================================
# ContentScorer
# =====================================================================
def _make_scorer(seed=0):
    torch.manual_seed(seed)
    mat = torch.randn(N_ITEMS + 1, DIM)
    mat[0] = 0.0
    return ContentScorer(mat)


class TestContentScorer:
    def test_score_shape_and_contract(self):
        s = _make_scorer()
        x = torch.randint(1, N_ITEMS + 1, (4, SEQ))
        cand = torch.randint(1, N_ITEMS + 1, (4, 6))
        out = s.score(x, cand)
        assert out.shape == (4, 6)

    def test_profile_is_mean_of_history(self):
        s = _make_scorer()
        x = torch.tensor([[3, 5, 5, 0]])                      # 含重复与 PAD
        prof = s.user_profile(x)
        expect = (s.content[3] + s.content[5] + s.content[5]) / 3
        expect = expect / expect.norm()
        assert torch.allclose(prof, expect, atol=1e-6)

    def test_empty_history_gives_zero_scores(self):
        s = _make_scorer()
        x = torch.zeros(2, SEQ, dtype=torch.long)             # 全 PAD
        cand = torch.randint(1, N_ITEMS + 1, (2, 5))
        assert torch.all(s.score(x, cand) == 0.0)

    def test_score_bounded_by_one(self):
        s = _make_scorer()
        x = torch.randint(1, N_ITEMS + 1, (16, SEQ))
        cand = torch.randint(1, N_ITEMS + 1, (16, 20))
        assert s.score(x, cand).abs().max() <= 1.0 + 1e-5

    def test_self_candidate_scores_one(self):
        s = _make_scorer()
        item = torch.tensor([[7]])
        assert torch.allclose(s.score(item, item), torch.ones(1, 1), atol=1e-5)

    def test_pad_items_never_scored(self):
        # 候选里混入 PAD（idx 0）时内容分应为 0（content 第 0 行全零）
        s = _make_scorer()
        x = torch.randint(1, N_ITEMS + 1, (2, SEQ))
        cand = torch.tensor([[0, 3, 0, 9], [0, 5, 0, 2]])      # 第 0/2 列整列是 PAD
        out = s.score(x, cand)
        assert torch.all(out[:, 0] == 0.0) and torch.all(out[:, 2] == 0.0)


# =====================================================================
# fusion_score
# =====================================================================
class TestFusionScore:
    def test_warm_user_uses_73(self):
        b = torch.tensor([[1.0, 0.0]])
        c = torch.tensor([[0.0, 1.0]])
        n = torch.tensor([COLD_START_THRESHOLD])              # 恰在阈值上 = 非冷启动
        out = fusion_score(b, c, n)
        assert torch.allclose(out, torch.tensor([[0.7, 0.3]]))

    def test_cold_user_uses_55(self):
        b = torch.tensor([[1.0, 0.0]])
        c = torch.tensor([[0.0, 1.0]])
        n = torch.tensor([COLD_START_THRESHOLD - 1])
        out = fusion_score(b, c, n)
        assert torch.allclose(out, torch.tensor([[0.5, 0.5]]))

    def test_mixed_batch_row_independent(self):
        b = torch.rand(8, 5)
        c = torch.rand(8, 5)
        n = torch.tensor([0, 5, 9, 10, 11, 50, 1, 10])
        out = fusion_score(b, c, n)
        for i, ni in enumerate(n.tolist()):
            wb, wc = (0.5, 0.5) if ni < COLD_START_THRESHOLD else (0.7, 0.3)
            assert torch.allclose(out[i], wb * b[i] + wc * c[i], atol=1e-6)

    def test_weight_sum_constraint(self):
        with pytest.raises(ValueError, match="必须为 1"):
            fusion_score(torch.rand(2, 3), torch.rand(2, 3),
                         torch.tensor([10, 10]), w_behavior=0.8, w_content=0.3)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="形状不一致"):
            fusion_score(torch.rand(2, 3), torch.rand(2, 4), torch.tensor([10, 10]))


# =====================================================================
# ContentFusedModel
# =====================================================================
class _StubBackbone(torch.nn.Module):
    """行为分数恒等于 (item_id % 7) / 7 的确定性假模型。"""

    def score(self, input_ids, candidate_ids):
        return (candidate_ids.float() % 7) / 7.0


class TestContentFusedModel:
    def _fused(self, **kw):
        torch.manual_seed(1)
        return ContentFusedModel(
            _StubBackbone(), _make_scorer(),
            w_behavior=kw.get("w_behavior", 0.7),
            w_content=kw.get("w_content", 0.3))

    def test_score_contract(self):
        m = self._fused()
        x = torch.randint(1, N_ITEMS + 1, (4, SEQ))
        cand = torch.randint(1, N_ITEMS + 1, (4, 11))
        assert m.score(x, cand).shape == (4, 11)

    def test_equals_manual_fusion(self):
        m = self._fused()
        x = torch.randint(1, N_ITEMS + 1, (8, SEQ))
        cand = torch.randint(1, N_ITEMS + 1, (8, 6))
        b = m.backbone.score(x, cand)
        c = m.scorer.score(x, cand)
        n = (x > 0).sum(dim=1)
        assert torch.allclose(m.score(x, cand), fusion_score(b, c, n), atol=1e-6)

    def test_fusion_adds_zero_params(self):
        m = self._fused()
        n_backbone = sum(p.numel() for p in m.backbone.parameters())
        assert m.n_params == n_backbone

    def test_wraps_real_sasrec(self):
        from models.sasrec.config import SASRecConfig
        from models.sasrec.model import SASRec
        cfg = SASRecConfig(n_items=N_ITEMS, max_seq_len=SEQ + 2,
                           hidden_size=16, num_layers=1, num_heads=2,
                           dropout=0.0, ffn_ratio=2)
        m = ContentFusedModel(SASRec(cfg), _make_scorer())
        x = torch.randint(1, N_ITEMS + 1, (3, SEQ))
        cand = torch.randint(1, N_ITEMS + 1, (3, 5))
        out = m.score(x, cand)
        assert out.shape == (3, 5)
        assert torch.isfinite(out).all()
