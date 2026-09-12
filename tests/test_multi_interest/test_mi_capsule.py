# -*- coding: utf-8 -*-
"""动态路由胶囊层的单元测试 —— `models/multi_interest/capsule.py`"""

import math

import pytest
import torch

from models.multi_interest.capsule import (
    InterestRouting,
    interest_diversity,
    squash,
)

H, K, B, L = 16, 4, 5, 12


# =====================================================================
# squash
# =====================================================================
class TestSquash:
    def test_norm_bounded_by_one(self):
        x = torch.randn(B, H) * 10.0
        y = squash(x)
        assert y.norm(dim=-1).max() <= 1.0 + 1e-6

    def test_direction_preserved(self):
        x = torch.randn(B, H)
        y = squash(x)
        # 余弦相似度为 1（方向不变，只压缩范数）
        cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
        assert torch.allclose(cos, torch.ones(B), atol=1e-5)

    def test_zero_input_is_zero_not_nan(self):
        """全 PAD 用户的票全为零 —— 必须输出全零而不是 NaN（fp16 安全前提）。"""
        y = squash(torch.zeros(B, H))
        assert torch.isfinite(y).all()
        assert y.abs().max() == 0.0

    def test_tiny_input_is_finite(self):
        y = squash(torch.full((B, H), 1e-8))
        assert torch.isfinite(y).all()

    def test_dim_argument(self):
        x = torch.randn(B, K, H)
        y = squash(x, dim=-1)
        assert y.shape == x.shape
        assert y.norm(dim=-1).max() <= 1.0 + 1e-6


# =====================================================================
# InterestRouting：构造与校验
# =====================================================================
class TestRoutingConstruction:
    def test_valid_construction(self):
        r = InterestRouting(H, K, routing_iters=3)
        # 唯一可训练参数是低维投影 W：H x (K*H)
        n_train = sum(p.numel() for p in r.parameters() if p.requires_grad)
        assert n_train == H * K * H

    @pytest.mark.parametrize("k", [0, 1, -2])
    def test_num_interests_must_be_at_least_two(self, k):
        with pytest.raises(ValueError, match="num_interests"):
            InterestRouting(H, k, routing_iters=3)

    @pytest.mark.parametrize("iters", [0, -1])
    def test_routing_iters_must_be_positive(self, iters):
        with pytest.raises(ValueError, match="routing_iters"):
            InterestRouting(H, K, routing_iters=iters)

    def test_dropout_out_of_range(self):
        with pytest.raises(ValueError, match="dropout"):
            InterestRouting(H, K, routing_iters=3, dropout=1.0)

    def test_bad_hidden_dim_on_forward(self):
        r = InterestRouting(H, K, routing_iters=3)
        with pytest.raises(ValueError, match="最后一维"):
            r(torch.randn(B, L, H + 1), torch.ones(B, L, dtype=torch.bool))


# =====================================================================
# InterestRouting：前向行为
# =====================================================================
class TestRoutingForward:
    @pytest.fixture()
    def routing(self):
        torch.manual_seed(7)
        return InterestRouting(H, K, routing_iters=3, dropout=0.0).eval()

    def test_output_shape(self, routing):
        hidden = torch.randn(B, L, H)
        valid = torch.ones(B, L, dtype=torch.bool)
        out = routing(hidden, valid)
        assert out.shape == (B, K, H)

    def test_interests_norm_bounded(self, routing):
        out = routing(torch.randn(B, L, H), torch.ones(B, L, dtype=torch.bool))
        assert out.norm(dim=-1).max() <= 1.0 + 1e-5

    def test_all_pad_row_gives_zero_no_nan(self):
        """整行 PAD（滑动窗口第 0 个窗口的空序列）：输出必须全零且无 NaN。"""
        r = InterestRouting(H, K, routing_iters=3).eval()
        hidden = torch.randn(B, L, H)          # PAD 位置的 hidden 即使有值也必须被挡掉
        valid = torch.zeros(B, L, dtype=torch.bool)
        out = r(hidden, valid)
        assert torch.isfinite(out).all()
        assert out.abs().max() == 0.0

    def test_pad_positions_do_not_contribute(self, routing):
        """PAD 位置票数置零：带垃圾值的 PAD 与置零的 PAD 输出严格一致。"""
        torch.manual_seed(3)
        valid = torch.zeros(B, L, dtype=torch.bool)
        valid[:, L // 2:] = True               # 左填充：前半是 PAD

        hidden = torch.randn(B, L, H)
        hidden_zeros = hidden.clone()
        hidden_zeros[:, :L // 2] = 0.0         # PAD 置零版

        out_garbage = routing(hidden, valid)
        out_zeroed = routing(hidden_zeros, valid)
        assert torch.allclose(out_garbage, out_zeroed, atol=1e-6)

    def test_deterministic_in_eval(self, routing):
        hidden = torch.randn(B, L, H)
        valid = torch.ones(B, L, dtype=torch.bool)
        out1 = routing(hidden, valid)
        out2 = routing(hidden, valid)
        assert torch.equal(out1, out2)

    def test_iterations_change_output(self):
        """路由迭代次数不同，输出应当不同（退化成恒等映射就说明写错了）。"""
        torch.manual_seed(11)
        r1 = InterestRouting(H, K, routing_iters=1).eval()
        r3 = InterestRouting(H, K, routing_iters=3).eval()
        r3.load_state_dict(r1.state_dict())    # 同权重，只差迭代次数
        hidden = torch.randn(B, L, H)
        valid = torch.ones(B, L, dtype=torch.bool)
        assert not torch.allclose(r1(hidden, valid), r3(hidden, valid))

    def test_gradients_flow_to_projection(self):
        r = InterestRouting(H, K, routing_iters=3)
        hidden = torch.randn(B, L, H, requires_grad=False)
        valid = torch.ones(B, L, dtype=torch.bool)
        out = r(hidden, valid)
        out.sum().backward()
        assert r.proj.weight.grad is not None
        assert torch.isfinite(r.proj.weight.grad).all()
        # PAD 位置的票被置零 => proj 对 PAD 位置的输入行没有梯度贡献
        # （全 PAD 输入时梯度应为 0）
        r.zero_grad()
        out = r(torch.randn(B, L, H), torch.zeros(B, L, dtype=torch.bool))
        out.sum().backward()
        assert r.proj.weight.grad.abs().max() == 0.0


# =====================================================================
# interest_diversity 诊断
# =====================================================================
class TestInterestDiversity:
    def test_identical_interests_near_one(self):
        x = torch.randn(1, H)
        interests = x.unsqueeze(1).expand(B, K, H).clone()
        assert interest_diversity(interests) > 0.99

    def test_orthogonal_interests_near_zero(self):
        eye = torch.eye(K, H)
        interests = eye.unsqueeze(0).expand(B, K, H)
        assert interest_diversity(interests) < 1e-4

    def test_too_few_interests_returns_zero(self):
        assert interest_diversity(torch.randn(B, 1, H)) == 0.0

    def test_returns_python_float(self):
        v = interest_diversity(torch.randn(B, K, H))
        assert isinstance(v, float) and math.isfinite(v)
