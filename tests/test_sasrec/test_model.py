# -*- coding: utf-8 -*-
"""`models/sasrec/model.py` 的单元测试。

按"最容易静默出错"的顺序组织：

1. **因果性** —— 改未来物品，前面位置的输出必须逐位不变。
   错了会让离线指标虚高，而且"变好看了"没人会查。
2. **mask 的正确性** —— PAD 位置不得被真实位置 attend。
   错了表现为"填充长度会轻微影响指标"，极难定位。
3. **全 PAD 行不得产生 NaN** —— 滑动窗口的第 0 个样本输入是空序列；
   softmax 全 -inf 会出 NaN，并顺着残差污染整个 batch。
4. **score 契约** —— 第 0 列是正样本、返回形状 [B, C]，
   这是 `models/eval/evaluator.py` 直接吃 `model.score` 的前提。

期望值全部手算或由 torch 的基本算子独立算出（不用模型自己的中间结果
当期望值），否则模型一起算错时测试会跟着一起错。
"""

from __future__ import annotations

import pytest
import torch

from models.sasrec.config import SASRecConfig
from models.sasrec.model import PAD_ITEM, SASRec, causal_pad_allow_mask

# 小配置：input_cap = 6 - 2 = 4，便于手算
N_ITEMS = 50
MAX_SEQ_LEN = 6
CAP = MAX_SEQ_LEN - 2


def _cfg(**kw) -> SASRecConfig:
    base = dict(n_items=N_ITEMS, max_seq_len=MAX_SEQ_LEN, hidden_size=16,
                num_layers=2, num_heads=2, dropout=0.0, ffn_ratio=2)
    base.update(kw)
    return SASRecConfig(**base)


def _model(**kw) -> SASRec:
    torch.manual_seed(0)
    m = SASRec(_cfg(**kw))
    m.eval()          # 关掉 dropout，保证确定性
    return m


def _seqs():
    """三行：正常、只有 1 个物品、整行全 PAD（滑动窗口的第 0 个样本）。"""
    x = torch.zeros(3, CAP, dtype=torch.long)
    x[0, :] = torch.tensor([3, 7, 11, 5])
    x[1, -1] = torch.tensor(9)
    # x[2] 保持全 PAD
    return x


# =====================================================================
# 1. 形状与基础契约
# =====================================================================
def test_encode_shapes():
    m = _model()
    x = _seqs()
    with torch.no_grad():
        hidden, urep = m.encode(x)
    assert hidden.shape == (3, CAP, 16)
    assert urep.shape == (3, 16)


def test_forward_returns_hidden_only():
    m = _model()
    with torch.no_grad():
        out = m(_seqs())
    assert out.shape == (3, CAP, 16)


def test_input_cap_matches_dataset_convention():
    """模型侧的输入上限必须与 Dataset 的 `max_seq_len - 2` 一致。

    两处曾经是各写各的常量（`model.py` 与 `dataset.py`），
    只改一处就会出现"Dataset 填 48、模型位置嵌入只到 46"这类越界。
    """
    from models.sasrec.dataset import TARGET_SLOTS

    assert _cfg().input_cap == MAX_SEQ_LEN - TARGET_SLOTS
    assert _cfg().input_cap == CAP


def test_rejects_too_long_input():
    m = _model()
    x = torch.zeros(2, CAP + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="超过上限"):
        m.encode(x)


# =====================================================================
# 2. 因果性（最重要）
# =====================================================================
def test_future_items_do_not_affect_past_positions():
    """把最后一个位置换成别的东西，前面位置的输出必须逐位不变。

    这是对因果 mask 最容易写错的地方（例如误用对称 mask、
    或忘了 `tril`）。若失败，模型在训练时就能看到 target，
    loss 会异常低、指标会异常高。
    """
    m = _model()
    x = _seqs()
    with torch.no_grad():
        h1, _ = m.encode(x)
        x2 = x.clone()
        x2[0, -1] = 999 % N_ITEMS + 1        # 改末位
        h2, _ = m.encode(x2)

    # 位置 0..CAP-2 只能看到自己与更早的位置，不受末位影响
    assert torch.allclose(h1[0, :-1], h2[0, :-1], atol=1e-6)
    # 而末位自身必须变了（否则说明输入根本没接进去）
    assert not torch.allclose(h1[0, -1], h2[0, -1], atol=1e-6)


def test_causal_mask_is_lower_triangular_with_diagonal():
    """mask 的因果部分必须是下三角且含对角线。"""
    seq = torch.zeros(1, 4, dtype=torch.long)
    valid = torch.ones(1, 4, dtype=torch.bool)
    allow = causal_pad_allow_mask(seq, valid)
    assert allow.shape == (1, 4, 4)
    expect = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(allow[0], expect)


# =====================================================================
# 3. PAD 不得被 attend
# =====================================================================
def test_pad_positions_do_not_leak_into_real_positions():
    """改 PAD 的嵌入，真实位置的输出必须完全不变。

    做法：把 PAD（索引 0）的嵌入向量整体改掉，重新前向。
    若 mask 写错（真实 query 能 attend 到 PAD 的 key/value），
    真实位置的输出就会跟着漂移 —— 表现为"填充长度会影响指标"，
    是极难定位的一类问题。
    """
    m = _model()
    x = _seqs()
    with torch.no_grad():
        before = m.encode(x)[0].clone()
        original = m.item_emb.weight[PAD_ITEM].clone()
        m.item_emb.weight[PAD_ITEM] = 3.14          # 任意非零向量
        after = m.encode(x)[0]
        m.item_emb.weight[PAD_ITEM] = original      # 还原，别影响其他测试

    valid = x != PAD_ITEM
    for b in range(x.size(0)):
        rows = valid[b]
        if not rows.any():
            continue
        assert torch.allclose(before[b][rows], after[b][rows], atol=1e-6), \
            f"第 {b} 行真实位置受 PAD 嵌入影响，说明 padding mask 失效"


# =====================================================================
# 4. 全 PAD 行不得产生 NaN
# =====================================================================
def test_all_pad_row_has_no_nan():
    """空历史序列（滑动窗口的第 0 个样本）必须能前向，且不产生 NaN。

    机制：若所有 key 都被 mask 成 -inf，softmax 分母为 0 → NaN，
    再顺着残差传遍整行甚至整个 batch。所以 mask 规则里
    「对角线永远可见」不是可有可无的装饰。
    """
    m = _model()
    x = _seqs()
    assert (x[2] == PAD_ITEM).all()          # 前提：第 3 行确实全 PAD
    with torch.no_grad():
        hidden, urep = m.encode(x)
    assert not torch.isnan(hidden).any(), "出现了 NaN"
    assert not torch.isinf(hidden).any(), "出现了 inf"
    assert not torch.isnan(urep).any()


def test_all_pad_row_does_not_pollute_other_rows():
    """全 PAD 那行的存在与否，不能改变其他行的用户表示。

    ⚠️ 注意 self-attention 本身没有跨样本信息流，这条测试真正锁的是
    「NaN 没有扩散」—— 一旦全 PAD 行算出 NaN，它会通过同一 batch 的
    矩阵运算传播出去，让其他行的指标一起崩掉。
    """
    m = _model()
    x = _seqs()
    with torch.no_grad():
        u_full = m.encode(x)[1]
        u_two = m.encode(x[:2])[1]
    assert torch.allclose(u_full[:2], u_two, atol=1e-6)


def test_user_repr_is_last_valid_position():
    """用户表示必须取「最后一个有效位置」的隐状态。

    左填充下等价于 `hidden[:, -1]`，但这里用「最后一个**有效**位置」
    定义 —— 两者只在整行全 PAD 时才不同，而那种样本的表示无意义。
    """
    m = _model()
    x = _seqs()
    with torch.no_grad():
        hidden, urep = m.encode(x)
    # 第 1 行（只有最后一位有物品，前面全是 PAD）应取 hidden[1, -1]
    assert torch.allclose(urep[1], hidden[1, -1], atol=1e-6)
    # 第 0 行同理，最后一个位置是真实物品
    assert torch.allclose(urep[0], hidden[0, -1], atol=1e-6)


def test_explicit_mask_is_respected():
    """显式传 mask 时以它为准（契约要求 `encode(seq, mask)` 可用）。"""
    m = _model()
    x = _seqs()
    mask = (x != PAD_ITEM)
    with torch.no_grad():
        h_auto = m.encode(x)[0]
        h_expl = m.encode(x, mask)[0]
    assert torch.allclose(h_auto, h_expl, atol=1e-6)


# =====================================================================
# 5. 打分契约
# =====================================================================
def test_score_shape_and_contract():
    m = _model()
    x = _seqs()
    cand = torch.randint(1, N_ITEMS + 1, (3, 101))
    with torch.no_grad():
        s = m.score(x, cand)
    assert s.shape == (3, 101)
    assert torch.isfinite(s).all()


def test_score_equals_manual_dot_product_when_sharing_embedding():
    """共享嵌入时，score 应等于 `user_repr · item_emb[cand]` 的手算结果。"""
    m = _model(share_item_emb=True)
    x = _seqs()
    cand = torch.randint(1, N_ITEMS + 1, (3, 7))
    with torch.no_grad():
        urep = m.encode(x)[1]
        got = m.score(x, cand)
        expect = (m.item_emb.weight[cand] * urep.unsqueeze(1)).sum(dim=-1)
    assert torch.allclose(got, expect, atol=1e-5)


def test_score_is_equivariant_to_candidate_permutation():
    """候选列序变化时，分数必须跟着同样地变 —— 第 0 列没有任何特殊待遇。

    这是"第 0 列恒为正样本"这套约定能成立的前提：评估器靠
    `positive_rank(pos_index=0)` 找正样本的排名，如果模型对第 0 列
    有位置偏置（例如错误地把第 0 列当成了别的语义），
    所有 HR/NDCG 都会被系统性带偏，而且"看起来还挺正常"。
    """
    m = _model()
    x = _seqs()
    cand = torch.randint(1, N_ITEMS + 1, (3, 8))
    perm = torch.tensor([5, 2, 7, 0, 3, 1, 6, 4])
    with torch.no_grad():
        s = m.score(x, cand)
        s_perm = m.score(x, cand[:, perm])
    assert torch.allclose(s_perm, s[:, perm], atol=1e-5)


def test_score_works_without_shared_embedding():
    """`share_item_emb=false` 时仍然能打分（消融开关必须可用）。"""
    m = _model(share_item_emb=False)
    x = _seqs()
    cand = torch.randint(1, N_ITEMS + 1, (3, 5))
    with torch.no_grad():
        s = m.score(x, cand)
    assert s.shape == (3, 5)
    assert torch.isfinite(s).all()


# =====================================================================
# 6. 初始化
# =====================================================================
def test_pad_embedding_row_is_zero_after_init():
    """PAD 行必须全 0，否则会给"空位"一个固定的偏置方向。"""
    m = _model()
    assert torch.equal(m.item_emb.weight[PAD_ITEM],
                       torch.zeros_like(m.item_emb.weight[PAD_ITEM]))


def test_pad_row_receives_no_gradient():
    """`padding_idx` 的行不应有梯度（否则 PAD 嵌入会被训练得漂移）。"""
    m = SASRec(_cfg())
    x = _seqs()
    cand = torch.randint(1, N_ITEMS + 1, (3, 2))
    loss = m.score(x, cand).sum()
    loss.backward()
    grad = m.item_emb.weight.grad
    assert grad is not None
    assert torch.equal(grad[PAD_ITEM], torch.zeros_like(grad[PAD_ITEM]))


def test_model_is_deterministic_in_eval_mode():
    """eval 模式下（dropout 关闭）同一输入两次前向必须逐位相同。"""
    m = _model()
    x = _seqs()
    with torch.no_grad():
        a = m.encode(x)[1]
        b = m.encode(x)[1]
    assert torch.equal(a, b)


def test_dropout_actually_active_in_train_mode():
    """train 模式下 dropout 应生效，两次前向不同 —— 防止 dropout 被写死。"""
    m = _model(dropout=0.5)
    m.train()
    x = _seqs()
    with torch.no_grad():
        a = m.encode(x)[1]
        b = m.encode(x)[1]
    assert not torch.equal(a, b)


# =====================================================================
# 7. 配置校验
# =====================================================================
def test_config_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="整除"):
        _cfg(hidden_size=16, num_heads=3)


def test_config_rejects_bad_dropout():
    with pytest.raises(ValueError, match="dropout"):
        _cfg(dropout=1.0)


def test_config_from_dict_ignores_foreign_keys():
    """`from_dict` 只吃自己认得的字段 —— M2.5 的开关不该让它报错。"""
    cfg = SASRecConfig.from_dict(
        {"hidden_size": 32, "num_layers": 1, "num_interests": 4,
         "use_multi_interest": True, "routing_iters": 3},
        n_items=100)
    assert cfg.hidden_size == 32 and cfg.num_layers == 1
    assert cfg.n_items == 100


def test_n_params_is_positive_and_snapshot_is_jsonable():
    import json
    m = _model()
    assert m.n_params > 0
    snap = m.config_snapshot()
    json.dumps(snap)          # 必须能直接落盘（checkpoint 的 meta 要用）


# =====================================================================
# 8. 半精度不产生 NaN（AMP 场景）
# =====================================================================
def test_half_precision_has_no_nan_on_all_pad_row():
    """fp16 下全 PAD 行同样不能出 NaN —— AMP 训练时这是真实场景。

    用 `model.half()` 而不是 CPU autocast：CPU 上 fp16 autocast 支持有限，
    而我们要测的是"fp16 的 softmax 遇到全 -inf 会怎样"，`half()` 更直接。
    """
    m = _model().half()
    x = _seqs()
    with torch.no_grad():
        hidden, urep = m.encode(x)
    assert hidden.dtype == torch.float16
    assert not torch.isnan(hidden).any()
    assert not torch.isinf(hidden).any()
    assert not torch.isnan(urep).any()


# =====================================================================
# 9. mask 工具自身的边界
# =====================================================================
def test_mask_shape_mismatch_raises():
    m = _model()
    x = _seqs()
    with pytest.raises(ValueError, match="不一致"):
        m.encode(x, torch.ones(2, CAP, dtype=torch.bool))


def test_mask_diagonal_open_even_for_pad_queries():
    """mask 的对角线必须恒为 True（含 PAD 位置的 query）。

    这条性质是"全 PAD 行不出 NaN"的唯一保证：只要 query 至少能看自己，
    softmax 分母就非 0。
    """
    seq = torch.zeros(1, 4, dtype=torch.long)          # 全 PAD
    valid = seq != PAD_ITEM
    allow = causal_pad_allow_mask(seq, valid)
    for q in range(4):
        assert bool(allow[0, q, q]), f"query {q} 看不到自己"
