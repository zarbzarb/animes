# -*- coding: utf-8 -*-
"""`models/checkpoint/` 的单元测试。

checkpoint 是"实验可复现"的最后一环，最容易出的两类问题：

1. **存下来的东西读不回来**（结构快照里混入派生字段、格式版本不匹配）
   —— 三个月后要复现某个数字时才发现，代价极高；
2. **保存的是引用而不是副本** —— 参数继续更新后，"最优权重"跟着变。

这里同时锁住这两条，并锁定"只靠 meta 就能重建模型"这个能力
（`load_model_from_checkpoint(model=None)`）。
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from models.checkpoint.io import (
    CHECKPOINT_FORMAT,
    checkpoint_name,
    load_checkpoint,
    load_model_from_checkpoint,
    load_state_into,
    save_checkpoint,
)
from models.sasrec.config import SASRecConfig
from models.sasrec.model import SASRec


def _model(seed: int = 0, **kw) -> SASRec:
    torch.manual_seed(seed)
    base = dict(n_items=30, max_seq_len=6, hidden_size=16, num_layers=1,
                num_heads=2, dropout=0.0, ffn_ratio=2)
    base.update(kw)
    return SASRec(SASRecConfig(**base))


def _meta(model: SASRec, **extra) -> dict:
    meta = {
        "experiment_name": "unit_test",
        "scale": "smoke",
        "seed": 42,
        "epoch": 7,
        "metrics": {"ndcg@10": 0.1234},
        "model_config": model.config_snapshot(),
    }
    meta.update(extra)
    return meta


# =====================================================================
# 1. 往返
# =====================================================================
def test_save_and_load_roundtrip(tmp_path):
    model = _model()
    path = save_checkpoint(str(tmp_path / "a.pt"), model, _meta(model))
    assert os.path.exists(path)

    payload = load_checkpoint(path)
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["meta"]["epoch"] == 7
    for k, v in model.state_dict().items():
        assert torch.equal(payload["state_dict"][k], v)


def test_saved_tensors_are_on_cpu_and_detached(tmp_path):
    """存盘必须脱离计算图、落在 CPU —— 否则会白占显存。"""
    model = _model()
    path = save_checkpoint(str(tmp_path / "a.pt"), model, _meta(model))
    payload = load_checkpoint(path)
    for v in payload["state_dict"].values():
        assert v.device.type == "cpu"
        assert not v.requires_grad


def test_saved_state_is_a_copy_not_a_reference(tmp_path):
    """存盘之后继续训练，文件里的权重不能跟着变。"""
    model = _model()
    path = save_checkpoint(str(tmp_path / "a.pt"), model, _meta(model))
    before = load_checkpoint(path)["state_dict"]["item_emb.weight"].clone()

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)          # 模拟继续训练

    after = load_checkpoint(path)["state_dict"]["item_emb.weight"]
    assert torch.equal(before, after)
    assert not torch.equal(after, model.item_emb.weight.detach())


def test_parent_directory_is_created(tmp_path):
    """父目录不存在时自动创建（CI 与实验脚本的首次运行都会遇到）。"""
    path = tmp_path / "deep" / "nested" / "b.pt"
    save_checkpoint(str(path), _model(), _meta(_model()))
    assert path.exists()


def test_load_state_into_restores_weights(tmp_path):
    a, b = _model(seed=1), _model(seed=2)
    assert not torch.equal(a.item_emb.weight, b.item_emb.weight)

    path = save_checkpoint(str(tmp_path / "c.pt"), a, _meta(a))
    load_state_into(b, load_checkpoint(path))
    assert torch.equal(a.item_emb.weight, b.item_emb.weight)


def test_load_state_into_is_strict_by_default(tmp_path):
    """形状不匹配必须报错 —— 静默跳过会让"加载了但没完全加载"。"""
    a = _model(hidden_size=16)
    b = _model(hidden_size=32)
    path = save_checkpoint(str(tmp_path / "d.pt"), a, _meta(a))
    with pytest.raises(RuntimeError):
        load_state_into(b, load_checkpoint(path))


# =====================================================================
# 2. 只靠 meta 重建模型（checkpoint 自描述）
# =====================================================================
def test_model_can_be_rebuilt_from_meta_alone(tmp_path):
    """不传模型时，应能按 meta["model_config"] 重建出结构一致的模型。

    ⚠️ 这条测试曾经抓到过一个真实 bug：`config_snapshot()` 里含
    `input_cap` / `n_params` 这类**派生字段**，早期实现直接
    `SASRecConfig(**cfg_dict)` 展开，导致加载必然 TypeError。
    现实现走 `SASRecConfig.from_dict()`，只取 dataclass 认得字段。
    """
    model = _model(seed=3, hidden_size=32, num_layers=2)
    path = save_checkpoint(str(tmp_path / "e.pt"), model, _meta(model))

    rebuilt, meta = load_model_from_checkpoint(path, model=None)
    assert rebuilt.cfg.hidden_size == 32
    assert rebuilt.cfg.num_layers == 2
    assert rebuilt.cfg.n_items == 30
    assert meta["epoch"] == 7
    assert torch.equal(rebuilt.item_emb.weight, model.item_emb.weight)


def test_meta_is_json_serializable(tmp_path):
    """meta 必须能直接 json.dump —— 训练日志会用同一份结构。"""
    model = _model()
    payload_path = save_checkpoint(str(tmp_path / "f.pt"), model, _meta(model))
    json.dumps(load_checkpoint(payload_path)["meta"], allow_nan=False)


def test_rebuild_fails_loudly_without_model_config(tmp_path):
    model = _model()
    path = save_checkpoint(str(tmp_path / "g.pt"), model, {"seed": 1})
    with pytest.raises(ValueError, match="model_config"):
        load_model_from_checkpoint(path, model=None)


# =====================================================================
# 3. 错误处理
# =====================================================================
def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(tmp_path / "nope.pt"))


def test_wrong_format_version_raises(tmp_path):
    """格式版本不匹配时立刻报错，而不是"能读但语义不对"。"""
    path = str(tmp_path / "h.pt")
    torch.save({"format": "anirec.checkpoint.v0",
                "meta": {}, "state_dict": {"w": torch.zeros(2)}}, path)
    with pytest.raises(ValueError, match="格式"):
        load_checkpoint(path)


def test_non_checkpoint_payload_raises(tmp_path):
    path = str(tmp_path / "i.pt")
    torch.save({"something": "else"}, path)
    with pytest.raises(ValueError, match="缺少 state_dict"):
        load_checkpoint(path)


# =====================================================================
# 4. 文件名
# =====================================================================
def test_checkpoint_name_encodes_scale_and_seed():
    name = checkpoint_name("exp1", "dev", 42)
    assert name == "exp1_dev_seed42_best.pt"
    assert checkpoint_name("exp1", "main", 7, suffix="last") == "exp1_main_seed7_last.pt"


def test_checkpoint_name_sanitizes_path_separators():
    """实验名里混进路径分隔符时不能把文件写到别的目录去。"""
    name = checkpoint_name("a/b\\c:d", "dev", 1)
    assert "/" not in name and "\\" not in name and ":" not in name


def test_time_suffix(tmp_path):
    from models.checkpoint.io import checkpoint_name as cn
    assert cn("exp", "dev", 1, "best").endswith(".pt")
