# -*- coding: utf-8 -*-
"""checkpoint 的读写实现（格式定义见 `models/checkpoint/__init__.py`）。"""

from __future__ import annotations

import os
import time
from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "CHECKPOINT_FORMAT",
    "checkpoint_name",
    "save_checkpoint",
    "load_checkpoint",
    "load_state_into",
    "load_model_from_checkpoint",
]

# 版本号：一旦 state_dict 的组织方式或 meta 的语义发生变化就要改它。
# 加载时会校验，避免"用新代码读旧权重"却一路静默跑出错误结果。
CHECKPOINT_FORMAT = "anirec.checkpoint.v1"


def checkpoint_name(experiment_name: str, scale: str, seed: int,
                    suffix: str = "best") -> str:
    """统一的文件名：`{实验名}_{档位}_seed{种子}_{后缀}.pt`。

    把档位与种子写进文件名，是为了避免"同一实验名换档位重跑后，
    新结果把旧结果覆盖掉，事后无法分辨论文里的数字来自哪一档"。
    这是本项目实际踩过的隐患（早期只按实验名命名）。

    非字母数字字符会被替换成 `_`，防止实验名里的斜杠等把路径带偏。
    """
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_"
                   for c in str(experiment_name))
    return f"{safe}_{scale}_seed{int(seed)}_{suffix}.pt"


def save_checkpoint(
    path: str,
    model: nn.Module,
    meta: Optional[dict] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> str:
    """把模型权重与元信息存到 `path`，返回实际写入的绝对路径。

    参数
    ----
    path       目标文件路径（`.pt`）。父目录不存在会自动创建。
    model      已训练好的模型。权重会被 `detach().cpu()` 后保存，
               因此**存盘动作不会影响训练中的参数，也不占显存**。
    meta       任意可 JSON 化的元信息。约定至少包含：
               `experiment_name` / `scale` / `seed` / `epoch` / `metrics` /
               `model_config`（结构快照，用于重建模型）。
    optimizer  可选。传入则一并保存优化器状态，用于**断点续训**。
               M2.4 的训练循环还不需要它（单次训练最长 2.3 小时），
               但格式先留好，避免将来加续训时又要改一次格式版本号。

    返回
    ----
    写入文件的绝对路径（便于调用方直接打印/记日志）。
    """
    path = os.path.abspath(os.path.expanduser(str(path)))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload = {
        "format": CHECKPOINT_FORMAT,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": dict(meta or {}),
        "state_dict": {k: v.detach().to("cpu", copy=True)
                       for k, v in model.state_dict().items()},
    }
    if optimizer is not None:
        # 优化器状态里的 tensor 也挪到 CPU；保留引用会在续训时把显存占回去
        payload["optimizer_state"] = {
            "state": {k: {kk: (vv.detach().to("cpu", copy=True)
                               if isinstance(vv, torch.Tensor) else vv)
                          for kk, vv in v.items()}
                      for k, v in optimizer.state.items()},
            "param_groups": optimizer.param_groups,
        }

    torch.save(payload, path)
    return path


def load_checkpoint(path: str, map_location=None, check_format: bool = True) -> dict:
    """读一个 checkpoint，返回原始 payload 字典。

    ⚠️ `weights_only=False` 是必需的：payload 里除 tensor 外还有
    dict / list / str 组成的 `meta`（`weights_only=True` 会拒绝其中
    部分类型）。这带来反序列化的安全前提 —— **只加载本项目自己生成的
    checkpoint，不要加载外部来源的 `.pt`**。
    """
    if map_location is None:
        map_location = "cpu"
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint 不存在：{path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(
            f"{path} 不是本项目的 checkpoint（缺少 state_dict 字段）")

    fmt = payload.get("format")
    if check_format and fmt != CHECKPOINT_FORMAT:
        raise ValueError(
            f"checkpoint 格式为 {fmt!r}，当前代码期望 {CHECKPOINT_FORMAT!r}；"
            "请用生成它时的代码版本读取，或重新训练")
    return payload


def load_state_into(model: nn.Module, payload: dict, strict: bool = True) -> nn.Module:
    """把 payload 里的 `state_dict` 灌进模型（原地），返回同一个模型。

    `strict=True` 时形状不匹配会直接报错 —— 这正是我们要的：
    悄悄跳过不匹配的键会让"加载了但没完全加载"，指标自然不对，
    而排查时很难想到是 checkpoint 的问题。
    """
    model.load_state_dict(payload["state_dict"], strict=strict)
    return model


def _project_root() -> str:
    """项目根目录（`F:\\pj`）：配置与产物路径一律相对它锚定。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def _rebuild_content_fused(cfg_dict: dict, fusion: dict) -> nn.Module:
    """按 `meta["content_fusion"]` 重建带候选侧内容融合的模型（M2.6b）。

    为什么必须走 `build_model` 工厂：训练与重建若各写一套装配逻辑，
    迟早出现「训练用 concat、加载建成 add」——权重形状一致、加载不报错，
    但语义变了，指标对不上时极难排查。
    """
    from models.content_encoder.model import (
        build_model,
        load_content_matrix_for_fusion,
    )

    rel = fusion.get("file")
    if not rel:
        raise ValueError(
            "checkpoint 声明了 content_fusion 但 meta 里没有 file 字段，"
            "无法定位内容矩阵")
    root = _project_root()
    path_abs = rel if os.path.isabs(rel) else os.path.join(root, rel)
    if not os.path.exists(path_abs):
        raise FileNotFoundError(
            f"checkpoint 需要内容矩阵 {path_abs}，但文件不存在；"
            "请先跑 scripts/build_content_vectors.py 或修正 meta")

    n_items = cfg_dict.get("n_items")
    if n_items is None:
        raise ValueError("checkpoint 的 model_config 里没有 n_items，无法重建模型")
    n_items = int(n_items)
    matrix = load_content_matrix_for_fusion(path_abs, n_items)
    # 兜底口径必须与 build_model / configs 的默认一致（M2.6b 起默认 = add）；
    # 正常情况下 meta 里一定带 mode，这行只是防手工构造 checkpoint 时的静默错配。
    mode = str(fusion.get("mode") or "add")

    if cfg_dict.get("arch") == "multi_interest":
        from models.multi_interest.config import MultiInterestConfig
        mi_cfg = MultiInterestConfig.from_dict(cfg_dict, n_items=n_items)
        return build_model("multi_interest", None, n_items,
                           content_matrix=matrix, content_mode=mode,
                           mi_cfg=mi_cfg)

    from models.sasrec.config import SASRecConfig
    return build_model(
        "sasrec", SASRecConfig.from_dict(cfg_dict, n_items=n_items), n_items,
        content_matrix=matrix, content_mode=mode)


def load_model_from_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    map_location=None,
) -> tuple:
    """便捷入口：读 checkpoint，重建/加载模型。

    参数
    ----
    model  已构造好的模型（结构必须与 checkpoint 一致）。
           **为 None 时自动按 `meta["model_config"]` 重建 SASRec** ——
           这正是 meta 里冗余保存结构快照的意义。

    返回
    ----
    `(model, meta)`。meta 至少含 `epoch` / `metrics` / `model_config`。

    用法
    ----
    >>> model, meta = load_model_from_checkpoint("data/checkpoints/sasrec_dev_seed42_best.pt")
    >>> meta["metrics"]["ndcg@10"]
    0.xxxx
    """
    payload = load_checkpoint(path, map_location=map_location)
    meta = dict(payload.get("meta") or {})

    if model is None:
        cfg_dict = meta.get("model_config")
        if not isinstance(cfg_dict, dict):
            raise ValueError(
                "checkpoint 的 meta 里没有 model_config，无法自动重建模型；"
                "请显式传入已构造的 model")

        # ---------- 候选侧内容融合（M2.6b）----------
        # meta["content_fusion"] 为 None / 缺失 = 无融合（M2.4 基线口径）。
        fusion = meta.get("content_fusion")
        if fusion:
            model = _rebuild_content_fused(cfg_dict, fusion)
        # 局部导入：避免 checkpoint 子包在 import 时就依赖具体模型包，
        # 将来 baselines 的 checkpoint 复用本模块时不会被迫拉起全部模型。
        elif cfg_dict.get("arch") == "gru4rec":
            # M2.7 对比基线。识别靠 `arch` —— 缺了它 GRU4Rec 的权重会被
            # 当成 SASRec 加载，然后爆一堆形状不匹配（能发现，但是无谓返工）。
            from models.baselines.gru4rec import GRU4Rec, GRU4RecConfig

            if "n_items" not in cfg_dict:
                raise ValueError(
                    "checkpoint 的 model_config 里没有 n_items，无法重建模型")
            model = GRU4Rec(GRU4RecConfig.from_dict(
                cfg_dict, n_items=int(cfg_dict["n_items"])))
        elif cfg_dict.get("arch") == "multi_interest":
            from models.multi_interest.config import MultiInterestConfig
            from models.multi_interest.model import MultiInterestSASRec

            if "n_items" not in cfg_dict:
                raise ValueError(
                    "checkpoint 的 model_config 里没有 n_items，无法重建模型")
            model = MultiInterestSASRec(MultiInterestConfig.from_dict(
                cfg_dict, n_items=int(cfg_dict["n_items"])))
        else:
            # 旧 checkpoint 没有 arch 字段，一律按 SASRec 兜底
            from models.sasrec.config import SASRecConfig
            from models.sasrec.model import SASRec

            if "n_items" not in cfg_dict:
                raise ValueError(
                    "checkpoint 的 model_config 里没有 n_items，无法重建模型")
            # 用 from_dict 而不是 SASRecConfig(**cfg_dict)：结构快照里包含
            # `input_cap` / `n_params` 这类**派生/诊断字段**，它们不是构造参数，
            # 直接展开会 TypeError。from_dict 只取 dataclass 认得的字段，
            # 这样将来往快照里加诊断量也不会让旧权重变得读不了。
            model = SASRec(SASRecConfig.from_dict(
                cfg_dict, n_items=int(cfg_dict["n_items"])))

    load_state_into(model, payload)
    return model, meta
