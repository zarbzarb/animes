# -*- coding: utf-8 -*-
"""A2 的模型与数据适配层。

本文件是 `agents/` 里**唯一**会碰到 `torch` 和 checkpoint 的地方，
而且它自己也不算张量 —— 真正的计算在 `models/retrieval/catalog.py`。

三件容易静默出错的事，这里都显式处理
------------------------------------
**1. 物品索引 ≠ src_anime_id。** `smap` 非恒等（实测 51.7% 的条目不相等）。
   所以序列构造与候选回译都必须过 `item_index.json`，绝不能用
   `anime.src_anime_id` 直接当索引。见 `scripts/export_item_index.py`。

**2. 序列必须左填充且 ≤ input_cap（48）。** `models/sasrec/model.py::encode`
   会在超长时直接 raise，而不是悄悄截断 —— 因为位置嵌入会越界。
   本模块按 `input_cap` 截取**最近**的 48 条。

**3. checkpoint 要按 `meta.arch` 重建。** 内容融合（M2.6b）的权重必须经
   `load_model_from_checkpoint` 的工厂重建，不能手写 `SASRec(cfg)` ——
   训练用 `add`、加载建成无融合，权重形状一样、加载不报错，但语义变了。
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, Optional, Sequence

from agents.common.registry import ModelRegistry, ModelSpec
from agents.recall.config import RecallConfig

logger = logging.getLogger(__name__)

# 路径锚定：`agents/recall/adapter.py` → 上溯三层 = 项目根
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


# ---------------------------------------------------------------- 物品索引映射


def item_index_path() -> str:
    return os.path.join(_ROOT, "data", "features", "item_index.json")


@lru_cache(maxsize=1)
def load_item_index(path: Optional[str] = None) -> dict:
    """读 `item_index.json`。**缺文件时不静默**：直接抛，让 A2 走降级。"""
    p = path or item_index_path()
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"缺少物品索引映射 {p}；请先跑 scripts/export_item_index.py")
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "n_items": int(raw["n_items"]),
        "src_to_index": {int(k): int(v) for k, v in raw["src_to_index"].items()},
        "index_to_src": {int(k): int(v) for k, v in raw["index_to_src"].items()},
        "checksum": raw.get("checksum_sha1", ""),
    }


def reset_item_index_cache() -> None:
    """测试用。"""
    load_item_index.cache_clear()


def src_to_item_index(src_ids: Sequence[int]) -> list[int]:
    """数据集 animeID → 池内索引。**不在池内的直接丢弃**（不是塞 0）。"""
    m = load_item_index()["src_to_index"]
    return [m[int(s)] for s in src_ids if int(s) in m]


def item_index_to_src(indices: Sequence[int]) -> list[int]:
    m = load_item_index()["index_to_src"]
    return [m[int(i)] for i in indices if int(i) in m]


# ---------------------------------------------------------------- 模型加载


def _build_factory(ckpt_path: str) -> Any:
    def _factory() -> Any:
        # 局部 import：没装 torch 的环境（纯 API 测试）也不该在 import 期失败
        from models.checkpoint.io import load_model_from_checkpoint

        model, meta = load_model_from_checkpoint(ckpt_path, map_location="cpu")
        model.eval()
        logger.info("A2 已加载召回模型 %s（arch=%s epoch=%s）",
                    os.path.basename(ckpt_path), meta.get("arch"),
                    meta.get("epoch"))
        model._anirec_meta = meta          # type: ignore[attr-defined]
        return model

    return _factory


def register_recall_model(cfg: RecallConfig, model_dir: Optional[str] = None) -> str:
    """把 checkpoint 注册成延迟加载的模型工厂，返回模型名。

    **延迟加载**是刻意的：`torch.load` 一个模型要几百毫秒并占显存，
    如果 import 期就加载，单元测试与 `/health` 都会被拖慢。
    """
    base = _abs(model_dir or "data/checkpoints")
    name = f"recall:{cfg.arch}"
    for candidate in (cfg.checkpoint, cfg.fallback_checkpoint):
        if not candidate:
            continue
        path = os.path.join(base, candidate)
        if os.path.exists(path):
            ModelRegistry.register(ModelSpec(
                name=name, factory=_build_factory(path), version=candidate,
                kind="torch", description=f"A2 召回模型（{cfg.arch}）",
                meta={"path": path, "arch": cfg.arch}))
            return name
    raise FileNotFoundError(
        f"在 {base} 下找不到 {cfg.checkpoint!r} / {cfg.fallback_checkpoint!r}；"
        "请先训练或调整 RecallConfig.checkpoint")


# ---------------------------------------------------------------- 前向


def recall_sequence(seq_indices: Sequence[int], cfg: RecallConfig,
                    *, model_name: Optional[str] = None,
                    exclude_indices: Sequence[int] = (),
                    profile: Optional[dict] = None) -> dict:
    """纯模型召回：序列 → 每个兴趣的 TopK。

    返回
    ----
    ``{"per_interest": [[(index, score), ...], ...], "n_items": int,
       "model_ver": str, "arch": str}``
    """
    import torch

    from models.retrieval.catalog import (
        build_input_ids, catalog_scores, resolve_item_table, topk_per_interest,
    )

    name = model_name or f"recall:{cfg.arch}"
    model = ModelRegistry.get(name)                 # 失败抛 MODEL_UNAVAILABLE

    input_ids, _valid = build_input_ids(seq_indices, cfg.input_cap,
                                        device=torch.device(cfg.device))
    forbid = _forbidden_mask(cfg)
    scores = catalog_scores(model, input_ids, forbid_mask=forbid,
                            item_table=_item_table(name, model))       # [1, K, V]

    exclude = [list(exclude_indices)] if exclude_indices else None
    per_interest = topk_per_interest(scores, cfg.topk_per_interest,
                                     exclude=exclude)[0]

    meta = getattr(model, "_anirec_meta", {}) or {}
    return {
        "per_interest": per_interest,
        "n_items": int(scores.shape[-1]) - 1,
        "model_ver": str(meta.get("experiment_name") or name),
        "arch": str(meta.get("arch") or cfg.arch),
    }


# 物品表示表缓存：推理期 `item_emb.weight` 冻结 → 表是常量。
# 内容融合时这张表是 (V, 576) @ (576, 64) 的矩阵乘（≈58M MAC），
# 每次请求重算纯属浪费；缓存后单次前向从 ~50ms 降到 ~2ms。
_TABLE_CACHE: dict[str, object] = {}


def _item_table(model_name: str, model: Any):
    if model_name not in _TABLE_CACHE:
        import torch
        from models.retrieval.catalog import resolve_item_table
        with torch.no_grad():
            _TABLE_CACHE[model_name] = resolve_item_table(model).detach().clone()
    return _TABLE_CACHE[model_name]


def reset_item_table_cache() -> None:
    """测试用（换了模型/权重后必须清）。"""
    _TABLE_CACHE.clear()


@lru_cache(maxsize=1)
def _forbidden_mask(cfg: RecallConfig):
    """违规题材物品的屏蔽掩码（`[V]` bool）。文件缺失 → None（不屏蔽）。"""
    if not cfg.block_forbidden:
        return None
    path = os.path.join(_ROOT, "data", "processed", "preprocess_report.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        idxs = rep.get("forbidden_item_indices") or rep.get("data", {}) \
            .get("forbidden_item_indices") or []
        if not idxs:
            return None
        import torch
        n_items = load_item_index()["n_items"]
        mask = torch.zeros(n_items + 1, dtype=torch.bool)
        for i in idxs:
            i = int(i)
            if 0 <= i <= n_items:
                mask[i] = True
        return mask
    except Exception as exc:                # 掩码不可用不该让召回失败
        logger.warning("读取违规题材屏蔽掩码失败（已忽略）：%s", exc)
        return None


def reset_forbidden_cache() -> None:
    _forbidden_mask.cache_clear()


# ---------------------------------------------------------------- 降级数据


def popular_candidates(cfg: RecallConfig, gateway: Any,
                       exclude_indices: Sequence[int] = ()) -> list[tuple[int, float]]:
    """降级：按 `n_interactions` 的热门榜（**不含任何模型**）。

    ⚠️ 返回的是**池内索引**。热门榜来自 DB 的 `src_anime_id`，
    必须过一次 `src_to_index`；不在池内的（新番/被过滤的）自然被丢掉。
    """
    src_ids = gateway.popular_anime_ids(limit=int(cfg.fallback_topk))
    excl = set(int(x) for x in exclude_indices)
    out: list[tuple[int, float]] = []
    m = load_item_index()["src_to_index"]
    for rank, src in enumerate(src_ids):
        idx = m.get(int(src))
        if idx is None or idx in excl:
            continue
        # 分数用 rank 的倒数，量级与模型分不同但**只用于降级**，
        # 且 A4 会做 min-max 归一化，不会与模型分混用
        out.append((idx, 1.0 / (1.0 + rank)))
    return out
