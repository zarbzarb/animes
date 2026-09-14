# -*- coding: utf-8 -*-
"""服务端侧的「池内索引 ↔ 数据集 animeID」映射（`data/features/item_index.json`）。

为什么服务端也要有一份
--------------------
`agents/recall/adapter.py` 里已有一份（Agent 层用）。这里再有一份**不是重复实现**，
而是**不同的使用场景**：

* Agent 层那份用于「构造模型输入序列」与「候选回译」；
* 服务端这份用于**落库翻译** —— `recommend_result.anime_id` 是 `anime.id`
  外键，而 Agent 传给网关的是池内索引（A4 的候选空间）。两者必须能互相翻译，
  否则写进去的是池内索引、读出来当外键用，**不会报错但全是错的行**。

⚠️ 口径相同、来源相同（同一个 json 文件），但**读法独立**：如果哪天
`scripts/export_item_index.py` 的输出格式变了，两处都要改。所以这里在加载时
做一次结构校验（键的集合与长度一致），一旦对不上就直接抛 —— 宁可启动失败，
也不要静默地用半个映射把数据写歪。
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "index_to_src",
    "item_index_path",
    "load_item_index",
    "n_items",
    "pool_to_src",
    "reset_item_index_cache",
    "src_to_pool",
]

# `server/db/item_index.py` → 上溯三层 = 项目根
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def item_index_path() -> str:
    return os.path.join(_ROOT, "data", "features", "item_index.json")


@lru_cache(maxsize=1)
def load_item_index(path: Optional[str] = None) -> dict:
    p = path or item_index_path()
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"缺少物品索引映射 {p}；请先跑 scripts/export_item_index.py")
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    s2i = {int(k): int(v) for k, v in raw["src_to_index"].items()}
    i2s = {int(k): int(v) for k, v in raw["index_to_src"].items()}
    n = int(raw["n_items"])
    if len(s2i) != len(i2s):
        raise ValueError(
            f"item_index.json 结构异常：src_to_index({len(s2i)}) 与 "
            f"index_to_src({len(i2s)}) 长度不一致，拒绝加载（避免写歪 recommend_result）")
    return {"n_items": n, "src_to_index": s2i, "index_to_src": i2s,
            "checksum": raw.get("checksum_sha1", "")}


def reset_item_index_cache() -> None:
    """测试 / 重新导出索引后调用。"""
    load_item_index.cache_clear()


def n_items() -> int:
    return int(load_item_index()["n_items"])


def src_to_pool(src_anime_id: int) -> Optional[int]:
    """数据集 animeID → 池内索引（不在池内返回 `None`，**不塞 0**）。"""
    try:
        return load_item_index()["src_to_index"].get(int(src_anime_id))
    except Exception:                   # 映射文件缺失 → 整个召回链路早该走降级
        return None


def pool_to_src(index: int) -> Optional[int]:
    """池内索引 → 数据集 animeID。"""
    try:
        return load_item_index()["index_to_src"].get(int(index))
    except Exception:
        return None


def index_to_src() -> dict[int, int]:
    return dict(load_item_index()["index_to_src"])
