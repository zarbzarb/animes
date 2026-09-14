# -*- coding: utf-8 -*-
"""在线召回纯计算层（`models/retrieval/`）。

* `catalog.py` —— 序列模型：序列构造 / 全库打分 / 逐兴趣 TopK。
* `content.py` —— 内容模型：用户内容画像 / 候选子集内的余弦 TopK（A3 用）。

分层纪律与其它 `models/` 子包相同：**纯函数、无 IO** —— 不读 checkpoint、
不读 pkl、不查库。谁来加载权重（`scripts/` 或 Agent 的 adapter）谁负责，
本包只接受"已经加载好的模型"与"已经构造好的张量"。
"""

from models.retrieval.catalog import (  # noqa: F401
    PAD_ITEM,
    build_input_ids,
    catalog_scores,
    n_items_of,
    resolve_item_table,
    topk_per_interest,
    user_vectors,
)
from models.retrieval.content import (  # noqa: F401
    content_topk,
    load_scorer,
    user_profile_vector,
)

__all__ = [
    "PAD_ITEM",
    "build_input_ids",
    "catalog_scores",
    "content_topk",
    "load_scorer",
    "n_items_of",
    "resolve_item_table",
    "topk_per_interest",
    "user_profile_vector",
    "user_vectors",
]
