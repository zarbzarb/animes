# -*- coding: utf-8 -*-
"""内容语义编码与融合（M2.6 / M2.6b）。

* `fusion.py` —— 排序侧（后验）融合：打分时按 7:3 / 5:5 加权。
* `model.py`  —— 候选侧（结构级）融合：内容向量拼进物品表示、端到端训练。
"""

from models.content_encoder.fusion import (  # noqa: F401
    COLD_START_THRESHOLD,
    ContentFusedModel,
    ContentScorer,
    fusion_score,
    load_content_matrix,
)
from models.content_encoder.model import (  # noqa: F401
    ItemContentFusion,
    build_model,
)
