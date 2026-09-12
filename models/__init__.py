# -*- coding: utf-8 -*-
"""算法层（`docs/project-structure.md` §三）。

**分层硬约束**：依赖方向只能是 `server/ → agents/ → models/`，本包不得反向依赖，
也不得 import `server` / `agents`。CI 用 `scripts/check_imports.py` 检查。

**本包的纪律**（写新模块前请先读）：
1. **纯函数化**：给定相同输入必得相同输出；不依赖全局随机状态（除非显式传 seed）。
2. **无 IO**：不查数据库、不读 `.env`、不写日志文件、不调用 LLM。用 `logging` 输出即可。
   需要读数据时，由调用方（脚本或 Agent）读好再传进来。
3. **配置用 dataclass**：超参从各子包的 `config.py` 注入，不在网络定义里硬编码。
4. **指标唯一实现**：HR / NDCG / Recall / MRR 只能来自 `models/eval/metrics.py`。

子包一览：
    sasrec/          序列推荐基座
    multi_interest/  多兴趣胶囊网络
    content_encoder/ 内容语义编码与双路融合
    baselines/       对比基线（ItemCF / GRU4Rec）
    eval/            指标与评估循环（唯一实现）
"""
