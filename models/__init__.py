# -*- coding: utf-8 -*-
"""算法层（`docs/project-structure.md` §三）。

**分层硬约束**：依赖方向只能是 `server/ → agents/ → models/`，本包不得反向依赖，
也不得 import `server` / `agents`。CI 用 `scripts/check_imports.py` 检查。

**本包的纪律**（写新模块前请先读）：
1. **纯函数化**：给定相同输入必得相同输出；不依赖全局随机状态（除非显式传 seed）。
2. **无 IO**：不查数据库、不读 `.env`、不写日志文件、不调用 LLM。用 `logging` 输出即可。
   需要读数据时，由调用方（脚本或 Agent）读好再传进来。
3. **配置用 dataclass**：超参从各子包的 `config.py` 注入，不在网络定义里硬编码。
4. **「唯一实现」清单**（不得在别处重写）：
   - 指标公式 → `models/eval/metrics.py`（HR / NDCG / Recall / MRR）
   - 抽哪些负样本 → `models/data/negatives.py`（所有模型必须共用同一批）
   - 历史序列怎么拼 → `models/sasrec/dataset.py`
   - 串起来跑前向 → `models/eval/evaluator.py`
5. **评估器不 import 模型**：`evaluator.py` 通过 `score_fn` 回调解耦，
   以便基线与本文模型共用同一套评估代码（公平性硬约束）。

子包一览：
    data/            数据侧纯函数（确定性用户抽样、负采样）
    sasrec/          序列推荐基座（含滑动窗口 Dataset）
    multi_interest/  多兴趣胶囊网络
    content_encoder/ 内容语义编码与双路融合
    baselines/       对比基线（ItemCF / GRU4Rec）
    eval/            指标与评估循环（唯一实现）
"""
