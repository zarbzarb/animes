# 多兴趣胶囊模型（M2.5）—— `models/multi_interest/`

MIND 式动态路由兴趣胶囊，架在 M2.4 的 SASRec 基座上。
**新增参数只有路由投影 W 的 4,096 个**（hidden=64 时），编码主干与训练
协议与基线完全一致 —— 这是 E2 消融「增益不来自参数/协议差异」的结构保证。

## 文件

| 文件 | 内容 |
|---|---|
| `config.py` | `MultiInterestConfig = SASRecConfig + num_interests / routing_iters`，yaml 扁平键自动拆分 |
| `capsule.py` | `squash` / `InterestRouting`（动态路由）/ `interest_diversity`（塌缩诊断） |
| `model.py` | `MultiInterestSASRec`：`encode_interests()` / `score()` / checkpoint 快照 |

## 与基线的两条可比性红线（改一条，E2 就废）

1. **编码主干不改**：复用 `SASRec.encode()` 的全部路径（嵌入、post-LN 块、
   对角恒开的因果 mask）。多兴趣只换"用户表示怎么聚合"这一层。
2. **训练协议不改**：`fit()` 只调 `model.score(x, cand)`，损失、
   1 正 1 负候选、BCE、早停全部照旧。打分从单向量点积换成
   max-over-interests 是唯一行为差异。

## 路由算法（MIND 口径，3 次迭代）

```
votes:  u_i = W · hidden_i              # 每位置一张票，PAD 位置整票置零
route:  c_ij = softmax_j(b_ij)          # softmax 在胶囊维 j（MIND 式，不是位置维）
        e_j = squash(Σ_i c_ij · u_i)
agree:  b_ij += u_i · e_j               # 下一次迭代的 logits
```

⚠️ softmax 在 **j 维**：部分开源实现写在位置维 i，与 MIND 论文不符，不可比。

## 踩过的坑（已修，勿回退）

### 坑 1：squash 与 0.02² 初始化的量级冲突（最重要）

squash 把兴趣向量范数压进 (0,1]，而 item_emb 按 SASRec 口径 N(0,0.02²)
初始化（H=64 时范数仅 ~0.16），点积量级 |score| ≤ **0.008**；基线
user_repr 是 LayerNorm 输出（范数 ≈ √H），logits ~0.3。同样 lr/epochs 下
多兴趣模型会系统性欠收敛 —— **E2 会把"训练预算不足"误判成"多兴趣没用"**。

修复：打分时 `interests * sqrt(H)`（与基座 `scale_emb` 同一条惯例；
MIND 原实现无此步是因为它的嵌入初始化范数≈1，天然匹配 squash 口径）。
正数缩放不改评估排序（max 对正数缩放保序），只影响训练 loss 动态。
玩具任务实测：修复前 10 epoch 才 0.693→0.566；修复后 3 epoch 即
0.694→0.664 且 |score| 0.008→0.57。

### 坑 2：PAD 位置必须"票数置零"而不是"mask logits"

softmax 在胶囊维 j，PAD 行的 c_ij 本来就合法（不会 NaN）；把该位置的票
u_i 直接置零即可让它对 Σ_i 无贡献。SASRec 的 `encode()` 已声明 PAD 位置
输出无意义，`capsule.py` 的 `u = u * valid` 是唯一把它挡在门外的关口。
测试 `test_pad_positions_do_not_contribute` 锁定（带垃圾值的 PAD 与
置零的 PAD 输出必须严格一致）。

## 用法

```bash
# 训练（与基线同一入口，只多一个 --model 开关）
python scripts/train.py --scale debug --model multi_interest --seed 42

# checkpoint 自动重建（meta["arch"]="multi_interest" 分发）
python -c "from models.checkpoint.io import load_model_from_checkpoint; \
           m, meta = load_model_from_checkpoint('data/checkpoints/xxx.pt')"
```

```python
from models.multi_interest.config import MultiInterestConfig
from models.multi_interest.model import MultiInterestSASRec

cfg = MultiInterestConfig.from_dict(
    {"hidden_size": 64, "num_interests": 4, "routing_iters": 3}, n_items=15687)
model = MultiInterestSASRec(cfg)
scores = model.score(input_ids, candidate_ids)   # [B, 1+C]，第 0 列恒正样本
```

## 诊断：路由塌缩

`interest_diversity(interests)` 返回 K 个兴趣的平均两两余弦相似度：
趋近 1 = 塌缩（所有位置挤进同一胶囊），健康状态通常 0.2~0.6。
塌缩时的处置顺序：先查 routing_iters 是否过少，再考虑 K 是否过大。

## 测试

`tests/test_multi_interest/`（42 例）：squash 数值边界、路由确定性、
PAD 隔离、参数增量恰好 = H·K·H、score 契约、checkpoint 按 arch 自动重建、
`fit()` 接线冒烟。跑全仓 `python -m pytest tests/ -q`（318 例）。
