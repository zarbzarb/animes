# 对比基线 —— `models/baselines/`

E1（整体效果对比）所需的三个基线。三条硬约束（`docs/evaluation-plan.md` §4.1）：

| 约束 | 落地方式 |
|---|---|
| 同一数据划分 | 用户抽样唯一实现 `models/data/user_subset.py::resolve_scale_users`（被 `scripts/train.py` 与 `scripts/run_baselines.py` 共用） |
| 同一负样本池 | 训练侧 `sample_train_negatives()`；评估侧 1 正 100 负由 `evaluator.build_eval_data` 统一生成 |
| 同一评估器 | 三个基线都实现 `score_fn` 契约，评估代码里**没有**「如果是基线就走另一条分支」 |
| 同一早停策略 | GRU4Rec 直接调 `models/sasrec/train.py::fit()`，连 patience 的单位换算都共用 |

## 一、清单

| 模块 | 类型 | 需训练 | 参数量 | 打分公式 |
|---|---|---|---|---|
| `popularity.py` | 热度（零信息） | 否 | 0 | `score(u,i) = freq_train[i]` |
| `itemcf.py` | 传统协同过滤 | 否 | 0 | `score(u,i) = Σ_{j∈H(u)} sim(i,j)` |
| `gru4rec.py` | 经典序列模型 | **是** | ≈1.03M（H=64, L=1） | `dense(GRU(emb(H))) · emb(i)` |

```python
from models.baselines import (
    count_train_frequency, normalize_frequency, PopularityScorer,
    build_itemcf_similarity, ItemCFScorer,
    GRU4Rec, GRU4RecConfig,
)

# 无训练的两个：直接吃评估器
scorer = PopularityScorer(normalize_frequency(
    count_train_frequency(train_seqs, n_items, rows=train_rows)))
evaluate(scorer.score, eval_data, ks=(5, 10))

# GRU4Rec：走 fit()，与本文模型完全同一套训练循环与早停
model = GRU4Rec(GRU4RecConfig(n_items=n_items, hidden_size=64, num_layers=1))
fit(model=model, train_loader=loader, cfg=train_cfg, optim_cfg=oc,
    eval_fn=lambda m: evaluate(m.score, eval_data, ks=(5, 10)), n_items=n_items)
```

一键跑三个：

```bash
python scripts/run_baselines.py --baseline all --scale dev
python scripts/run_baselines.py --baseline popularity --scale main   # CPU、秒级
```

## 二、三处必须写清楚的实现约定

### 1. 热度只用**训练集**频次（`freq_source: train`）

⚠️ 与 `scripts/diagnose_popularity_bias.py` 的口径**不同**，两边都要报：

| | 本模块 | 诊断脚本 |
|---|---|---|
| 来源 | 本档位训练用户的序列 | `item_stats.parquet` 的 `n_positive`（**全量**，含 val/test） |
| 用途 | 论文正式数字 | 协议可信度自查 |
| 偏差方向 | 无 | **略微乐观**（偷看了一点测试集） |

「偷看测试集」这个方向的偏差恰好是最不容易被察觉的：它让**基线**更好看，
于是会**缩小**本文模型的相对增益，看起来像"我们很保守"。

### 2. ItemCF：二值化共现 + 余弦 + 逐行 TopK

* 用户内**取重**（本数据集用户会重复消费同一部番；不去重会把重复消费放大成共现强度）；
* `sim(i,j) = C[i,j] / sqrt(pop_i · pop_j)` —— 不做 popularity 归一的话，
  "热门物品的邻居"里全是热门物品；
* TopK 截断作用在**归一化后的余弦**上（不是原始共现次数），且是**逐行**的 ——
  打分只用 `sim` 的第 i 行，逐行截断与公式一致；
* 相似度用**全量训练序列**算（对应神经模型"在完整训练序列上训练"），
  但打分只用**评估器给的历史**（上限 48）—— 神经模型也只能看到这 48 个。

### 3. ⚠️ GRU4Rec 不能用 `pack_padded_sequence`（本机实测踩到的坑）

`torch.nn.utils.rnn.pack_padded_sequence` 的前提是**右填充**：它按 `lengths`
取每行**开头**的若干位置。本项目是**左填充**（有效物品在行尾），
于是它取到的是每行最前面的 PAD。

实测（H=16、1 层 GRU、3 个有效物品的历史）：

| | 「只喂有效位置」手算 | `pack_padded_sequence` |
|---|---|---|
| L=4（垫 1 个 PAD） | 基准 | 偏离 **1.5e-2** |
| L=6（垫 3 个 PAD） | 与基准**逐位相同** | 偏离 1.7e-2，且两次互相不同 |

CPU 与 CUDA 结果一致 ⇒ **不是内核问题，是用法错误**。最危险的是它不报错、
指标也算得出来。

本实现改为「滚动对齐 + 掩码取末位」，并显式校验「有效位置必须是每行的后缀」
（不满足直接报错，不静默算错）。回归测试：
`tests/test_baselines/test_baselines.py::test_repr_invariant_to_left_padding`。

## 三、并列口径的假象（必须披露）

`models/eval/metrics.py` 的并列约定是「稳定排序 + 正样本固定在第 0 列
⇒ 完全并列时正样本 rank = 1」。于是若某用户的 101 个候选**全是 0 分**
（ItemCF 的历史与所有候选都没有共同邻居），该样本会**白拿 HR=1**。

`ItemCFScorer.last_stats` 会报 `n_zero_rows` / `zero_row_share`，
`scripts/run_baselines.py` 也会打 warning 并额外跑一次
`itemcf_tb_pop`（只把这些全零行用热度填充）用来量化假象大小。
实测 smoke 档为 **0/647 = 0.00%**，两个变体数字完全相同。

## 四、测试

```bash
python -m pytest tests/test_baselines/ -q      # 40 例
```

覆盖：热度频次手算对拍、ItemCF 余弦手算对拍与 TopK 截断、重复消费只算一次、
分块不影响数值、全零行诊断与 tiebreak 只填全零行、GRU4Rec 左填充不变性
（★ 核心回归）、同批长短序列互不影响、PAD 行不接收梯度、参数量公式对拍、
checkpoint 按 `arch` 重建往返逐位一致、三个基线都能被同一评估器直接吃。
