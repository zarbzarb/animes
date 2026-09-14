# models/content_encoder —— 内容语义编码与融合

本包负责「动漫文本语义 → 可用的特征」，以及把它接进推荐模型的两条不同路径。
设计前提：`animes.csv` **没有剧情简介字段**，内容向量由
`标题 + 12 类题材 + 细分标签` 经 DistilBERT(multilingual) 编码、PCA 降到 512 维
（解释方差 0.9991），产物在 `data/features/`，**已 L2 归一**。

| 文件 | 职责 |
|---|---|
| `fusion.py` | **排序侧（后验）融合**：行为分与内容分按 7:3 / 5:5 加权 |
| `model.py` | **候选侧（结构级）融合**：内容向量拼进物品表示，端到端训练 |
| `build_vectors.py` | 离线批量生成 512 维内容向量（阶段一，`scripts/` 侧同名脚本调用） |
| `pretrained/` | 本地 DistilBERT 权重（517MB，不入库） |

---

## 一、两条融合路径，别搞混

**排序侧（后验，`fusion.py`）** —— 行为模型照旧训练，只在打分时混合：

```
final = w_b · behavior_score + w_c · content_cosine
```

**候选侧（结构级，`model.py`）** —— 内容作为**输入特征**参与训练：

```
table = f(item_emb, content)      # 输入嵌入与输出打分共用这张表
```

为什么必须两条都实现：后验加权补不动「物品嵌入没学过」这件事 ——
冷启动物品的问题在表示层，不在分数层。

---

## 二、接口契约（改动会使实验不可比）

### 排序侧

```python
fusion_score(behavior, content, n_inter, w_behavior=0.7, w_content=0.3,
             w_content_cold=0.5, cold_threshold=10) -> Tensor
```

* 签名与 `docs/project-structure.md` 一致，**纯函数、无 IO、无可学参数**；
* `n_inter < 10`（`COLD_START_THRESHOLD`）逐样本切 5:5，同 batch 冷热混排安全；
* `content` 与 `behavior` 列序必须一致（第 0 列都是正样本）。

### 候选侧

```python
ItemContentFusion(content_matrix, hidden_size, mode="concat"|"add", dropout=0.0)
    .forward(item_weight [V,H]) -> table [V,H]

build_model(arch, sasrec_cfg, n_items, content_matrix=None,
            content_mode="concat", mi_cfg=None) -> nn.Module
```

* `SASRec` / `MultiInterestSASRec` 都通过 `repr_provider` 注入；
  **输入侧与输出侧都必须经 `SASRec.item_table()` 取表**，否则会出现
  「输入用融合表示、输出用纯嵌入」的隐性不一致（指标看着还行，语义是错的）；
* `model.repr_provider` 在**两种 arch 上都可直接取**（多兴趣是转发到 backbone；
  曾因路径分歧在 `--model multi_interest --content-fusion` 首次运行时报
  AttributeError，已加回归测试锁死）。

### 三条硬约束

1. **PAD 行恒零**。投影层带 bias，必须显式乘 `_pad_mask` 清零；同时嵌入查表
   必须用 `F.embedding(..., padding_idx=0)` 而不是裸张量索引 ——
   后者会丢掉「PAD 行不接收梯度」的保障，PAD 行会随训练漂移出零。
2. **内容矩阵是 buffer，不是参数**。离线 PCA 产物，不可学；
   参数增量只剩投影层，可精确核算。
3. **`mode` 决定参数增量**：
   * `concat`：`(H + D)·H + H` = **36,928**（H=64, D=512），占基线 1,107,328 的 3.3%
   * `add`：`D·H + H` = **32,832**（+ 基座嵌入本身不变）

---

## 三、`concat` 与 `add` 的关键差异（实测教训）

| | concat | add（零初始化残差） |
|---|---|---|
| 形式 | `Linear([emb, content])` | `emb + Linear(content)` |
| 起点 | 不等价基线：线性层可自由重学整个物品空间 | **严格等价基线**（零权重+零 bias） |
| 实测后果 | 训练后融合表与原嵌入余弦仅 **0.0097** —— 等于把 tie-embedding 已学到的表示又学一遍，白付优化代价 | 内容增量从 0 学起，正向就是"内容有用"，负向就是"没用"，可归因 |

`add` 必须零初始化：内容向量是单位向量，`‖proj(content)‖ ≈ σ_w·√H`，
σ_w=0.02 时约 0.16，与 `item_emb` 初始化范数**同量级** —— 非零初始化下
残差增量与嵌入一样大，"起点等价基线"这个前提就不成立。梯度不会因此消失
（`dL/dW = δᵀ·contentᵀ`），测试里有一条专测"一步梯度后残差必须离开 0"。

---

## 四、实测结果（debug 档 8 epoch 固定预算、seed 42、同评估子集 2,572 样本）

| 配置 | best ndcg@10 | vs 同 arch 无内容 | 参数量 | s/epoch |
|---|---|---|---|---|
| SASRec 纯行为（M2.4 基线） | 0.7871 | — | 1,107,328 | 66.5 |
| SASRec + 内容（concat） | 0.7760 | −0.0111 | 1,144,256 | 63.3 |
| **SASRec + 内容（add）** | **0.7895** | **+0.0024** | 1,140,160 | 58.2 |
| 多兴趣（M2.5） | 0.7624 | — | 1,123,712 | 84.5 |
| 多兴趣 + 内容（concat） | 0.7680 | +0.0056 | 1,160,640 | 78.8 |
| 多兴趣 + 内容（add） | 0.7717 | **+0.0093** | 1,156,544 | 79.2 |

**同权重内容消融**（把内容侧权重置零，重评同一个 checkpoint —— 这份权重里
内容到底值多少分；评估用 fp32、默认批大小，与训练中的 AMP 评估路径不同，
故绝对值与上表差 3~5e-3）：

| checkpoint | 内容开 | 内容关 | 内容净贡献 |
|---|---|---|---|
| SASRec + concat | 0.7803 | 0.7272 | **+0.0531** |
| SASRec + add | 0.7927 | 0.7576 | **+0.0350** |
| 多兴趣 + add | 0.7742 | 0.7291 | **+0.0451** |
| 纯基线（对照，同评估路径） | 0.7878 | — | — |

排序侧（后验）实测（`logs/content_fusion_eval.json`）：

| 配置 | val ndcg@10 | 冷启动 holdout ndcg@10 |
|---|---|---|
| 纯行为 w=0 | 0.7878 | — |
| 7:3 | 0.7874 | 0.6414 |
| 5:5 | 0.7845 | 0.6350 |
| 纯内容 w=1 | 0.1739 | 0.1368 |

### 怎么读这张表（论文口径）

* **后验加权无增益**：7:3 与纯行为同水平，5:5 更差，冷启动上 5:5 也不比 7:3 好。
  纯内容有真信号（0.1739 vs 随机 0.099）但远弱于行为通路。
* **候选侧 concat 也没赢**：SASRec 上 −1.1pp、多兴趣上 +0.6pp，方向不一致。
  **但内容不是被忽略了**：同权重消融显示内容通路净贡献 +0.0531 ndcg@10，
  内容贡献了物品表示范数的 ~30%，内容块权重范数从 3.62 训到 32.72（9 倍）。
  诊断：concat 的投影把物品空间**重学了一遍**（融合表与原嵌入余弦仅 0.0097），
  白付优化代价 —— 问题在实现方式，不在内容本身。
* **候选侧 add（零初始化残差）把两个架构都拉成正增益**：
  SASRec +0.0024、多兴趣 +0.0093；同权重消融也一致为正（+0.0350 / +0.0451）。
  两种架构、两种评估口径下方向一致 ⇒ **残差口径优于 concat** 这个判断有支撑。
  但增益幅度很小（同评估路径下 0.7927 vs 基线 0.7878，+0.0049 ndcg@10）。

⚠️ 所有训练对比都是**同预算快照**：8 epoch 时各配置都还在上升（best 都在最后一轮），
单种子、组间差 ≤1.3pp 与评估噪声同量级。**"add 略优"只能说方向可信，
不能说幅度可信**；E2 正式结论用 dev 档（30 epoch + 早停）+ 多种子。
评估口径提醒：主协议里**每个物品训练集至少 6 次交互**，即没有真正的冷启动物品，
内容融合的预期收益主要在 E3 冷启动协议（新番在训练集中不可见）验证。

---

## 五、复现

```bash
# 内容向量（阶段一产物，重跑需下载 DistilBERT）
python scripts/build_content_vectors.py

# 排序侧后验融合：val 扫权重 + test 确认 + 冷启动快检
python scripts/eval_content_fusion.py

# 候选侧结构级融合（训练）
python scripts/train.py --scale debug --model sasrec --content-fusion --content-mode add
python scripts/train.py --scale debug --model sasrec --content-fusion            # concat
python scripts/train.py --scale debug --model sasrec --no-content-fusion         # 纯基线

# 单测
python -m pytest tests/test_content/ -q
```

checkpoint 的 `meta["content_fusion"]` 记录 `{mode, content_dim, hidden_size,
dropout, n_params, file}`，`load_model_from_checkpoint` 靠它自动装上 provider；
`file` 按项目根锚定。**缺文件会报错而不是静默降级** —— 静默降级会得到一份
"能跑但语义不同"的模型，是最难排查的一类错误。
