# 实验方案与评价体系

> **本文是毕业论文「实验设计」章节的素材来源**，所有实验脚本、参数、口径以本文为准。
> 指标实现唯一来源：`models/eval/metrics.py`。**禁止在任何脚本中重写指标计算。**
> 消融实验细节见 [ablation-study.md](ablation-study.md)，结果填写见 [result-analysis.md](result-analysis.md)。

---

## 一、实验目标

验证四个递进命题：

| # | 命题 | 对应实验 |
|---|---|---|
| H1 | 序列建模优于传统协同过滤 | 基线对比（ItemCF vs GRU4Rec vs SASRec） |
| H2 | 自注意力优于循环神经网络 | 基线对比（SASRec vs GRU4Rec） |
| H3 | 多兴趣建模提升小众题材覆盖 | 消融实验 + 分题材实验 |
| H4 | 内容融合缓解新番冷启动 | 消融实验 + 冷启动专项 |

---

## 二、数据集与预处理

### 2.1 数据集概况

来源：**MyAnimeList（MAL）**，已落位于 `dataset/` 目录。

| 文件 | 规模 | 用途 |
|---|---|---|
| `animes.csv` | 20,237 条动漫元数据 | 物品侧信息（标题/类型/年份/评分/题材/细分标签） |
| `ratings.npy` | 148,170,496 条评分（int32×3） | **权威交互源**，见下方「数据源判定」 |
| `ratings.csv` / `ratings.dat` | 同为 148,170,496 条，两者完全相同 | 参考源（相对 npy 有整体 +1 偏移，见下） |
| `id_to_genreids.json` | 20,237 条映射 | 动漫 → 21 类 MAL 主题材 ID |
| `dataset.pkl` | 用户 1,306,691 / 物品 15,687 | 已切分的 train/val/test 序列 + id 映射（**序列顺序的唯一口径**） |
| `random-sample_size100-seed98765.pkl` | 每用户 100 个负样本 | 评估负采样池 |
| `pretrained_bert.pth` | 116 MB | **⚠️ 不是内容编码器**，见下方说明 |

#### 数据源判定（必须记录，否则实验不可复现）

三个交互文件**并非同一口径**，实测结论如下：

| 用哪个文件重建 `dataset.pkl` 的物品集合 | 1,500 个抽样用户的一致率 |
|---|---|
| `ratings.npy` | **100.0%** |
| `ratings.csv`（= `ratings.dat`） | 90.7% |

全量抽样 1,500 用户的实测明细：可比评分单元 **166,567** 对，其中不同的 **8,896** 对（**5.34%**），
差异以 **+1 偏移为主（8,825 对，占差异的 99.2%）**，即 csv 相对 npy 被整体 +1 微调；
落在 6/7 正样本边界上、会造成判定翻转的有 **1,494 对（0.90%）**。
**结论：`dataset.pkl` 由 `ratings.npy` 口径构建（集合复现率 100%），本项目一律以 `ratings.npy` 为权威交互源。**
依据可复现：运行 `python scripts/diagnose_dataset_source.py`，结论落盘于 `data/processed/dataset_source_verdict.json`。

#### ⚠️ `ratings.npy` 存在重复评分行（去重规则必须写进论文）

实测发现**同一 `(userID, animeID)` 在 `ratings.npy` 中出现多行**：
抽样 300 个用户中有 **16 户**存在重复（共 **541 条冗余行**），
其中 **1 户**出现跨 7 分冲突（同一物品既有 `7` 也有 `<7` 的评分行）。

`dataset.pkl` 采用的是**逐行判定**口径——只要该物品存在**任一行** `rating >= 7`，
即计入该用户的正样本（不是「取最后一行」、也不是「取平均」）。这一点已用
`scripts/verify_stage1.py` 的 C 组检查固定下来：若误用 last-value 语义，该用户会少 1 个物品。

> 影响范围：这是清洗章节必须披露的细节。若后续自行重建数据集，
> 去重规则应写成「按 `(user, anime)` 取 max(rating)」，与官方切分保持一致。

#### ⚠️ `pretrained_bert.pth` 的真实身份

文档早期版本把它写成 DistilBERT 权重，**这是错的**。实际检查其 state_dict：

```
bert.embedding.token_ids            (15687,)        物品 id 表
bert.embedding.genre_ids            (15687, 5)      每物品的题材 id
bert.embedding.token.weight         (15689, 256)    物品嵌入
bert.embedding.genre_embed.weight   (22, 256)       题材嵌入
bert.transformer_blocks.0/1         (256 维, 2 层)
out.weight                          (15688, 256)    物品输出投影
```

这是一个**跑在本数据集上的 BERT4Rec 式序列推荐模型**（epoch=3 的中断训练检查点），
不是文本编码器——它没有词表、没有 word_embeddings，无法编码任何自然语言文本。
内容语义向量改用真正下载的 `distilbert-base-multilingual-cased`，见 2.4。

#### 已完成的过滤（已用 `scripts/preprocess.py` 逐条复现验证）

| 过滤项 | 更正后的真实规则 | 结果 |
|---|---|---|
| 正样本判定 | `rating >= 7`（评分 1~6 一律丢弃，7~10 一律保留，无例外） | 77.24% 的交互为正样本 |
| 用户过滤 | **正样本交互数 >= 10** | 保留 **1,306,691** 用户 |
| 物品过滤 | **正样本交互数 >= 10** | 保留 **15,687** 物品 |
| 缺失值 | `score` 列的 `?` 占位符（593 条）、`year` 非数值（136 条）转 NaN | 已处理 |
| 不合规题材 | **`dataset.pkl` 实际并未剔除**，见第十节 | 1,551 个此类物品在池内，评估阶段屏蔽 |

> 早期文档把用户阈值写作「< 5」、物品阈值写作「< 10」，与数据实际不符，已更正。
> 注意：物品阈值 >= 10 意味着**池内不存在交互数 < 10 的物品**，
> 因此「交互数 < 10 即新番」的冷启动口径会得到空集，2.5 节改用长尾分位口径。

### 2.2 题材体系（12 类）

从 21 类 MAL 主题材 + `genres_detailed` 细分标签归并为 12 类，用于分题材实验与兴趣胶囊打标。
口径定义在 `configs/genre_taxonomy.yaml`（唯一事实来源）。

| id | 题材 | MAL 映射 | 覆盖数（可多标签） | 主标签数 |
|---|---|---|---|---|
| 1 | 热血战斗 | Action | 4,664 | 680 |
| 2 | 冒险奇幻 | Adventure, Fantasy | 5,205 | 2,849 |
| 3 | 日常治愈 | Slice of Life, Gourmet | 1,084 | 780 |
| 4 | 恋爱校园 | Romance (+Girls/Boys Love) | 2,287 | 1,258 |
| 5 | 悬疑推理 | Mystery, Suspense | 1,145 | 1,047 |
| 6 | 科幻机战 | Sci-Fi | 2,728 | 2,173 |
| 7 | 喜剧搞笑 | Comedy | 5,389 | 1,304 |
| 8 | **运动竞技** | Sports | 618 | **567** |
| 9 | 超自然灵异 | Supernatural, Horror | 1,582 | 1,005 |
| 10 | 剧情文艺 | Drama, Award Winning, Avant Garde | 2,934 | 680 |
| 11 | **青春音乐** | 细分标签词边界匹配（idol/band/musical/…） | 1,364 | 1,364 |
| 12 | **后宫福利** | Ecchi | 813 | 732 |

> 统计范围为**推荐池的 15,687 个物品**（非全部 20,237 条元数据）。
> 「覆盖数」为包含该题材的物品数（一物可多题材）；「主标签数」为 `is_primary` 唯一的物品数。

**主标签（`is_primary`）的确定**：一部动漫常有多个题材，分题材实验（E4）需要唯一主标签，
故采用**特定优先**原则——越具体的题材优先级越高，避免小众番被 Comedy/Action 这类泛题材吞掉：

```
青春音乐 > 运动竞技 > 后宫福利 > 悬疑推理 > 科幻机战 > 超自然灵异
        > 日常治愈 > 恋爱校园 > 冒险奇幻 > 热血战斗 > 剧情文艺 > 喜剧搞笑
```

若不做此处理，2,000 多部运动/音乐番会被归到喜剧或热血里，**直接摧毁 H3 假设的检验效力**。

**「青春音乐」的判定为什么不用 `music` 标签**：语料中带 `music` 标签的动漫有 3,616 部，
但其中 2,339 部只是配乐出色（Cowboy Bebop、Samurai Champloo、Initial D…），并非音乐番。
改用「表演性」强信号词（idol/band/musical/singer/concert/…）并做**词边界**匹配后，
命中 2,100 部，语义准确度显著提升；词边界也避免了 `contraband` 命中 `band` 这类误判。

> **小众题材关注对象**：运动竞技（567）、后宫福利（732）、青春音乐（1,364）、悬疑推理（1,047）。
> 这四类的 Recall@10 提升是验证「多兴趣建模价值」的关键证据。
>
> 注：1,248 个池内物品**没有任何 12 类题材标签**（8.0%）——它们的 MAL 题材只有
> Hentai/Erotica，即下文第十节的合规题材。这些物品在题材实验中被排除。


### 2.3 序列构造与数据集划分

采用**留一法（Leave-One-Out）**，直接复用 `dataset.pkl` 的划分：

| 划分 | 内容 | 样本量 |
|---|---|---|
| 训练集 `train` | 用户序列去掉最后 2 条 | 1,306,691 用户 |
| 验证集 `val` | 倒数第 2 条 | 1,306,691 条 |
| 测试集 `test` | 倒数第 1 条 | 1,306,691 条 |
| 负采样池 | 每用户 100 个随机负样本 | 130,669,100 条 |

#### ⚠️ 时序口径（必须写进论文「局限性」）

原始三个交互文件**都没有时间戳字段**（只有 `userID, animeID, rating`），
因此 rq.md 要求的「按时间戳升序排序」**在数据层面无法执行**。
进一步实测：`dataset.pkl` 的序列顺序也无法从原始文件复现——

| 顺序复现（1,500 抽样用户） | 一致率 |
|---|---|
| 物品**集合** | **100.0%**（规则已完全还原） |
| 物品**顺序** | 60.7%（npy 口径） |

即：**能重建出"用户看过哪些番"，但重建不出"按什么顺序看的"**。
`dataset.pkl` 是唯一保留了排序信息的载体（推测其构建者使用了带时间戳的原始版本，
而落盘的 csv/dat/npy 是剥离时间戳后的再导出）。

**处理方式**：以 `dataset.pkl` 的既有顺序作为唯一时序口径，不做重排。
**论文表述要求**：不得声称序列是"按真实播放/收藏时间排序"，
应表述为「采用公开数据集提供的序列顺序，该顺序承载了数据集构建时的时序信息」。
这是本项目最需要如实披露的数据局限。

**序列长度**：`max_len = 50`。超出截断（保留最近 50 条），不足左侧补齐 `PAD=0`。
实测 592,753 个用户的训练序列长度超过 48，会被截断。

**训练样本扩充**：滑动窗口。对长度为 `n` 的训练序列，构造 `n` 个训练样本 `(prefix[0:t] → item[t])`（`t` 从 1 到 `n`）。

```
原始序列: [v1, v2, v3, v4, v5]
样本1: []            → v1
样本2: [v1]          → v2
样本3: [v1, v2]      → v3
样本4: [v1, v2, v3]  → v4
样本5: [v1..v4]      → v5
```

**实测规模**：滑动窗口展开后共 **109,081,471** 个训练样本（均值 83.5 / 用户，最大 8,917）。
样本量过大不宜落盘，实际训练时由 `models/sasrec/dataset.py` **在线生成**，
`scripts/preprocess.py` 只负责统计规模（见 `data/processed/preprocess_report.json`）。

### 2.4 内容特征库构建

| 步骤 | 做法 |
|---|---|
| 输入文本 | `title + alternative_title + genres(中文12类) + genres_detailed(细分标签，最多 24 个)` |
| 编码模型 | `distilbert-base-multilingual-cased`（本地权重 `models/content_encoder/pretrained/`） |
| 池化 | 取 `[CLS]` 向量（768 维） |
| 降维 | 在语料上拟合 **PCA** 投影到 **512 维**（累计解释方差比见 `content_meta.json`） |
| 归一化 | L2 归一化（使内积等价于余弦相似度） |
| 输出 | `data/features/content_vec_512.npy`（15,687 × 512，float32，约 32MB）<br>`data/features/content_vec_512_all.npy`（20,237 × 512，含未入池新番） |
| 索引 | FAISS `IndexFlatIP`（内积即余弦，因已归一化） |

#### 两处与早期文档的偏差（已更正）

**偏差一：没有剧情简介。** `animes.csv` 的列只有
`animeID/title/alternative_title/type/year/score/episodes/mal_url/sequel/image_url/genres/genres_detailed`，
**不存在 summary / synopsis 字段**。rq.md 要求的「题材标签 + 剧情简介」只能取前一半。
替代方案是用 `genres_detailed` 的数十个细分标签（如 `battle of wits`、`time skip`、
`based on a light novel`），其信息密度足以支撑语义向量。
若后续要补简介，可用 `mal_url` 爬取（20,237 次请求），或引入外部带简介的数据集。

**偏差二：降维不用「随机线性投影」。** 早期文档写「[CLS] → 线性投影到 512 维」，
但未训练的随机线性层会破坏语义结构。改为在语料上拟合 PCA（确定性、可复现），
投影矩阵落盘为 `content_pca.npz`，新番上线时用同一矩阵投影即可。
若要严格照搬文档口径，加 `--reduce random`。

```bash
# ① 一次性下载权重到本地（脚本内置代理，避开 huggingface_hub 的代理兼容问题）
python scripts/download_content_encoder.py

# ② 构建向量（CPU 可跑，默认离线加载本地权重）
python scripts/build_content_vectors.py --dim 512 --reduce pca --batch-size 64

# ③ 仅对进入推荐池的 15,687 个物品构建
python scripts/build_content_vectors.py --subset pool
```

> 注意：CPU 环境（本机 torch 为 `2.14.0+cpu`）下编码 20,237 条文本约数分钟。
> 有 GPU 时加 `--device cuda` 显著加速。

### 2.5 冷启动子集构造（E3 的核心，口径经过重设计）

| 项 | 定义 |
|---|---|
| 构建脚本 | `scripts/build_cold_start_subset.py` |
| **口径** | **holdout：上映年份 >= 2021 的动漫全部从训练集剥离**（模拟新番上线） |
| 新番规模 | **2,783 个物品**（占推荐池 17.7%） |
| 冷启动测试集 | **209,912 条**测试样本（目标物品属于新番，远超 5,000 的统计显著性下限） |
| 训练集剥离 | **15,344,636** 条交互（占训练总量 **14.07%**），涉及 743,556 个用户 |
| 负样本 | 池限定为新番集合（同分布），每样本 100 个，`cold_start_negatives.npy` |
| 出厂配置 | `configs/data.yaml` 的 `cold_start` 段 |

#### ⚠️ 为什么必须改口径：本数据集不存在严格意义的物品冷启动

早期文档定义新番为「交互数 < 10」。逐条核对后发现**这个口径不可用**：

1. 物品过滤规则本身就是「保留正样本交互数 >= 10 的物品」，故池内**不存在**交互数 < 10
   的物品，「< 10」得到空集；
2. 更根本的是，留一法只把每个用户的最后 2 条划给 val/test，
   因此**池内每个物品在训练集中至少出现 6 次**（实测最小值 = 6，
   训练中出现次数 <= 5 的物品数为 0）——**没有任何物品是模型「没见过」的**。

也就是说，这个数据集在构造时就把冷启动物品过滤掉了。必须用**模拟**口径。

#### 两种可选口径与其代价（实测）

| 口径 | 新番数 | 冷启动测试样本 | 剥离训练交互 | 占训练总量 |
|---|---|---|---|---|
| 年份 >= 2018 | 4,657 | 466,952 | 34,871,712 | **31.97%** |
| 年份 >= 2019 | 3,997 | 380,670 | 27,664,655 | 25.36% |
| **年份 >= 2021（默认）** | **2,783** | **209,912** | **15,344,636** | **14.07%** |
| 长尾后 20%（交互数 <= 38） | 3,200 | 446 | 67,199 | 0.06% |

选择 2021 的理由：剥离 14% 训练交互意味着「这些新番在训练时还不存在」，
正是真实的冷启动场景，而模型仍有 86% 的学习信号可用于学习物品嵌入与序列模式。
若把阈值调到 2018，剥离比例升至 32%，训练信号损失过大，会造成 E1/E2 与 E3 不可比。

> 长尾口径虽然几乎不动训练集，但测试样本仅 446 条，**达不到统计显著性要求**，
> 故仅作为备选（`--mode longtail`），不用于主实验。

#### 两条「否则实验无效」的约束

1. **新番在训练集中必须不可见**——否则模型从交互中学到了它，内容融合的贡献被高估。
   本方案通过 `strip_from_train_item_ids` 强制剥离，实现方式是训练时对序列做掩码过滤。
2. **负样本必须与新番同分布**——若负样本来自热门物品，模型可凭「热度先验」区分正负，
   而不是靠内容向量。故 `cold_start_negatives.npy` 的负样本池**限定为新番集合**。

#### 训练时的具体做法

```python
# models/sasrec/dataset.py / train.py
strip = set(cold_split["strip_from_train_item_indices"])   # 2,783 个 smap 索引
# 构造训练序列时过滤掉这些索引（等价于它们当时尚未上线）
seq = [x for x in user_seq if x not in strip]
```

> E3 必须**单独训练一次模型**（用剥离后的训练集），不能复用 E1/E2 的权重，
> 否则「新番不可见」的前提不成立。

---

### 2.6 阶段一产物的验收方式

阶段一产物（`data/processed/`、`data/features/`）体积大且不入库，无法靠「看文件」验收；
也不应只信 `*_report.json`——**报告是产物，本身可能写错**。因此专门提供了独立检验器：

```bash
python scripts/verify_stage1.py              # 快速档，跑 A~F 共 31 项，约 40 秒
python scripts/verify_stage1.py --full       # 追加 G 段权威源比对，共 33 项，约 2 分钟
python scripts/verify_stage1.py --sample-users 500   # 加大抽样用户数（更稳但更慢）
```

`verify_stage1.py` 的核心原则是**独立重算**：它不把任何 `*_report.json` 当作依据，
而是回到 `dataset/` 与 `configs/` 原始文件重新推导每一条结论，再与产物比对。
所以「报告写错」或「产物与代码不一致」都会在这里暴露。

检查覆盖七组：快速档 31 项（A~F），加 `--full` 后 33 项（含 G）。
最近一次运行 2026-09-12：快速档 **31/31 通过，耗时 40 秒**；`--full` **33/33 通过，耗时 122 秒**。

| 组 | 检查内容 | 关键判据 |
|---|---|---|
| A | 产物完整性 | 15 个文件存在且非空（合计约 625 MB）；不齐则直接退出 |
| B | 序列数据集结构 | 用户 1,306,691 / 物品 15,687；**输出 pkl 与官方切分逐键一致（抽 2,000 用户）**；每用户正样本数 >= 10 |
| C | 交互口径 | 全量扫描 `ratings.npy` 独立复现：正样本阈值 = 7（逐项比对）；正样本>=10 的用户 1,306,705 ≈ 1,306,691；物品恰 15,687；重复评分按「任一行 >= 7」的逐行口径 |
| D | 题材与合规 | 池内仍含 1,551 个 Hentai/Erotica 物品；12 类题材每类非空 |
| E | 内容向量 | 形状 (15687, 512) / (20237, 512)；L2 归一化；**池内矩阵与全量矩阵按 anime_id 精确对齐**；无空行 |
| F | 时序与冷启动 | `ratings.csv` 无时间戳列；holdout 新番集合 == {year>=2021} ∩ 池；负样本仅取自新番；训练集内物品最少出现 6 次（无严格冷启动） |
| G | 权威源口径（仅 `--full`） | `ratings.npy` 集合复现率 100%，`ratings.csv` 明显偏低（300 用户抽样 95.0%）——证明必须用 npy |

结构化结果落盘 `data/processed/stage1_acceptance.json`，可直接贴入论文附录。
退出码 `0` = 全部通过，`1` = 存在失败项，可直接接入 CI。

> 注意：B 组的「与官方切分逐键一致」是**回归护栏**——任何后续改动若误改
> `train/val/test`，会立即 FAIL。验收标准的口径常量集中在脚本顶部 `EXPECT` 字典，
> 改口径必须同步本文件。

### 2.7 阶段一脚本执行顺序

阶段一共 6 个脚本（1 条主链路 + 1 个一次性下载 + 1 个取证器 + 1 个验收器），
依赖关系如下。**顺序不可随意调换**：`build_content_vectors.py` 与
`build_cold_start_subset.py` 都依赖 `preprocess.py` 的产物，跑在它前面会直接报错；
而只重跑后者而不重跑前者，会得到「内容向量与物品集合对不上」的静默错误。

| # | 脚本 | 依赖 | 产物 | 性质 |
|---|---|---|---|---|
| ① | `preprocess.py` | `dataset/` 原始文件 + `configs/` | `data/processed/` 7 个文件 | **必须最先** |
| ② | `download_content_encoder.py` | 无（联网） | `models/content_encoder/pretrained/` | 一次性，权重已存在则跳过 |
| ③ | `build_content_vectors.py` | ① 的 `anime_meta` / `anime_detailed_tags` | `data/features/` 内容向量 | 与 ④ 独立 |
| ④ | `build_cold_start_subset.py` | ① 的 `item_stats` / `seq_dataset` | `cold_start_*.pkl/npy/json` | 与 ③ 独立 |
| ⑤ | `diagnose_dataset_source.py` | 仅 `dataset/` 原始文件 | `dataset_source_verdict.json` | 与 ① 无依赖，可随时跑 |
| ⑥ | `verify_stage1.py` | ①③④⑤ 的全部产物 | `stage1_acceptance.json` | **必须最后** |

为此提供了固化顺序的一键执行器（已内置前置检查与产物检查，顺序跑错会明确报出应先跑哪一步）：

```bash
python scripts/run_stage1.py                 # 缺什么跑什么；产物已存在的步骤自动跳过
python scripts/run_stage1.py --dry-run       # 只打印将执行的命令与跳过理由
python scripts/run_stage1.py --force         # 忽略「产物已存在」，全部重跑
python scripts/run_stage1.py --from content  # 从 ③ 开始（前面已完成的跳过）
python scripts/run_stage1.py --only verify --verify-full
python scripts/run_stage1.py --list          # 列出步骤、依赖、产物与当前状态
```

> 完全重跑一次全流水线（`.gitignore` 已排除产物，新克隆仓库必须全跑）的顺序即 ①→⑥；
> 日常开发通常只需 ⑥，约 40 秒。

> **工作目录无关**：以上命令（含单独执行某一个脚本）在**任意目录**下结果一致。
> `configs/*.yaml` 里的相对路径在读取配置时会被锚定到项目根，因此
> `cd F:/pj && python scripts/preprocess.py` 与
> `python F:/pj/scripts/preprocess.py`（在别的目录下）完全等价。
> 新增脚本必须遵守该约定，见 `docs/dev-conventions.md` 第 3.0 节。

---

## 三、实验环境

### 3.1 硬件与软件

> ⚠️ **下表为实测值**（2026-09-12 核对）。早期版本写的是「RTX 3090 24GB / 双端验证」，与实际不符，已更正。
> 本机**没有** CUDA 版 PyTorch：`torch.cuda.is_available() == False`，是阶段 2 的开工阻塞项，
> 详见 [progress.md](progress.md) §3.3 与 §6 的路线选项。

| 项 | 实测配置 | 备注 |
|---|---|---|
| 操作系统 | Windows（内核 10.0.26200 / DisplayVersion 25H2） | Linux 端**未验证** |
| CPU | Intel 16 逻辑核 | — |
| GPU | **NVIDIA RTX 2060 / 6 GB / 驱动 457.85（CUDA 11.1）** | 驱动偏旧：CUDA 11.x 需 ≥ 452.39、CUDA 12.x 需 ≥ 527.41 |
| 内存 | 31.7 GB | 加载 `ratings.npy`（6.9 GB）前建议先释放内存 |
| 磁盘 | F: 余 66 GB | 充足 |
| Python | 3.11.16（conda env `py3_11`） | `E:\tools\anaconda\envs\py3_11\python.exe` |
| PyTorch | 2.14.0 **+cpu** ⚠️ | 需按 progress.md §6 D1 换装 CUDA 轮子 |
| 关键库 | numpy 2.4.6, pandas 2.3.3, transformers 4.57.1, scikit-learn 1.9.0 | faiss 未安装（阶段 2 可先用 numpy 精确内积） |

### 3.2 环境固化

```bash
# 完整环境快照（论文附录用）
python -c "import torch,sys;print(sys.version);print(torch.__version__, torch.version.cuda)"
pip freeze > experiments/{exp_id}/requirements_freeze.txt
```

每次实验输出目录必须包含 `requirements_freeze.txt` 与 `git commit hash`，保证可复现。

> **待办**：M2.0 装好 CUDA 版 torch 后，本节需重新生成一次快照并把实际版本号固化进论文附录。

### 3.3 随机性控制

| 项 | 值 |
|---|---|
| 随机种子 | `42`（主实验）；`{42, 2024, 2025}` 三种子取平均（最终报告） |
| 控制范围 | `random` / `numpy` / `torch` / `cuda` 全部设种 |
| cuDNN | `torch.backends.cudnn.deterministic = True`，`benchmark = False` |
| DataLoader | `shuffle=False`（序列推荐按用户顺序可固定） |
| 结果波动 | 同种子重跑指标必须完全一致；否则视为有未控随机源，必须排查 |

```python
# scripts/train.py 中的 set_seed
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 四、对比基线

| 类型 | 模型 | 实现位置 | 实现的公平性要求 |
|---|---|---|---|
| 传统协同过滤 | **ItemCF** | `models/baselines/itemcf.py` | 基于训练集共现矩阵，余弦相似度，TopK=200 邻居 |
| 经典序列模型 | **GRU4Rec** | `models/baselines/gru4rec.py` | 隐藏层 64，1 层 GRU，与本文模型同嵌入维度 |
| 基准序列模型 | **原版 SASRec** | `models/sasrec/model.py` | 2 层、2 头、hidden=64、dropout=0.2 |
| 本文模型 | **多兴趣 + 内容融合 SASRec** | `models/multi_interest/` + `content_encoder/` | 在上者基础上加 K=4 胶囊 + 双路融合 |

### 4.1 公平性控制（论文答辩必问）

| 控制项 | 做法 |
|---|---|
| 同一数据划分 | 全部模型用同一个 `dataset.pkl` 的 train/val/test |
| 同一负采样池 | 全部模型共用 `random-sample_size100-seed98765.pkl` |
| 同一嵌入维度 | 物品嵌入统一 64 维（ItemCF 无需嵌入） |
| 同一评估协议 | 全部走 `models/eval/evaluator.py`，候选 = 1 正 + 100 负 |
| 同一早停策略 | 看验证集 NDCG@10，patience=10 |
| 超参搜索预算一致 | 每个模型同样的网格规模（见 4.2） |

### 4.2 超参搜索网格

| 参数 | 搜索范围 | 最优值（待实验确定） |
|---|---|---|
| `hidden_size` | {32, 64, 128} | 待填 |
| `num_layers` | {1, 2, 3} | 待填 |
| `num_heads` | {1, 2, 4} | 待填 |
| `dropout` | {0.1, 0.2, 0.3, 0.5} | 待填 |
| `lr` | {1e-4, 5e-4, 1e-3, 3e-3} | 待填 |
| `batch_size` | {128, 256, 512} | 待填 |
| `K`（兴趣数） | {2, 4, 6, 8} | 待填（预期 4） |

搜索方式：先粗网格定范围，再对 `hidden/lr/dropout` 做细搜。**记录完整搜索日志到 `experiments/{exp_id}/hpo_log.csv`**，论文可放热力图。

---

## 五、评价指标

### 5.1 指标定义

统一采用留一法 + 100 负采样（1 正 : 100 负）的标准序列推荐评估协议。

**HR@K（Hit Ratio，命中率）**

```
HR@K = (1/|U|) · Σ_{u∈U} 1( rank(positive_u) ≤ K )
```

含义：推荐列表 TopK 中出现真实下一部动漫的用户占比。

**NDCG@K（Normalized Discounted Cumulative Gain）**

```
DCG@K  = Σ_{i=1..K} (2^rel_i - 1) / log2(i + 1),   rel_i = 1 若位置 i 是正样本，否则 0
IDCG@K = 1 / log2(2) = 1                            (理想情况正样本排在第 1 位)
NDCG@K = DCG@K / IDCG@K
```

含义：不但要看"有没有命中"，还要看"排在第几"。命中位置越靠前得分越高。

**Recall@K（分题材实验用）**

```
Recall@K = Σ_u |TopK_u ∩ Relevant_u| / Σ_u |Relevant_u|
```

含义：在分题材实验中，`Relevant_u` 限定为该题材的测试正样本，用于衡量模型对该题材的覆盖能力。

**MRR（辅助指标）**

```
MRR = (1/|U|) · Σ_u 1 / rank(positive_u)
```

**K 取值**：`K ∈ {5, 10}`，与 rq.md 要求一致。

### 5.2 评估流程

```mermaid
flowchart LR
    T[测试用户序列] --> M[模型前向打分]
    N[每用户100负样本] --> M
    M --> R[对 101 个候选排序]
    R --> IDX[取正样本 rank]
    IDX --> CAL[models/eval/metrics.py<br/>计算 HR/NDCG/Recall]
    CAL --> OUT[metrics.json]
```

**严格实现要求**：

| 要求 | 说明 |
|---|---|
| 候选集 | 1 个正样本 + 100 个负样本 = 101 个候选，**不是全库排序** |
| 负样本 | 使用预生成的 `random-sample_size100-seed98765.pkl`，所有模型完全一致 |
| 排名定义 | 从 1 开始计数（`rank=1` 表示排第一） |
| 并列处理 | 得分并列时用稳定排序（`torch.argsort(stable=True)`） |
| 无效样本 | 若正样本在训练/验证集已出现，该样本剔除并记录数量 |
| 聚合方式 | 全部测试用户的**算术平均**，宏平均（macro） |

### 5.3 统计显著性

仅报告平均值不够，需给出显著性检验：

| 检验 | 用途 |
|---|---|
| 配对 t 检验（paired t-test） | 本文模型 vs 各基线，p < 0.05 视为显著 |
| 三随机种子取均值 ± 标准差 | 主实验表格中报 `mean ± std` |
| 样本量 | 1,306,691 用户，远超显著性所需样本量 |

---

## 六、实验分组

### 6.1 实验组 E1：整体效果对比

| 项 | 内容 |
|---|---|
| **目的** | 验证 H1、H2，证明本文模型综合最优 |
| **变量** | 模型（4 个：ItemCF / GRU4Rec / SASRec / 本文模型） |
| **控制** | 数据划分、负采样池、评估协议、早停策略全部一致 |
| **测试集** | 全量测试集（1,306,691 用户，每用户 101 候选） |
| **指标** | HR@5、HR@10、NDCG@5、NDCG@10、MRR |
| **输出** | `experiments/E1_overall/metrics.json` + 对比柱状图 |

### 6.2 实验组 E2：消融实验

| 项 | 内容 |
|---|---|
| **目的** | 验证 H3、H4，量化每个模块的独立增益与协同增益 |
| **变量** | 4 组配置（纯 SASRec / +多兴趣 / +内容融合 / 完整模型） |
| **控制** | 其余超参完全一致（同一 `hidden/layers/heads/lr/epochs`） |
| **测试集** | 全量测试集 |
| **指标** | HR@5/10、NDCG@5/10 |
| **输出** | 详见 [ablation-study.md](ablation-study.md) |

### 6.3 实验组 E3：冷启动专项

| 项 | 内容 |
|---|---|
| **目的** | 验证 H4——内容融合对无交互新番的作用 |
| **变量** | 完整模型 vs 纯 SASRec |
| **测试集** | **冷启动子集**（目标物品交互数 < 10 的测试样本） |
| **控制** | 训练时新番交互已剥离；负采样同为 100 个随机负样本 |
| **额外控制** | 对比「内容融合权重 0.3 vs 0.5」两档，验证动态权重设计 |
| **指标** | HR@10、NDCG@10 |
| **输出** | `experiments/E3_coldstart/metrics.json` |

**关键注意事项（否则实验无效）**：
1. 新番在训练集中**必须不可见**——否则模型从交互中学到了，内容融合的贡献被高估。
2. 冷启动子集的负样本也应该是新番（或至少同分布），否则模型可能靠"热门"而非"内容"区分正负。
3. 需报告冷启动子集的样本量。样本量过小时结论不可靠。

### 6.4 实验组 E4：分题材细分指标

| 项 | 内容 |
|---|---|
| **目的** | 验证 H3——多兴趣建模对小众题材的覆盖提升 |
| **变量** | 4 组消融配置（同 E2）× 12 类题材 |
| **测试集** | 全量测试集，按目标物品的主题材分组 |
| **指标** | Recall@10（每类题材单独计算） |
| **重点** | 小众题材（运动竞技/后宫福利/悬疑推理/青春音乐）的提升幅度 |
| **输出** | 12 × 4 的指标矩阵 + 分组柱状图 |

> **假设**：多兴趣模型对**小众题材**的提升应显著大于对主流题材（如喜剧搞笑、热血战斗）的提升——因为主流题材的单一兴趣向量已经能覆盖，小众题材必须靠多向量才能召回。**如果结果相反，需要在论文中如实讨论。**

### 6.5 实验矩阵总览

| 实验 | 变量数 | 配置组合数 | 模型训练次数 | 预估耗时 |
|---|---|---|---|---|
| E1 整体对比 | 4 模型 | 4 | 4 | ~12 h |
| E2 消融 | 4 配置 | 4 | 4 | ~12 h |
| E3 冷启动 | 2 模型 × 2 权重 | 4 | 4 | ~12 h |
| E4 分题材 | 复用 E2 结果 | 4 | 0（复用） | ~1 h（评估） |
| 超参搜索 | 7 参数 | ~60 组 | 60 | ~40 h |
| **合计** | — | — | **72** | **~77 h** |

> 消融组 2、3 与 E1 的 SASRec 基准可复用同一次训练结果，实际训练次数可压缩。
>
> ⚠️ **上表的 ~77 h 是早期按 RTX 3090 估的，不可作为排期依据。** 本机为 RTX 2060 6GB，
> 且当前 torch 无 CUDA 支持；阶段一实测每 epoch 的训练样本量为 **109,081,471**（约 1.09 亿），
> 必须采用「开发档抽样 / 正式档全量」双档位策略，超参搜索只在开发档进行。
> 重估流程与决策项见 [progress.md](progress.md) §5–§6。

---

## 七、变量控制与测试集划分说明

### 7.1 三类变量的处理

| 变量类型 | 实验中的处理 |
|---|---|
| **自变量** | 要验证的模块开关（多兴趣有/无、内容融合有/无）、模型种类 |
| **控制变量** | 数据划分、负采样池、嵌入维度、层数、学习率、训练轮数、早停策略、随机种子 |
| **因变量** | HR@5/10、NDCG@5/10、Recall@10、MRR |

### 7.2 测试集划分的唯一性

**整个项目只有一份测试集**：`dataset.pkl` 的 `test` 字段（1,306,691 用户，每用户 1 条正样本）。所有实验从它派生：

| 派生子集 | 派生规则 | 仅用于 |
|---|---|---|
| 全量测试集 | 原样 | E1、E2、E4 |
| 冷启动子集 | 正样本属于 `n_interactions < 10` 的新番 | E3 |
| 分题材子集 | 按正样本的 `is_primary` 题材分组 | E4 |

> ⚠️ **绝不允许**为了让指标好看而重新划分测试集。任何划分变更必须记录在 `experiments/SPLIT_CHANGELOG.md`。

### 7.3 训练超参一致性检查

消融实验最容易犯的错是"某个配置多训了几轮"。必须用**同一份配置基线**：

```yaml
# configs/experiment.yaml
base:                        # 所有消融组共享的基线配置
  max_seq_len: 50
  hidden_size: 64
  num_layers: 2
  num_heads: 2
  dropout: 0.2
  batch_size: 256
  lr: 0.001
  epochs: 200
  early_stop_patience: 10
  seed: 42

groups:
  E2_1_pure_sasrec:      { use_multi_interest: false, use_content_fusion: false }
  E2_2_multi_interest:   { use_multi_interest: true,  use_content_fusion: false }
  E2_3_content_fusion:   { use_multi_interest: false, use_content_fusion: true  }
  E2_4_full:             { use_multi_interest: true,  use_content_fusion: true  }
```

跑实验时只允许覆盖 `groups` 里的开关，`base` 里的任何值被改动都必须记录原因。

---

## 八、实验执行

### 8.1 一键跑全部实验

```bash
# 全部实验（E1 + E2 + E3 + E4）
python scripts/run_experiments.py --config configs/experiment.yaml --all

# 只跑消融
python scripts/run_experiments.py --config configs/experiment.yaml --group E2

# 只跑冷启动
python scripts/run_experiments.py --config configs/experiment.yaml --group E3

# 指定输出目录与种子
python scripts/run_experiments.py --all --seed 42 --out experiments/20260912_E_all
```

### 8.2 输出目录规范

```
experiments/20260912_E_all/
├── config.yaml              # 本次实验的完整配置快照
├── requirements_freeze.txt  # 环境快照
├── git_commit.txt           # 代码版本
├── hpo_log.csv              # 超参搜索日志
├── E1_overall/
│   ├── itemcf/metrics.json
│   ├── gru4rec/metrics.json
│   ├── sasrec/metrics.json
│   ├── ours/metrics.json
│   └── summary.csv          # 汇总表
├── E2_ablation/
│   ├── pure_sasrec/  multi_interest/  content_fusion/  full/
│   └── summary.csv
├── E3_coldstart/
│   └── summary.csv
├── E4_by_genre/
│   └── recall10_matrix.csv
└── figures/                 # 生成的图表
    ├── fig1_overall_bar.png
    ├── fig2_ablation_grouped.png
    ├── fig3_genre_recall.png
    └── fig4_coldstart.png
```

### 8.3 结果回写

实验脚本跑完后，指标回写数据库，供管理后台展示：

```bash
python scripts/backfill_metrics.py \
  --exp-dir experiments/20260912_E_all \
  --model-ver multi_interest_content_v1
# → 写入 metric_snapshot 表（metric_type = hr5/hr10/ndcg5/ndcg10/recall10）
```

---

## 九、实验结果模板

> 数值待实验完成后填写，**禁止编造**。填写规范见 [result-analysis.md](result-analysis.md)。

### 9.1 E1 整体效果对比

| 模型 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| ItemCF | 待填 | 待填 | 待填 | 待填 | 待填 |
| GRU4Rec | 待填 | 待填 | 待填 | 待填 | 待填 |
| SASRec（原版） | 待填 | 待填 | 待填 | 待填 | 待填 |
| **本文模型** | **待填** | **待填** | **待填** | **待填** | **待填** |

### 9.2 E2 消融实验

| 配置 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | 相对基准 |
|---|---|---|---|---|---|
| 纯 SASRec | 待填 | 待填 | 待填 | 待填 | — |
| + 多兴趣 | 待填 | 待填 | 待填 | 待填 | 待填 |
| + 内容融合 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 完整模型 | 待填 | 待填 | 待填 | 待填 | 待填 |

### 9.3 E3 冷启动专项

| 模型 | HR@10 | NDCG@10 | 样本量 |
|---|---|---|---|
| 纯 SASRec | 待填 | 待填 | 待填 |
| 完整模型（内容权重 0.3） | 待填 | 待填 | 待填 |
| 完整模型（内容权重 0.5） | 待填 | 待填 | 待填 |

### 9.4 E4 分题材 Recall@10

| 题材 | 动漫数 | 纯 SASRec | +多兴趣 | +内容融合 | 完整模型 |
|---|---|---|---|---|---|
| 热血战斗 | 5,387 | 待填 | 待填 | 待填 | 待填 |
| 冒险奇幻 | 8,423 | 待填 | 待填 | 待填 | 待填 |
| 日常治愈 | 1,615 | 待填 | 待填 | 待填 | 待填 |
| 恋爱校园 | 2,218 | 待填 | 待填 | 待填 | 待填 |
| 悬疑推理 | 1,431 | 待填 | 待填 | 待填 | 待填 |
| 科幻机战 | 3,255 | 待填 | 待填 | 待填 | 待填 |
| 喜剧搞笑 | 6,902 | 待填 | 待填 | 待填 | 待填 |
| **运动竞技** | 743 | 待填 | 待填 | 待填 | 待填 |
| 超自然灵异 | 2,114 | 待填 | 待填 | 待填 | 待填 |
| 剧情文艺 | 3,907 | 待填 | 待填 | 待填 | 待填 |
| **青春音乐** | 待填 | 待填 | 待填 | 待填 | 待填 |
| **后宫福利** | 837 | 待填 | 待填 | 待填 | 待填 |

---

## 十、伦理与合规

| 项 | 说明 |
|---|---|
| 数据合规 | MAL 数据为公开数据集，仅用于学术研究；不二次分发原始数据 |
| 内容合规 | `Hentai` / `Erotica` 题材不进入推荐候选池与全部实验，见下方实测说明 |
| 隐私 | 实验仅使用匿名 `userID`，不涉及任何真实身份信息 |
| 可复现 | 所有实验提供配置、种子、环境快照与代码版本 |

### ⚠️ 合规题材的实测情况（必须如实披露）

早期文档称「已剔除 Hentai(1,602) / Erotica(73) 共 1,675 条，不入推荐池」。
**逐条核对后发现这不成立**——所提供的 `dataset.pkl` 并未做这项剔除：

| 指标 | 实测值 |
|---|---|
| 元数据中 Hentai/Erotica 动漫数 | 1,675 |
| **其中进入推荐池（`smap`）的** | **1,551（占推荐池 9.89%）** |
| test 正样本属于合规题材的样本数 | 5,594（占 0.428%） |
| 训练集中涉及合规题材的交互占比 | 约 0.56%（30,000 用户抽样） |
| 池内无任何 12 类题材标签的物品 | 1,248（其 MAL 题材仅剩合规题材） |

**处理方式**：**不改动 `dataset.pkl` 的原始切分**（改动会破坏与负采样文件的配套关系、
并使所有已记录指标失去可比性），而是在**候选池与评估阶段屏蔽**——

1. 评估打分时把 1,551 个合规物品的得分置为 `-inf`；
2. 负采样池中若出现合规物品，同样剔除后重采（由 `models/eval/evaluator.py` 保证）；
3. 分题材实验（E4）中排除无 12 类标签的 1,248 个物品。

屏蔽清单落盘在 `data/processed/preprocess_report.json` 的
`compliance.forbidden_item_indices`，可随时审计。

> **论文表述要求**：不能写「已从数据集中剔除不合规内容」，应写
> 「公开数据集包含 1,551 个不合规题材物品，本文在候选池与评估阶段将其屏蔽，
> 未纳入任何推荐结果与指标计算」。这是答辩时可能被追问的点，需提前准备。


---

## 十一、相关文档

- 消融实验细节 → [ablation-study.md](ablation-study.md)
- 结果分析与图表 → [result-analysis.md](result-analysis.md)
- 指标代码位置 → [project-structure.md](project-structure.md) `models/eval/`
- 指标回写接口 → [api-specification.md](api-specification.md) `/admin/metrics`
