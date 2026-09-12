# 项目进度总览 · AniRec

> **快照时间**：2026-09-12 16:20 ｜ **当前提交**：`f9f9a58` ｜ **下一里程碑**：M2.0 训练环境就绪
>
> 本文件只回答三个问题：**做到哪了**（§1–2）／**环境撑不撑得住**（§3）／**下一步做什么**（§4–5）。
> 每完成一个里程碑请更新本文件；文档索引同步维护在 [README.md](README.md)。

---

## 一、总进度

| 阶段 | 权重 | 状态 | 交付物 / 卡点 |
|---|---|---|---|
| 阶段 0　项目初始化 | — | ✅ 完成 | 目录骨架 + 14 份设计文档 + GitHub 仓库（`zarbzarb/animes`） |
| **阶段 1　数据预处理与数据集构建** | 15% | ✅ **完成并通过验收** | 7 个脚本 + 全部数据产物 + 8 条实测口径，验收 31/31 |
| 阶段 2　核心算法与对比实验 | 35% | ⬜ 未开始 | **卡在训练环境**：当前 torch 是 CPU 版，见 §3.3 |
| 阶段 3　全栈系统与联调 | 35% | ⬜ 未开始 | 依赖阶段 2 的模型权重与嵌入矩阵 |
| 阶段 4　测试 | 15% | ⬜ 未开始 | 依赖阶段 3 的可运行系统 |

```mermaid
flowchart LR
    S0["阶段 0<br/>项目初始化"] --> S1["阶段 1<br/>数据预处理<br/>✅ 15%"]
    S1 --> S2["阶段 2<br/>核心算法与实验<br/>⬜ 35%"]
    S2 --> S3["阶段 3<br/>全栈系统与联调<br/>⬜ 35%"]
    S3 --> S4["阶段 4<br/>测试<br/>⬜ 15%"]
    S2 -.模型权重/嵌入.-> S3
    S3 -.接口/页面.-> S4
```

---

## 二、阶段一交付（已完成，可随时独立复验）

### 2.1 脚本（`scripts/`，共 7 个）

| 脚本 | 角色 | 说明 |
|---|---|---|
| `preprocess.py` | 主链路 1-A | 清洗 `animes.csv` → 12 类题材；全量扫 `ratings.npy` 统计；按 `dataset.pkl` 顺序重建留一法切分；与官方切分交叉校验 |
| `build_content_vectors.py` | 主链路 1-B | 标题+12 类题材+细分标签 → DistilBERT `[CLS]` → PCA 降到 512 维 → L2 归一化 |
| `build_cold_start_subset.py` | 主链路 1-D | holdout 口径（year≥2021）模拟新番冷启动 + 新番专用负样本池 |
| `download_content_encoder.py` | 辅助 | 离线拉取 DistilBERT 权重（517MB，`.gitignore` 排除） |
| `diagnose_dataset_source.py` | 取证器 | 可复现生成「为什么用 npy / 为什么阈值 7」的证据 JSON |
| `verify_stage1.py` | **验收器** | 不读任何报告，回到原始文件独立重算 31 项结论并打 PASS/FAIL；退出码可接 CI |
| `run_stage1.py` | 编排器 | 把 6 个脚本的依赖顺序固化成代码（前置检查 + 产物检查） |

一键复现整个阶段一：

```bash
E:/tools/anaconda/envs/py3_11/python.exe scripts/run_stage1.py
```

### 2.2 数据产物（本地约 665 MB，已被 `.gitignore` 排除）

| 路径 | 体量 | 内容 |
|---|---|---|
| `data/processed/seq_dataset.pkl` | 361 MB | `{train, val, test, umap, smap, meta}`，1,306,691 用户 × 15,687 物品 |
| `data/processed/cold_start_subset.pkl` | 89 MB | 2,783 个新番 / 209,912 条冷启动测试样本 |
| `data/processed/cold_start_negatives.npy` | 84 MB | 209,912 × 100，负样本池**限定为新番** |
| `data/processed/user_stats.parquet` | 13 MB | 每用户序列长度 |
| `data/processed/anime_meta.parquet` | 1.4 MB | 元数据 + 12 类题材 + 主题材 + `is_forbidden` |
| `data/processed/anime_detailed_tags.parquet` | 0.9 MB | 细分标签（内容向量用） |
| `data/processed/item_stats.parquet` | 0.6 MB | 每物品交互数 / 是否新番 |
| `data/processed/preprocess_report.json` | 25 KB | 全量统计报告（论文「数据集」节素材） |
| `data/processed/validation_report.json` | 1.1 KB | 与官方切分的交叉校验 |
| `data/processed/dataset_source_verdict.json` | 1.2 KB | 数据源判定证据 |
| `data/processed/cold_start_report.json` | 1.2 KB | 冷启动子集统计 |
| `data/processed/stage1_acceptance.json` | 5.4 KB | 最近一次验收结果（31/31） |
| `data/features/content_vec_512.npy` | 32 MB | (15687, 512) 池内矩阵，第 `i-1` 行对齐 `smap` 索引 |
| `data/features/content_vec_512_all.npy` | 41 MB | (20237, 512) 全量矩阵 |
| `data/features/content_pca.npz` | 1.6 MB | PCA 投影矩阵（新番上线可复用） |
| `data/features/content_ids.npy` / `content_meta.json` | 81 KB | 行序对照与元信息 |

### 2.3 配置

| 文件 | 说明 |
|---|---|
| `configs/genre_taxonomy.yaml` | 12 类题材**单一事实源**（21 类 MAL 主题材 + 细分标签兜底 + 禁用题材） |
| `configs/data.yaml` | 已按逐条实测口径对齐（`rating≥7`、正样本≥10、npy 权威源、PCA 降维） |

### 2.4 口径锁定（改动会使数据集与后续实验不可比）

1. **权威交互源 = `dataset/ratings.npy`**（csv/dat 存在整体 +1 偏移，集合复现率仅 90.7%）
2. **正样本 = `rating ≥ 7`**（1~6 全丢、7~10 全留，二分无例外）
3. **过滤 = 用户与物品「正样本交互数 ≥ 10」**（非早期文档写的 <5 / <10）
4. **时序 = `dataset.pkl` 既有顺序**（原始文件**无时间戳**，顺序不可复现）
5. **序列长度 `max_len = 50`**，留一法切分（倒数 1 = test，倒数 2 = val）
6. **内容特征 = 标题 + 12 类题材 + 细分标签**（`animes.csv` **无剧情简介字段**）
7. **内容编码 = DistilBERT 多语言版 `[CLS]` → PCA → 512 维 → L2 归一化**
8. **冷启动 = holdout 模拟口径（year ≥ 2021）**，因为池内物品训练集最少出现 6 次，**不存在严格物品冷启动**

> 另有两条必须如实披露的事项：`dataset.pkl` **未剔除** Hentai/Erotica（池内 1,551 个，占 9.89%），
> 本项目在候选池与评估阶段屏蔽；负样本池为 100 个随机负样本。

### 2.5 复验方式（三级，由硬到软）

```bash
# ① 机器判定：独立重算，最难造假
E:/tools/anaconda/envs/py3_11/python.exe scripts/verify_stage1.py          # 31 项，约 40 s
E:/tools/anaconda/envs/py3_11/python.exe scripts/verify_stage1.py --full   # 33 项，约 2 min

# ② 一键复跑（缺什么跑什么）
E:/tools/anaconda/envs/py3_11/python.exe scripts/run_stage1.py

# ③ 自己挑一个反例：任选一个 userID，在 ratings.npy 里筛出 rating≥7 的 animeID，
#    与 seq_dataset.pkl 中该用户 train+val+test 的并集比对 —— 应完全一致
```

---

## 三、当前环境实况

### 3.1 本机硬件（实测，2026-09-12）

| 项 | 实测值 | 备注 |
|---|---|---|
| 操作系统 | Windows（内核 10.0.26200 / DisplayVersion 25H2） | 注册表 `ProductName` 仍写 "Windows 10 Pro"（Win11 已知显示问题）；Linux 端**未验证** |
| CPU | Intel 16 逻辑核 | — |
| 内存 | 31.7 GB（可用约 14 GB） | 加载 `ratings.npy`（6.9GB）时建议先关掉其他大程序 |
| **GPU** | **NVIDIA RTX 2060 / 6 GB / 驱动 457.85（CUDA 11.1）** | **与 `evaluation-plan.md` 3.1 写的 RTX 3090 24GB 不符** |
| 磁盘 F: | 总 86 GB / 余 66 GB | 单模型 checkpoint 预计 < 10 MB，够用 |
| Python | 3.11.16（conda env `py3_11`） | `E:\tools\anaconda\envs\py3_11\python.exe` |

### 3.2 已装依赖

| 已装 | 版本 | | 缺失（阶段 2/3 需要） |
|---|---|---|---|
| torch | **2.14.0+cpu** ⚠️ | | `faiss`（阶段 2 后期检索，可选） |
| numpy / pandas | 2.4.6 / 2.3.3 | | `pytest`（阶段 4） |
| transformers | 4.57.1 | | `fastapi` / `uvicorn`（阶段 3） |
| scikit-learn / scipy | 1.9.0 / 1.17.1 | | `sqlalchemy` / `pymysql`（阶段 3） |
| matplotlib / seaborn | 3.11.0 / 0.13.2 | | `redis` / `apscheduler`（阶段 3） |

### 3.3 ⚠️ 阻塞项：torch 是 CPU 版，阶段 2 无法按现有环境开工

`configs/model.yaml` 已写 `device: cuda`，但本机 `torch.cuda.is_available() == False`。
原因有两个，必须一起解决：

1. **torch 装的是 CPU 轮子**（`2.14.0+cpu`），本机根本没装 CUDA 运行时；
2. **显卡驱动偏旧**：457.85 只对应 CUDA 11.1，而新版 torch 的 CUDA 轮子要求
   驱动 ≥ 452.39（CUDA 11.x 小版本兼容）或 ≥ 527.41（CUDA 12.x）。

三条可选路线（需要你拍板，见 §6）：

| 路线 | 做法 | 优点 | 代价 |
|---|---|---|---|
| **A. 更新驱动 + 装新 CUDA torch** | 驱动升到 5xx+ → `pip install torch --index-url .../cu12x` | 性能最好，保留 torch 2.14 | 需自行下载安装驱动（可能要重启） |
| **B. 不动驱动，装 cu118 版 torch** | 把 py3_11 的 torch 换成 `2.5.x+cu118` | 免装驱动 | 需降 torch 版本；与 numpy 2.4 存在兼容风险，建议单开一个 conda env |
| **C. 纯 CPU** | 保持现状 | 零改动 | **不可行**：见 §5，每 epoch 量级为千万级样本 |

> 无论走哪条路，**M2.0 的第一件事都是实测吞吐**（用 5 万用户子集跑 1 个 epoch，记录 s/step），
> 再据此定训练档位。**不要凭感觉估时间。**

### 3.4 需要随本次进度一起更正/留意的文档不实之处

| 位置 | 现状 | 处理 |
|---|---|---|
| `evaluation-plan.md` 3.1 硬件表 | 写「RTX 3090 24GB / 32GB / 双端验证」 | 应改为本机实测（RTX 2060 6GB / Windows），已在本次提交更正 |
| `evaluation-plan.md` 6.5 耗时预估 | 「合计 ~77 h」基于 3090 + 全量网格 | 待 M2.0 实测后重估，当前数字不可作为排期依据 |
| `evaluation-plan.md` 3.2 环境固化 | 依赖 `requirements.txt` | 阶段 2 装完 CUDA torch 后需重新 `freeze` |

---

## 四、阶段二拆解（下一步做什么）

阶段 2 的验收标准是：**E1~E4 四组实验的真实指标落进 `docs/result-analysis.md`，且能被一键复现。**

| # | 里程碑 | 做什么 | 产物 | 验收标准 |
|---|---|---|---|---|
| **M2.0** | 训练环境就绪 | 按 §6 选定路线装 GPU 版 torch；写吞吐基准脚本，实测 s/step 并外推单 epoch 耗时 | `scripts/bench_throughput.py` + 环境快照 | `torch.cuda.is_available()==True`；能跑通 1 个 epoch 并给出耗时 |
| **M2.1** | 指标唯一实现 | 按 `evaluation-plan.md` 5.1 实现 HR@K / NDCG@K / Recall@K / MRR，含并列稳定排序 | `models/eval/metrics.py` | 单元测试用手算样例比对，允许误差 1e-6 |
| **M2.2** | 评估器 | 1 正 + 100 负候选，走预生成的负样本 pkl；支持全量 / 分题材 / 冷启动三种切分 | `models/eval/evaluator.py` | 同一批预测重复跑结果一致；两种切分都能出 `metrics.json` |
| **M2.3** | 滑动窗口 Dataset | 按 `max_len=50` 在线生成样本，不落盘 | `models/sasrec/dataset.py` | **因果性测试**：第 t 个样本不得看到 t 之后的信息 |
| **M2.4** | SASRec 基座 | 2 层 2 头 hidden 64 dropout 0.2 + BCE 负采样 + 早停（看 val NDCG@10, patience=10） | `models/sasrec/{model,config,train}.py` | 在开发档上 loss 稳定下降，val NDCG@10 明显高于随机基线 |
| **M2.5** | 多兴趣胶囊 | K=4 动态路由（3 次迭代）接在 SASRec 输出后；记录胶囊两两余弦相似度检测塌缩 | `models/multi_interest/{capsule,model,train}.py` | 4 个胶囊不塌缩（相似度 < 0.7，配置里已有告警阈值） |
| **M2.6** | 内容融合 | 候选侧拼接 + 排序侧 7:3 加权；冷启动样本权重切 5:5 | `models/content_encoder/fusion.py` | `fusion_score()` 签名与 `project-structure.md` 契约一致 |
| **M2.7** | 对比基线 | ItemCF（共现 + 余弦，TopK 200 邻居）、GRU4Rec（隐藏 64，1 层） | `models/baselines/*.py` | 与本文模型**共用同一划分 / 负样本池 / 评估器 / 早停策略** |
| **M2.8** | 实验编排 | `run_experiments.py` 按 `configs/experiment.yaml` 串起 E1~E4，落盘 `experiments/{id}/`，回写 `result-analysis.md` | `scripts/run_experiments.py`、`scripts/plot_results.py` | 一次命令跑完四组；输出目录结构与 `evaluation-plan.md` 8.2 一致 |
| **M2.9** | 填表与作图 | 真实指标填入 `result-analysis.md` / `ablation-study.md`，生成 4 张图 | 指标表 + `figures/*.png` | **每个数字都有出处（实验目录名）**，无出处不许填 |

推荐执行顺序即上表自上而下；**M2.1–M2.3 与 GPU 环境无关，可以立刻开工**（这也是我建议的下一步起点）。

---

## 五、训练规模与时间预算（必须先实测）

阶段一产物给出的训练规模：

| 指标 | 值 |
|---|---|
| 训练样本数（滑动窗口，每 epoch） | **109,081,471**（约 1.09 亿） |
| 每用户训练序列长度 | 均值 83.5，最大 8,917 |
| 用户 / 物品 | 1,306,691 / 15,687 |
| 测试用户 | 全量 1,306,691（每用户 101 候选） |
| 冷启动测试样本 | 209,912 |

**结论：1.09 亿样本/epoch 在 CPU 上不可行**（量级上每 epoch 是小时~天级），
必须用 GPU；即便用 RTX 2060，也建议采用**双档位**策略：

| 档位 | 用途 | 用户抽样 | 预计每 epoch（**待 M2.0 实测标定**） |
|---|---|---|---|
| **开发档** | 架构调试、超参搜索、快速迭代 | 5%~10%（6.5 万~13 万用户） | 分钟级 |
| **正式档** | E1~E4 最终数字 | 全量 130 万用户 | 小时级 |

要点：
1. **训练与评估可以不同规模**——「训练抽样、评估全量测试集」是合法且常见的做法，
   但必须在论文中明确披露抽样比例；且**所有对比模型必须用完全相同的抽样与划分**，否则对比无效。
2. 超参搜索（`configs/experiment.yaml` 的 HPO 段，约 60 组）**只在开发档做**，
   最终只把最优配置在全量上重训一遍，否则 77 h 的量级会翻好几倍。
3. 三种子（`seeds: [42, 2024, 2025]`）建议**只对最终模型与关键基线**做，
   全矩阵三种子在当前算力下不现实。

---

## 六、待决策（不决定就无法开工）

| # | 决策项 | 选项 | 建议 |
|---|---|---|---|
| D1 | **GPU 路线** | A 更新驱动+新 CUDA torch ／ B cu118 降版 ／ C 纯 CPU | **A**（性能与后续维护最好）；若不便动驱动则 B（单开新 env） |
| D2 | **训练档位** | 开发档抽样比例（5% / 10% / 20%） | **先 5% 跑通链路，再按实测吞吐决定正式档** |
| D3 | **正式实验规模** | 全量 E1+E2+E3 ／ 只全量跑最终模型、基线用开发档 | 视 D1+D2 实测结果再定，**不以文档预估的 77 h 为排期依据** |
| D4 | 是否现在装 `faiss` | 装 / 不装 | 阶段 2 召回可先用 numpy 精确内积（15,687 物品规模完全够），**faiss 推迟** |

---

## 七、风险清单

| 风险 | 影响 | 缓解 |
|---|---|---|
| 驱动 457.85 过旧，CUDA 轮子装不上 | 阶段 2 直接停摆 | D1 决策；B 路线已确认可行 |
| 6 GB 显存 | 批量受限，可能需 `batch_size` 从 256 降到 128 | SASRec 很小（hidden 64），预计不成为瓶颈；实测确认 |
| 1.09 亿样本/epoch | 训练与网格搜索时间爆炸 | 开发档 / 正式档双档位 + HPO 只在开发档 |
| 无时间戳导致的时序口径 | 论文「局限性」必须写明 | 已在 `evaluation-plan.md` 2.3 记录，阶段 2 沿用同一口径 |
| 合规题材仍在池内 | 答辩可能被问 | 已在 `evaluation-plan.md` 第十节如实披露 + 评估期屏蔽方案 |
| 冷启动为模拟口径 | 结论需限定表述 | 已记录 holdout 口径与 14.07% 训练信号剥离代价 |
| 指标表格可能被"先填个好看的数" | 学术不端 | `result-analysis.md` 填报规范：**无出处不许填** |

---

## 八、相关文档

| 想了解 | 看 |
|---|---|
| 实验口径、数据集处理细节 | [evaluation-plan.md](evaluation-plan.md) |
| 各模块增益如何验证 | [ablation-study.md](ablation-study.md) |
| 指标汇总与图表模板 | [result-analysis.md](result-analysis.md) |
| 算法代码落在哪、接口契约 | [project-structure.md](project-structure.md) §三 |
| 编码与提交规范、路径硬约束 | [dev-conventions.md](dev-conventions.md) |
| 配置项含义 | [config-guide.md](config-guide.md) |
| 原始需求 | [../rq.md](../rq.md) |
