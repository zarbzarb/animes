# 项目进度总览 · AniRec

> **快照时间**：2026-09-14 19:20 ｜ **最新完成**：**M3 系统开发主干贯通（能启动、能调用、能看见：`uvicorn server.main:app` → `/` 演示页 + `/docs`；冒烟 43/43、单测 643 例全绿）** ｜ **下一里程碑**：M3.6 收尾 ｜ **进行中**：阶段 2 与阶段 3 **并行**（GPU 空档跑实验，人工时间推 M3）
>
> 🔀 **2026-09-14 决策 D6：阶段 2 与阶段 3 并行推进。**
> 阶段 2 剩余实验需 ≈ 42 h GPU（1.8 天）且**不占用人力**，而阶段 3（全栈系统）**不依赖 GPU**。
> 故 GPU 排队跑实验、同一时间开发系统；阶段 2 的待跑清单、命令与结果回写位置统一记在
> **[gpu-queue.md](gpu-queue.md)**（防遗忘的唯一入口，勿在别处另记一份）。
> **当前提交**：以 `git log -1 --oneline` 为准（本文档不写死哈希，避免每次提交后过期）
>
> 本文件只回答三个问题：**做到哪了**（§1–2）／**环境撑不撑得住**（§3）／**下一步做什么**（§4–5）。
> 每完成一个里程碑请更新本文件；文档索引同步维护在 [README.md](README.md)。

---

## 一、总进度

| 阶段 | 权重 | 状态 | 交付物 / 卡点 |
|---|---|---|---|
| 阶段 0　项目初始化 | — | ✅ 完成 | 目录骨架 + 14 份设计文档 + GitHub 仓库（`zarbzarb/animes`） |
| **阶段 1　数据预处理与数据集构建** | 15% | ✅ **完成并通过验收** | 7 个脚本 + 全部数据产物 + 8 条实测口径，验收 31/31 |
| 阶段 2　核心算法与对比实验 | 35% | 🔵 **进行中 85%** | **M2.6 两条融合路径均完成**：后验加权无增益（7:3≈纯行为、冷启动 5:5 更差）；候选侧 concat 亦无稳定增益（SASRec −1.1pp / 多兴趣 +0.6pp），但**同权重消融证明内容通路净贡献 +0.053 ndcg@10**（问题在 concat 重学了物品空间）；**零初始化残差（add）模式首次跑正 0.7895 > 基线 0.7871**。⚠️ 主协议无真冷启动物品，内容收益预期在 E3 验证；**内容融合默认口径已锁定为 `add`**（configs 五处对齐 + 配置一致性单测锁死，dev 档 30 epoch 复核进行中）。**M2.7 对比基线实现完成**：热度 / ItemCF 零训练即拿到 main-test **hr@10 0.8996 / 0.9597**，ndcg@10 **0.6645 / 0.7514** —— ⚠️ 本协议下 ItemCF 极强，论文增益必须相对它报告；GRU4Rec 待 GPU。下一步 M2.8 实验编排 |
| 阶段 3　全栈系统与联调 | 35% | ⬜ 未开始 | 依赖阶段 2 的模型权重与嵌入矩阵 |
| 阶段 4　测试 | 15% | ⬜ 未开始 | 依赖阶段 3 的可运行系统 |

```mermaid
flowchart LR
    S0["阶段 0<br/>项目初始化"] --> S1["阶段 1<br/>数据预处理<br/>✅ 15%"]
    S1 --> S2["阶段 2<br/>核心算法与实验<br/>🔵 进行中 35%"]
    S2 --> S3["阶段 3<br/>全栈系统与联调<br/>⬜ 35%"]
    S3 --> S4["阶段 4<br/>测试<br/>⬜ 15%"]
    S2 -.模型权重/嵌入.-> S3
    S3 -.接口/页面.-> S4
```

---

## 二、阶段一交付（已完成，可随时独立复验）

### 2.1 脚本（`scripts/`，共 13 个）

| 脚本 | 角色 | 说明 |
|---|---|---|
| `preprocess.py` | 主链路 1-A | 清洗 `animes.csv` → 12 类题材；全量扫 `ratings.npy` 统计；按 `dataset.pkl` 顺序重建留一法切分；与官方切分交叉校验 |
| `build_content_vectors.py` | 主链路 1-B | 标题+12 类题材+细分标签 → DistilBERT `[CLS]` → PCA 降到 512 维 → L2 归一化 |
| `build_cold_start_subset.py` | 主链路 1-D | holdout 口径（year≥2021）模拟新番冷启动 + 新番专用负样本池 |
| `download_content_encoder.py` | 辅助 | 离线拉取 DistilBERT 权重（517MB，`.gitignore` 排除） |
| `diagnose_dataset_source.py` | 取证器 | 可复现生成「为什么用 npy / 为什么阈值 7」的证据 JSON |
| `verify_stage1.py` | **阶段一验收器** | 不读任何报告，回到原始文件独立重算 31 项结论并打 PASS/FAIL；退出码可接 CI |
| `run_stage1.py` | 编排器 | 把 6 个脚本的依赖顺序固化成代码（前置检查 + 产物检查） |
| `bench_throughput.py` | **吞吐基准**（阶段二） | 精确重算各档位规模 + 实测 train/infer 吞吐 + AMP 对比 + 显存上限 + 外推全矩阵耗时 |
| `verify_eval_stack.py` | **评估栈验收器**（阶段二） | 32 项独立重算：窗口数对账、因果性、负采样不变量、评估链路无偏性 |
| `check_metrics_mutation.py` | **变异测试**（阶段二） | 注入 6 类典型改错，要求单测逐一变红；先断言基线为绿再开始 |
| `train.py` | **训练入口**（阶段二） | 承担全部 IO 与用户抽样口径；`--model sasrec\|multi_interest`、`--content-fusion`、`--content-mode concat\|add` |
| `eval_content_fusion.py` | **融合评估**（阶段二 M2.6） | 复用已训练基座：val 扫融合权重 → test 确认 → 冷启动 holdout 快检 |
| `diagnose_popularity_bias.py` | **诊断器**（阶段二 M2.4） | 复现「随机 / 仅热度 / 模型」三组对照，量化热度先验对指标的支配 |
| `run_baselines.py` | **基线评测**（阶段二 M2.7） | 三个对比基线（热度 / ItemCF / GRU4Rec）统一入口；复用主链路的抽样、负样本、评估器与早停 |

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
| **GPU** | **NVIDIA RTX 2060 / 6 GB**（Turing sm_75，桌面版） | **与 `evaluation-plan.md` 3.1 原写的 RTX 3090 24GB 不符，已更正** |
| GPU 驱动 | **616.92**（Windows 显示版本 `32.0.16.1692`，2026-09-04）✅ **已验证可用** | 2026-09-12 由 457.85（CUDA 11.1）升级；`nvlddmkm` RUNNING、设备 ErrCode=0、`cuda.is_available()==True` |
| 磁盘 | C: 余 92.9 GB / F: 余 66.3 GB（SSD `WDC PC SN730`，健康） | 单模型 checkpoint < 10 MB，够用 |
| Python | 3.11.16（conda env `py3_11`） | `E:\tools\anaconda\envs\py3_11\python.exe` |

### 3.2 已装依赖

| 已装 | 版本 | | 缺失（阶段 2/3 需要） |
|---|---|---|---|
| torch | **2.14.0+cu130** ✅ | | `faiss`（阶段 2 后期检索，可选，可推迟） |
| numpy / pandas | 2.4.6 / 2.3.3 | | `fastapi` / `uvicorn`（阶段 3） |
| transformers | 4.57.1 | | `sqlalchemy` / `pymysql`（阶段 3） |
| scikit-learn / scipy | 1.9.0 / 1.17.1 | | `redis` / `apscheduler`（阶段 3） |
| matplotlib / seaborn | 3.11.0 / 0.13.2 | | — |
| pytest | 9.1.1（用 `python -m pytest`） | | — |

**GPU 可用性验证（2026-09-12 17:27 实测，M2.0）**：

| 检查项 | 实测值 |
|---|---|
| `torch.cuda.is_available()` | **True** |
| 设备 / 架构 / 显存 / SM 数 | `NVIDIA GeForce RTX 2060` / `sm_75` / 6.0 GB / 30 |
| CUDA 运行时 / cuDNN | 13.0 / 92400 |
| fp32 矩阵乘实测 | 4.6 TFLOPS |
| **fp16 矩阵乘实测** | **12.8 TFLOPS**（Tensor Core 已启用，AMP 可用） |
| AMP 前向 + 反向 | 通过（峰值显存 0.32 GB） |

**torch 安装口径（勿走默认源）**：PyPI 默认源上的 torch 是 `+cpu` 轮子，装完不报错、
但 `torch.cuda.is_available()` 恒为 `False`（本项目踩过一次）。必须显式指定索引：

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.0+cu130
```

更换后回归：`python -m pytest tests/ -q` → **115 passed**（指标口径未受影响）。
旧环境快照留档 `logs/pip_freeze_before_cuda.txt`（149 行，可回滚）。

### 3.3 ✅ 已解决的阻塞项：驱动文件损坏（2026-09-12 定位并修复）

**症状**：驱动 616.92 装完后 `torch.cuda.is_available()` 仍为 `False`，
且 `nvidia-smi` 报"权限不足"（**误导性提示**，沙箱下每次都是这句）。

**根因**：安装写盘时 **`nvlddmkm.sys` 这一个 109 MB 文件被写坏**（签名 HashMismatch），
Windows 因哈希不符拒绝加载 → 设备 Code 52 → CUDA 报告无设备。
同目录其余文件完好、安装包签名有效 ⇒ **不是下载损坏，无需重下**。

| 检查项 | 修复前 | 修复后 |
|---|---|---|
| `cuInit(0)`（ctypes 直调） | 100 = `CUDA_ERROR_NO_DEVICE` | 0 = 成功 |
| 设备管理器 | `Status=Error`、**Code 52** | `Status=OK`、**ErrCode=0** |
| 内核服务 `nvlddmkm` | **STOPPED**（退出码 1077） | **RUNNING** |
| Kernel-PnP 事件 219 | `0xC0000428` = `STATUS_INVALID_IMAGE_HASH` | — |
| `nvlddmkm.sys` 数字签名 | **HashMismatch** ← 根因 | `Valid` |
| 同目录其余 17 个大文件 | 全部 `Valid` | — |
| 安装包 `616.92-...whql.exe` | 签名 `Valid`（NVIDIA Corporation） | — |
| 磁盘 | C: 余 92.9 GB、SSD 健康 | — |

**修复过程**（用户操作）：设备管理器卸载设备并删除驱动 → 重启 → 管理员重跑安装包
（自定义安装 + 勾选执行清洁安装）→ 重启 → `nvidia-smi` 正常输出。

> 排查经验（值得记住）：
> 1. `nvidia-smi` 的"权限不足"会掩盖真实原因（Code 52），必须交叉验证
>    **设备状态、内核服务状态、驱动文件签名**三者。
> 2. `Get-AuthenticodeSignature` 对单个驱动文件返回 `HashMismatch` 是"文件被写坏"的
>    决定性证据；若同目录其他大文件都 `Valid`，即可排除整个包与下载环节的问题。
> 3. 本机 **HVCI（内存完整性）已开启**、VBS 开启、`VulnerableDriverBlocklistEnable=1`，
>    排查驱动类问题时这些状态会影响判断。

### 3.4 需要随本次进度一起更正/留意的文档不实之处

| 位置 | 现状 | 处理 |
|---|---|---|
| `evaluation-plan.md` 3.1 硬件表 | 原写「RTX 3090 24GB」 | ✅ 已改为本机实测（RTX 2060 6GB / Windows / 驱动 616.92） |
| `evaluation-plan.md` 6.5 耗时预估 | 原写「合计 ~77 h」基于 3090 + 全量网格 | ✅ **已用实测替换**：整个矩阵保守口径 **1.66 天**，见 §5.0 |
| `evaluation-plan.md` 3.2 环境固化 | 依赖 `requirements.txt` | ✅ 已生成 `requirements.lock`（152 包，记录真实版本号，避开 conda 的本地路径问题） |

---

## 四、阶段二拆解（下一步做什么）

阶段 2 的验收标准是：**E1~E4 四组实验的真实指标落进 `docs/result-analysis.md`，且能被一键复现。**

| # | 里程碑 | 做什么 | 产物 | 验收标准 |
|---|---|---|---|---|
| **M2.0** ✅ | 训练环境就绪 | 装 GPU 版 torch；写吞吐基准脚本，实测 s/step 并外推单 epoch 耗时 | `scripts/bench_throughput.py`、`requirements.lock` | ✅ **已完成**：GPU 可用（驱动 616.92 / torch 2.14.0+cu130）；实测最优 **bs2048+AMP = 40,083 samples/s**；全矩阵保守口径 **1.66 天**（见 §5.0） |
| **M2.1** ✅ | 指标唯一实现 | 按 `evaluation-plan.md` 5.1 实现 HR@K / NDCG@K / Recall@K / MRR，含并列稳定排序 | `models/eval/metrics.py`、`tests/test_metrics/`、`scripts/check_metrics_mutation.py` | ✅ **已完成**：单测 88/88 通过；变异测试 6/6 被捕获 |
| **M2.2** ✅ | 评估器 | 1 正 + 100 负候选，确定性负采样；支持全量 / 分题材 / 冷启动三种切分 | `models/eval/evaluator.py`、`models/data/negatives.py`、`tests/test_eval/` | ✅ **已完成**：单测 22/22；验收 C/D/E 组全过；随机打分器 HR@10 实测 0.1023（期望 0.0990±0.0064，评估链路无偏） |
| **M2.3** ✅ | 滑动窗口 Dataset | 按 `max_len=50` 在线生成样本，不落盘（输入上限 48） | `models/sasrec/dataset.py`、`tests/test_sasrec/` | ✅ **已完成**：单测 29/29；验收 B 组在真实数据（最长序列 8,917）上通过因果性验证 |
| **M2.4** ✅ | SASRec 基座 | 2 层 2 头 hidden 64 dropout 0.2 + BCE 负采样 + 早停（看 val NDCG@10, patience=10） | `models/sasrec/{model,config,train}.py`、`models/checkpoint/`、`scripts/train.py` | ✅ **已完成**：smoke 档 loss 0.66→0.20 稳定下降，val ndcg@10 **0.716**（随机基线 0.045）；复评与记录**逐位一致**；端到端 **17.6 s/epoch**。⚠️ **附带发现热度先验主导指标**（见 §4.2） |
| **M2.5** ✅ | 多兴趣胶囊 | K=4 动态路由（3 次迭代）接在 SASRec 输出后；记录胶囊两两余弦相似度检测塌缩 | `models/multi_interest/{config,capsule,model}.py` | ✅ **已完成**：全仓 318 例全绿（新增 42）；debug 档 8 epoch 固定预算 val ndcg@10 **0.7624**（基线同预算 0.7871，路由未收敛属预期）；相似度 0.699（阈值 <0.7 边缘达标，无死胶囊）；新增参数恰好 +16,384（=路由 W）。⚠️ 发现并修复 **squash 与 0.02² 初始化的量级冲突**（见 §4.1 M2.5） |
| **M2.6** ✅ | 内容融合（排序侧） | 排序侧 7:3 加权 + 冷启动 5:5 自动切权 | `models/content_encoder/fusion.py`、`scripts/eval_content_fusion.py` | ✅ `fusion_score()` 契约达成；实测**后验融合无增益**（见 §4.1 M2.6），E2 按此口径如实报告 |
| **M2.6b** ✅ | 内容融合（候选侧） | 内容向量拼进物品表示、端到端训练；`concat` 与零初始化残差 `add` 两臂 | `models/content_encoder/model.py`、`models/sasrec/model.py` 的 `repr_provider` | ✅ 实现 + 39 例单测（全仓 375 例全绿）；`add` 臂 **0.7895 > 基线 0.7871**（单种子 8 epoch）；**默认口径锁定 `add`**（决策 D5，dev 档 30 epoch 复核中）；concat 无稳定增益（见 §4.1 M2.6b） |
| **M2.7** 🔵 | 对比基线 | ItemCF（共现 + 余弦，TopK 200 邻居）、GRU4Rec（隐藏 64，1 层）、热度 | `models/baselines/`、`scripts/run_baselines.py` | 🔵 **实现完成 + 热度/ItemCF 已出正式数字**（main/test：热度 0.8996/0.6645、ItemCF **0.9597/0.7514**，零训练零参数）；GRU4Rec 走 `fit()` 同早停，smoke 已跑通，dev/main 待 GPU（见 §4.1 M2.7） |
| **M2.8** | 实验编排 | `run_experiments.py` 按 `configs/experiment.yaml` 串起 E1~E4，落盘 `experiments/{id}/`，回写 `result-analysis.md` | `scripts/run_experiments.py`、`scripts/plot_results.py` | 一次命令跑完四组；输出目录结构与 `evaluation-plan.md` 8.2 一致 |
| **M2.9** | 填表与作图 | 真实指标填入 `result-analysis.md` / `ablation-study.md`，生成 4 张图 | 指标表 + `figures/*.png` | **每个数字都有出处（实验目录名）**，无出处不许填 |

推荐执行顺序即上表自上而下。**M2.0–M2.6b 已完成；M2.7 实现完成、热度与 ItemCF 已出数字，GRU4Rec 待 GPU**（排在 M2.6b 的 dev 档复核之后）；环境已无阻塞。

### 4.1 已完成里程碑记录

#### M2.1 指标唯一实现 ✅（2026-09-12）

| 项 | 内容 |
|---|---|
| 交付 | `models/eval/metrics.py`（`hr_at_k` / `ndcg_at_k` / `recall_at_k` / `mrr` / `positive_rank` / `ranking_metrics`）、`models/__init__.py`、`models/eval/__init__.py`、`pytest.ini` |
| 测试 | `tests/test_metrics/` 3 个文件 88 个用例，覆盖手工算例、**暴力通用公式对拍**（rank 1~20 × K∈{5,10}）、并列口径、批量独立性、错误输入、空集 |
| 质量证据 | `scripts/check_metrics_mutation.py` —— 注入 6 类典型改错（HR 边界、NDCG 分母、去掉 `stable=True`、排名少 +1、Recall micro→macro、批量全局排序），**6/6 被测试抓住** |
| 复现命令 | `python -m pytest tests/ -v`（约 2 s）、`python scripts/check_metrics_mutation.py`（约 15 s） |

**两条实测发现（已写入 `evaluation-plan.md` 5.2）：**

1. **必须显式写 `stable=True`，不能依赖默认行为。**
   实测 torch 2.14.0：n ≤ 32 且并列少时非稳定排序"碰巧"按列序输出，
   **n ≥ 64 时才暴露**（101 候选的真实规模必然满足，实测 30/30 个随机种子都出现差异）。
   漏掉它会让 NDCG 随实现/线程数/设备变化而**不可复现**，且指标数字上看不出来。
   回归测试用纯 Python 参考实现 `sorted(key=(-score, col))` 在 101 规模对拍锁定。

2. **并列时对正样本取乐观口径**（正样本放第 0 列 + 稳定排序 ⇒ 完全并列时排第 1）。
   这是**口径选择**而非实现细节，需在论文「评价协议」写一句；换规则会得到不同的 NDCG。

> 顺带记录的流程教训：第一版测试**漏掉了 `stable=True` 的断言**（改成 `False` 后 88 个测试仍全绿），
> 是变异测试发现的；而在修这个漏洞时又出现「基线本身在失败」，导致那一轮变异结论**全部作废**。
> 因此 `check_metrics_mutation.py` 现在会**先断言基线为绿**再开始变异。

#### M2.2 + M2.3 评估栈 ✅（2026-09-12）

| 项 | 内容 |
|---|---|
| 交付 | `models/data/negatives.py`（负采样唯一实现）、`models/sasrec/dataset.py`（滑动窗口 + 评估输入构造）、`models/eval/evaluator.py`（分批前向 + 剔除泄漏 + `metrics.json`） |
| 测试 | `tests/test_data/test_negatives.py` 18 例、`tests/test_sasrec/test_dataset.py` 29 例、`tests/test_eval/test_evaluator.py` 22 例；**全仓 184 例全绿**（原 115 + 69） |
| 验收 | `scripts/verify_eval_stack.py` 五组 **32 项**，`32/32 通过`，约 21 秒；结果落 `data/processed/eval_stack_acceptance.json` |
| 复现 | `python -m pytest tests/ -q`（约 3 s）、`python scripts/verify_eval_stack.py`（约 21 s） |

**五条实测发现（均已写入文档）：**

1. **答案泄漏率首次量化**：抽样 2 万用户，按完整历史判断 test 泄漏 **1.665%**、val **1.995%**；
   按模型可见输入（截断到 48）判断为 **0.95%**。成因是**用户重复消费同一部番**
   （例：用户行 693545 的 test 目标在训练历史里已出现 3 次）。
   这是数据固有性质而非 bug，但 `evaluation-plan.md` 5.2 早就规定"正样本已出现则剔除并
   记录数量"——**此前只有规定、没有数字**。`main` 档实际评估样本约 **129,400** 而非 130,669。
2. **输入上限是 48 而非 50**：完整序列截断到 50 后，val/test 各占 1 位，故模型单次前向
   最多看 48 条历史。**窗口数（1.09 亿）不因截断而减少**——截断的是"每个样本能看多长
   的历史"，不是"每个用户贡献几个样本"。这两者极易混淆，`evaluation-plan.md` 2.3 已列表澄清。
3. **负采样 67 μs / 用户**：`dev` 档 0.9 秒、`main` 档 8.7 秒、全量 130 万用户 **87 秒**。
   故必须落盘缓存跨模型复用。实现上为每行派生独立随机流，做到
   **同 seed + 同用户 ⇒ 永远同一批负样本，与子集/顺序/分块方式全都无关**。
4. **训练集 DataLoader 必须 `shuffle=True`**：滑窗样本按用户行号连续排列，不打乱会导致
   一个 batch 内全是同一批用户的连续历史、梯度方向严重偏斜。早期文档写的
   `shuffle=False` 是针对"每用户一样本"的写法，已更正。
5. **验收器自身的坑**：第一版用 `(id * C) % P` 线性哈希冒充随机打分器，正负样本的 id
   分布差异变成系统性偏移，HR@10 稳定停在 0.0783（真值 0.0990）**且样本量再大也不收敛**。
   换成独立同分布的随机分数表后恢复到 0.1023。已在文档中标注该陷阱。

> 另外修掉一个真实设计缺陷：`SlidingWindowDataset` 的 `user_rows` 校验原先用
> `max(user_rows) < len(train_seqs)`，而 `seq_dataset.pkl` 里 `train` 是 **dict**
> （key = 用户行号），该判据会误报越界。现按 `Mapping` / 序列分情况校验。

#### M2.4 SASRec 基座 ✅（2026-09-12）

| 项 | 内容 |
|---|---|
| 交付 | `models/sasrec/config.py`（超参 dataclass + 校验 + 学习率调度）、`model.py`（因果自注意力网络）、`train.py`（BCE + AMP + warmup + 早停）、`models/checkpoint/{__init__,io}.py`（权重格式唯一实现）、`scripts/train.py`（训练入口，承担全部 IO） |
| 测试 | 新增 92 例（模型 26 / 训练 24 / 批量取数 13 / 训练负采样 18 / checkpoint 15 等），**全仓 276 例全绿**（原 184 + 92） |
| 冒烟验收 | smoke 档 3 epoch：**loss 0.66 → 0.20** 单调下降，val **ndcg@10 = 0.716**（随机基线 0.045）；最优权重复评与训练中记录 **差 0.00e+00**（逐位一致）；完整流程 72.6 s |
| 复现 | `python -m pytest tests/ -q`（约 6 s）、`python scripts/train.py --scale smoke`（约 73 s） |

**四条实测发现与修复（均已写入代码注释与文档）：**

1. ⚠️ **热度先验主导指标（最重要，影响论文论证）**。
   三组对照（smoke 档，1 正 + 100 均匀负）：

   | 打分器 | hr@10 | ndcg@10 |
   |---|---|---|
   | 随机 | 0.074 | 0.030 |
   | **仅按物品热度**（零信息） | **0.866** | **0.607** |
   | smoke 档模型 | 0.920 | 0.716 |

   模型只比"零信息的广度先验"高 **6% / 18%**。成因是结构性的：正样本必然是
   用户交互过（且交互数 ≥10 才进池）的物品，天然偏热门；负样本从全池均匀抽，
   而全池热度极度长尾（中位 288 vs 均值 6,687）。实测**正样本热度均值是负样本的 24.7 倍**。
   → **E1 必须包含 popularity 基线**（SASRec 原论文表 3 也有这一行），
   论文里只报 0.716 而不报热度基线属于选择性报告。
   诊断脚本：`scripts/diagnose_popularity_bias.py`。

2. **训练侧负采样不能用"环绕 +1"避让正样本**。该写法把偏差集中转嫁给 `target+1`，
   使其被抽中次数变成其他物品的**两倍**（实测构造 5 万条同 target 样本时：
   2036 次 vs 期望 1020）。真实数据下这点偏差小到看不出，但只要批内 target 集中
   （按物品分桶、冷启动里同一新番被大量用户当目标）就会成规模。现改为**重抽**，
   期望轮数 1/(1−1/n_items) 几乎总是一轮内解决。

3. **批量取数必须绕开 `DataLoader`**。三组实测（bs=1024）：`DataLoader(num_workers=0)`
   17.1 ms/batch、`DataLoader(num_workers=4)` **慢 15 倍**（Windows spawn 把整个
   序列数据集 pickle 到每个 worker）、本项目的 `WindowBatchIterator` **约 5 ms/batch**。
   端到端因此从 60 ms/step 降到 **39 ms/step（1.54x）**。

4. **收尾复评必须显式 `model.eval()`**。`fit()` 在每次评估后会调回 `model.train()`
   以便继续训练，若复评前不切回 eval，dropout 会重新打开，指标比真值低且每次不同
   （实测差 2.8e-3）。这是"指标差一点点、找不到原因"的典型来源。

5. **lr 重标定结论：batch 256→1024 后 `lr=0.001` 仍最优，无需上调**。
   debug 档（2% 用户 / 2,572 评估用户）8 epoch 固定预算、seed 42，扫描 5 组
   （val ndcg@10）：

   | lr | warmup | best ndcg@10 | loss@ep8 |
   |---|---|---|---|
   | 0.0005 | 5% | 0.7842 | 0.1346 |
   | **0.001** | 0 | **0.7871** | **0.1291** |
   | 0.002 | 5% | 0.7827 | 0.1294 |
   | 0.002 | 0 | 0.7807 | 0.1294 |
   | 0.004 | 0 | 0.7739 | 0.1363 |

   结论可信度有两点要注意：① 所有组到 ep8 仍在涨（未收敛），比较的是
   **同等预算下的排序**，不是最终性能；② 组间差 ≤1.3%，接近该评估集的噪声水平
   —— 这条扫描只回答「**batch 放大后不需要把 lr 调大**」这一个实际问题，
   正式最优仍由 evaluation-plan 4.2 的 HPO 网格在 dev 档确定。
   原计划在 dev 档做，实际改用 debug 档（4 组 × 30 epoch 在 dev 档约 7 h，
   debug 档排序结论足够且快 4 倍）。扫描日志：`logs/lr_sweep.log`、
   `logs/train_*lrsweep*.json`。

> 另外修掉一个真实 bug：`config_snapshot()` 里含 `input_cap` / `n_params` 这类
> **派生字段**，早期 `load_model_from_checkpoint` 直接 `SASRecConfig(**cfg_dict)`
> 展开，导致**任何** checkpoint 都加载失败（TypeError）。现改走
> `SASRecConfig.from_dict()` 只取 dataclass 认得字段，并加了回归测试。

#### M2.5 多兴趣胶囊 ✅（2026-09-12）

| 项 | 内容 |
|---|---|
| 交付 | `models/multi_interest/{config,capsule,model}.py`：MIND 式动态路由（K=4、3 次迭代、softmax 在胶囊维、PAD 票数置零），复用 SASRec 编码主干；`scripts/train.py --model sasrec\|multi_interest` 切换；checkpoint 按 `meta["arch"]` 自动重建 |
| 测试 | 新增 42 例（路由 20 / 模型 22），**全仓 318 例全绿**（原 276 + 42） |
| 对照 | debug 档 8 epoch 固定预算、seed 42、同评估集：MI **ndcg@10 = 0.7624** vs SASRec 0.7871（−2.5pp）；loss 0.1525 vs 0.1291。⚠️ 这是**同预算快照不是结论**：两组到 ep8 都未收敛，MI 的路由需要更多轮次分化，E2 的正式对比在 dev 档跑满 30 epoch + 早停 |
| 塌缩诊断 | 兴趣两两余弦相似度 **0.699**（阈值 <0.7，边缘达标）；4 个胶囊全部在被使用（max 选中占比 10%/22%/61%/7%，无死胶囊），但主胶囊偏强，dev 档长预算下需复查 |
| 开销 | 87 s/epoch vs 基线 71 s（**+22%**，路由迭代 + 第 5 个 einsum 的代价）；参数 +16,384（恰好 = 64×4×64 路由 W） |
| 复现 | `python scripts/train.py --scale debug --model multi_interest --seed 42 --epochs 8 --no-final-eval --tag m2_5_mi`（约 11 分钟） |

**M2.5 两条实测发现：**

1. ⚠️ **squash 与 0.02² 初始化存在量级冲突（已修，勿回退）**。
   squash 把兴趣向量范数压进 (0,1]，而 item_emb 按论文口径 N(0,0.02²)
   初始化（H=64 范数仅 ~0.16），点积量级 |score|≤0.008；基线 user_repr 是
   LayerNorm 输出（范数≈√H），logits ~0.3。不修正的话 BCE 在线性区学习极慢
   （玩具任务 10 epoch 才 0.693→0.566），**同样 lr/epochs 下多兴趣模型会系统性
   欠收敛，E2 会把"训练预算不足"误判成"多兴趣没用"**。修复：打分时
   `interests × sqrt(H)`（与基座 scale_emb 同一条惯例；MIND 原实现无此步
   是因为它的嵌入初始化范数≈1）。正数缩放不改评估排序，只改训练 loss 动态。
   详见 `models/multi_interest/README.md`。

2. **softmax 维度是 MIND 与部分开源实现的分歧点**：论文口径 softmax 在
   **胶囊维 j**（每个位置的票在 K 个兴趣间分配），不是位置维 i。capsule.py
   docstring 已写死该口径，改则与 MIND 不可比。

#### M2.6 排序侧内容融合 🔵（2026-09-12，后验融合部分完成并出结论）

**实现**（`models/content_encoder/fusion.py` + `scripts/eval_content_fusion.py`）：
- `load_content_matrix`：`content_vec_512.npy`（第 i−1 行 = 物品 idx i，阶段一定稿）
  装载为 (n_items+1, D)，PAD 行恒零，防御性 L2 归一；
- `ContentScorer`：用户内容画像 = 有效历史内容向量均值（L2 归一），与候选算余弦；
  空历史画像为零向量 → 内容分中性，不 NaN；
- `fusion_score(behavior, content, n_inter, ...)`：与 project-structure.md 契约一致，
  `n_inter < 10` 自动切 5:5，逐样本独立选权；纯函数无 IO；
- `ContentFusedModel`：包任意满足 score 契约的基座，对外仍是同一契约。

**关键设计决定：融合只发生在打分/评估时，训练只训行为基座**（BCE 只看行为分）。
理由：① checkpoint 完全复用，不同融合权重不用重训；② 行为基座收敛动态与
E2_1 纯基线完全一致，消融差异可完全归因于融合；③ 内容向量是离线 PCA 产物
无可学参数。

**实测结论（debug 档，复用 lr=0.001 最优 checkpoint，2,572 val / 2,581 test 样本）**：

| 配置 | val ndcg@10 | 说明 |
|---|---|---|
| 纯行为 w=0 | **0.7878** | ≈ 训练日志 0.7871（自检有效，差 7e-4 来自 AMP/数值序） |
| 7:3 | 0.7874 | 无增益（噪声级） |
| 5:5 | 0.7845 | 更差 |
| 纯内容 w=1 | 0.1739 | 有真信号（随机 0.0990）但远弱于行为通路 |

冷启动 holdout（新番 2,000 样本，负样本池限定新番）：7:3 **0.6414** vs 5:5 0.6350
vs 纯内容 0.1368 —— **5:5 并不更好**。

**解读（论文须如实写）**：后验打分融合在当前协议下无增益，与 §4.2 热度先验发现
互相印证 —— 行为模型 + 均匀负样本协议下，"历史题材均值"级别的内容信号提不出
增量信息。**这否定的只是"后验融合"，不是内容通路本身**。

#### M2.6b 候选侧内容融合 ✅（2026-09-14）

**实现**（`models/content_encoder/model.py` + `SASRec.repr_provider`）：

* `SASRec` 增加可选 `repr_provider`：`table = provider(item_emb.weight)`，
  **输入嵌入与输出打分都经 `item_table()` 取表**（保证两侧一致）；
  不传时为 None，行为与 M2.4 逐位相同（回归测试锁死）。
* `ItemContentFusion` 两臂：
  * `concat`：`Linear([emb, content])`，参数 +36,928（= (64+512)·64+64）；
  * `add`：`emb + Linear(content)`，**零初始化**（严格等价基线起点），参数 +32,832。
* `build_model` 工厂：训练与 checkpoint 重建**共用同一条装配路径**；
  `meta["content_fusion"]` 记录 mode/维度/参数/file，缺文件报错而非静默降级。
* 新增 32 例测试；全仓 **368 例全绿**。

**实测（debug 档 8 epoch 固定预算、seed 42、同一评估子集 2,572 样本）**：

| 配置 | best ndcg@10 | vs 同 arch 无内容 | 参数量 |
|---|---|---|---|
| SASRec 纯行为（M2.4） | 0.7871 | — | 1,107,328 |
| SASRec + concat | 0.7760 | −0.0111 | 1,144,256 |
| **SASRec + add** | **0.7895** | **+0.0024** | 1,140,160 |
| 多兴趣（M2.5） | 0.7624 | — | 1,123,712 |
| 多兴趣 + concat | 0.7680 | +0.0056 | 1,160,640 |
| 多兴趣 + add | 0.7717 | **+0.0093** | 1,156,544 |

**同权重内容消融**（置零内容侧权重后重评同一 checkpoint，量化"内容值多少分"）：
concat +0.0531、SASRec+add +0.0350、多兴趣+add +0.0451 —— 三者都显著为正，
**内容通路确实被用上了**。

**三条结论**：

1. **concat 的问题在实现方式，不在内容**：训练后融合表与原嵌入余弦仅 0.0097，
   即 `Linear` 把 tie-embedding 已学到的物品空间**重学了一遍**，白付优化代价；
   而内容本身在这份权重里贡献了 +0.053 ndcg@10 与 ~30% 的表示范数。
2. **残差（零初始化）口径在两种架构上都为正增益**（+0.0024 / +0.0093），
   且两臂的消融方向一致 ⇒ "内容有增量"方向可信；**但幅度很小**
   （同评估路径 0.7927 vs 基线 0.7878），单种子、8 epoch 未收敛，
   正式结论必须等 dev（30 epoch）+ 多种子。
3. **主协议没有真正的冷启动物品**（池内每物品训练集至少 6 次交互），
   内容融合的预期价值在 E3（新番训练集不可见）；这是 H4 的验证出口，
   也是下一步 E3 必须单独训练（剥离后）的原因。

**过程中的两个真 bug（都加了回归测试）**：

* 裸张量索引取嵌入会丢掉 `padding_idx` 的「PAD 行不接收梯度」保障 →
  必须用 `F.embedding(..., padding_idx=0)`；
* 多兴趣模型的 provider 挂在 `backbone` 上，训练脚本按 `model.repr_provider`
  取时 `AttributeError` → 顶层统一暴露该属性。

#### 内容融合口径锁定 ✅（2026-09-14，决策 D5）

**决定**：候选侧内容融合的**默认口径 = `add`（零初始化残差）**，`concat` 降级为**对照臂**。

**依据**：上表 debug 档实测 —— `concat` 在 SASRec 上 −1.1pp、在多兴趣上只 +0.6pp（方向不一致）；
`add` 在两架构上都转正（+0.0024 / +0.0093），同权重消融方向也一致。
`concat` 的失败可归因到**实现方式**（把物品空间重学一遍），不是内容没信号。

**落点（五处，任何一处不一致都会"静默换口径"）**：

| 位置 | 默认 | 说明 |
|---|---|---|
| `ItemContentFusion(mode=...)` | `add` | 类默认 |
| `build_model(content_mode=...)` | `add` | 工厂默认（训练与 checkpoint 重建共用同一路径） |
| `models/checkpoint/io.py` 兜底值 | `add` | 防手工构造 meta 时错配 |
| `configs/model.yaml` → `content_fusion.mode` | `add` | 训练脚本读它 |
| `configs/experiment.yaml` → E2_3 / E2_4 | `add` | 消融组口径 |

新增 `TestDefaultModeConfigConsistency`（3 例，直接断言代码默认 + 两个 yaml 的默认）：
口径一旦漂回 `concat` 立刻变红。这类错配的特点是**权重形状一致、指标也算得出来**，
肉眼看不出来，只能靠测试挡。
顺带把 E1 的开关名从早期占位的 `use_multi_interest / use_content_fusion`
对齐为已落地的 `arch / content_fusion`（旧名会被 `from_dict` **静默忽略**）。

**复核状态**：dev 档 30 epoch（`eval_user_ratio=0.20` ≈ 13,067 评估用户）四组串行：
`sasrec` 纯行为 → `sasrec+add` → `multi_interest` 纯行为 → `multi_interest+add`，
`--tag m26b_dev_*`，日志 `logs/dev_m26b_*.log`、报告 `logs/train_multi_interest_content_v1_m26b_dev_*_dev_seed42.json`。
⚠️ **若 dev 档 `add` 不再为正，则回退默认值并同步本文档** ——
当前默认值的依据是 8 epoch 单种子，属"方向可信、幅度不可信"。

#### M2.7 对比基线 🔵（2026-09-14，实现完成 + 2/3 已出正式数字）

**实现**：`models/baselines/`（新包）+ `scripts/run_baselines.py`（入口）。

| 基线 | 需要训练 | 参数量 | 状态 |
|---|---|---|---|
| `popularity.py` 热度（零信息） | 否 | 0 | ✅ 三档全跑（smoke/dev/main） |
| `itemcf.py` 协同过滤（共现+余弦，TopK 200） | 否 | 0 | ✅ 三档全跑，val+test |
| `gru4rec.py` 经典序列模型（H=64，1 层 GRU） | **是** | 1,033,088 | 🔵 smoke 档已跑通；dev/main 待 GPU（排在 M2.6b dev 复核之后） |

公平性控制全部落在**复用唯一实现**上，没有一处重写：

| 环节 | 复用的唯一实现 |
|---|---|
| 用户抽样 | `models/data/user_subset.py::resolve_scale_users`（**本次新抽出的公共函数**） |
| 输入序列 / 负样本 / 指标 / 前向串联 | `dataset.build_eval_inputs` → `negatives.sample_negatives` → `metrics` → `evaluator.evaluate` |
| 训练循环 + 早停 | `models/sasrec/train.py::fit`（GRU4Rec 直接用，连 patience 单位换算都共用） |

⚠️ **本次抽出了 `resolve_scale_users()` 并**真的把两个旧调用方迁了过去**：原先
「档位 → 训练/评估用户行号」这段逻辑在 `scripts/train.py` 与
`scripts/diagnose_popularity_bias.py` 里各有一份，M2.7 是第三个调用方。
三份拷贝意味着"某天有人只改了其中一处"——而这类不一致**不报错**，
只会让两个脚本报出不可比的指标。现在三处（含 `run_baselines.py`）都调同一个函数，
并用 `tests/test_data/test_user_subset.py::test_resolve_scale_users_matches_legacy_formula`
对**抽取前的内联公式**做逐位对拍（smoke/debug/dev/main 四档比例 × 三种
`eval_user_ratio`）。迁移后三脚本实跑一致：smoke 档均为「训练 6,533 / 评估 653」。

⚠️ 顺带修掉一个**预先存在的语法错误**：`scripts/diagnose_popularity_bias.py`
有一行 f-string 里嵌了未转义的双引号（`…越容易"蒙对"`），该脚本自那次编辑起
**一直无法执行**（`SyntaxError`）。因为它是"自查脚本"、不在主链路上，
所以直到 M2.7 要把抽样口径收敛过去时才发现。修复后实跑通过，
并与本文件的 train-only 热度口径互为交叉验证：
诊断脚本（全量 `n_positive`）smoke 0.8655 / 0.6073，
`run_baselines.py`（仅训练频次）0.8640 / 0.6030 —— 两组数字一致。

**已出的正式数字**（`main` 档 = 10% 用户，`seed 42`，1 正 100 负，
剔除答案泄漏 1,287/1,532 条；出处见 `logs/baseline_*_main_*.json`）：

| 基线 | 集 | n | hr@5 | hr@10 | ndcg@5 | ndcg@10 | mrr |
|---|---|---|---|---|---|---|---|
| popularity | val | 129,137 | 0.7257 | 0.8544 | 0.5561 | 0.5980 | 0.5256 |
| popularity | **test** | 129,382 | 0.7934 | **0.8996** | 0.6298 | **0.6645** | 0.5956 |
| itemcf | val | 129,137 | 0.8654 | 0.9479 | 0.6934 | 0.7205 | 0.6506 |
| itemcf | **test** | 129,382 | 0.8930 | **0.9597** | 0.7295 | **0.7514** | 0.6865 |
| gru4rec | val(smoke, 3ep) | 647 | 0.8497 | 0.8733 | 0.5906 | 0.6179 | 0.5439 |

**⚠️ 必须写进论文「局限性」的一条硬事实：本协议下 ItemCF 极强。**
一个**零训练、零参数**的共现模型在 test 集拿到 **hr@10 = 0.9597 /
ndcg@10 = 0.7514**，而仅按热度（同样零训练）就有 **0.8996 / 0.6645**。
成因是结构性的（M2.4 已定性、M2.7 定量）：正样本必然是「交互数 ≥ 10」的
热门物品（实测热度均值 17,381），负样本从全池均匀抽（均值 629），
**正/负热度倍数 27.6×**；再加上共现信号，1 正 100 均匀负的 Top-10 几乎是
"送分"。⇒ 论文里本文模型的增益**必须相对 ItemCF 报告**，
"相对随机基线 +5 倍"这种说法在本协议下没有信息量。

**两处实现决定（都写进了 README 与单测）**

1. **ItemCF 的并列口径假象**：若某用户的 101 个候选全为 0 分
   （历史与所有候选项无共同邻居），按 `metrics.positive_rank` 的并列约定
   （稳定排序 + 正样本在第 0 列）该样本**白拿 HR=1**。
   实测该占比 **0.00%**（main：2 / 129,137；dev/main 其余为 0），
   故这一次不构成影响，但脚本仍强制上报 `n_zero_rows` 并额外跑一次
   `itemcf_tb_pop` 对照，两个变体数字完全相同。
2. **热度只用训练集频次**（与 `diagnose_popularity_bias.py` 用
   `item_stats.parquet` 全量 `n_positive` 的口径**不同**）。
   全量口径会混进 val/test 交互，方向是让**基线更好看**——
   正好是最不容易被察觉的那个方向。两套口径互为交叉验证：
   smoke 档本实现 0.8640/0.6030 vs 诊断脚本 0.866/0.607，一致。

**⚠️ 踩坑记录：`pack_padded_sequence` 假设右填充，本项目是左填充。**
GRU4Rec 初版按常规写法用 `pack_padded_sequence(..., enforce_sorted=False)`
跳过 PAD，实测**取到的是每行最前面的 PAD**（它按 `lengths` 取 `data[:, :length]`）：

| 实现 | 「只喂 3 个有效位置」手算 | `pack_padded_sequence` |
|---|---|---|
| L=4（垫 1 个 PAD） | 基准 | 偏离 1.5e-2 |
| L=6（垫 3 个 PAD） | 与基准**逐位相同** | 偏离 1.7e-2，且两次互相不同 |

CPU 与 CUDA 结果一致 ⇒ **不是内核问题，是用法错误**；且它不报错、
指标也算得出来。改为「滚动对齐 + 掩码取末位」，并显式校验
「有效位置必须是每行的后缀」（不满足直接报错）。
回归测试 `test_repr_invariant_to_left_padding`。

**另一个上报 bug**：`ItemCFScorer` 的诊断量最初只记**最后一次** `score()`
调用，而 `collect_ranks` 会按 batch 反复调用 —— main 档据此报出
`全零行 0 / 2,161`（最后一个残缺 batch）而不是 `/ 129,137`，**少报 98%**。
改为跨调用累计 + 显式 `reset_stats()`，并加了「累计行数必须等于评估样本数」
的自检。

**验收**：全仓 **421 例全绿**（新增 41 例基线单测 + 5 例抽样口径对拍）；三档端到端跑通。


#### M3 系统开发 🔵（2026-09-14 起，与阶段 2 并行）

阶段 2 的 GPU 排期太长，改为**两个阶段并行**：GPU 空档跑实验，人工时间推进
M3 系统开发。待办队列见 [gpu-queue.md](gpu-queue.md)（防遗忘的唯一入口）。

**已完成**（截至 2026-09-14）

| 项 | 状态 |
|---|---|
| 工程骨架：`server/`（core / db / api / services / runtime）+ `agents/` 十个 Agent + `models/` | ✅ |
| 统一响应体：全部路由经 `EnvelopeRoute` 包成 `{code,message,data,meta,trace_id,elapsed_ms}` | ✅ 42 个路径 / 44 个操作 |
| 数据库与演示数据：14,136 番剧、29,265 题材关联、6 用户（5 普通 + 1 管理员）、221 追番记录 | ✅ `scripts/init_db.py` |
| A0 编排流水线：意图 → Stage 内并行 / Stage 间串行；总预算 800ms | ✅ |
| 端到端冒烟脚本 `scripts/smoke_api.py`（HTTP 层，不 mock） | ✅ **43/43** |
| 分层依赖静态检查 `scripts/check_imports.py`（R1–R5，含 `--self-test` 15 例） | ✅ |
| 演示页 `web/index.html`（单文件零依赖，`/` 直接可看） | ✅ |
| 单测：**643 例全绿**（M2 的 421 + M3 新增 222） | ✅ |

**本轮修掉的真实 bug**（每条都有回归测试，且做过变异检验确认测试**抓得住**）

1. **排序不变量被破坏**：`final_score` 导出的是 MMR 的**输入相关性**，
   而 `rank_no` 是 MMR 的**输出顺序** —— 实测 `#2 rel=0.742` 排在了
   `#3 rel=0.969` 前面（Fate/Zero 与其第二季几乎同题材，被重复惩罚压低，
   这是**正确**的 MMR 行为）。但两者必须一致：`rank_no` 要落库、要被 A9
   用于位置-CTR 分析。改为导出 **MMR 目标值**（贪心过程可证明单调不增），
   新增 `strategy.export_final_scores`。回归见 `tests/test_fusion/`。
2. **推荐缓存失效完全失效**：A4 写的键是 `rec:{uid}:{scene}[:{gid}]`，
   而失效逻辑删的是 `rec:{uid}` —— **一个真实键都删不中且不报错**，
   表现为"用户新增追番后推荐 24 小时内毫无变化"。改为
   `rec_scope_prefix(uid)` + `Cache.delete_prefix()`（Redis 侧用 `SCAN`
   分批删，不用阻塞性的 `KEYS`）。新增 `tests/test_agents/test_cache_keys.py`。
   > 两条 docstring 当时都写着"两侧同构、有测试断言一致"，而那个测试**并不存在** ——
   > 于是这次把被引用却不存在的测试真正补上了。
3. **`/chat/message` 对不带 `session_id` 的客户端返回 500**：
   `agent_bridge` 显式传 `session_id=None`，**覆盖**了 schema 里 `""` 的默认值。
   冒烟脚本因为显式传了 `""` 而一直没暴露。改为在输入契约处归一化 `None → ""`。
4. **`meta.cache_hit` 结构性地恒为 false**：A0 的产物是两层的
   （`{intent, result:{...}}`），bridge 从第一层取该字段，而 A0 从不设置它。
   客户端会以为"缓存从未生效"。现已透传（实测：首次 `False` → 再次 `True`，419ms → 110ms）。
5. **`/static/*` 静默 404**：`app.mount` 在 import 时判断目录存在，
   "服务先起、`web/` 后建"就永远不会挂载。改用 `check_dir=False`。
6. **`A3` 75% 超时降级**：预算 150ms 是按文档拍的，实测 p95=141ms 且 A2∥A3
   在 GIL 上并发顶牛 → 降到 300ms（实测 p95×2 安全系数），12 次连测降级率 0%。
7. **A1 永久返回半成品画像**：`user_profile` 里存在 `top_genres=[]` 的行
   （预热时 `anime_genre` 尚未建好），旧逻辑把它当权威结果永久返回 ——
   雷达图恒为全 0 且不报错。加 `_usable()` 守卫强制重算。

**待办**：M3.6 收尾（`tests/test_api/` 其余接口测试）→ M3.7 文档与部署说明；
阶段 2 剩余 GPU 实验按 [gpu-queue.md](gpu-queue.md) 补跑。


---

## 五、训练规模与时间预算（✅ M2.0 已实测校准）

阶段一产物给出的训练规模：

| 指标 | 值 |
|---|---|
| 训练样本数（滑动窗口，每 epoch） | **109,081,471**（约 1.09 亿）= **Σ `train_len`** ✅ 已用 `user_stats.parquet` 逐位复核 |
| 每用户训练序列长度 | 均值 83.5、**中位数 41**、最大 8,917；**截断到 50 后的有效平均长度 34.8** |
| 序列长度分布 | 仅 **44.8%** 的用户 `train_len ≥ 50`（会被截断到满长） |
| 用户 / 物品 | 1,306,691 / 15,687 |
| 测试用户 | 全量 1,306,691（每用户 101 候选） |
| 冷启动测试样本 | 209,912 |

> ✅ **M2.3 已落地：采用「统一填充到 `input_cap = 48`」**，不做按批动态填充。
> 理由：`SlidingWindowDataset` 是在线生成样本的，动态填充要在每个 batch 内重排长度，
> 会让「窗口号 ↔ (用户, t)」这份确定性索引变复杂，收益抵不上额外实现与验证成本。
> 实测端到端 39 ms/step 已满足排期（见 §5.0）。
> 若要进一步提速，**按批动态填充是首选方向**（理论可省约 27% 的序列编码算力）。

**结论：1.09 亿样本/epoch 在 CPU 上不可行**（量级上每 epoch 是小时~天级），必须用 GPU。

### 5.0 ✅ 实测吞吐与总耗时（2026-09-12，RTX 2060 6GB）

**两层口径，请分清：**

**① 纯计算吞吐**（M2.0，`bench_throughput.py` 的参考计算图）
配置：batch 2048 + AMP + 训练候选 2（1 正 1 负），
**train 40,083 samples/s、infer 98,088 samples/s**。
复现：`python scripts/bench_throughput.py --batch-sizes 512 1024 2048 --train-negatives 1`

| 档位 | 用户数 | 样本数/epoch | 纯计算 sec/epoch | 跑满 30 epoch |
|---|---|---|---|---|
| `smoke` | 6,533 | 535,046 | 13 s | 3 分钟 |
| `debug` | 26,134 | 2,180,315 | 54 s | 27 分钟 |
| `dev` | 65,335 | 5,442,340 | 136 s | 1.13 h |
| `main` | 130,669 | 10,927,444 | 273 s | 2.27 h |
| `full` | 1,306,691 | 109,081,471 | 2,721 s | 22.7 h |

**② 端到端实测**（M2.4，`scripts/train.py` 的真实训练循环，batch 1024）
**39 ms/step** —— 比纯计算慢约 35%，差额是取数、训练侧负采样与 CPU↔GPU 搬运。

| 档位 | step/epoch | 端到端 sec/epoch | 跑满 30 epoch |
|---|---|---|---|
| `smoke` | 522 | **17.6 s（实测）** | 9 分钟 |
| `debug` | 2,129 | 约 83 s | 42 分钟 |
| `dev` | 5,315 | 约 207 s | 1.7 h |
| `main` | 10,671 | 约 416 s | 3.5 h |
| `full` | 106,525 | 约 69 分钟 | 34.6 h |

**整个实验矩阵（约 26 次训练）**：纯计算口径 1.66 天 → **端到端口径 ≈ 2.5 天**。
口径是"每档都跑满 `max_epochs`"，实际有早停（patience=10），会明显更短。
**论文里报训练耗时请用端到端数字。**
早期按 RTX 3090 估的「~77 h」已废弃，不得再引用。

一条提速实测（M2.4 新增）：**批量取数必须绕开 `torch.utils.data.DataLoader`**。
`bs=1024` 下 `DataLoader(num_workers=0)` 是 17.1 ms/batch、
`num_workers=4` **慢 15 倍**（Windows spawn 把整个序列数据集 pickle 到每个 worker），
而项目自己的 `WindowBatchIterator` 只要约 **5 ms/batch**。
端到端因此从 60 ms/step 降到 39 ms/step（**1.54x**）。

四条实测发现（都会影响后续调参）：

1. **AMP 只在 batch ≥ 512 时才有收益**。bs=128/256 时 AMP 反而**变慢**
   （0.80x / 0.91x —— kernel 启动开销盖过 Tensor Core 的收益）；bs=1024 达 **1.48x**、
   bs=2048 达 **1.54x**。故 `model.yaml` 的 `batch_size` 由 256 提到 **1024**，
   且必须与 `amp: true` 搭配使用（该条已写进配置注释）。
2. **训练候选集大小不是主要开销**。训练侧 1 / 10 / 100 个负样本 →
   40,083 / 32,555 / 31,171 samples/s，**只差 26%**。主开销是**序列编码器**
   （因果自注意力，O(L²)），这也解释了为什么 MFU 只有 6.2%。
   推论：想再提速应走「缩短序列 / 按批动态填充」，而不是缩减候选集。
   > 顺带修正一处口径混淆：`neg_sample_num: 100` 是**评估**口径（1 正 + 100 负），
   > 训练侧按 SASRec 原论文取「每位置 1 个负样本」，两者是**两件事**。
   > 修正前基准脚本把评估口径误用到训练侧，会让训练耗时被高估。
   > 现已在 `model.yaml` 用 `train.negatives_per_position` 显式区分。
3. **显存不是瓶颈，但 batch 有硬上限**。bs2048 只用 2.36 GB / 6 GB；
   而 bs4096 触顶（5.46 GB）后性能**崩塌到 5,220 samples/s（慢 7 倍）**，
   属 WDDM 显存超限颠簸，不要使用。
4. **批量取数不要用 `torch.utils.data.DataLoader`**（M2.4 实测，见上表②）。
   单进程下 `default_collate` 要逐个样本建 tensor 再 stack，实测 6.5 ms/batch 的固定开销；
   多进程在 Windows 下更糟 —— spawn 会把 dataset 持有的整个序列数据集 pickle 到
   每个 worker，实测 4 workers 比单进程**慢 15 倍**。
   现改用 `models/sasrec/dataset.py` 的 `WindowBatchIterator`（约 5 ms/batch）。
   > 通用教训：**"标准做法"在进程启动成本高的平台上会失效**，必须实测而非照搬。

> ⚠️ 上面两张表的关系：① 是 GPU 上的纯计算，② 是本项目的端到端。
> **排期、预算与论文里的"训练耗时"请用 ②**。
> `bench_throughput.py` 目前仍跑参考计算图；M2.4 的真实模型落地后可以用
> `--mode model` 复测（优先级低 —— 端到端数字本身更权威）。

### 5.1 训练档位（✅ 已定 2026-09-12，配置见 `configs/scale.yaml`）

分档是**为了让毕设做得完**，不是可选优化。理由：按原设计，
E1~E4 + HPO（60 组）+ 补充实验，再乘 3 个随机种子，合计约 **230 次训练**，
总时长为「年」量级 —— 即便单次训练只需 1 小时也无法完成。

| 档位 | 用户比例 | 用户数 | **每 epoch 样本数** | **评估用户数** | epoch 上限 | 种子 | 用途 |
|---|---|---|---|---|---|---|---|
| `smoke` | 0.5% | 6,533 | 535,046 | 653 | 3 | 1 | 冒烟：确认代码没崩，分钟级出结果 |
| `debug` | 2% | 26,134 | 2,180,315 | 2,613 | 30 | 1 | 看趋势（指标涨了还是跌了） |
| `dev` | 5% | 65,335 | 5,442,340 | 13,067 | 30 | 1 | HPO 细搜、消融、基线对比 |
| `main` | 10% | 130,669 | 10,927,444 | 130,669 | 30 | 3 | E1 主表 + E2 消融的正式数字 |
| `full` | 100% | 1,306,691 | 109,081,471 | 261,338 | 30 | 1 | 附录的全量规模验证，有时间才跑 |

> 样本数是**精确值**，不是「全量 × 比例」外推——抽样按 uid 哈希取，各用户序列长度不同，
> 实测 `dev` 档样本占比是 4.99% 而非 5.00%。复算命令：
> `python scripts/bench_throughput.py --no-bench`

要点：
1. **只抽用户、不抽物品**（物品池恒为 15,687）。模型结构与输出层维度跨档位完全一致，
   因此 `dev` 档调出的超参可以直接拿到 `main` 档用，不必按档重调。
2. 抽样**确定性且严格嵌套**：`0.5% ⊂ 2% ⊂ 5% ⊂ 10% ⊂ 100%`，
   实现见 `models/data/user_subset.py`（uid 的 blake2b 哈希排序取前缀，与数组顺序无关）。
   换档位是「放大验证」，不是「换一套数据重做」。
3. **HPO 只在 `smoke` 档粗搜**（8 组单种子）→ 前 3 组到 `dev` 档细搜。
   原先 60 组 × 3 种子 = 180 次训练，是本项目最大的时间黑洞。
4. **跨实验组复用训练结果**：E2_1 复用 E1 的 sasrec、E2_4 复用 E1 的 ours、
   E3 的 pure_sasrec 复用 E2_1、E4 完全复用 E2 → 训练次数 **230 次压缩到约 26 次**。
5. **种子数是最优先牺牲项**：只有 `main` 档跑 3 种子，其余单种子（用户已确认此取舍）。
6. **评估子集口径**（2026-09-12 修正）：**评估用户数 = 本档位用户数 × `eval_user_ratio`**，
   不是相对全量用户池 —— 否则会出现「冒烟档训练 6,533 人、却评估 65,335 人」的荒谬配比。
   留一法协议下每个用户的测试物品已被剥离，**在训练过的用户上评估是标准做法、不构成泄漏**。
   `main` 档取 `eval_user_ratio: 1.0`（正式数字用本档全部用户），其余档位按成本控制取 0.1~0.2。
   > ✅ **已实测证实为高估**：infer 98,088 samples/s ⇒ `main` 档全量评估（130,669 用户）
   > 每轮仅约 **1.6 秒**，30 epoch 共 6 轮 ≈ **10 秒**，相对该档训练耗时（端到端 3.5 h）可忽略。
   > `model.yaml` 中「评估与训练同量级」的旧注释已删除。
7. 所有对比模型必须使用**完全相同的抽样与划分**，否则对比无效。

完整的三条抽样口径、算力预算表与四条不变式见 [evaluation-plan.md](evaluation-plan.md) §6.6。

---

## 六、待决策（不决定就无法开工）

| # | 决策项 | 状态 / 选项 | 结论 |
|---|---|---|---|
| D1 | **GPU 路线** | ✅ **已定（2026-09-12）** | **路线 A：更新显卡驱动 + 装 CUDA 版 torch**，实施清单见 §6.1 |
| D2 | **训练档位** | ✅ **已定（2026-09-12）** | 五档位制 `smoke / debug / dev / main / full`，配置见 `configs/scale.yaml`；正式档取 `main` = 10% 用户 |
| D3 | **正式实验规模** | ✅ **已定稿（2026-09-12，M2.4 端到端实测后最终确认）** | **`main` 档 = 10% 用户，不上调 20%**。理由：M2.4 实测端到端 416 s/epoch，10% 档 E1 约 **1.3 天**、全矩阵约 **2.5 天**；20% 会让 E1 单独吃掉 2.6 天，而 10% 已有 13 万用户 / 1,092 万样本，结论强度足够。`full` 档保持"有余力才跑"。 |
| D4 | 是否现在装 `faiss` | 待定 | 阶段 2 召回可先用 numpy 精确内积（15,687 物品规模完全够），**faiss 推迟** |
| D5 | **内容融合口径** | ✅ **已定（2026-09-14，M2.6b）** | **候选侧 `add`（零初始化残差）为默认，`concat` 降为对照臂**。依据 debug 档实测（concat 在 SASRec 上 −1.1pp、add 在双架构上转正），五处默认值已对齐 + `TestDefaultModeConfigConsistency` 锁死；**dev 档 30 epoch 复核进行中，不达预期即回退**（§4.1） |

### 6.1 D1 实施清单（路线 A：更新驱动 + 装 CUDA 版 torch）

> 下面 1–3 步需要**你本人操作**（装驱动要管理员权限，我无法代做）；
> 4–5 步我可以接手，装完顺带把 `evaluation-plan.md` 3.2 的环境快照重新固化。

| 步 | 操作 | 状态 / 校验方式 |
|---|---|---|
| 1 | 到 NVIDIA 官网下载 RTX 2060（Turing）适配的新版驱动 | ✅ **已完成**：`616.92-desktop-win10-win11-64bit-international-nsd-dch-whql.exe`（945 MB，签名 Valid） |
| 2 | 安装驱动（勾选「执行清洁安装」），重启 | ✅ **已完成**：首次安装失败（`nvlddmkm.sys` 写坏 → Code 52），按 §3.3 的 5 步重装后设备 `ErrCode=0`、`nvlddmkm` RUNNING |
| 3 | 装 CUDA 版 torch | ✅ **已完成**：`torch==2.14.0+cu130`，`cuda.is_available() == True`，sm_75 / 6 GB / 30 SM |
| 4 | 写吞吐基准脚本，实测单 epoch 耗时 T | ✅ **已完成**：纯计算实测 40,083 samples/s（1.66 天）；M2.4 落地后修正为端到端 **39 ms/step（约 2.5 天）**（§5.0） |
| 5 | 重新固化环境快照并更新文档 | ✅ **已完成**：`requirements.lock`（152 包）+ `model.yaml`/`scale.yaml` 实测校准 + 本文档 §3/§5 |

**✅ M2.0 已闭环**：驱动修复 → GPU 可用 → 实测吞吐 → 校准档位与预算 → 固化依赖，
五步全部完成，**当前无任何环境阻塞**。
**M2.4（SASRec 基座）也已于同日完成**，下一步进 **M2.5（多兴趣胶囊）**。

> 按 `result-analysis.md` 的填报规范，**无实测出处的数字一律不许填**。
> 现在 `logs/bench_throughput.json` 已产出真实吞吐，档位与预算均有实测依据。
> 注意该文件在 `.gitignore` 内，故关键数字已内嵌到本文档 §5.0 与 `configs/*.yaml`。

---

## 七、风险清单

| 风险 | 影响 | 缓解 |
|---|---|---|
| ~~驱动 457.85 过旧，CUDA 轮子装不上~~ | — | ✅ 已解决：升级到 616.92 |
| ~~驱动文件 `nvlddmkm.sys` 写坏 → 设备 Code 52~~ | — | ✅ 已解决（§3.3）：清洁重装后 `ErrCode=0`、`nvlddmkm` RUNNING |
| ~~6 GB 显存不够用~~ | — | ✅ 已证伪：bs2048 仅用 2.36 GB。但**上限确实存在**：bs4096 触顶（5.46 GB）后崩塌到 5,220 samples/s |
| ~~1.09 亿样本/epoch 导致毕设做不完~~ | — | ✅ 已证伪：端到端实测全矩阵约 **2.5 天**（§5.0） |
| ~~评估开销与训练同量级~~ | — | ✅ 已证伪：`main` 档全量评估每轮约 1.6 秒（§5.1 要点 6） |
| ~~基准是"纯计算"吞吐，不含取数开销~~ | — | ✅ 已解决：M2.4 端到端实测 39 ms/step，预算表已改为两层口径（§5.0） |
| ⚠️ **热度先验主导指标**（M2.4 新发现） | 一个只看物品热度、**完全不知道用户是谁**的打分器就能拿到 hr@10=0.866 / ndcg@10=0.607，而模型是 0.920 / 0.716 —— 只高 6% / 18%。若论文只报模型数字而不报热度基线，等于把广度先验的功劳记给序列建模。**这是学术诚信问题，不只是技术问题** | ① E1 **强制包含 popularity 基线**（`configs/experiment.yaml` 已加入）；② 论文如实披露，并在「局限性」说明该协议下绝对指标偏高；③ 复现脚本 `scripts/diagnose_popularity_bias.py`；④ **不得**为了"让指标更真实"而私自改负采样口径 —— 阶段一已定稿，改则全部历史实验不可比 |
| ~~`batch_size` 由 256 提到 1024 后，`lr` 未重标定~~ | — | ✅ 已解决（M2.4 收尾）：debug 档 8 epoch 固定预算扫描 5 组，**0.001 仍最优**（ndcg@10：0.001→0.7871、0.002→0.7807、0.004→0.7739、0.0005→0.7842），无需上调；组间差 ≤1.3% 接近噪声，正式最优先由 HPO 定（`model.yaml` 注释已固化） |
| **用户重复消费导致答案泄漏（test 1.7% / val 2.0%）** | 不剔除会让指标虚高，且"变好了"不易察觉 | `evaluator.drop_leaked_samples()` 强制剔除并上报数量；论文必须披露剔除数与实际评估样本量 |
| 负样本若各模型各抽一套 | 对比实验直接失效 | 唯一实现 `models/data/negatives.py` + `(seed, row)` 派生独立随机流（顺序/子集无关）+ 落盘缓存复用 |
| 无时间戳导致的时序口径 | 论文「局限性」必须写明 | 已在 `evaluation-plan.md` 2.3 记录，阶段 2 沿用同一口径 |
| 合规题材仍在池内 | 答辩可能被问 | 已在 `evaluation-plan.md` 第十节如实披露 + 评估期屏蔽方案 |
| 冷启动为模拟口径 | 结论需限定表述 | 已记录 holdout 口径与 14.07% 训练信号剥离代价 |
| 指标表格可能被"先填个好看的数" | 学术不端 | `result-analysis.md` 填报规范：**无出处不许填** |
| **实验口径分散在代码默认值与 `configs/` 里** | 任一处不同步（如训练用 `concat`、重建建成 `add`）就会在「权重形状一致、指标也算得出来」的情况下换掉口径，肉眼极难发现 | ① M2.6b 起五处默认值统一为 `add`；② `TestDefaultModeConfigConsistency` 直接断言两个 yaml 与代码默认一致；③ 口径变更必须同时更新本文档 §6 决策表与 `models/content_encoder/README.md` |
| ⚠️ **本协议下 ItemCF 基线极强**（M2.7 定量） | 零训练零参数的 ItemCF 在 main/test 拿到 **hr@10 0.9597 / ndcg@10 0.7514**（热度单独就 0.8996 / 0.6645）。若论文的增益表述相对「随机基线」，读者会以为模型很强；实际相对 ItemCF 可能只有几个百分点 | ① E1 主表**必须有** popularity + itemcf 两行（`configs/experiment.yaml` 已列）；② 论文增益一律**相对 ItemCF** 报告，并在「局限性」说明 1 正 100 均匀负的送分性质；③ 复现 `python scripts/run_baselines.py --baseline all --scale main` |
| **基线或模型侧"零信息样本"被并列口径白送分** | `metrics.positive_rank` 的并列约定是「稳定排序 + 正样本在第 0 列 ⇒ 完全并列时正样本 rank=1」。ItemCF 的 101 个候选全为 0 分时该行**白拿 HR=1** —— 指标变好且不报错 | `ItemCFScorer` 跨 batch 累计并上报 `n_zero_rows` / `zero_row_share`，`run_baselines.py` 打 warning 并跑 `itemcf_tb_pop` 对照；实测占比 0.00%（main 2/129,137），本次不构成影响但必须披露 |
| **`pack_padded_sequence` 用于左填充序列会静默算错**（M2.7 踩到） | 它按 `lengths` 取每行**开头**的若干位置（前提是右填充）。本项目左填充 ⇒ 取到的是 PAD，且结果随"垫了几个 PAD"变化。不报错、指标也算得出来 | GRU4Rec 改为「滚动对齐 + 掩码取末位」并校验「有效位置必须是后缀」；回归测试 `test_repr_invariant_to_left_padding`。⚠️ 将来若有人给 GRU4Rec 加 pack，这条测试会变红 |

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
