# 项目进度总览 · AniRec

> **快照时间**：2026-09-12 17:00 ｜ **最新完成里程碑**：M2.1 指标唯一实现 ｜ **下一里程碑**：M2.2 评估器
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
| 阶段 2　核心算法与对比实验 | 35% | 🔵 **进行中** | M2.1 指标层已完成；M2.0 驱动已升级、CUDA torch 已装、吞吐基准脚本已就位，待修复驱动文件损坏后出实测耗时 |
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
| **GPU** | **NVIDIA RTX 2060 / 6 GB**（Turing sm_75，桌面版） | **与 `evaluation-plan.md` 3.1 原写的 RTX 3090 24GB 不符，已更正** |
| GPU 驱动 | **616.92**（Windows 显示版本 `32.0.16.1692`，2026-09-04） | 2026-09-12 由 457.85（CUDA 11.1）升级；包内 CUDA 13 |
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

**torch 安装口径（勿走默认源）**：PyPI 默认源上的 torch 是 `+cpu` 轮子，装完不报错、
但 `torch.cuda.is_available()` 恒为 `False`（本项目踩过一次）。必须显式指定索引：

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.0+cu130
```

更换后回归：`python -m pytest tests/ -q` → **115 passed**（指标口径未受影响）。
旧环境快照留档 `logs/pip_freeze_before_cuda.txt`（149 行，可回滚）。

### 3.3 ⚠️ 剩余阻塞项：驱动文件损坏，GPU 仍不可用（2026-09-12 17:00 已定位根因）

驱动与 torch 都已就绪，但 `torch.cuda.is_available()` 仍为 `False`。
逐步排除后的完整证据链：

| 检查项 | 实测结果 | 排除的猜测 |
|---|---|---|
| `cuInit(0)`（ctypes 直调） | 返回 **100 = CUDA_ERROR_NO_DEVICE** | 排除「torch 装错」「没装 CUDA 运行时」 |
| 设备管理器 | RTX 2060 `Status=Error`、**`ConfigManagerErrorCode=52`**（无法验证驱动签名） | — |
| 内核服务 `nvlddmkm` | **STOPPED**，退出码 **1077**（本次开机从未尝试启动） | — |
| Kernel-PnP 事件 219 | `\Driver\nvlddmkm 加载失败`，状态 **0xC0000428** = `STATUS_INVALID_IMAGE_HASH` | 排除权限问题 |
| CodeIntegrity 事件 3004 | 无法验证 `...\nvlesi.inf_amd64_33018540a16fd177\nvlddmkm.sys` 映像完整性 | — |
| 该文件数字签名 | **HashMismatch** | **← 根因** |
| 同目录其余 17 个大文件 | 全部 `Valid` | 排除「整个驱动包坏了」 |
| 安装包 `616.92-...whql.exe` | 签名 `Valid`（NVIDIA Corporation） | 排除「下载损坏」，无需重下 |
| 磁盘 | C: 余 92.9 GB，SSD 健康 | 排除空间不足与坏盘 |

**结论**：安装写盘时 **`nvlddmkm.sys` 这一个 109 MB 文件被写坏了**，Windows 因哈希不符
拒绝加载 → 设备 Code 52 → CUDA 报告无设备。其余文件完好，所以不是包的问题。

**修复步骤**（需管理员，由用户执行）：

1. 设备管理器 → 显示适配器 → `NVIDIA GeForce RTX 2060` → 右键**卸载设备** → 勾选「尝试删除此设备的驱动程序」→ 确定
2. **重启**
3. 重新运行 `C:\Users\32683\Downloads\616.92-desktop-win10-win11-64bit-international-nsd-dch-whql.exe`（右键管理员）→ 选「自定义（高级）」→ 勾选**执行清洁安装**
4. **重启**
5. 复验：`nvidia-smi` 应能正常输出；再由 AI 跑 `python scripts/bench_throughput.py`

> 若 1–4 仍不生效：用 DDU（Display Driver Uninstaller）在安全模式下彻底清理后重装。
> 本机未装 7-Zip / DDU，需要时现下。
>
> 顺带记录：本机 **HVCI（内存完整性）已开启**、VBS 开启、`VulnerableDriverBlocklistEnable=1`，
> 排查驱动类问题时这些状态会影响判断。

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
| **M2.0** | 训练环境就绪 | 按 §6 D1 选定路线装 GPU 版 torch；写吞吐基准脚本，实测 s/step 并外推单 epoch 耗时 | `scripts/bench_throughput.py` + 环境快照 | 进度：驱动已升到 **616.92**、torch 已换 **2.14.0+cu130**、基准脚本已就位并通过数据侧自检；**卡在驱动文件损坏（§3.3），`cuda.is_available()` 仍为 False**，修复后即可产出实测吞吐表 |
| **M2.1** ✅ | 指标唯一实现 | 按 `evaluation-plan.md` 5.1 实现 HR@K / NDCG@K / Recall@K / MRR，含并列稳定排序 | `models/eval/metrics.py`、`tests/test_metrics/`、`scripts/check_metrics_mutation.py` | ✅ **已完成**：单测 88/88 通过；变异测试 6/6 被捕获 |
| **M2.2** | 评估器 | 1 正 + 100 负候选，走预生成的负样本 pkl；支持全量 / 分题材 / 冷启动三种切分 | `models/eval/evaluator.py` | 同一批预测重复跑结果一致；两种切分都能出 `metrics.json` |
| **M2.3** | 滑动窗口 Dataset | 按 `max_len=50` 在线生成样本，不落盘 | `models/sasrec/dataset.py` | **因果性测试**：第 t 个样本不得看到 t 之后的信息 |
| **M2.4** | SASRec 基座 | 2 层 2 头 hidden 64 dropout 0.2 + BCE 负采样 + 早停（看 val NDCG@10, patience=10） | `models/sasrec/{model,config,train}.py` | 在开发档上 loss 稳定下降，val NDCG@10 明显高于随机基线 |
| **M2.5** | 多兴趣胶囊 | K=4 动态路由（3 次迭代）接在 SASRec 输出后；记录胶囊两两余弦相似度检测塌缩 | `models/multi_interest/{capsule,model,train}.py` | 4 个胶囊不塌缩（相似度 < 0.7，配置里已有告警阈值） |
| **M2.6** | 内容融合 | 候选侧拼接 + 排序侧 7:3 加权；冷启动样本权重切 5:5 | `models/content_encoder/fusion.py` | `fusion_score()` 签名与 `project-structure.md` 契约一致 |
| **M2.7** | 对比基线 | ItemCF（共现 + 余弦，TopK 200 邻居）、GRU4Rec（隐藏 64，1 层） | `models/baselines/*.py` | 与本文模型**共用同一划分 / 负样本池 / 评估器 / 早停策略** |
| **M2.8** | 实验编排 | `run_experiments.py` 按 `configs/experiment.yaml` 串起 E1~E4，落盘 `experiments/{id}/`，回写 `result-analysis.md` | `scripts/run_experiments.py`、`scripts/plot_results.py` | 一次命令跑完四组；输出目录结构与 `evaluation-plan.md` 8.2 一致 |
| **M2.9** | 填表与作图 | 真实指标填入 `result-analysis.md` / `ablation-study.md`，生成 4 张图 | 指标表 + `figures/*.png` | **每个数字都有出处（实验目录名）**，无出处不许填 |

推荐执行顺序即上表自上而下；**M2.1–M2.3 与 GPU 环境无关，可以立刻开工**（这也是我建议的下一步起点）。

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


---

## 五、训练规模与时间预算（必须先实测）

阶段一产物给出的训练规模：

| 指标 | 值 |
|---|---|
| 训练样本数（滑动窗口，每 epoch） | **109,081,471**（约 1.09 亿）= **Σ `train_len`** ✅ 已用 `user_stats.parquet` 逐位复核 |
| 每用户训练序列长度 | 均值 83.5、**中位数 41**、最大 8,917；**截断到 50 后的有效平均长度 34.8** |
| 序列长度分布 | 仅 **44.8%** 的用户 `train_len ≥ 50`（会被截断到满长） |
| 用户 / 物品 | 1,306,691 / 15,687 |
| 测试用户 | 全量 1,306,691（每用户 101 候选） |
| 冷启动测试样本 | 209,912 |

> 序列长度分布这条对 `max_seq_len` 的实现方式有直接影响：若 M2.3 采用「统一填充到 50」，
> 算力按 L=50 计；若采用「按批动态填充」，有效长度只有 34.8，**能省下可观算力**。
> 这个取舍在 M2.3 落地时明确，吞吐基准脚本 `scripts/bench_throughput.py` 已按两种口径分别打印。

**结论：1.09 亿样本/epoch 在 CPU 上不可行**（量级上每 epoch 是小时~天级），必须用 GPU。

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
   > 另注：早期文档称「全量评估 1.32 亿次打分与一个训练 epoch 同量级」是**高估**——
   > 1.32 亿里绝大多数是 101 候选的**点积**（dim 64，几乎免费），真正的开销只有
   > n_eval_users 次序列编码。待实测后正式更正 `model.yaml` 与 `evaluation-plan.md` 的这段描述。
7. 所有对比模型必须使用**完全相同的抽样与划分**，否则对比无效。

完整的三条抽样口径、算力预算表与四条不变式见 [evaluation-plan.md](evaluation-plan.md) §6.6。

---

## 六、待决策（不决定就无法开工）

| # | 决策项 | 状态 / 选项 | 结论 |
|---|---|---|---|
| D1 | **GPU 路线** | ✅ **已定（2026-09-12）** | **路线 A：更新显卡驱动 + 装 CUDA 版 torch**，实施清单见 §6.1 |
| D2 | **训练档位** | ✅ **已定（2026-09-12）** | 五档位制 `smoke / debug / dev / main / full`，配置见 `configs/scale.yaml`；正式档取 `main` = 10% 用户 |
| D3 | **正式实验规模** | ✅ **已定（2026-09-12）** | 先用 `main` 档（10%）出论文主表；`full` 档作为「有余力再跑」的附录验证。训练次数由约 230 次压到约 26 次，**不再以早期文档的 77 h 为排期依据** |
| D4 | 是否现在装 `faiss` | 待定 | 阶段 2 召回可先用 numpy 精确内积（15,687 物品规模完全够），**faiss 推迟** |

### 6.1 D1 实施清单（路线 A：更新驱动 + 装 CUDA 版 torch）

> 下面 1–3 步需要**你本人操作**（装驱动要管理员权限，我无法代做）；
> 4–5 步我可以接手，装完顺带把 `evaluation-plan.md` 3.2 的环境快照重新固化。

| 步 | 操作 | 状态 / 校验方式 |
|---|---|---|
| 1 | 到 NVIDIA 官网下载 RTX 2060（Turing）适配的新版驱动 | ✅ **已完成**：`616.92-desktop-win10-win11-64bit-international-nsd-dch-whql.exe`（945 MB，签名 Valid） |
| 2 | 安装驱动（勾选「执行清洁安装」），重启 | ⚠️ **已执行但失败**：`nvlddmkm.sys` 写坏、签名 HashMismatch，设备 Code 52，详见 §3.3 |
| 3 | 装 CUDA 版 torch | ✅ **已完成**：`torch==2.14.0+cu130`，`torch.version.cuda == 13.0`；但 `cuda.is_available()` 仍 False（受步骤 2 牵连） |
| 4 | 写吞吐基准脚本，实测单 epoch 耗时 T | ✅ 脚本 `scripts/bench_throughput.py` 已就位并通过数据侧自检；⏳ 实测待 GPU 可用 |
| 5 | 重新固化环境快照并更新 `evaluation-plan.md` 3.1/3.2 与 `progress.md` §3 | 🟡 部分完成：`progress.md` §3 已更新；`pip freeze` 与环境快照待 GPU 打通后一并固化 |

**🔴 当前唯一待办（需管理员操作，AI 无法代做）**：按 §3.3 的 5 步修复损坏的驱动文件
（设备管理器卸载设备并删除驱动 → 重启 → 重跑安装包 → 重启 → 复验 `nvidia-smi`）。

**不浪费等待时间的并行路径**：M2.2（评估器）与 M2.3（滑动窗口 Dataset）都不需要 GPU，
可以先做，等驱动修好再回头补 M2.0 的实测吞吐表。

---

## 七、风险清单

| 风险 | 影响 | 缓解 |
|---|---|---|
| 驱动 457.85 过旧，CUDA 轮子装不上 | 阶段 2 直接停摆 | D1 决策；B 路线已确认可行 |
| 6 GB 显存 | 批量受限，可能需 `batch_size` 从 256 降到 128 | SASRec 很小（hidden 64），预计不成为瓶颈；实测确认 |
| 1.09 亿样本/epoch 且全矩阵约 230 次训练 | 训练与网格搜索时间爆炸，毕设做不完 | 五档位制 + HPO 降档 + 跨实验组复用结果 + 单种子，训练次数压到约 26 次（`configs/scale.yaml`） |
| 抽样训练可能被质疑代表性 | 答辩被追问 | 只抽用户不抽物品、确定性嵌套抽样、所有对比模型同口径；论文「实验设置」如实披露抽样比例与理由 |
| 评估开销与训练同量级（1.32 亿次打分/轮） | 容易被忽略，实际吃掉近半时间 | `eval.every_n_epochs: 5` + 同档位内固定的评估子集 |
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
