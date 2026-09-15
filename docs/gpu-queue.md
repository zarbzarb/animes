# GPU 实验待办队列（阶段 2 未跑完的部分）

> **建立时间**：2026-09-14 17:05
> **建立原因**：阶段 2 剩余实验需 **≈ 42 h GPU**（约 1.8 天），而阶段 3（全栈系统）不依赖 GPU。
> 因此**两阶段并行**：GPU 排队跑实验，人在同一时间开发阶段 3 系统。
> **本文件是「不许忘」清单** —— 任何时刻中断开发回来，照此文件逐条执行即可。
>
> ⚠️ **本文件不做进度快照**（那是 [progress.md](progress.md) 的职责）。
> 本文件只回答：**还有哪些实验没跑、命令是什么、结果写到哪里**。
> 每完成一条就在下面的复选框打勾，并把数字回写到 progress.md 对应小节。

---

## 一、当前 GPU 上正在跑什么

| 项 | 内容 |
|---|---|
| 任务 | dev 档 30 epoch 四组复核（M2.6b 口径确认）——**第 4 组 mi_add 第三次启动** |
| 启动时间 | 第 1–3 组 2026-09-14 完成；第 4 组 14 日 21:20 重启中断于 23/30 → 21:25 二启中断于 **11/30（22:02 关机）** → **15 日 11:26 三启** |
| 日志 | `logs/dev_m26b_base.log` / `_add` / `_mi_base`（✅ 完成）/ `_mi_add`（三跑中） |
| 已出数字（val，n=12913） | base ndcg@10 **0.79854** → add **0.80090（+0.24pp 转正）**；mi_base 0.78492 |

**时间线（按实测 30.1 ms/step 折算）**：

| 顺序 | 组 | 状态 |
|---|---|---|
| 1/4 | `sasrec` 纯行为 | ✅ 17:34（ndcg@10 0.79854） |
| 2/4 | `sasrec` + 内容（add） | ✅ 18:53（ndcg@10 0.80090，**+0.24pp**） |
| 3/4 | `multi_interest` 纯行为 | ✅ 20:08（ndcg@10 0.78492） |
| 4/4 | `multi_interest` + 内容（add） | 🔄 21:25 重启，预计 ~23:20 |

> ✅ 跑完后**必做**：按 §2.2 做「同权重内容消融」复评，再把结论回写
> `docs/progress.md` §4.1「口径锁定」与 `models/content_encoder/README.md`。
> **若 dev 档 `add` 不再为正 → 回退默认口径为纯行为基线，并同步五处默认值**（决策 D5）。

---

## 二、待跑实验清单（按推荐顺序）

### 2.1 ☐ dev 档四组复核（**正在跑**，无需再启动）

复现命令（已在后台串行执行，此处留档）：

```bash
PY=E:/tools/anaconda/envs/py3_11/python.exe
$PY scripts/train.py --scale dev --model sasrec         --no-content-fusion --tag m26b_dev_base
$PY scripts/train.py --scale dev --model sasrec         --content-fusion    --tag m26b_dev_add
$PY scripts/train.py --scale dev --model multi_interest --no-content-fusion --tag m26b_dev_mi_base
$PY scripts/train.py --scale dev --model multi_interest --content-fusion    --tag m26b_dev_mi_add
```

产物：`logs/train_multi_interest_content_v1_<tag>_dev_seed42.json`、`data/checkpoints/...best.pt`

### 2.2 ☐ 同权重内容消融复评（≈ 10 min，跑完 2.1 立刻做）

把内容侧权重置零、用同一 checkpoint 重评，量化「内容在这份权重里值多少分」。

```bash
$PY scripts/eval_content_fusion.py --ckpt data/checkpoints/<content_best.pt> --zero-content
```

→ 数字写入 `logs/content_fusion_eval.json`，并在 `progress.md` §4.1 补一行对照。

### 2.3 🔵 M2.7 收尾：GRU4Rec 的 dev / main（≈ 6.4 h）

```bash
$PY scripts/run_baselines.py --baseline gru4rec --scale dev  --split val      # 0.9 h，方向确认  ✅ 2026-09-15
$PY scripts/run_baselines.py --baseline gru4rec --scale main --split val  --seed 42   # ✅ 2026-09-15（2838s）
$PY scripts/run_baselines.py --baseline gru4rec --scale main --split test --seed 42   # ✅ 2026-09-15（2950s，0.9709/0.7878）→ ×3 种子
```

> ✅ dev 档已完成：val hr@10 0.9639 / ndcg@10 0.7935（seed 42，30ep，1731s），
> `logs/baseline_gru4rec_dev_val_seed42.json`。低于 SASRec base（0.79854）、高于多兴趣两组。
>
> ✅ main/val 已完成：hr@10 **0.9658** / ndcg@10 **0.7966** / mrr 0.7436（seed 42，最优 ep15，25ep 早停，
> `logs/baseline_gru4rec_main_val_seed42.json`）。main 实测 2,838 s ≈ 0.79 h（与 20.7 ms/step × 10,671 step 口径吻合）。

> ⚠️ E1 主表要求 3 个神经模型 × 3 种子。种子 42 已有阈值参照，2024 / 2025 需补。

→ 回写 `progress.md` §4.1 M2.7 与 `configs/experiment.yaml` 的 E1 note。

### 2.4 ☐ **E1 主表**（≈ 18.2 h，阶段 2 最大的一块）

| 模型 | 次数 | 单次 | 小计 |
|---|---|---|---|
| `ours`（多兴趣 + 内容 add） | 3 种子 | 3.4 h | 10.2 h |
| `sasrec` | 3 种子 | 2.7 h | 8.0 h |
| `gru4rec` | 3 种子 | 1.85 h | 5.6 h（已计入 §2.3） |

```bash
for SEED in 42 2024 2025; do
  $PY scripts/train.py --scale main --model sasrec --no-content-fusion --seed $SEED --tag e1_sasrec
  $PY scripts/train.py --scale main --model multi_interest --content-fusion --seed $SEED --tag e1_ours
done
```

> ⚠️ **必须先有 `scripts/run_experiments.py`（M2.8）**，否则种子/档位/评估子集容易各写各的。
> 口径：main 档 `user_ratio=0.10`、`eval_user_ratio=1.0`、30 epoch、早停 patience=10。

### 2.5 ☐ HPO（≈ 4.3 h）

```bash
# 粗搜 8 组（smoke 档，3 epoch，每组约 1.5 min）
$PY scripts/run_experiments.py --group HPO --phase coarse
# 入选前 3 组到 dev 档细搜（每组 1.4 h）
$PY scripts/run_experiments.py --group HPO --phase fine
```

### 2.6 ☐ E3 冷启动（≈ 2.6 h）— **前置代码未完成，见 §4.1**

```bash
$PY scripts/run_experiments.py --group E3 --scale dev
```

口径：`data/processed/cold_start_report.json`（holdout 年份 ≥ 2021；**2,783** 个新番、
**209,912** 条冷启动测试样本、剥离 **14.07%** 训练交互）。
⚠️ E3 **必须用剥离后的训练集重新训练**，不能复用 E1/E2 权重。

### 2.7 ☐ 补充实验：K 敏感性（≈ 5.1 h）

```bash
for K in 2 4 8; do
  $PY scripts/train.py --scale dev --model multi_interest --num-interests $K --tag hpo_k$K
done
```

> 原设计的「融合权重 6 组」**删掉**：排序侧后验加权已由 M2.6 证伪
> （7:3 ≈ 纯行为、冷启动 5:5 更差），再扫权重没有信息量。仅保留 K 敏感性。

### 2.8 ☐ E4 分题材 + 汇总（≈ 0.5 h，零训练）

复用消融训练结果，仅做分组评估（`evaluate_grouped`，12 类题材 × 4 组），
输出 `experiments/E4_by_genre/metrics.json`。**不得使用低于 main 的档位**。

---

## 三、算力与日历

| 项 | 数值 |
|---|---|
| 剩余 GPU 合计 | **≈ 42 h**（其中 E1 占 18.2 h） |
| 24 h 不停机 | 9/14 17:00 起 → **9/16 中午**跑完 |
| 夜间挂机 12 h/天 | 约 3.5 天 → 9/18 |
| 每天 8 h | 5～6 天 → 9/20 |

**折算依据（实测，别再用旧口径）**：`bs1024 + AMP + 训练负样本 1` 下每步耗时与档位无关 ——
SASRec **30.1 ms**、多兴趣（含内容）**≈ 38–40 ms**、GRU4Rec **20.7 ms**；
steps/epoch = smoke 522 / debug 2 129 / dev 5 315 / main 10 671。
⚠️ `configs/scale.yaml` 与 `evaluation-plan.md` 里记的 **39 ms/step、main 416 s/epoch** 是
M2.4 的一次性测量，比实测慢约 22% —— **等 main 档真跑一次拿到 `total_elapsed_sec` 后再统一校正文档**，
不要现在改成另一个未验证的数。

**可砍时间的三个杠杆（需明确同意）**：

1. E1 只对 `ours` 跑 3 种子，`sasrec` / `gru4rec` 各 1 种子 → 省 **≈ 8 h**；
   代价：违反 `scale.yaml` 的 `final_model_seeds: 3`，论文须写明。
2. HPO 细搜 3 组 → 2 组 → 省 1.4 h。
3. 双进程并行（实测单进程 2.4 GB / 6 GB，util 86%）→ 单进程慢 30–50%，净收益约 1.2–1.3×，
   可把 42 h 压到 ≈ 33 h。**先跑小规模对照确认再开**。

---

## 四、阶段 2 遗留的「非 GPU」事项（不占卡，但要补）

### 4.1 🔴 E3 现在直接跑会出**假结果**

`configs/experiment.yaml` 的 `E3_coldstart` 段还是早期占位开关：

```yaml
- { name: pure_sasrec,  cfg: { use_multi_interest: false, use_content_fusion: false } }
- { name: ours_w03,     cfg: { ..., w_content_cold: 0.3 } }
```

问题：① `use_multi_interest` / `use_content_fusion` 是**早期占位名**，`from_dict` 会
**静默忽略**未知键；② 候选侧融合里**根本没有 `w_content_cold`** 这个参数（那是排序侧口径）。
⇒ 现在跑会给出一组「全默认配置」的模型，指标照样算得出来。

**必须做**：重写 E3 组配置（`arch` / `content_fusion` / `content_mode`）+
训练时按 `strip_from_train_item_ids` 过滤序列 + 冷启动评估切分。

### 4.2 ☐ `scripts/run_experiments.py`（M2.8）不存在

需要它统一 E1~E4 的种子、档位、评估子集与目录结构（`experiments/{id}/`），
按 `configs/experiment.yaml` 驱动。**这是 2.4~2.8 的前置**。

### 4.3 ☐ `scripts/plot_results.py`（M2.9）不存在

4 张图（主对比柱状、消融、题材雷达、冷启动）+ 指标表回填
`docs/result-analysis.md` / `docs/ablation-study.md`，**每个数字必须标出处**。

### 4.4 ☐ 文档口径待统一

`configs/scale.yaml`、`docs/evaluation-plan.md` §6.6 的 sec/epoch 与
`configs/experiment.yaml` 头部的「≈2.5 天」—— 均基于旧的 39 ms/step。
拿到 main 档实测后**一次性校正**（别边跑边改，否则引用对不上）。

---

## 五、恢复执行的检查清单

中断开发、回来继续跑实验前，逐条确认：

1. ☐ `nvidia-smi` 里没有别的进程占卡（本项目只允许一个训练进程）
2. ☐ 当前 dev 复核四组是否已全部完成 → 完成才做 §2.2，否则先等
3. ☐ §4.2 的 `run_experiments.py` 是否就绪 → E1 之前必须有
4. ☐ §4.1 的 E3 配置是否已重写 → E3 之前必须有
5. ☐ 磁盘余量（当前 F: 67 GB 可用；main 档 checkpoint 每个约 37 MB，无压力）
6. ☐ 每跑完一组：结果回写 `docs/progress.md` → 提交（代码与文档同一 PR）→ 勾掉本文件对应条目
