# `models/sasrec/` —— SASRec 基座

> 本文件回答：**这个子包干什么、接口是什么、哪些地方改了就不可比**。
> 设计依据见 `docs/evaluation-plan.md` 第 4 节与 `docs/project-structure.md` 三。

---

## 一、文件清单与职责

| 文件 | 职责 | 是否允许 IO |
|---|---|---|
| `config.py` | 超参 dataclass（`SASRecConfig` / `OptimConfig` / `TrainConfig`）+ 合法性校验 + 学习率调度函数 | ❌ 纯计算 |
| `dataset.py` | 滑动窗口训练集、批量取数、评估输入构造 | ❌ 纯计算 |
| `model.py` | SASRec 网络（嵌入 + 因果自注意力 + FFN），对外 `encode()` / `score()` | ❌ 纯计算 |
| `train.py` | 训练循环（BCE + AMP + warmup + 早停），通过回调与外部解耦 | ❌ 纯计算 |

**所有 IO 都在 `scripts/train.py`**：读 YAML、读 `seq_dataset.pkl`、写 checkpoint、写训练日志。
这样可以放心地让 `agents/` 进程直接 import 本子包，不产生任何副作用。

---

## 二、对外接口契约

```python
# ---- 模型（Agent 与评估器都依赖）----
model.encode(seq: LongTensor[B, L], mask: LongTensor[B, L] | None)
    -> (hidden [B, L, H], user_repr [B, H])
model.score(input_ids: LongTensor[B, L], candidate_ids: LongTensor[B, 1+C])
    -> FloatTensor[B, 1+C]        # 第 0 列恒为正样本，分越大越相关

# ---- 训练 ----
fit(model, train_loader, cfg: TrainConfig, optim_cfg: OptimConfig,
    eval_fn, n_items, ...) -> FitResult
    # train_loader 只需满足 len() + 迭代协议，产出 (x [B,L], y [B])
    # eval_fn(model) -> dict，键名与 evaluator 输出一致（如 "ndcg@10"）
```

`score()` 的签名与 `models/eval/evaluator.py` 要求的 `score_fn` **逐字一致**，
所以评估器可以直接 `evaluate(model.score, data)`，不需要适配层 ——
这是「所有模型共用同一套评估代码」这条公平性约束能成立的前提。

### 输入长度口径

| 量 | 值 | 出处 |
|---|---|---|
| 完整序列上限 | 50 | `configs/data.yaml` 的 `sequence.max_len` |
| val / test 各占位 | 1 / 1 | 留一法 |
| **单次前向输入上限** | **48** | `SASRecConfig.input_cap = max_seq_len - 2` |

`model.py` 与 `dataset.py` 各自声明了这个约束（一处是 property、一处是常量），
`tests/test_sasrec/test_model.py::test_input_cap_matches_dataset_convention`
断言两者一致，防止将来只改一处导致位置嵌入越界。

---

## 三、四个「改了就不可比」的实现细节

### 1. 因果 mask 的规则：`(k <= q) and (seq[k] 有效 or k == q)`

对角**恒开**。原因：左填充下，PAD 位置作为 query 时能看到的历史全是 PAD；
若把它们全 mask 成 `-inf`，softmax 分母为 0 → NaN → 顺着残差污染整个 batch。
而 `SlidingWindowDataset` 的第 0 个窗口（`t=0`，无历史）就是一个全 PAD 样本，
`main` 档有 13 万个这样的样本，不可能绕过。

同时这条规则保证了真实位置**不会** attend 到左侧的 PAD，否则填充长度会
轻微影响指标（极难定位）。

回归防线：`test_model.py` 的
`test_pad_positions_do_not_leak_into_real_positions` 与
`test_all_pad_row_has_no_nan`。

### 2. 负采样：训练侧 1 个 / 评估侧 100 个，是两件事

| | 训练侧 | 评估侧 |
|---|---|---|
| 函数 | `sample_train_negatives()` | `sample_negatives()` |
| 每位置几个 | **1** | **100** |
| 排除训练物品 | 否（SASRec 原论文口径） | 是 |

M2.0 实测：训练侧负样本从 1 增到 100 只慢 26%（主开销在序列编码器），
所以"多采样负样本"既不是提速手段、也不是提精度的手段。
详见 `models/data/negatives.py` 的模块 docstring。

### 3. 早停 `patience` 的单位是 **epoch**，不是评估次数

`patience=10` + `eval_every_n_epochs=5` → 连续 2 次评估无提升即停。

若按"评估次数"理解，10 次评估 × 5 epoch = 50 轮，超过 `max_epochs=30`，
早停**永远不会触发**，配置形同虚设。两种理解都常见（RecBole 用后者），
本项目选前者并在此写明。

### 4. 批量取数不走 `torch.utils.data.DataLoader`

| 路径 | 每 batch 取数（bs=1024） |
|---|---|
| `DataLoader(num_workers=0)` | 17.1 ms |
| `WindowBatchIterator`（本项目） | **约 5 ms** |
| `DataLoader(num_workers=4)` | 慢 15 倍（Windows spawn 复制整个序列数据集） |

Windows 用 spawn 启动 worker，dataset 持有的序列会被 pickle 到每个 worker。
实测 `num_workers=4` 让 15 个 step 从 3.6s 变成 55.6s。
所以 `configs/model.yaml` 的 `train.num_workers` 默认 **0**。

---

## 四、端到端实测（2026-09-12，RTX 2060 6GB）

| 档位 | 窗口数 | epoch 耗时（实测） | 说明 |
|---|---|---|---|
| smoke | 535,046 | 17.6 s | 完整跑 3 epoch + 评估共 72.6 s |
| debug | 2,180,315 | 约 83 s | 按 39 ms/step 推算 |
| dev | 5,442,340 | 约 207 s | 按 39 ms/step 推算 |

**M2.0 基准表写的是"纯计算"耗时（39,344 samples/s 量级），端到端会慢约 40%** ——
差额是取数、负采样、CPU↔GPU 搬运与 Python 循环开销。
写论文时用本节数字，不要用 `bench_throughput.py` 的纯计算数字。

---

## 五、已知问题（M2.4 结束时点）

### ⚠️ 热度先验主导指标

在「1 正 + 100 均匀负采样」协议下，**一个完全不知道用户是谁、
只看物品全局热度的打分器**就能拿到 `hr@10 = 0.866 / ndcg@10 = 0.607`
（smoke 档），而 smoke 档模型是 `0.920 / 0.716` —— 只高 6% / 18%。

成因是结构性的：正样本必然是用户交互过的物品（偏热门），
负样本从全池均匀抽，而全池热度极度长尾。
实测正样本热度均值是负样本的 **24.7 倍**。

处置：
* `configs/experiment.yaml` 的 E1 **必须包含 popularity 基线**；
* 诊断脚本：`scripts/diagnose_popularity_bias.py`；
* 论文里必须同时报告热度基线，否则属于选择性报告。

### 其他待办

* `lr` 在 `batch_size` 256→1024 后需要重标定（`configs/model.yaml` 已标注）。
* `bench_throughput.py` 用的是参考计算图，M2.4 落地后可用 `--mode model` 复测。
