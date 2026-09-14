# -*- coding: utf-8 -*-
"""热度基线（Popularity / MostPop）—— `models/baselines/popularity.py`

这是 E1 里**零训练成本**的一行，但它是整张表最不能省的一行。

为什么必须有这一行（M2.4 实测发现，是一条学术诚信约束）
------------------------------------------------------
本数据集在「1 正 + 100 均匀负采样」协议下：

| 打分器 | hr@10 | ndcg@10 |
|---|---|---|
| 随机 | 0.074 | 0.030 |
| **仅按物品热度（本文件）** | **0.866** | **0.607** |
| smoke 档 SASRec 模型 | 0.920 | 0.716 |

一个**完全不知道用户是谁**的打分器就能拿到 0.866 / 0.607，而模型只高
6% / 18%。成因是结构性的：正样本必然是「用户交互过、且交互数 ≥ 10」
的物品（天然偏热门），负样本从全池均匀抽（全池中位热度仅 288 次）。
实测正/负样本热度均值差 **24.7 倍**。

所以：论文里只报 0.716 而不报热度基线 = 把"热度先验"的功劳算给了序列建模。
`docs/evaluation-plan.md` 6.7 把这一行列为**强制项**。

实现口径
--------
::

    score(u, i) = freq[i]        # 与 u 无关，完全忽略 input_ids

`freq` 取**训练集**物品频次（`docs/evaluation-plan.md` 4.1「同一数据划分」）。

⚠️ 与 `scripts/diagnose_popularity_bias.py` 的口径差异（有意为之，两边都要报）
--------------------------------------------------------------------------
那个脚本用的是 `item_stats.parquet` 的 `n_positive`，即**全量**（含 val/test）
交互数 —— 作为基线会**略微乐观**，它的用途是"快速自查协议可信度"。
本文件默认只用**训练集**频次，严格满足"只用训练数据"的公平性要求；
需要在同一份报告里对照两种口径时，把 `freq_source` 分别传 `"train"` / `"full"`。

与评估器的契约
--------------
`PopularityScorer.score(input_ids, candidate_ids) -> [B, 1+C]`
与 `models/eval/evaluator.py` 的 `score_fn` **逐字一致**（第 0 列恒正样本、
分越大越相关）。`input_ids` 参数**保留但被忽略** —— 为了能和其他模型
放进同一个 `evaluate()` 调用里，不写特例分支。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

__all__ = [
    "PAD_ITEM",
    "count_train_frequency",
    "normalize_frequency",
    "PopularityScorer",
]

# 池内索引 0 保留给 PAD，物品从 1 开始（与 models/data/negatives.py 同一口径）
PAD_ITEM = 0


# =====================================================================
# 频次统计
# =====================================================================
def count_train_frequency(
    train_seqs,
    n_items: int,
    rows=None,
    chunk_rows: int = 4096,
) -> np.ndarray:
    """统计训练集里每个物品被消费的次数，返回 `[n_items + 1]` float64。

    * 下标 0 恒为 0（PAD 不是物品）；
    * 长度按 `n_items + 1` 开，直接可用池内索引取值。

    为什么按「行块」累加而不是一次性 `np.concatenate` 全部序列：
    `main` 档训练用户约 13 万、平均序列长 83，一次性拼是 1,100 万个整数
    （88MB）—— 不算大，但每个 epoch 都要重算的话就没必要；分块累加让
    峰值内存与用户数无关（仅 `chunk_rows × 平均长度`）。

    参数
    ----
    train_seqs  可按下标取到序列的对象（`seq_dataset.pkl` 的 `train`）。
    n_items     物品池规模（= `len(smap)`），不含 PAD。
    rows        参与统计的用户行号；None = 全部用户。
                ⚠️ 档位制下**必须传本档位的 `train_rows`**，
                否则不同档位的热度基线来自不同的数据量，"同数据划分"
                这条公平性约束就破了。
    chunk_rows  每累积多少行做一次 bincount（只影响内存峰值与速度）。

    返回
    ----
    `(n_items + 1,)` float64，第 i 项 = 物品 i 在训练集中的出现次数。
    """
    n_items = int(n_items)
    freq = np.zeros(n_items + 1, dtype=np.float64)
    if n_items <= 0:
        raise ValueError(f"n_items 必须为正，收到 {n_items}")

    idx_iter = range(len(train_seqs)) if rows is None else \
        np.asarray(rows, dtype=np.int64).reshape(-1)

    buf: list = []
    step = max(1, int(chunk_rows))
    for r in idx_iter:
        seq = train_seqs[int(r)]
        if len(seq):
            buf.append(np.asarray(seq, dtype=np.int64))
        if len(buf) >= step:
            freq += np.bincount(np.concatenate(buf), minlength=n_items + 1)
            buf = []
    if buf:
        freq += np.bincount(np.concatenate(buf), minlength=n_items + 1)

    # 防御：序列里若混入越界索引（例如 PAD 之外的哨兵值），
    # bincount 会静默把它塞进同一个桶或报错；这里显式核对一次。
    if freq.size != n_items + 1:
        raise ValueError(
            f"频次数组长度 {freq.size} 与 n_items+1={n_items + 1} 不一致，"
            "序列里可能含越界物品索引")
    freq[PAD_ITEM] = 0.0
    return freq


def normalize_frequency(freq: np.ndarray, power: float = 1.0) -> np.ndarray:
    """把频次变成可直接当分数的形式：先幂次、再除以最大值（归一到 0~1）。

    归一不是必须的（评估只看**排序**），但有两个好处：
    分数落在一个有界的量纲里，多个基线拼在一起看时不会出现
    "热度分 1e5、相似度分 1e-3"这种读不出来的对照；
    也让 `eps` 之类的容差判断有可解释的绝对尺度。

    `power` 是标准 MostPop 变体（`freq**power`），默认 1.0 = 原始计数。
    ⚠️ 本项目**不做** power 调参 —— 调了就不再是"零信息基线"，
    热度的每一分增益都会被算成"模型设计"的功劳。留着这个参数只是为了
    复现别人论文里的 `freq**0.5` 口径时有地方落笔。
    """
    f = np.asarray(freq, dtype=np.float64).copy()
    f[f < 0] = 0.0
    if float(power) != 1.0:
        f = np.power(f, float(power))
    m = float(f.max()) if f.size else 0.0
    if m > 0:
        f = f / m
    f[PAD_ITEM] = 0.0
    return f


# =====================================================================
# 打分器
# =====================================================================
class PopularityScorer:
    """`score_fn` 契约下的热度打分器（**无参数、无需训练**）。

    用法
    ----
    >>> freq = count_train_frequency(train_seqs, n_items, rows=train_rows)
    >>> scorer = PopularityScorer(normalize_frequency(freq))
    >>> evaluate(scorer.score, eval_data, ks=(5, 10))      # 直接吃评估器

    为什么还要包一个类而不是裸函数：评估结果要落盘成"模型卡"，
    需要统一的 `config_snapshot()` / `n_params`；用类可以让基线、
    本文模型、检查点重建三条路径的报告字段长得一样。
    """

    def __init__(
        self,
        freq: np.ndarray,
        device: Optional[str] = None,
        name: str = "popularity",
    ):
        """
        参数
        ----
        freq      `[V]` 分数表（通常来自 `normalize_frequency`）。
                  长度必须 = `n_items + 1`，下标 0 = PAD。
        device    仅作记录（打分在 CPU 上做：一次查表，搬到 GPU 不划算）。
        """
        self.scores = np.asarray(freq, dtype=np.float32).reshape(-1)
        self.n_items = int(self.scores.size - 1)
        if self.n_items <= 0:
            raise ValueError(f"freq 长度应为 n_items+1 ≥ 2，收到 {self.scores.size}")
        self.device = device
        self.name = name

    # ------------------------------------------------------------------
    def score(self, input_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """`[B, L] x [B, 1+C] -> [B, 1+C]`。

        `input_ids` **刻意忽略** —— 这就是"零信息基线"的定义。
        保留该参数是为了与 `evaluator.ScoreFn` 契约一致，从而复用
        同一套评估代码（`docs/evaluation-plan.md` 4.1）。
        """
        cand = candidate_ids.detach().to("cpu").numpy()
        # 越界防御：候选索引若超过分数表长度，np.take 会抛错；
        # 这里先裁剪到合法范围，避免"评估跑到一半崩在基线"。
        cand = np.clip(cand, 0, self.n_items)
        vals = self.scores[cand]
        return torch.as_tensor(vals, dtype=torch.float32, device=candidate_ids.device)

    # ------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        """可训练参数量恒为 0（E1 表里"训练成本"一列要写这个）。"""
        return 0

    def config_snapshot(self) -> dict:
        return {
            "arch": "popularity",
            "n_items": int(self.n_items),
            "n_params": 0,
            "trained": False,
            "freq_source": "train",
        }

    def __repr__(self) -> str:                       # pragma: no cover - 诊断用
        return f"PopularityScorer(n_items={self.n_items}, n_params=0)"
