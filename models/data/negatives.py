# -*- coding: utf-8 -*-
"""负采样（negative sampling）的**唯一实现**。

为什么必须唯一实现
------------------
负样本决定了「模型在跟什么比」。若不同模型、不同实验组各抽一套负样本，
指标差异里就混进了「谁的负样本更容易区分」这一无关变量，对比直接失效。
`docs/evaluation-plan.md` 4.1 把「共用负样本池」列为公平性硬约束，
本文件就是这条约束的唯一落点。

约定（改则实验不可比）
----------------------
* 每样本 **100 个负样本**（`configs/base.yaml` 的 `model.neg_sample_num`），
  **无放回**抽取，故 100 个负样本互不相同。
* **排除该用户的训练物品**：否则负样本可能是用户已经看过的番，
  模型凭「序列里出现过」就能答对，指标虚高。
* **排除正样本本身**：否则同一物品既是正例又是负例，BCE 目标自相矛盾。
* 索引口径 = **池内索引**（`smap` 的 value）。`0` 保留给 PAD，物品从 `1` 开始。
* 种子默认 **98765**，与阶段一 `scripts/build_cold_start_subset.py` 及
  `configs/data.yaml` 的 `negative_sampling.seed` 一致。

为什么「逐行独立随机流」而不是「一个 rng 顺序跑」
------------------------------------------------
后者会让「第 k 个用户的负样本」依赖「在它之前抽样了多少次」。于是：

* 换一个评估子集 → 同一个用户的负样本变了；
* 换个遍历顺序 → 结果又变；
* 想并行分块生成 → 结果不可复现。

而本项目的负样本要**落盘缓存后跨模型复用**（`scale.yaml` 的不变式
`eval_subset_fixed_within_scale`），一旦上述任何一件事发生，
"所有模型共用同一批负样本"就名存实亡。

因此这里为**每一行**派生一个独立的随机流，种子取自 `(seed, row)`：

    同 seed + 同 row  =>  永远同一批负样本，与子集、顺序、分块方式全都无关。

代价是每行要新建一个 `Generator`（实测约 35~80 微秒），所以**必须在调用侧
落盘缓存**、而不是每次训练都重算。130 万行全量约需 1~2 分钟，26 万行约 20 秒。

关于「候选不足」
----------------
正常情况物品池 15,687 个，扣掉用户训练物品（均值 85 个）后仍有 1.5 万以上候选，
远大于 100。但在冷启动场景负样本池被限定为新番集合（2,783 个），
若某用户的训练里正好覆盖了大量新番，候选可能不足 100 ——
此时降级为**有放回**补齐，并把这类样本数记进 `n_shortfall` 如实上报，
不做静默处理（论文中若该值非 0 必须说明）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "DEFAULT_NEG_SEED",
    "NegativeSampleResult",
    "sample_negatives",
]

# 与 configs/data.yaml 的 negative_sampling.seed、
# scripts/build_cold_start_subset.py 的 --seed 默认值保持一致。
# 改这个数等于换一套负样本，所有实验必须重跑。
DEFAULT_NEG_SEED = 98765


# =====================================================================
# 返回结构
# =====================================================================
@dataclass(frozen=True)
class NegativeSampleResult:
    """负采样结果 + 必须随指标一起上报的诊断量。

    为什么不只返回一个 ndarray：`n_shortfall` 这类「降级了多少条」的信息
    一旦丢失，读结果的人就无法判断负样本是否真的满足「同分布、无放回」，
    而这是 E3 冷启动实验有效性的前提（evaluation-plan 2.5 的两条硬约束）。
    """

    negatives: np.ndarray            # (N, n_negatives) int32，池内索引
    seed: int                        # 实际使用的种子
    pool_size: int                   # 候选池大小（不是物品总数：冷启动时是新番池）
    n_rows: int                      # 样本数 N
    n_negatives: int                 # 每样本负样本数 C
    n_shortfall: int                 # 候选不足、只能有放回补齐的样本数
    excluded_items_mean: float       # 平均每条样本被排除的候选数（诊断用）

    def as_report(self) -> dict:
        """转成可 `json.dump` 的字典，供 metrics.json / 实验报告落盘。"""
        return {
            "seed": int(self.seed),
            "pool_size": int(self.pool_size),
            "n_rows": int(self.n_rows),
            "n_negatives": int(self.n_negatives),
            "n_shortfall": int(self.n_shortfall),
            "excluded_items_mean": round(float(self.excluded_items_mean), 4),
            "replacement": "without" if self.n_shortfall == 0 else "partially_with",
        }


# =====================================================================
# 主函数
# =====================================================================
def sample_negatives(
    user_rows: Sequence[int],
    positives: Sequence[int],
    n_items: int,
    n_negatives: int = 100,
    seed: int = DEFAULT_NEG_SEED,
    pool: Optional[Sequence[int]] = None,
    train_seqs=None,
    exclude_train: bool = True,
    exclude_positive: bool = True,
) -> NegativeSampleResult:
    """为每条评估样本抽取 `n_negatives` 个负样本。

    参数
    ----
    user_rows   (N,) 每条样本对应的**用户行号**（`umap` 的 value，0-based）。
                扮演「随机流地址」的角色：同一行号永远得到同一批负样本。
    positives   (N,) 每条样本的正样本池内索引（测试目标物品）。
    n_items     物品池规模（`len(smap)`），用于确定索引上界 = n_items。
    n_negatives 每条样本抽几个负样本，默认 100。
    seed        随机种子，默认 98765（与阶段一口径一致）。
    pool        候选池（池内索引序列）。None = 全池 `1..n_items`。
                冷启动实验传入新番集合的池内索引，使负样本与正样本同分布。
    train_seqs  可按下标取到用户训练序列的对象（list / dict 均可）。
                `exclude_train=True` 时必须提供，否则无法排除。
    exclude_train     是否排除该用户训练集中出现的物品（默认 True）。
    exclude_positive  是否排除正样本本身（默认 True）。

    返回
    ----
    NegativeSampleResult（见上方定义）。

    用法
    ----
    >>> res = sample_negatives([3, 7], [101, 202], n_items=15687, train_seqs=train)
    >>> res.negatives.shape
    (2, 100)
    """
    rows = np.asarray(user_rows, dtype=np.int64).reshape(-1)
    pos = np.asarray(positives, dtype=np.int64).reshape(-1)
    if rows.shape != pos.shape:
        raise ValueError(
            f"user_rows {rows.shape} 与 positives {pos.shape} 长度必须一致")

    if exclude_train and train_seqs is None:
        raise ValueError("exclude_train=True 时必须提供 train_seqs，否则无法排除训练物品")

    n_rows = int(rows.size)
    c = int(n_negatives)
    if c <= 0:
        raise ValueError(f"n_negatives 必须为正，收到 {n_negatives}")

    # ---------- 候选池 ----------
    if pool is None:
        pool_arr = np.arange(1, int(n_items) + 1, dtype=np.int64)
    else:
        pool_arr = np.asarray(pool, dtype=np.int64).reshape(-1)
        if pool_arr.size and (pool_arr.min() < 1 or pool_arr.max() > int(n_items)):
            raise ValueError(
                f"候选池索引越界：池内索引应在 [1, {n_items}]，"
                f"实际 [{pool_arr.min()}, {pool_arr.max()}]")
    pool_size = int(pool_arr.size)
    if pool_size == 0:
        raise ValueError("候选池为空，无法抽样")
    # 池内正样本必须真的在池里，否则「排除正样本」无意义、且说明调用方口径错了
    if exclude_positive and n_rows:
        oob = pos[(pos < 1) | (pos > int(n_items))]
        if oob.size:
            raise ValueError(
                f"正样本索引越界（应在 [1, {n_items}]）：{oob[:5].tolist()} ...")

    out = np.zeros((n_rows, c), dtype=np.int32)
    if n_rows == 0:
        return NegativeSampleResult(out, int(seed), pool_size, 0, c, 0, 0.0)

    # 「代数戳」数组：stamp[x] == g 表示物品 x 在本条样本里已被占用。
    # 用递增的代数 g 代替每条样本重置一个 n_items 大小的布尔数组，
    # 133 万条样本下省掉 133 万次 memset。
    # int32 上限 2.1e9 >> 用户数上限 1.3e6，不会溢出。
    stamp = np.zeros(int(n_items) + 1, dtype=np.int64)

    n_shortfall = 0
    excluded_total = 0
    # 每轮从池里无放回抽「还缺的数量」，被排除的位置下一轮再补。
    # 排除率极小（训练物品 ~85 / 池 15687 ≈ 0.54%），故期望 1 轮完成；
    # 上限 64 轮是防御性设置，正常永远走不到。
    max_rounds = 64

    for i in range(n_rows):
        row = int(rows[i])
        g = i + 1

        # --- 本行需要排除的物品 ---
        n_blocked = 0
        if exclude_train:
            tr = train_seqs[row]
            if len(tr):
                stamp[np.asarray(tr, dtype=np.int64)] = g
                n_blocked += len(tr)
        if exclude_positive:
            stamp[int(pos[i])] = g
            n_blocked += 1
        excluded_total += n_blocked

        # --- 抽样：随机流由 (seed, row) 派生，与遍历顺序无关 ---
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), row]))

        picked = np.empty(c, dtype=np.int64)
        n_got = 0
        rounds = 0
        # 已选集合也用 stamp 标记（复用同一张表，代数号错开避免与排除标记冲突）
        sel_g = -g          # 负数代数，与排除用的正数代数天然隔离
        # 池本身装不下 c 个不同物品时，「无放回」在数学上就不可能，
        # 直接走下面的有放回补齐分支，不要在这里空转到轮次耗尽。
        if pool_size > c:
            while n_got < c and rounds < max_rounds:
                rounds += 1
                need = c - n_got
                cand = rng.choice(pool_arr, size=min(need, pool_size), replace=False)
                cand = cand[stamp[cand] != g]        # 去掉被排除的（训练物品 / 正样本）
                cand = cand[stamp[cand] != sel_g]    # 去掉已选中的（防跨轮重复）
                if cand.size == 0:
                    continue
                stamp[cand] = sel_g
                take = min(int(cand.size), need)
                picked[n_got:n_got + take] = cand[:take]
                n_got += take

        if n_got < c:
            # 池太小，只能有放回补齐 —— 如实记录，不静默
            n_shortfall += 1
            avail = pool_arr[stamp[pool_arr] != g] if exclude_positive or exclude_train \
                else pool_arr
            if avail.size == 0:
                avail = pool_arr
            picked[n_got:] = rng.choice(avail, size=c - n_got, replace=True)

        # 输出转 int32：池内索引上限 15,687，int32 绰绰有余，
        # 而 1.31 亿个负样本用 int64 会白占 524MB。
        out[i] = picked.astype(np.int32, copy=False)

    return NegativeSampleResult(
        negatives=out,
        seed=int(seed),
        pool_size=pool_size,
        n_rows=n_rows,
        n_negatives=c,
        n_shortfall=int(n_shortfall),
        excluded_items_mean=(excluded_total / n_rows) if n_rows else 0.0,
    )
