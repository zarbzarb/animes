# -*- coding: utf-8 -*-
"""负采样（negative sampling）的**唯一实现**。

本文件里有两个**完全不同**的负采样口径，改任何一个都会影响实验可比性，
所以放在同一个文件里，逼着人看到它们的区别：

| | 训练侧 | 评估侧 |
|---|---|---|
| 函数 | `sample_train_negatives()` | `sample_negatives()` |
| 每位置几个 | **1 个** | **100 个** |
| 用途 | 算 BCE 损失（学参数） | 算 HR/NDCG（报指标） |
| 候选池 | 全池均匀随机 | 全池均匀随机（冷启动时限定新番池）|
| 排除训练物品 | **否**（原论文口径） | **是**（否则模型凭"见过"就能答对）|
| 排除正样本 | 是 | 是 |
| 落盘缓存 | 否（每 batch 现采） | 是（跨模型复用，见下） |

⚠️ 最容易混淆的一点：`configs/model.yaml` 里同时存在
`eval.protocol = leave_one_out_neg100`（100）与
`train.negatives_per_position = 1`（1）。**它们不是同一个数**，
一个是评估协议、一个是训练超参，把 100 套到训练侧会让训练慢 26%
却换不来指标提升（M2.0 实测：训练开销主要花在序列编码器上）。

为什么评估侧必须"逐行独立随机流"
--------------------------------
一个 rng 顺序跑会让「第 k 个用户的负样本」依赖「在它之前抽样了多少次」。于是：

* 换一个评估子集 → 同一个用户的负样本变了；
* 换个遍历顺序 → 结果又变；
* 想并行分块生成 → 结果不可复现。

而本项目的负样本要**落盘缓存后跨模型复用**（`scale.yaml` 的不变式
`eval_subset_fixed_within_scale`），一旦上述任何一件事发生，
"所有模型共用同一批负样本"就名存实亡。

因此 `sample_negatives()` 为**每一行**派生一个独立的随机流，种子取自 `(seed, row)`：

    同 seed + 同 row  =>  永远同一批负样本，与子集、顺序、分块方式全都无关。

代价是每行要新建一个 `Generator`（实测约 35~80 微秒），所以**必须在调用侧
落盘缓存**、而不是每次训练都重算。130 万行全量约需 1~2 分钟，26 万行约 20 秒。

训练侧为什么不需要这么讲究
--------------------------
训练负样本**不进指标、不需要跨模型一致**（每个模型自己的训练过程本来就是
独立的），且每个 batch 都要重采（否则模型会把固定负样本背下来）。
一个 epoch 有 ~26 万个 batch，逐行建 `Generator` 的 80 微秒 × 1024 行
会让取负样本比前向还慢。因此训练侧直接复用调用方传入的 `Generator`
（按 batch 顺序消耗），允许结果依赖调用顺序 —— 这是**有意的取舍**，
不是偷懒。

关于「候选不足」
----------------
评估侧：正常情况物品池 15,687 个，扣掉用户训练物品（均值 85 个）后仍有
1.5 万以上候选，远大于 100。但在冷启动场景负样本池被限定为新番集合
（2,783 个），若某用户的训练里正好覆盖了大量新番，候选可能不足 100 ——
此时降级为**有放回**补齐，并把这类样本数记进 `n_shortfall` 如实上报，
不做静默处理（论文中若该值非 0 必须说明）。

训练侧：池恒为全池（15,687），永不不足；唯一的边界是「抽到正样本自己」，
用环绕 +1 避开（见 `sample_train_negatives`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "DEFAULT_NEG_SEED",
    "NegativeSampleResult",
    "sample_negatives",
    "sample_train_negatives",
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


# =====================================================================
# 训练侧负采样（与评估侧的对照见模块 docstring 的表格）
# =====================================================================
def sample_train_negatives(
    targets: Sequence[int],
    n_items: int,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """为训练批的每个 target 位置抽 **1 个**负样本。

    口径（SASRec 原论文，`docs/evaluation-plan.md` 6.6.5）
    -------------------------------------------------------
    * 从**全物品池** `1..n_items` 均匀随机抽，每位置 1 个；
    * 只排除**正样本本身**（否则同一物品既正又负，BCE 目标自相矛盾），
      **不排除**该用户的其他训练物品。

    关于「不排除训练物品」这条，值得解释清楚（它看起来像个疏漏，其实不是）：

    * 原论文的实现（以及 RecBole）就是这么做的，改掉会让本模型的
      训练目标与公开基线不一致，E1 的对比就多了一个说不清的变量；
    * 代价很小：用户平均 85 个训练物品 / 池 15,687 ≈ **0.54%** 的负样本
      会撞上用户看过的番，它们相当于少量标签噪声；
    * 收益是每个 batch 不需要去查「这个用户看过什么」——
      在 `main` 档一个 epoch 有 ~1.07 万个 batch（1,092 万样本 / 1024），
      省掉的是 1,092 万次序列查询，而训练侧负采样本身不该成为瓶颈。

    参数
    ----
    targets  `(n,)` 本批的 target 池内索引。**必须都是真实物品索引**
             （即不含 PAD）。传 PAD 会在下面被 `n_items` 上界校验挡住。
    n_items  物品池规模（`len(smap)`）。索引上界。
    rng      随机数生成器。**传入同一个 rng 并让它顺序消耗**是有意设计：
             训练数据每个 epoch 都重新洗牌，负样本没有"跨模型必须一致"
             的要求，所以不需要评估侧那套逐行独立随机流（那样做
             80 微秒 × 1024 行会比前向还慢）。None 时用 `seed` 新建一个。
    seed     `rng` 为 None 时用它新建 `Generator`；两者都为 None 则用
             `DEFAULT_NEG_SEED`，保证不传参时结果可复现。

    返回
    ----
    `(n,)` int64 数组，每个元素落在 `[1, n_items]` 且**不等于**对应 target。
    可直接作为 `candidate_ids` 的第 1 列（第 0 列是正样本）。

    用法
    ----
    >>> sample_train_negatives([5, 9], n_items=100, seed=0).shape
    (2,)
    """
    t = np.asarray(targets, dtype=np.int64).reshape(-1)
    n = int(t.size)
    n_items = int(n_items)

    if n_items < 2:
        # n_items == 1 时「避开正样本」在数学上不可能（池里只剩它自己）
        raise ValueError(f"n_items 至少为 2 才能抽负样本，收到 {n_items}")
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    # 上界校验放在抽样前：让「误把 PAD 传进来」当场报错，
    # 而不是悄悄产出一个越界索引、等 embedding 层才抛 CUDA 断言。
    bad = t[(t < 1) | (t > n_items)]
    if bad.size:
        raise ValueError(
            f"targets 含越界索引（应在 [1, {n_items}]，PAD=0 不能作为 target）："
            f"{bad[:5].tolist()} ...")

    if rng is None:
        rng = np.random.default_rng(DEFAULT_NEG_SEED if seed is None else int(seed))

    neg = rng.integers(1, n_items + 1, size=n, dtype=np.int64)

    # 撞上正样本就**重抽**。
    #
    # 为什么不用"环绕 +1"这类确定性避让：那会把偏差**集中**转移给
    # `target + 1`，使它被抽中的概率变成其他物品的**两倍**。
    # 在真实场景下（target 分布多样、n_items=15,687）这点偏差小到可忽略
    # ——因为每个物品额外多抽到的次数只有 `N / n_items²` 量级。
    # 但只要出现「某个 target 特别集中」的批（按物品分桶采样、冷启动里
    # 同一部新番被大量用户当作目标），偏差就会成规模显现：
    # 实测构造 5 万条 target 全相同的样本时，`target+1` 的命中次数
    # 达到 2036 vs 期望 1020（+100%）。
    #
    # 重抽的期望轮数是 1/(1 - 1/n_items)，n_items≥2 时几乎总在一轮内解决，
    # 成本可忽略，换来的是完全不偏的分布。
    clash = neg == t
    guard = 0
    while clash.any() and guard < 64:
        neg[clash] = rng.integers(1, n_items + 1, size=int(clash.sum()),
                                  dtype=np.int64)
        clash = neg == t
        guard += 1
    if clash.any():
        # 理论上到不了这里：每轮都是独立重抽，撞车概率 1/n_items。
        # 真到了说明 n_items 极小或随机源异常 —— 显式报错，不返回脏数据。
        raise RuntimeError(
            f"负采样重抽 {guard} 轮后仍有 {int(clash.sum())} 条撞上正样本，"
            "请检查 n_items 与随机源")

    return neg
