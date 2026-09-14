# -*- coding: utf-8 -*-
"""ItemCF 基线（基于物品的协同过滤）—— `models/baselines/itemcf.py`

口径（`docs/evaluation-plan.md` §4）
-----------------------------------
::

    共现矩阵   C = B^T · B            B = 训练集「用户×物品」0/1 矩阵
    相似度     sim(i, j) = C[i,j] / sqrt(C[i,i] · C[j,j])      （余弦）
    截断       每个物品只保留 TopK = 200 个邻居
    打分       score(u, i) = Σ_{j ∈ H(u)} sim(i, j)

其中 `H(u)` 是**模型可见的历史**（= 评估器给的 `input_ids`，左填充、上限 48）。
这一点是刻意的公平性控制：神经模型只能看到最后 48 个物品，ItemCF 也必须
只看这 48 个 —— 否则"基线看得到更长的历史"会变成一个说不清的变量。

⚠️ 相似度矩阵本身用**全量训练序列**算（不截断）—— 这对应神经模型的
"在完整训练序列上训练"，是训练数据，不是推理输入。两者不要混。

三个必须写清楚的实现约定
------------------------
**1. 相似度按「余弦」而不是原始共现次数。**
原始共现次数会被热门物品主导：一个 10 万次交互的番与任何物品的共现次数
都很大，于是"热门物品的邻居"里全是热门物品。除以 `sqrt(pop_i · pop_j)`
是标准的 popularity 归一，也是 RecBole `ItemKNN` 的默认口径。

**2. TopK 截断作用在「归一化后的余弦」上，且是**逐行**的（不强制对称）。**
打分只用 `sim` 的**第 i 行**（候选物品那一行），所以逐行截断与公式一致；
对称化会引入 `min(row_i, row_j)` 这种没有依据的口径。

**3. ⚠️ 「全零行」是这份基线最大的一个指标陷阱，必须上报。**
若某用户的历史与**所有** 101 个候选都没有共同邻居，则 101 个分数全是 0。
本项目 `models/eval/metrics.py` 的并列口径是「稳定排序 + 正样本固定在第 0 列
⇒ 完全并列时正样本 rank = 1」，于是这种"模型毫无信息"的样本会**白拿 HR=1**。
所以本文件把 `n_zero_rows` 作为一等诊断量报出来（`ItemCFScorer.last_stats`），
并提供一个 `tiebreak="popularity"` 开关（**只**填充这些全零行）用来量化
这个假象对指标的影响有多大。默认 `None` = 纯 ItemCF，与公开实现一致。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch

__all__ = [
    "PAD_ITEM",
    "build_cooccurrence",
    "build_itemcf_similarity",
    "ItemCFScorer",
]

PAD_ITEM = 0


# =====================================================================
# 共现矩阵
# =====================================================================
def build_cooccurrence(
    train_seqs,
    n_items: int,
    rows=None,
    chunk_rows: int = 8192,
    verbose: bool = False,
) -> sp.csr_matrix:
    """由训练序列构造「物品×物品」共现次数矩阵 `[V, V]`（V = n_items + 1）。

    * 用户内部**取重**（一件物品在一个用户的历史里只算一次）——
      本数据集用户会重复消费同一部番，若不去重，重度重复消费的用户
      会把自己的口味差异放大成共现强度，这是标准 ItemCF 不做的；
    * 分块累加：`main` 档一次拼 1,100 万个整数虽然放得下，但
      `C = C + C_chunk` 的分块写法让峰值内存与用户数无关，
      也便于在 `full` 档（130 万用户）上跑。

    参数
    ----
    train_seqs  可按下标取到序列的对象（`seq_dataset.pkl` 的 `train`）。
    n_items     物品池规模（不含 PAD）。
    rows        参与统计的用户行号；None = 全部。
                ⚠️ 档位制下必须传本档位的 `train_rows`，与热度基线同理。
    chunk_rows  每次矩阵乘的行块大小（只影响内存峰值与速度）。

    返回
    ----
    `scipy.sparse.csr_matrix`，float32，`C[i, j]` = 同时消费过 i 与 j 的
    用户数。下标 0 行/列恒为 0（序列里没有 PAD）。
    """
    n_items = int(n_items)
    if n_items <= 0:
        raise ValueError(f"n_items 必须为正，收到 {n_items}")
    V = n_items + 1

    idx_iter = (range(len(train_seqs)) if rows is None
                else np.asarray(rows, dtype=np.int64).reshape(-1))
    all_rows = np.asarray(list(idx_iter), dtype=np.int64)
    n_rows = int(all_rows.size)
    if n_rows == 0:
        return sp.csr_matrix((V, V), dtype=np.float32)

    C: Optional[sp.csr_matrix] = None
    step = max(1, int(chunk_rows))
    for s in range(0, n_rows, step):
        block = all_rows[s:s + step]
        rr: list = []
        cc: list = []
        for local, r in enumerate(block):
            seq = np.asarray(train_seqs[int(r)], dtype=np.int64).reshape(-1)
            seq = seq[(seq >= 1) & (seq <= n_items)]
            if seq.size == 0:
                continue
            seq = np.unique(seq)
            rr.append(np.full(seq.size, local, dtype=np.int64))
            cc.append(seq)
        if not rr:
            continue
        n_local = int(block.size)
        B = sp.csr_matrix(
            (np.ones(sum(x.size for x in rr), dtype=np.float32),
             (np.concatenate(rr), np.concatenate(cc))),
            shape=(n_local, V))
        chunk_C = (B.T @ B).tocsr().astype(np.float32, copy=False)
        C = chunk_C if C is None else (C + chunk_C).tocsr()
        if verbose:
            print(f"  [itemcf] 共现累加 {s + block.size:,}/{n_rows:,} 用户，"
                  f"nnz={C.nnz:,}", flush=True)

    if C is None:
        return sp.csr_matrix((V, V), dtype=np.float32)
    # 数值上 C 是对称的；强行对称化一次消除浮点累加顺序造成的 1e-7 级差异，
    # 让"同一个矩阵"这件事有唯一形状（便于测试对拍）。
    C = ((C + C.T) * 0.5).tocsr()
    C.eliminate_zeros()
    return C


# =====================================================================
# 余弦相似度 + TopK 截断
# =====================================================================
def build_itemcf_similarity(
    train_seqs,
    n_items: int,
    topk: int = 200,
    rows=None,
    chunk_rows: int = 8192,
    verbose: bool = False,
) -> tuple:
    """构造 ItemCF 的 TopK 邻居矩阵。

    返回 `(sim, stats)`：

    * `sim`   `[V, V]` csr，float32，第 i 行是物品 i 与它 TopK 邻居的余弦相似度
              （对角线已置 0 —— 自己不是自己的邻居）；
    * `stats` 诊断字典：`n_items` / `topk` / `nnz` / `avg_neighbors` /
              `n_items_without_neighbor` / `cooc_nnz`。

    实现顺序很关键（也是内存安全的唯一顺序）：先算完整共现 → 再逐行算余弦并
    当场截断 → 最后丢掉完整共现矩阵。反过来（先把完整余弦矩阵物化出来）
    在 `main` 档会多占几百 MB，且没有任何信息增益。
    """
    n_items = int(n_items)
    topk = int(topk)
    if topk <= 0:
        raise ValueError(f"topk 必须为正，收到 {topk}")
    V = n_items + 1

    C = build_cooccurrence(train_seqs, n_items, rows=rows,
                           chunk_rows=chunk_rows, verbose=verbose)
    cooc_nnz = int(C.nnz)

    # 归一化因子：1 / sqrt(用户数)。注意用 C 的对角线作为物品用户数
    # （只有二值化之后它才等于"交互过 i 的用户数"）。
    pop = np.asarray(C.diagonal(), dtype=np.float64)
    inv = np.zeros_like(pop)
    nz = pop > 0
    inv[nz] = 1.0 / np.sqrt(pop[nz])

    indptr, indices, data = C.indptr, C.indices, C.data
    out_r: list = []
    out_c: list = []
    out_v: list = []
    n_no_nbr = 0

    for i in range(1, V):
        s, e = int(indptr[i]), int(indptr[i + 1])
        if e <= s:
            n_no_nbr += 1
            continue
        idx = indices[s:e]
        keep = idx != i                      # 对角线不是邻居
        idx = idx[keep]
        if idx.size == 0:
            n_no_nbr += 1
            continue
        cos = (data[s:e][keep].astype(np.float64)
               * inv[i] * inv[idx]).astype(np.float32)

        if cos.size > topk:
            # argpartition 取前 topk 大；再按 (相似度降序, 物品索引升序) 排一次，
            # 让并列时的输出**确定**（浮点并列很常见：两个邻居的相似度可能
            # 都等于 1/3 这种可精确表示的值）。
            sel = np.argpartition(-cos, topk)[:topk]
            sel = sel[np.lexsort((idx[sel], -cos[sel]))]
        else:
            sel = np.lexsort((idx, -cos))
        out_r.append(np.full(sel.size, i, dtype=np.int32))
        out_c.append(idx[sel].astype(np.int32))
        out_v.append(cos[sel])

    if out_r:
        sim = sp.csr_matrix(
            (np.concatenate(out_v), (np.concatenate(out_r), np.concatenate(out_c))),
            shape=(V, V), dtype=np.float32)
    else:
        sim = sp.csr_matrix((V, V), dtype=np.float32)

    stats = {
        "n_items": n_items,
        "topk": topk,
        "cooc_nnz": cooc_nnz,
        "nnz": int(sim.nnz),
        "avg_neighbors": (round(float(sim.nnz) / max(1, n_items), 3)),
        "n_items_without_neighbor": int(n_no_nbr),
    }
    return sim, stats


# =====================================================================
# 打分器
# =====================================================================
class ItemCFScorer:
    """`score_fn` 契约下的 ItemCF 打分器（**无参数、无需训练**）。

    用法
    ----
    >>> sim, stats = build_itemcf_similarity(train_seqs, n_items, topk=200,
    ...                                      rows=train_rows)
    >>> scorer = ItemCFScorer(sim, n_items=n_items, topk=200)
    >>> evaluate(scorer.score, eval_data, ks=(5, 10))

    `tiebreak="popularity"` 只用来**量化全零行的假象**（见模块 docstring 第 3 条），
    正式 E1 的 ItemCF 一行取默认值（`None`，纯 ItemCF 口径）。
    """

    def __init__(
        self,
        sim: sp.csr_matrix,
        n_items: int,
        topk: int = 200,
        popularity: Optional[np.ndarray] = None,
        tiebreak: Optional[str] = None,
        score_chunk: int = 512,
        name: str = "itemcf",
    ):
        """
        参数
        ----
        sim          `[V, V]` csr 邻居相似度矩阵（`build_itemcf_similarity` 的输出）。
        n_items      物品池规模（不含 PAD），用于校验 `sim` 的行数。
        topk         只作记录与报告（打分不依赖它）。
        popularity   可选 `[V]` 归一化热度，仅 `tiebreak="popularity"` 时使用。
        tiebreak     `None`（默认，纯 ItemCF）/ `"popularity"`（只填全零行）。
        score_chunk  打分内部分块大小。打分要在 [V, B] 的稠密中间量上做，
                     B 越大中间量越大（V=15,688 时 B=512 约 32MB）；
                     固定 512 让显存/内存占用与调用方批大小解耦。
        """
        self.sim = sim.tocsr()
        self.n_items = int(n_items)
        self.topk = int(topk)
        if self.sim.shape != (self.n_items + 1, self.n_items + 1):
            # 形状不符会在打分时静默取到错行，属于"指标算得出来但没意义"，
            # 所以宁可当场报错。
            raise ValueError(
                f"sim 形状 {self.sim.shape} 与 n_items+1={self.n_items + 1} 不一致")
        self.popularity = (None if popularity is None
                           else np.asarray(popularity, dtype=np.float32).reshape(-1))
        if tiebreak not in (None, "popularity"):
            raise ValueError(f"tiebreak 只能是 None / 'popularity'，收到 {tiebreak!r}")
        if tiebreak == "popularity" and self.popularity is None:
            raise ValueError("tiebreak='popularity' 需要同时传入 popularity 数组")
        self.tiebreak = tiebreak
        self.score_chunk = max(1, int(score_chunk))
        self.name = name

        # 诊断量：**跨多次 `score()` 调用累计**（`evaluator.collect_ranks`
        # 会把评估集切成一堆 batch 逐批调用 `score()`）。
        # ⚠️ 若只记"最后一次调用"，`main` 档会报出 2,161 行（最后一个残缺
        #    batch）而不是 129,137 行 —— 数字看着像模像样，实际少报 98%。
        #    调用方必须在每次评估前 `reset_stats()`；`run_baselines.py` 已照此做。
        self._n_rows = 0
        self._n_zero_rows = 0

    # ------------------------------------------------------------------
    def reset_stats(self) -> None:
        """清零诊断计数器。**每次评估前都要调**（见 `_n_rows` 的注释）。"""
        self._n_rows = 0
        self._n_zero_rows = 0

    @property
    def last_stats(self) -> dict:
        """自上次 `reset_stats()` 以来累计的诊断量。"""
        return {
            "n_rows": int(self._n_rows),
            "n_zero_rows": int(self._n_zero_rows),
            "zero_row_share": round(self._n_zero_rows / max(1, self._n_rows), 6),
            "tiebreak": self.tiebreak,
        }

    # ------------------------------------------------------------------
    def score(self, input_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """`[B, L] x [B, 1+C] -> [B, 1+C]`：`Σ_{j ∈ H(u)} sim(i, j)`。

        实现要点：把每个 mini-batch 的历史写成**稀疏**矩阵 `H [V, B]`，
        再算 `sim @ H`。写成稀疏而不是稠密 `[V, B]` 不是为了省内存
        （V×512 也就 32MB），而是为了省**乘法次数**：
        稀疏乘法的代价正比于"真正重合的非零个数"（≈ B × |H| × TopK），
        稠密写法则是 `nnz(sim) × B`，实测差 2 个数量级。
        """
        x = input_ids.detach().to("cpu").numpy()
        cand = candidate_ids.detach().to("cpu").numpy()
        n = int(x.shape[0])
        if x.shape[0] != cand.shape[0]:
            raise ValueError(
                f"input_ids 行数 {x.shape[0]} 与 candidate_ids 行数 {cand.shape[0]} 不一致")

        V = self.n_items + 1
        out = np.zeros(cand.shape, dtype=np.float32)
        n_zero_rows = 0
        for s in range(0, n, self.score_chunk):
            e = min(s + self.score_chunk, n)
            b = e - s
            xb = x[s:e]
            rows_c: list = []
            cols_c: list = []
            for local in range(b):
                hist = xb[local]
                hist = hist[(hist >= 1) & (hist <= self.n_items)]
                if hist.size:
                    # 取重：相似度矩阵是按「二值化」口径建的（用户内取重），
                    # 打分侧若不去重，重复消费同一部番会把它的权重乘以次数 ——
                    # 同一个口径在两侧必须一致。用 np.unique 排序后去重，
                    # 顺带让"历史顺序无关"这条性质成立。
                    hist = np.unique(hist)
                    rows_c.append(hist)                       # 行 = 物品
                    cols_c.append(np.full(hist.size, local, dtype=np.int64))
            if rows_c:
                H = sp.csr_matrix(
                    (np.ones(sum(v.size for v in rows_c), dtype=np.float32),
                     (np.concatenate(rows_c), np.concatenate(cols_c))),
                    shape=(V, b))
                R = (self.sim @ H.tocsc()).toarray()          # [V, b]
                vals = R[cand[s:e], np.arange(b)[:, None]]    # [b, 1+C]
            else:
                vals = np.zeros((b, cand.shape[1]), dtype=np.float32)
            out[s:e] = vals

            # ---- 全零行诊断（见模块 docstring 第 3 条）----
            row_max = vals.max(axis=1)
            zero = row_max <= 0.0
            k = int(zero.sum())
            if k:
                n_zero_rows += k
                if self.tiebreak == "popularity":
                    cb = np.clip(cand[s:e], 0, self.n_items)
                    out[s:e][zero] = self.popularity[cb[zero]]

        self._n_rows += n
        self._n_zero_rows += int(n_zero_rows)
        return torch.as_tensor(out, dtype=torch.float32, device=candidate_ids.device)

    # ------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        """可训练参数量恒为 0（相似度矩阵是从数据算出来的，不是学出来的）。"""
        return 0

    def config_snapshot(self) -> dict:
        return {
            "arch": "itemcf",
            "n_items": int(self.n_items),
            "topk": int(self.topk),
            "n_params": 0,
            "trained": False,
            "similarity": "cosine_on_binary_cooccurrence",
            "neighbor_nnz": int(self.sim.nnz),
            "tiebreak": self.tiebreak,
        }

    def __repr__(self) -> str:                       # pragma: no cover - 诊断用
        return (f"ItemCFScorer(n_items={self.n_items}, topk={self.topk}, "
                f"nnz={self.sim.nnz})")
