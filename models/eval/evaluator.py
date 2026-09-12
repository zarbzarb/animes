# -*- coding: utf-8 -*-
"""评估器：把「打分函数」变成 `metrics.json`。

职责边界（三者不要互相渗透）
----------------------------
    负采样     -> models/data/negatives.py     (抽哪些候选)
    输入序列   -> models/sasrec/dataset.py     (历史怎么拼)
    指标公式   -> models/eval/metrics.py       (HR/NDCG/Recall/MRR 怎么算)
    本文件     -> 只负责【把这三块串起来并跑批量前向】

为什么用 `score_fn` 回调而不是直接接收模型
-------------------------------------------
`docs/dev-conventions.md` 的分层约束是 `server/ → agents/ → models/`，
且 `models/` 必须纯函数、无 IO。本文件若不 import 任何模型，
就可以同时服务三类调用方：

* SASRec 基座、多兴趣胶囊模型（`models/sasrec/`、`models/multi_interest/`）
* 对比基线 ItemCF / GRU4Rec（`models/baselines/`）
* 单元测试里的"手算模型"

只要满足一个契约即可 —— **所有模型共用同一套评估代码**，这正是
`docs/evaluation-plan.md` 4.1「公平性控制」要求的（否则指标差异里
会混进"谁的评估循环写得更划算"这种无关变量）。

打分函数契约（唯一的接口约定）
------------------------------
::

    score_fn(input_ids: LongTensor[B, L], candidate_ids: LongTensor[B, 1 + C])
        -> FloatTensor[B, 1 + C]

* `input_ids`     `[B, L]` 左填充的历史序列（PAD = 0）
* `candidate_ids` `[B, 1+C]` 第 0 列恒为**正样本**，其后 C 列为负样本
* 返回值          逐候选得分，**越大越相关**（不要求归一化、不要求是概率）

候选矩阵把正样本固定在第 0 列，是为了让 `metrics.positive_rank(pos_index=0)`
可以直接用，从而全项目共用同一套并列口径（见 metrics 模块 docstring 第 3 条）。

无效样本
--------
若测试目标出现在模型可见的输入历史里（实测本数据集为 0%，但必须防御），
该样本属于"答案泄漏"，指标无意义。`drop_leaked_samples()` 会把它剔除，
并把剔除数量随指标一起上报 —— 不静默。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch

from models.eval.metrics import ranking_metrics

__all__ = [
    "EvalData",
    "collect_ranks",
    "evaluate",
    "evaluate_grouped",
    "find_leaked_mask",
    "drop_leaked_samples",
]

# 模型打分函数的最小契约，仅用于文档与静态检查，不参与运行时校验
ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# =====================================================================
# 评估数据容器
# =====================================================================
@dataclass
class EvalData:
    """一批评估样本。所有字段都是**已就绪的普通数组**，本类不做 IO。

    形状约定
    --------
    input_ids  (N, L)      历史序列，左填充，PAD = 0
    positives  (N,)        正样本池内索引（= 测试目标物品）
    negatives  (N, C)      负样本池内索引，来自 `sample_negatives()`
    user_rows  (N,)        用户行号，用于把指标归因回用户（A6 兴趣漂移要用）
    genres     (N,) 或 None 目标物品的题材标签，用于 E4 分题材指标
    """

    input_ids: np.ndarray
    positives: np.ndarray
    negatives: np.ndarray
    user_rows: Optional[np.ndarray] = None
    genres: Optional[np.ndarray] = None
    label: str = "eval"

    # 预拼好的候选矩阵，避免每个 batch 现拼（N=26 万时能省掉上千次分配）
    candidates: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.input_ids = np.ascontiguousarray(self.input_ids, dtype=np.int64)
        self.positives = np.ascontiguousarray(self.positives, dtype=np.int64)
        self.negatives = np.ascontiguousarray(self.negatives, dtype=np.int64)

        if self.input_ids.ndim != 2:
            raise ValueError(f"input_ids 应为 (N, L)，实际 {self.input_ids.shape}")
        n = self.input_ids.shape[0]
        if self.positives.shape != (n,):
            raise ValueError(
                f"positives 应为 ({n},)，实际 {self.positives.shape}")
        if self.negatives.ndim != 2 or self.negatives.shape[0] != n:
            raise ValueError(
                f"negatives 应为 ({n}, C)，实际 {self.negatives.shape}")
        if self.user_rows is not None and np.asarray(self.user_rows).shape != (n,):
            raise ValueError(f"user_rows 应为 ({n},)，实际 {np.shape(self.user_rows)}")
        if self.genres is not None and np.asarray(self.genres).shape != (n,):
            raise ValueError(f"genres 应为 ({n},)，实际 {np.shape(self.genres)}")

        # 正样本固定第 0 列（见模块 docstring）
        self.candidates = np.concatenate(
            [self.positives[:, None], self.negatives], axis=1)

    # ------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def n_candidates(self) -> int:
        return int(self.candidates.shape[1])

    @property
    def n_negatives(self) -> int:
        return int(self.negatives.shape[1])

    def subset(self, mask: np.ndarray, label: Optional[str] = None) -> "EvalData":
        """按布尔掩码取子集，用于分题材 / 冷启动子集评估。"""
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if m.shape != (self.n_samples,):
            raise ValueError(f"掩码长度 {m.shape} 与样本数 {self.n_samples} 不一致")
        return EvalData(
            input_ids=self.input_ids[m],
            positives=self.positives[m],
            negatives=self.negatives[m],
            user_rows=None if self.user_rows is None else self.user_rows[m],
            genres=None if self.genres is None else self.genres[m],
            label=self.label if label is None else label,
        )

    def summary(self) -> dict:
        return {
            "label": self.label,
            "n_samples": self.n_samples,
            "n_candidates": self.n_candidates,
            "n_negatives": self.n_negatives,
            "window_len": int(self.input_ids.shape[1]),
            "n_distinct_users": (int(np.unique(self.user_rows).size)
                                 if self.user_rows is not None else None),
        }


# =====================================================================
# 无效样本（答案泄漏）
# =====================================================================
def find_leaked_mask(data: EvalData) -> np.ndarray:
    """返回 `(N,)` 布尔掩码：哪些样本的正样本出现在模型可见的输入历史里。

    判据直接取「正样本是否出现在 `input_ids` 里」，而不是"是否出现在 train 集合里"——
    因为序列做了尾部截断，早期物品模型其实看不到；按输入矩阵判断才与模型实际
    获得的信息一致。
    """
    if data.n_samples == 0:
        return np.zeros(0, dtype=bool)
    hit = (data.input_ids == data.positives[:, None]).any(axis=1)
    return hit


def drop_leaked_samples(data: EvalData) -> tuple:
    """剔除答案泄漏样本，返回 `(新 EvalData, 剔除数量)`。

    本数据集实测泄漏率为 0%（留一法保证了 test 目标不在 train 里），
    但保留这条防线：一旦将来换划分口径或改截断长度，泄漏会**提高**指标，
    属于"变好看了所以没人查"的那类 bug。
    """
    mask = find_leaked_mask(data)
    n = int(mask.sum())
    if n == 0:
        return data, 0
    kept = data.subset(~mask, label=data.label)
    return kept, n


# =====================================================================
# 批量前向
# =====================================================================
def collect_ranks(
    score_fn: ScoreFn,
    data: EvalData,
    batch_size: int = 512,
    device=None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """分批跑前向，返回每个样本中正样本的 1-based 排名 `(N,)`。

    为什么要单独暴露"排名"这一层：HR / NDCG / MRR 都是排名的函数，
    算一次排名就能得出全部指标，而且**分题材评估可以复用同一批排名**
    （按样本切片即可），不必为每个题材再跑一遍前向 —— 在 26 万用户规模上
    这是数量级的差别。
    """
    n = data.n_samples
    ranks = np.zeros(n, dtype=np.int64)
    if n == 0:
        return ranks

    bs = max(1, int(batch_size))
    n_batches = (n + bs - 1) // bs

    with torch.no_grad():
        for b in range(n_batches):
            s = b * bs
            e = min(s + bs, n)
            x = torch.as_tensor(data.input_ids[s:e], device=device)
            cand = torch.as_tensor(data.candidates[s:e], device=device)
            scores = score_fn(x, cand)

            # --- 契约校验：形状不对时立刻报错，别让它污染指标 ---
            if not isinstance(scores, torch.Tensor):
                raise TypeError(
                    f"score_fn 必须返回 torch.Tensor，实际 {type(scores).__name__}")
            want = (e - s, data.n_candidates)
            if tuple(scores.shape) != want:
                raise ValueError(
                    f"score_fn 返回形状 {tuple(scores.shape)}，期望 {want}；"
                    "注意第 0 列必须是正样本，其后是负样本")

            # 复用 metrics 的唯一排序口径（稳定排序 + 正样本在第 0 列）
            ranks[s:e] = _positive_ranks(scores)
            if progress is not None:
                progress(b + 1, n_batches)
    return ranks


def _positive_ranks(scores: torch.Tensor) -> np.ndarray:
    """内部：从打分矩阵取正样本排名（委托给 metrics，保持口径唯一）。"""
    from models.eval.metrics import positive_rank   # 局部导入，避免循环引用

    return positive_rank(scores, pos_index=0).detach().cpu().numpy()


# =====================================================================
# 指标
# =====================================================================
def _metrics_dict(ranks: np.ndarray, ks: Sequence[int], extra: dict) -> dict:
    """把排名转成实验表格用的字典（键名与 configs/experiment.yaml 对齐）。"""
    out = ranking_metrics(ranks, ks=tuple(ks))
    out.update(extra)
    return out


def evaluate(
    score_fn: ScoreFn,
    data: EvalData,
    ks: Sequence[int] = (5, 10),
    batch_size: int = 512,
    device=None,
) -> dict:
    """算出总体指标。

    返回的字典可直接 `json.dump` 成 `metrics.json`，含：
    指标本体 + `n_samples`（evaluation-plan 5.2 要求随指标一起报告样本量，
    否则分不清"空集返回 0"和"真的得 0 分"）+ 少量诊断字段。
    """
    t0 = time.time()
    ranks = collect_ranks(score_fn, data, batch_size=batch_size, device=device)
    meta = {
        "label": data.label,
        "n_candidates": data.n_candidates,
        "n_negatives": data.n_negatives,
        "input_len": int(data.input_ids.shape[1]) if data.n_samples else 0,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    return _metrics_dict(ranks, ks, meta)


def evaluate_grouped(
    score_fn: ScoreFn,
    data: EvalData,
    ks: Sequence[int] = (5, 10),
    group_field: str = "genres",
    batch_size: int = 512,
    device=None,
    min_samples: int = 1,
) -> dict:
    """总体指标 + 按 `group_field` 分组的分组指标（E4 分题材实验用）。

    关键点：**只跑一次前向**，用同一批排名切片出各分组的指标。
    若每个题材各跑一次前向，12 个题材就要 12 倍时间，且各组的候选集
    与并列口径会因批大小不同而漂移。

    参数
    ----
    group_field  `EvalData` 上的字段名，默认 `"genres"`。
    min_samples  样本数少于此值的分组不下结论（返回但仍标注样本量）。

    返回
    ----
    ``{"overall": {...}, "groups": {组名: {...}}, "group_field": ...}``
    每个分组至少包含 `n_samples`，便于按 evaluation-plan 6.3 披露样本量。
    """
    t0 = time.time()
    ranks = collect_ranks(score_fn, data, batch_size=batch_size, device=device)

    overall = _metrics_dict(ranks, ks, {
        "label": data.label,
        "n_candidates": data.n_candidates,
        "n_negatives": data.n_negatives,
        "elapsed_sec": round(time.time() - t0, 3),
    })

    labels = getattr(data, group_field, None)
    groups: Dict[str, dict] = {}
    if labels is not None:
        labels = np.asarray(labels)
        for g in sorted(set(labels.tolist())):
            mask = labels == g
            cnt = int(mask.sum())
            if cnt < min_samples:
                continue
            groups[str(g)] = _metrics_dict(ranks[mask], ks, {
                "n_samples": cnt,
                "share": round(cnt / data.n_samples, 6) if data.n_samples else 0.0,
            })

    return {
        "overall": overall,
        "groups": groups,
        "group_field": group_field,
        "n_groups": len(groups),
        "n_samples": data.n_samples,
    }


def evaluate_cold_start(
    score_fn: ScoreFn,
    data: EvalData,
    ks: Sequence[int] = (5, 10),
    batch_size: int = 512,
    device=None,
) -> dict:
    """冷启动专项评估（E3）。

    与 `evaluate()` 唯一的区别是**在结果里标注这是冷启动子集**，
    因为 evaluation-plan 2.5 要求：冷启动指标必须与它的样本量、
    负样本池口径（限定新番）一起披露，否则读者无法判断这个数字
    是和什么比出来的。样本量本身由 `ranking_metrics` 的 `n_samples` 给出。
    """
    res = evaluate(score_fn, data, ks=ks, batch_size=batch_size, device=device)
    res["protocol"] = "cold_start_holdout"
    res["label"] = data.label if data.label != "eval" else "cold_start"
    return res
