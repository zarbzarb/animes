# -*- coding: utf-8 -*-
"""评估协议可信度诊断：随机基线 / 热度基线 / 模型对比。

为什么需要这个脚本（这是一次真实发现的固化）
--------------------------------------------
M2.4 首次训出 smoke 档模型时，`val ndcg@10 = 0.716`。这个数字**好得可疑**
——随机打分器只有 0.045，公开数据集上 SASRec 通常也就 0.05~0.20。
于是做了三组对照，结果是：

| 打分器 | hr@10 | ndcg@10 |
|---|---|---|
| 随机 | 0.074 | 0.030 |
| **仅按物品热度** | **0.866** | **0.607** |
| smoke 档模型 | 0.920 | 0.716 |

**结论：本数据集在「1 正 + 100 均匀负采样」协议下，指标主要由热度先验贡献，
而不是序列建模能力。** 成因是结构性的，不是代码 bug：

* 正样本必然是「用户交互过、且交互数 ≥ 10 才进池」的物品，天然偏热门；
* 负样本从全池均匀抽，而全池热度极度长尾
  （中位 288 次 vs 均值 6,687 次）。

实测正样本热度均值是负样本的 **24.7 倍**，所以「给热门打高分」这一条
零信息的策略就能把正样本顶到前 10。

这条诊断的三个用途
------------------
1. **防止自我欺骗**：看到 0.716 时能立刻知道它有多少来自真实建模。
   论文里若只报 0.716 而不报热度基线，属于选择性报告。
2. **给 E1 提供必要对照**：`configs/experiment.yaml` 的 E1 必须包含
   popularity 基线（SASRec 原论文表 3 也有这一行），否则无法论证
   "序列建模带来了增益"。
3. **可复现**：本脚本每次运行都重新算这三组数字，不依赖任何报告。

关于热度口径
------------
热度取自 `data/processed/item_stats.parquet` 的 `n_positive`，它是**全量**
（含 val/test）的正样本交互数。作为基线这会略微乐观 —— 但 val/test 只占
全部交互的约 1.8%，且对物品热度**排序**几乎无影响，故不额外区分。
正式论文数字若要更严格，可在 M2.7 用训练集频次重算（见 `itemcf.py` 的做法）。

用法
----
    python scripts/diagnose_popularity_bias.py --scale smoke
    python scripts/diagnose_popularity_bias.py --scale dev \
        --ckpt data/checkpoints/xxx_seed42_best.pt

退出码
------
    0 = 诊断完成；1 = 前置条件缺失（缺数据/权重）
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from typing import Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.data.negatives import DEFAULT_NEG_SEED  # noqa: E402
from models.data.user_subset import subset_order  # noqa: E402
from models.eval.evaluator import build_eval_data, drop_leaked_samples, evaluate  # noqa: E402

# 与 scripts/train.py 保持一致：抽样种子固定 42，不随训练种子变化
SUBSET_SEED = 42

# 随机打分器的理论值（1 正 + 100 负，共 101 候选）
THEORY_HR10 = 10 / 101
THEORY_NDCG10 = sum(1.0 / np.log2(r + 1) for r in range(1, 11)) / 101


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="评估协议可信度诊断")
    ap.add_argument("--scale-config", default="configs/scale.yaml")
    ap.add_argument("--scale", default="smoke", help="用哪个档位的用户规模")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--ckpt", default=None, help="可选：待评估的 checkpoint 路径")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0, help="随机打分器的种子")
    ap.add_argument("--json", default=None, help="结果落盘路径（默认 logs/ 下）")
    return ap.parse_args()


def to_abs(p: str) -> str:
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(ROOT, p))


def main() -> int:
    args = parse_args()

    seq_path = os.path.join(ROOT, "data", "processed", "seq_dataset.pkl")
    stats_path = os.path.join(ROOT, "data", "processed", "item_stats.parquet")
    if not os.path.exists(seq_path):
        log(f"缺少 {seq_path}，请先跑 scripts/run_stage1.py")
        return 1

    log(f"加载 {os.path.relpath(seq_path, ROOT)} ...")
    with open(seq_path, "rb") as f:
        ds = pickle.load(f)
    train, val, test = ds["train"], ds["val"], ds["test"]
    n_users, n_items = len(train), len(ds["smap"])

    # ---------- 按档位抽用户（与 scripts/train.py 完全同一套口径）----------
    import yaml
    with open(to_abs(args.scale_config), encoding="utf-8") as f:
        scale_yaml = yaml.safe_load(f)
    spec = (scale_yaml.get("scales") or {})[args.scale]

    uids = np.empty(n_users, dtype=np.int64)
    for uid, row in ds["umap"].items():
        uids[int(row)] = int(uid)
    order = subset_order(uids, seed=SUBSET_SEED)

    n_keep = max(1, int(round(n_users * float(spec["user_ratio"]))))
    train_rows = np.sort(order[:n_keep].astype(np.int64))
    n_eval_cap = max(1, int(round(train_rows.size * float(spec["eval_user_ratio"]))))
    in_train = np.zeros(n_users, dtype=bool)
    in_train[train_rows] = True
    eval_rows = np.sort(order[in_train[order]][:n_eval_cap].astype(np.int64))
    log(f"档位 {args.scale}：训练用户 {train_rows.size:,} / 评估用户 {eval_rows.size:,}")

    # ---------- 评估集 ----------
    val_items = np.array([int(val[int(r)][0]) for r in eval_rows], dtype=np.int64)
    test_items = (np.array([int(test[int(r)][0]) for r in eval_rows], dtype=np.int64)
                  if args.split == "test" else None)
    data = build_eval_data(
        train, eval_rows, n_items, split=args.split,
        val_items=val_items, test_items=test_items, max_seq_len=50,
        n_negatives=100, seed=DEFAULT_NEG_SEED, label=args.split)
    data, n_leaked = drop_leaked_samples(data)
    log(f"评估集：{data.n_samples:,} 样本 × {data.n_candidates} 候选"
        f"（剔除答案泄漏 {n_leaked}）")

    out = {
        "scale": args.scale, "split": args.split,
        "n_train_users": int(train_rows.size),
        "n_eval_users": int(eval_rows.size),
        "n_samples": int(data.n_samples),
        "n_candidates": int(data.n_candidates),
        "n_leaked_dropped": int(n_leaked),
        "theory": {"hr@10": THEORY_HR10, "ndcg@10": THEORY_NDCG10},
    }

    # ---------- A 随机打分器 ----------
    rng = np.random.default_rng(args.seed)
    cache: dict = {}

    def rand_score(input_ids, candidate_ids):
        key = (tuple(candidate_ids.shape), str(candidate_ids.device))
        if key not in cache or cache[key].device != candidate_ids.device:
            cache[key] = torch.as_tensor(
                rng.random(tuple(candidate_ids.shape)),
                dtype=torch.float32, device=candidate_ids.device)
        return cache[key]

    ra = evaluate(rand_score, data, ks=(5, 10), batch_size=args.batch_size)
    out["random"] = {"hr@10": ra["hr@10"], "ndcg@10": ra["ndcg@10"]}
    log(f"A 随机打分器      hr@10={ra['hr@10']:.4f}  ndcg@10={ra['ndcg@10']:.4f}"
        f"  （理论 {THEORY_HR10:.4f} / {THEORY_NDCG10:.4f}）")

    # ---------- B 热度基线 ----------
    if not os.path.exists(stats_path):
        log(f"缺少 {stats_path}，跳过热度基线")
    else:
        import pandas as pd
        stats = pd.read_parquet(stats_path)
        pool_index = {int(a): int(i) for a, i in ds["smap"].items()}
        pop = np.zeros(n_items + 1, dtype=np.float64)
        for aid, n_pos in zip(stats["anime_id"].to_numpy(),
                              stats["n_positive"].to_numpy()):
            pi = pool_index.get(int(aid))
            if pi is not None:
                pop[pi] = float(n_pos)

        def pop_score(input_ids, candidate_ids):
            c = candidate_ids.detach().cpu().numpy()
            return torch.as_tensor(pop[c], dtype=torch.float32,
                                   device=candidate_ids.device)

        rb = evaluate(pop_score, data, ks=(5, 10), batch_size=args.batch_size)
        out["popularity"] = {"hr@10": rb["hr@10"], "ndcg@10": rb["ndcg@10"]}

        pos_pop, neg_pop = pop[data.positives], pop[data.negatives]
        out["popularity_stats"] = {
            "pos_mean": float(pos_pop.mean()),
            "pos_median": float(np.median(pos_pop)),
            "neg_mean": float(neg_pop.mean()),
            "neg_median": float(np.median(neg_pop)),
            "ratio_mean": float(pos_pop.mean() / max(1e-9, neg_pop.mean())),
        }
        log(f"B 仅按热度打分    hr@10={rb['hr@10']:.4f}  ndcg@10={rb['ndcg@10']:.4f}")
        log(f"   正样本热度 均值 {pos_pop.mean():,.0f} / 中位 {np.median(pos_pop):,.0f}")
        log(f"   负样本热度 均值 {neg_pop.mean():,.0f} / 中位 {np.median(neg_pop):,.0f}")
        log(f"   正负热度倍数 = {pos_pop.mean() / max(1e-9, neg_pop.mean()):.1f}x"
            "   ← 这个倍数越高，热度先验越容易"蒙对"")
        out["n_distinct_positives"] = int(np.unique(data.positives).size)

    # ---------- C 模型 ----------
    if args.ckpt:
        from models.checkpoint.io import load_model_from_checkpoint
        ckpt = to_abs(args.ckpt)
        if not os.path.exists(ckpt):
            log(f"checkpoint 不存在：{ckpt}")
            return 1
        model, meta = load_model_from_checkpoint(ckpt, map_location="cpu")
        model.eval()
        rc = evaluate(model.score, data, ks=(5, 10), batch_size=args.batch_size)
        out["model"] = {"hr@10": rc["hr@10"], "ndcg@10": rc["ndcg@10"],
                        "checkpoint": os.path.relpath(ckpt, ROOT),
                        "recorded_best_metric": meta.get("best_metric")}
        log(f"C 模型            hr@10={rc['hr@10']:.4f}  ndcg@10={rc['ndcg@10']:.4f}")
        if "popularity" in out:
            out["model_gain_over_popularity"] = {
                "hr@10": rc["hr@10"] / max(1e-9, out["popularity"]["hr@10"]),
                "ndcg@10": rc["ndcg@10"] / max(1e-9, out["popularity"]["ndcg@10"]),
            }
            log(f"   相对热度基线的增幅："
                f"hr@10 {out['model_gain_over_popularity']['hr@10']:.2f}x，"
                f"ndcg@10 {out['model_gain_over_popularity']['ndcg@10']:.2f}x")

    # ---------- 落盘 ----------
    json_path = to_abs(args.json or os.path.join(
        "logs", f"popularity_bias_{args.scale}_{args.split}.json"))
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"结果已保存：{os.path.relpath(json_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
