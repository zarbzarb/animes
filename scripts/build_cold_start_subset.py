# -*- coding: utf-8 -*-
"""阶段 1-D：构建冷启动新番子集（E3 实验用）。

用法
----
    # 默认：按上映年份留出（推荐）——把 2021 年及以后的新番全部从训练集剥离
    python scripts/build_cold_start_subset.py

    # 调整年份阈值（年份越早 -> 样本越多，但剥离的训练数据也越多）
    python scripts/build_cold_start_subset.py --min-year 2019

    # 长尾口径（按交互数分位；样本少但几乎不动训练集）
    python scripts/build_cold_start_subset.py --mode longtail --quantile 0.20

依赖：先跑 scripts/preprocess.py

产物（data/processed/）
-----------------------
    cold_start_subset.pkl      新番清单 + 测试样本索引 + 训练剥离清单 + 统计
    cold_start_negatives.npy   (n_samples, 100) 冷启动专用负样本（池限定为新番）
    cold_start_report.json     完整统计报告

⚠️ 必须先读：这个数据集里【不存在】严格意义的物品冷启动
-----------------------------------------------------
物品过滤规则是「正样本交互数 >= 10」，而留一法只把每个用户的最后 2 条划给 val/test，
因此**推荐池内每个物品在训练集中至少出现 6 次**（实测最小值 = 6，
训练集中出现次数 <= 5 的物品数为 0）。也就是说，池内没有任何“没见过”的物品。

所以「物品交互数 < 10 即新番」会得到空集，必须改用**模拟**口径。两种可选：

| 口径 | 定义 | 优点 | 代价 |
|---|---|---|---|
| `holdout`（默认） | 年份 >= min_year 的动漫全部从训练集剥离 | 真正「训练集不可见」，样本量大 | 剥离部分训练交互 |
| `longtail` | 正样本交互数 <= P{q} 的长尾物品 | 几乎不动训练集 | 测试样本很少 |

以 2026-09 实测的数据为例：

| 口径 | 物品数 | 冷启动测试样本 | 剥离训练交互 | 占训练总量 |
|---|---|---|---|---|
| 年份 >= 2018 | 4,657 | 466,952 | 34,871,712 | 31.97% |
| 年份 >= 2019 | 3,997 | 380,670 | 27,664,655 | 25.36% |
| **年份 >= 2021**（默认） | **2,783** | **209,912** | **15,344,636** | **14.07%** |
| 长尾后 20% | 3,200 | 446 | 67,199 | 0.06% |

年份口径剥离 14% 的训练交互，含义是「这些新番在训练时还不存在」——
这正是真实的冷启动场景，且模型仍有 86% 的学习信号，是可接受的协议。
若年份阈值调得过低（如 >= 2018 剥离 32%），训练信号损失偏大，需在论文中说明。

为什么负样本池必须限定为新番
----------------------------
若负样本来自热门物品，模型可凭「训练中学到的热度先验」区分正负，
而不是靠内容向量，内容融合的贡献会被系统性高估。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="构建冷启动新番子集")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "data.yaml"))
    ap.add_argument("--mode", choices=["holdout", "longtail"], default="holdout")
    ap.add_argument("--min-year", type=int, default=None, help="holdout 口径的年份阈值")
    ap.add_argument("--quantile", type=float, default=None, help="longtail 口径的分位")
    ap.add_argument("--n-negatives", type=int, default=100)
    ap.add_argument("--seed", type=int, default=98765)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    proc = os.path.join(ROOT, cfg["output"]["processed_dir"].lstrip("./"))
    cs_cfg = dict(cfg.get("cold_start") or {})
    min_year = args.min_year if args.min_year is not None else int(cs_cfg.get("min_year", 2021))
    quantile = args.quantile if args.quantile is not None else float(cs_cfg.get("quantile", 0.10))
    target_samples = int(cs_cfg.get("min_test_samples", 5000))

    item_stats = pd.read_parquet(os.path.join(proc, "item_stats.parquet"))
    with open(os.path.join(proc, cfg["output"]["sequence_file"]), "rb") as f:
        ds = pickle.load(f)
    train, test, smap = ds["train"], ds["test"], ds["smap"]
    n_users, n_items = len(train), len(smap)
    inv_smap = {int(v): int(k) for k, v in smap.items()}

    # ---------- 0. 训练集出现次数（判定“是否真的没见过”）----------
    log("统计各物品在训练集中的出现次数 ...")
    tot_train = sum(len(v) for v in train.values())
    flat = np.fromiter(itertools.chain.from_iterable(train.values()),
                       dtype=np.int32, count=tot_train)
    cnt_train = np.bincount(flat, minlength=n_items + 1)
    log(f"  训练交互总数 {tot_train:,}；物品训练出现次数 min={cnt_train[1:].min()} "
        f"max={cnt_train[1:].max()}")

    # ---------- 1. 新番判定 ----------
    idx_all = np.arange(1, n_items + 1)
    orig_of = np.array([inv_smap[int(i)] for i in idx_all])
    year_of = dict(zip(item_stats["anime_id"], item_stats["year"].fillna(0)))
    years = np.array([year_of.get(int(a), 0) for a in orig_of])
    npos = np.array([cnt_train[int(i)] for i in idx_all], dtype=np.int64)  # 池内出现次数

    if args.mode == "holdout":
        sel = years >= min_year
        rule = f"holdout：上映年份 >= {min_year} 的动漫整体留出（模拟新番上线）"
    else:
        thr = int(np.percentile(
            item_stats["n_positive"].to_numpy(), quantile * 100))
        sel = np.array([item_stats.set_index("anime_id").loc[a, "n_positive"]
                        <= thr for a in orig_of])
        rule = f"longtail：正样本交互数 <= P{int(quantile * 100)} = {thr}"

    new_idx = idx_all[sel]
    new_items = [int(a) for a in orig_of[sel]]
    new_set = set(new_idx.tolist())
    strip_ix = int(cnt_train[new_idx].sum())
    log(f"新番判定规则: {rule}")
    log(f"  新番数量: {len(new_items):,} 个；剥离训练交互 {strip_ix:,} "
        f"({strip_ix / tot_train:.2%} of train)")

    # ---------- 2. 冷启动测试样本 ----------
    log("抽取冷启动测试样本（test 目标物品属于新番）...")
    tgt_idx = np.array([int(test[i][0]) for i in range(n_users) if len(test[i]) == 1],
                       dtype=np.int64)
    tgt_user = np.array([i for i in range(n_users) if len(test[i]) == 1], dtype=np.int64)
    hit = np.isin(tgt_idx, new_idx)
    sample_users = tgt_user[hit]
    sample_tgts = tgt_idx[hit]
    log(f"  冷启动测试样本: {len(sample_users):,} 条（目标 >= {target_samples}）")

    # ---------- 3. 训练剥离清单（numpy 向量化，避免 1 亿次 Python 循环）----------
    log("统计训练集中需剥离的交互 ...")
    lens = np.fromiter((len(train[i]) for i in range(n_users)), dtype=np.int64,
                       count=n_users)
    offsets = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    in_new = np.isin(flat, new_idx)
    if lens.min() > 0:
        per_user = np.add.reduceat(in_new, offsets[:-1])
    else:
        per_user = np.zeros(n_users, dtype=np.int64)
        nz = lens > 0
        per_user[nz] = np.add.reduceat(in_new, offsets[:-1][nz])
    n_strip_pairs = int(per_user.sum())
    strip_users = np.nonzero(per_user)[0].tolist()
    log(f"  需剥离交互 {n_strip_pairs:,} 条，涉及 {len(strip_users):,} 个用户")

    # ---------- 4. 冷启动专用负样本（池 = 新番集合）----------
    log("生成冷启动专用负样本（池 = 新番集合，同分布）...")
    rng = np.random.default_rng(args.seed)
    pool_idx = new_idx.copy()
    negs = np.full((len(sample_users), args.n_negatives),
                   int(pool_idx[0]) if len(pool_idx) else 0, dtype=np.int32)
    skipped = 0
    if len(pool_idx) > args.n_negatives:
        # 代数戳：避免每条样本都重置 seen 掩码
        stamp = np.zeros(n_items + 1, dtype=np.int32)
        for r in range(len(sample_users)):
            g = r + 1
            ui, pos = int(sample_users[r]), int(sample_tgts[r])
            tr = train[ui]
            if tr:
                stamp[np.fromiter(tr, dtype=np.int32, count=len(tr))] = g
            stamp[pos] = g
            cand = pool_idx[stamp[pool_idx] != g]
            if len(cand) >= args.n_negatives:
                negs[r] = rng.choice(cand, size=args.n_negatives, replace=False)
            else:
                skipped += 1
                negs[r] = (rng.choice(cand, size=args.n_negatives, replace=True)
                           if len(cand) else 0)
    else:
        log(f"  ⚠ 新番池({len(pool_idx)}) <= 负样本数({args.n_negatives})，无法取样")

    # ---------- 5. 统计 ----------
    from collections import Counter
    genre_of = dict(zip(item_stats["anime_id"], item_stats["primary_genre"]))
    gc = Counter(genre_of.get(a) for a in new_items)
    yrs = years[sel]
    report = dict(
        mode=args.mode,
        rule=rule,
        min_year=min_year if args.mode == "holdout" else None,
        quantile=quantile if args.mode == "longtail" else None,
        n_new_items=len(new_items),
        n_test_samples=int(len(sample_users)),
        target_test_samples=target_samples,
        meets_target=len(sample_users) >= target_samples,
        n_strip_interactions=n_strip_pairs,
        strip_ratio_of_train=round(n_strip_pairs / max(tot_train, 1), 5),
        n_affected_users=len(strip_users),
        min_train_occurrence_in_pool=int(cnt_train[1:].min()),
        new_item_genre_distribution=dict(gc.most_common()),
        new_item_year_range=[int(yrs.min()), int(yrs.max())] if len(yrs) else None,
        new_item_year_median=float(np.median(yrs)) if len(yrs) else None,
        negative_pool="新番集合（同分布）",
        n_negatives_per_sample=args.n_negatives,
        seed=args.seed,
        warnings=[],
    )
    if not report["meets_target"]:
        report["warnings"].append(
            f"测试样本 {len(sample_users):,} < 目标 {target_samples:,}；"
            f"holdout 口径可下调 --min-year，longtail 口径可上调 --quantile")
    if skipped:
        report["warnings"].append(
            f"{skipped:,} 条样本可用负样本不足 {args.n_negatives}，已放回重复采样补齐")
    report["warnings"].append(
        f"⚠ 推荐池内每个物品在训练集中至少出现 {report['min_train_occurrence_in_pool']} 次，"
        f"本数据集不存在严格意义的物品冷启动；当前为 holdout 模拟口径")

    # ---------- 6. 落盘 ----------
    out = dict(
        mode=args.mode,
        new_item_ids=sorted(new_items),
        new_item_pool_indices=sorted(int(x) for x in new_idx),
        new_item_train_occurrences={int(a): int(cnt_train[int(i)])
                                    for a, i in zip(new_items, new_idx)},
        test_sample_user_indices=sample_users.tolist(),
        test_sample_target_indices=sample_tgts.tolist(),
        strip_from_train_item_indices=sorted(int(x) for x in new_idx),
        strip_user_indices=strip_users,
        n_strip_interactions=n_strip_pairs,
        negatives=negs,
        report=report,
    )
    with open(os.path.join(proc, cs_cfg.get("output", "cold_start_subset.pkl")), "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(proc, "cold_start_negatives.npy"), negs)
    with open(os.path.join(proc, "cold_start_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log("=" * 62)
    log(f"完成: 新番 {report['n_new_items']:,} 个 / 测试样本 {report['n_test_samples']:,} 条 / "
        f"剥离交互 {report['n_strip_interactions']:,} ({report['strip_ratio_of_train']:.2%})")
    for w in report["warnings"]:
        log("  " + w)
    log("新番题材分布（前 8）:")
    for g, nn in list(report["new_item_genre_distribution"].items())[:8]:
        log(f"    {g}: {nn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
