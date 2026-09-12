# -*- coding: utf-8 -*-
"""内容融合评估（M2.6）：复用已训练的行为基座，验证融合的增益。

流程
----
1. 加载行为基座 checkpoint（默认 debug 档 lr 扫描最优，val ndcg@10=0.7871）；
2. **val 上扫融合权重** `w_content ∈ {0, 0.1, ..., 0.5}`（w=0 即纯行为基线，
   应精确复现训练日志里的指标，作为脚本正确性的自检）；
3. 选 val 最优权重，在 **test** 上对比 w=0 与最优 w（权重在 val 选、
   test 只做确认，避免用测试集调参）；
4. 冷启动快检：新番 holdout 样本（year>=2021）上对比 7:3 与 5:5，
   验证「内容越冷越重要」的假设。

产物：logs/content_fusion_eval.json + 控制台表格。

运行
----
    python scripts/eval_content_fusion.py                  # 默认 debug 档
    python scripts/eval_content_fusion.py --ckpt <path.pt> # 指定其他基座
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.checkpoint.io import load_model_from_checkpoint  # noqa: E402
from models.content_encoder.fusion import (  # noqa: E402
    COLD_START_THRESHOLD,
    ContentFusedModel,
    ContentScorer,
    load_content_matrix,
)
from models.data.user_subset import subset_order  # noqa: E402
from models.eval.evaluator import (  # noqa: E402
    build_eval_data,
    drop_leaked_samples,
    evaluate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_content_fusion")

SUBSET_SEED = 42
DEFAULT_CKPT = os.path.join(
    ROOT, "data", "checkpoints",
    "multi_interest_content_v1_lrsweep_lr0.001_w0.0_debug_seed42_best.pt")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="行为基座 checkpoint")
    ap.add_argument("--scale", default="debug")
    ap.add_argument("--scale-config", default=os.path.join(ROOT, "configs", "scale.yaml"))
    ap.add_argument("--seed", type=int, default=SUBSET_SEED, help="用户抽样种子")
    ap.add_argument("--w-grid", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    help="val 上扫描的内容权重网格")
    ap.add_argument("--cold-users", type=int, default=2000,
                    help="冷启动快检最多取多少样本")
    ap.add_argument("--device", default=None)
    return ap.parse_args(argv)


def resolve_scale(scale_yaml: dict, name: str) -> dict:
    scales = scale_yaml.get("scales") or {}
    if name not in scales:
        raise SystemExit(f"未知档位 {name!r}，可选：{list(scales)}")
    return scales[name]


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---------- 数据集 ----------
    seq_path = os.path.join(ROOT, "data", "processed", "seq_dataset.pkl")
    with open(seq_path, "rb") as f:
        ds = pickle.load(f)
    train_seqs, val_seqs, test_seqs = ds["train"], ds["val"], ds["test"]
    n_users, n_items = len(train_seqs), len(ds["smap"])
    logger.info("数据集：用户 %s / 物品 %s", f"{n_users:,}", f"{n_items:,}")

    spec = resolve_scale(
        yaml.safe_load(open(args.scale_config, encoding="utf-8")), args.scale)
    uids = np.empty(n_users, dtype=np.int64)
    for uid, row in ds["umap"].items():
        uids[int(row)] = int(uid)
    order = subset_order(uids, seed=args.seed)
    n_train = max(1, int(round(n_users * float(spec["user_ratio"]))))
    train_rows = np.sort(order[:n_train].astype(np.int64))
    train_row_set = np.zeros(n_users, dtype=bool)
    train_row_set[train_rows] = True
    n_eval_cap = max(1, int(round(train_rows.size * float(spec["eval_user_ratio"]))))
    eval_rows = np.sort(order[train_row_set[order]][:n_eval_cap].astype(np.int64))

    # ---------- 评估数据（val / test）----------
    max_seq_len = 50
    val_items = np.array([int(val_seqs[int(r)][0]) for r in eval_rows], dtype=np.int64)
    test_items = np.array([int(test_seqs[int(r)][0]) for r in eval_rows], dtype=np.int64)
    t0 = time.time()
    data_val = drop_leaked_samples(build_eval_data(
        train_seqs, eval_rows, n_items, split="val", val_items=val_items,
        max_seq_len=max_seq_len, n_negatives=100, seed=42, label="val"))[0]
    data_test = drop_leaked_samples(build_eval_data(
        train_seqs, eval_rows, n_items, split="test", val_items=val_items,
        test_items=test_items, max_seq_len=max_seq_len,
        n_negatives=100, seed=42, label="test"))[0]
    logger.info("val %s / test %s 样本（构建 %.1fs，含泄漏剔除）",
                f"{data_val.n_samples:,}", f"{data_test.n_samples:,}",
                time.time() - t0)

    # ---------- 模型 + 融合 ----------
    model, meta = load_model_from_checkpoint(args.ckpt)
    model = model.to(device).eval()
    logger.info("基座：%s（%s 参数，val best %.4f）",
                meta.get("arch"), f"{sum(p.numel() for p in model.parameters()):,}",
                float(meta.get("best_metric", float("nan"))))
    scorer = ContentScorer(load_content_matrix(
        os.path.join(ROOT, "data", "features", "content_vec_512.npy"),
        n_items, device=device)).to(device)
    fused = ContentFusedModel(model, scorer).to(device)

    def run(data, w_content, split):
        fused.w_behavior = round(1.0 - w_content, 6)
        fused.w_content = w_content
        res = evaluate(fused.score, data, ks=(5, 10), device=device)
        res["w_content"] = w_content
        res["split"] = split
        return res

    # ---------- 1) val 扫权重 ----------
    val_results = [run(data_val, w, "val") for w in args.w_grid]
    best = max(val_results, key=lambda r: r["ndcg@10"])
    logger.info("val 扫描完成，最优 w_content=%.2f (ndcg@10=%.4f)",
                best["w_content"], best["ndcg@10"])

    # ---------- 2) test 确认 ----------
    w_best = float(best["w_content"])
    test_base = run(data_test, 0.0, "test")
    test_best = run(data_test, w_best, "test")

    # ---------- 3) 冷启动快检（新番 holdout，7:3 vs 5:5）----------
    cold_result = None
    cs_path = os.path.join(ROOT, "data", "processed", "cold_start_subset.pkl")
    if os.path.exists(cs_path):
        with open(cs_path, "rb") as f:
            cs = pickle.load(f)
        cs_rows_all = np.asarray(cs["test_sample_user_indices"], dtype=np.int64)
        cs_pos_all = np.asarray(cs["test_sample_target_indices"], dtype=np.int64)
        keep = train_row_set[cs_rows_all]
        if keep.sum() > args.cold_users:
            # 只取前 cold_users 条，但保持行号升序（评估切片要求顺序稳定）
            idx = np.sort(np.nonzero(keep)[0][:args.cold_users])
        else:
            idx = np.nonzero(keep)[0]
        cs_rows = cs_rows_all[idx]
        cs_pos = cs_pos_all[idx]
        cs_rows = np.sort(cs_rows) if False else cs_rows
        if cs_rows.size >= 100:
            new_pool = np.array(cs["new_item_pool_indices"], dtype=np.int64)
            data_cs = drop_leaked_samples(build_eval_data(
                train_seqs, cs_rows, n_items, split="val", val_items=cs_pos,
                max_seq_len=max_seq_len, n_negatives=100, seed=42,
                strip_items=new_pool, pool=new_pool,
                label="cold_start_holdout"))[0]
            # 冷启动口径：普通 7:3 对照、5:5 候选口径、纯内容 w=1.0 探针
            # （纯内容回答"内容通路本身有没有信号"，是解读融合失/增益的前提）
            cold_rows = {}
            for wb, wc, name in [(0.7, 0.3, "w_73"), (0.5, 0.5, "w_55"),
                                 (0.0, 1.0, "content_only")]:
                fused.w_behavior, fused.w_content, fused.w_content_cold = wb, wc, wc
                cold_rows[name] = evaluate(fused.score, data_cs, ks=(5, 10),
                                           device=device)
            cold_result = {
                "n_samples": data_cs.n_samples,
                **cold_rows,
                "threshold": COLD_START_THRESHOLD,
            }
            logger.info("冷启动 holdout：%s 样本 | 7:3 ndcg@10=%.4f | "
                        "5:5 %.4f | 纯内容 %.4f",
                        f"{data_cs.n_samples:,}",
                        cold_rows["w_73"]["ndcg@10"],
                        cold_rows["w_55"]["ndcg@10"],
                        cold_rows["content_only"]["ndcg@10"])
        else:
            logger.warning("冷启动样本在本档位用户内不足 100 条，跳过快检")

    # ---------- 汇总落盘 ----------
    out = {
        "ckpt": os.path.relpath(args.ckpt, ROOT),
        "scale": args.scale, "seed": args.seed,
        "n_eval_users": int(eval_rows.size),
        "val_sweep": val_results, "best_w_content": w_best,
        "test_baseline_w0": test_base, "test_best": test_best,
        "cold_start": cold_result,
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    out_path = os.path.join(ROOT, "logs", "content_fusion_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)

    print(f"\n{'w_content':>9} | {'val hr@10':>9} {'val ndcg@10':>11} | "
          f"{'test hr@10':>10} {'test ndcg@10':>12}")
    for r in val_results:
        tw = test_base if r["w_content"] == 0.0 else (
            test_best if r["w_content"] == w_best else None)
        print(f"{r['w_content']:>9.2f} | {r['hr@10']:>9.4f} {r['ndcg@10']:>11.4f} | "
              + (f"{tw['hr@10']:>10.4f} {tw['ndcg@10']:>12.4f}" if tw else "-"))
    logger.info("结果已写入 %s（总耗时 %.1fs）",
                os.path.relpath(out_path, ROOT), time.time() - t_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
