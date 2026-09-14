# -*- coding: utf-8 -*-
"""对比基线评测入口 —— `scripts/run_baselines.py`

用法
----
    # 零训练成本的两个基线（秒级出结果，不需要 GPU）
    python scripts/run_baselines.py --baseline popularity --scale main
    python scripts/run_baselines.py --baseline itemcf     --scale main

    # GRU4Rec 需要训练（与本文模型共用 fit() + 同一早停口径）
    python scripts/run_baselines.py --baseline gru4rec --scale main --seed 42

    # 三个一起跑
    python scripts/run_baselines.py --baseline all --scale dev

    # 快速自检（只评估前 200 个用户）
    python scripts/run_baselines.py --baseline all --scale smoke --eval-users 200

公平性控制（这是本脚本存在的全部意义）
------------------------------------
`docs/evaluation-plan.md` §4.1 要求全部模型共用同一划分 / 同一负样本池 /
同一评估协议 / 同一早停策略。本脚本的做法是**不自己实现任何一环**：

| 环节 | 复用的唯一实现 |
|---|---|
| 用户抽样 | `models/data/user_subset.py::resolve_scale_users` |
| 输入序列拼接 | `models/sasrec/dataset.py::build_eval_inputs`（经 `build_eval_data`） |
| 负样本 | `models/data/negatives.py::sample_negatives` |
| 指标 | `models/eval/metrics.py` |
| 串起来跑前向 | `models/eval/evaluator.py::evaluate` |
| 训练循环与早停 | `models/sasrec/train.py::fit` |

⚠️ 本脚本**不重算**「热度」的任何公式：它调用 `models/baselines/popularity.py`。
`scripts/diagnose_popularity_bias.py` 里的热度是另一套口径（全量 `n_positive`），
那个脚本是"协议可信度自查"，本脚本出的是**论文正式数字**（只用训练集频次）。

产物
----
* `logs/baseline_{name}_{scale}_{split}_seed{seed}.json` —— 指标 + 诊断量 + 结构快照
* `data/checkpoints/gru4rec_{scale}_seed{n}_best.pt` —— 仅 GRU4Rec（要训练的才有权重）

⚠️ 工作目录无关：只用 `ROOT = dirname(dirname(abspath(__file__)))` 定位项目根，
不出现 `os.getcwd()`（`docs/dev-conventions.md` 3.0）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import replace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.baselines.gru4rec import GRU4Rec, GRU4RecConfig  # noqa: E402
from models.baselines.itemcf import ItemCFScorer, build_itemcf_similarity  # noqa: E402
from models.baselines.popularity import (  # noqa: E402
    PopularityScorer,
    count_train_frequency,
    normalize_frequency,
)
from models.checkpoint.io import checkpoint_name, save_checkpoint  # noqa: E402
from models.data.negatives import DEFAULT_NEG_SEED  # noqa: E402
from models.data.user_subset import resolve_scale_users  # noqa: E402
from models.eval.evaluator import (  # noqa: E402
    build_eval_data,
    drop_leaked_samples,
    evaluate,
)
from models.sasrec.config import OptimConfig, TrainConfig  # noqa: E402
from models.sasrec.dataset import SlidingWindowDataset, WindowBatchIterator  # noqa: E402
from models.sasrec.train import fit, set_seed  # noqa: E402

# 用户抽样种子固定 42，**不随训练种子变化**（与 scripts/train.py 同一条约定：
# 否则换种子时评估子集也跟着换，"3 个种子的均值"就不是同一个测试集上的重跑）。
SUBSET_SEED = 42

# 需要训练的基线（其余是"给定数据直接算分数"，没有权重可存）
TRAINED_BASELINES = ("gru4rec",)
BASELINE_CHOICES = ("popularity", "itemcf", "gru4rec")

logger = logging.getLogger("run_baselines")


# =====================================================================
# 工具（与 scripts/train.py 同一约定：路径按项目根锚定）
# =====================================================================
def to_abs(p: str) -> str:
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(ROOT, p))


def load_yaml(path: str) -> dict:
    import yaml
    with open(to_abs(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="对比基线评测（M2.7）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--baseline", default="all",
                    choices=list(BASELINE_CHOICES) + ["all"])
    ap.add_argument("--config", default="configs/model.yaml",
                    help="模型/训练配置（基线取其中 model 段与 train 段）")
    ap.add_argument("--scale-config", default="configs/scale.yaml")
    ap.add_argument("--scale", default=None, help="档位；默认取 model.yaml 的 scale")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="评估用验证集还是测试集（E1 正式数字用 val 早停 + test 报数）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None, help="覆盖档位 max_epochs（仅 gru4rec）")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-users", type=int, default=None,
                    help="只评估前 N 个评估用户（快速自检）")
    ap.add_argument("--eval-batch-size", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=200, help="ItemCF 邻居数（论文口径 200）")
    ap.add_argument("--itemcf-max-train-users", type=int, default=None,
                    help="只用前 N 个训练用户建相似度矩阵（默认全用；仅用于省时探查）")
    ap.add_argument("--no-itemcf-tiebreak-check", action="store_true",
                    help="不再额外跑一次「全零行用热度填充」的 ItemCF，"
                         "只报全零行占比（默认两次都跑，用来量化并列口径的假象）")
    ap.add_argument("--device", default=None, help="cuda / cpu")
    ap.add_argument("--tag", default=None, help="产物文件名后缀标记")
    ap.add_argument("--out-dir", default=None, help="checkpoint 目录")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--no-save", action="store_true", help="不写 checkpoint / 报告")
    return ap.parse_args(argv)


# =====================================================================
# 评估数据（与 scripts/train.py 完全同一套口径）
# =====================================================================
def build_shared_eval_data(ds: dict, spec: dict, args, max_seq_len: int) -> dict:
    """构造三件事：训练用户行号、评估用户行号、评估集。

    刻意**与 `scripts/train.py` 用同一套函数**（`resolve_scale_users` +
    `build_eval_data`），而不是各写一遍。这类"看起来一样的抽样代码"
    一旦分叉，会产出一张每一行都来自不同用户子集的 E1 表 ——
    数字都算得出来，但彼此不可比。
    """
    train_seqs, val_seqs, test_seqs = ds["train"], ds["val"], ds["test"]
    n_users = len(train_seqs)
    n_items = len(ds["smap"])

    uids = np.empty(n_users, dtype=np.int64)
    for uid, row in ds["umap"].items():
        uids[int(row)] = int(uid)

    train_rows, eval_rows = resolve_scale_users(
        uids, spec["user_ratio"], spec["eval_user_ratio"], seed=SUBSET_SEED)
    if args.eval_users is not None:
        eval_rows = eval_rows[:int(args.eval_users)]

    val_items = np.array([int(val_seqs[int(r)][0]) for r in eval_rows], dtype=np.int64)
    test_items = (np.array([int(test_seqs[int(r)][0]) for r in eval_rows],
                           dtype=np.int64) if args.split == "test" else None)

    data = build_eval_data(
        train_seqs, eval_rows, n_items, split=args.split,
        val_items=val_items, test_items=test_items, max_seq_len=max_seq_len,
        n_negatives=100, seed=DEFAULT_NEG_SEED, label=args.split,
    )
    data, n_leaked = drop_leaked_samples(data)

    logger.info("档位 %s（%s）：训练用户 %s / 评估用户 %s；评估集 %s 样本 × %d 候选",
                spec["name"], args.split, f"{train_rows.size:,}",
                f"{eval_rows.size:,}", f"{data.n_samples:,}", data.n_candidates)
    if n_leaked:
        logger.warning("  剔除答案泄漏 %s 条（%.3f%%）—— 论文须注明",
                       f"{n_leaked:,}",
                       100.0 * n_leaked / max(1, n_leaked + data.n_samples))

    return {
        "train_seqs": train_seqs, "val_seqs": val_seqs, "test_seqs": test_seqs,
        "n_items": int(n_items), "train_rows": train_rows, "eval_rows": eval_rows,
        "eval_data": data, "n_leaked": int(n_leaked),
        "max_seq_len": max_seq_len,
    }


# =====================================================================
# 三个基线
# =====================================================================
def run_popularity(ctx: dict, args, ks) -> list:
    """热度基线：零训练成本。返回报告列表（本基线恒有 1 项）。

    ⚠️ `freq` 只统计**本档位的训练用户**。若图省事用全量用户统计，
    热度里就混进了 val/test 的交互 —— 那是"偷看测试集"的一种，
    而且它会让基线**更好看**，正好是最不容易被察觉的方向。
    """
    freq_raw = count_train_frequency(
        ctx["train_seqs"], ctx["n_items"], rows=ctx["train_rows"])
    freq = normalize_frequency(freq_raw)

    scorer = PopularityScorer(freq)
    t0 = time.time()
    metrics = evaluate(scorer.score, ctx["eval_data"], ks=tuple(ks),
                       batch_size=args.eval_batch_size)
    elapsed = time.time() - t0

    # 正/负样本热度对比：解释"为什么热度能拿高分"的核心证据
    edge = ctx["eval_data"]
    pos_pop = freq_raw[edge.positives]
    neg_pop = freq_raw[edge.negatives]
    diagnostics = {
        "freq_source": "train_only",
        "n_items_with_positive_freq": int((freq_raw > 0).sum()),
        "pos_pop_mean": float(pos_pop.mean()),
        "pos_pop_median": float(np.median(pos_pop)),
        "neg_pop_mean": float(neg_pop.mean()),
        "neg_pop_median": float(np.median(neg_pop)),
        "pos_over_neg_ratio": float(pos_pop.mean() / max(1e-9, neg_pop.mean())),
        "n_distinct_positives": int(np.unique(edge.positives).size),
    }
    logger.info("热度基线：hr@10=%.4f ndcg@10=%.4f（正/负热度倍数 %.1fx，%.1fs）",
                metrics.get("hr@10", float("nan")),
                metrics.get("ndcg@10", float("nan")),
                diagnostics["pos_over_neg_ratio"], elapsed)
    return [{
        "baseline": "popularity",
        "metrics": metrics,
        "diagnostics": diagnostics,
        "model_config": scorer.config_snapshot(),
        "n_params": 0,
        "trained": False,
        "elapsed_sec": round(elapsed, 3),
    }]


def run_itemcf(ctx: dict, args, ks) -> list:
    """ItemCF 基线：零训练成本（但建相似度矩阵要几十秒）。

    默认跑**两次**：纯 ItemCF（E1 主表用这一行）与「全零行填热度」。
    两次的差值就是「并列口径」这份假象的大小 —— 它是一个必须披露的量，
    因为 `metrics.positive_rank` 的并列约定会让"模型毫无信息"的行
    白拿 HR=1（见 `models/baselines/itemcf.py` 的模块 docstring）。
    """
    rows = ctx["train_rows"]
    if args.itemcf_max_train_users is not None:
        rows = rows[:int(args.itemcf_max_train_users)]
        logger.warning("⚠️ ItemCF 只用前 %s 个训练用户建相似度矩阵（探查用，"
                       "正式数字不要这样跑）", f"{rows.size:,}")

    t0 = time.time()
    sim, stats = build_itemcf_similarity(
        ctx["train_seqs"], ctx["n_items"], topk=args.topk, rows=rows)
    build_sec = time.time() - t0
    logger.info("ItemCF 相似度：TopK=%d，nnz=%s（均值 %.1f 邻居/物品），"
                "无邻居物品 %s，共现 nnz=%s，建矩阵 %.1fs",
                stats["topk"], f"{stats['nnz']:,}", stats["avg_neighbors"],
                f"{stats['n_items_without_neighbor']:,}",
                f"{stats['cooc_nnz']:,}", build_sec)

    freq = normalize_frequency(count_train_frequency(
        ctx["train_seqs"], ctx["n_items"], rows=ctx["train_rows"]))

    reports = []
    variants = [("itemcf", None)]
    if not args.no_itemcf_tiebreak_check:
        variants.append(("itemcf_tb_pop", "popularity"))

    for name, tiebreak in variants:
        scorer = ItemCFScorer(sim, n_items=ctx["n_items"], topk=args.topk,
                              popularity=freq, tiebreak=tiebreak)
        scorer.reset_stats()          # ⚠️ 必须：诊断量是按调用累计的
        t1 = time.time()
        metrics = evaluate(scorer.score, ctx["eval_data"], ks=tuple(ks),
                           batch_size=args.eval_batch_size)
        elapsed = time.time() - t1
        diag = {**stats, **scorer.last_stats,
                "build_sec": round(build_sec, 3)}
        if diag["n_rows"] != ctx["eval_data"].n_samples:
            # 累计口径与评估集对不上 ⇒ 诊断量不可信，先别用它下结论
            raise RuntimeError(
                f"ItemCF 诊断行数 {diag['n_rows']} 与评估样本数 "
                f"{ctx['eval_data'].n_samples} 不一致，全零行占比不可信")
        logger.info("%s：hr@10=%.4f ndcg@10=%.4f（全零行 %s / %s = %.2f%%，"
                    "打分 %.1fs）",
                    name, metrics.get("hr@10", float("nan")),
                    metrics.get("ndcg@10", float("nan")),
                    f"{diag['n_zero_rows']:,}", f"{diag['n_rows']:,}",
                    100.0 * diag["zero_row_share"], elapsed)
        if diag["n_zero_rows"]:
            logger.warning(
                "  ⚠️ 上表 %s 行里，有 %s 行（%.2f%%）101 个候选分数全为 0 —— "
                "按本项目并列口径，这些行正样本 rank=1，**白拿 HR=1**。"
                "论文必须披露这个占比；需要去掉假象时用 itemcf_tb_pop 的对照值。",
                name, f"{diag['n_zero_rows']:,}", 100.0 * diag["zero_row_share"])
        reports.append({
            "baseline": name,
            "metrics": metrics,
            "diagnostics": diag,
            "model_config": scorer.config_snapshot(),
            "n_params": 0,
            "trained": False,
            "elapsed_sec": round(elapsed, 3),
        })
    return reports


def run_gru4rec(ctx: dict, args, ks, model_yaml: dict, train_yaml: dict,
                spec: dict, device, ckpt_dir: str) -> list:
    """GRU4Rec：唯一需要训练的基线。走 `fit()`，与本文模型共享全部训练逻辑。"""
    set_seed(int(args.seed))

    # GRU4Rec 的超参来自 `configs/model.yaml` 的 `baselines.gru4rec` 段，
    # 缺省时回落到 `model` 段（于是"同 hidden、同 dropout"由代码保证）。
    # 唯一必须显式覆盖的是 num_layers：`model` 段写的是 2（本文模型的层数），
    # 而 GRU4Rec 的论文口径是 1 层（docs/evaluation-plan.md §4）。
    m_cfg = dict(model_yaml.get("model") or {})
    gru_yaml = dict((model_yaml.get("baselines") or {}).get("gru4rec") or {})
    base_cfg = GRU4RecConfig.from_dict({**m_cfg, **gru_yaml}, n_items=ctx["n_items"])

    tc = dict(train_yaml)
    tc["epochs"] = int(args.epochs if args.epochs is not None else spec["max_epochs"])
    tc["seed"] = int(args.seed)
    tc["device"] = str(device)
    if args.batch_size is not None:
        tc["batch_size"] = int(args.batch_size)
    train_cfg = TrainConfig.from_dict(
        tc, eval_cfg={"every_n_epochs": int(spec["eval_every_n_epochs"])})
    oc = OptimConfig.from_dict(train_yaml)
    if args.lr is not None:
        oc = replace(oc, lr=float(args.lr))

    model = GRU4Rec(base_cfg).to(device)
    logger.info("GRU4Rec：hidden=%d layers=%d dropout=%g 参数量 %s"
                "（公式值 %s，两者必须相等）",
                base_cfg.hidden_size, base_cfg.num_layers, base_cfg.dropout,
                f"{model.n_params:,}", f"{model.expected_n_params():,}")
    if model.n_params != model.expected_n_params():
        raise RuntimeError("GRU4Rec 参数量与公式不符，说明结构被改动过")

    train_ds = SlidingWindowDataset(
        ctx["train_seqs"], max_seq_len=ctx["max_seq_len"], user_rows=ctx["train_rows"])
    loader = WindowBatchIterator(
        train_ds, batch_size=int(train_cfg.batch_size),
        shuffle=True, seed=int(args.seed))

    edge = ctx["eval_data"]

    def eval_fn(m) -> dict:
        return evaluate(m.score, edge, ks=tuple(ks), batch_size=args.eval_batch_size,
                        device=device)

    t0 = time.time()
    result = fit(model=model, train_loader=loader, cfg=train_cfg, optim_cfg=oc,
                 eval_fn=eval_fn, n_items=ctx["n_items"],
                 grad_accum_steps=int(train_yaml.get("grad_accum_steps", 1)))
    train_sec = time.time() - t0

    final_metrics = None
    if result.best_state_dict is not None:
        model.load_state_dict(result.best_state_dict)
        model.eval()          # fit() 评估后会调回 train()，这里必须显式切回
        final_metrics = eval_fn(model)

    report = {
        "baseline": "gru4rec",
        "metrics": final_metrics if final_metrics is not None else {},
        "diagnostics": {
            "best_epoch": result.best_epoch,
            "best_metric": result.best_metric,
            "metric_name": result.metric_name,
            "n_epochs_run": result.n_epochs_run,
            "stopped_early": result.stopped_early,
            "first_loss": (result.records[0].train_loss if result.records else None),
            "last_loss": result.last_loss,
            "train_sec": round(train_sec, 3),
        },
        "model_config": model.config_snapshot(),
        "result": result.as_report(),
        "n_params": int(model.n_params),
        "trained": True,
        "elapsed_sec": round(train_sec, 3),
    }

    if not args.no_save:
        exp_tag = f"gru4rec{('_' + args.tag) if args.tag else ''}"
        path = os.path.join(
            ckpt_dir, checkpoint_name(exp_tag, spec["name"], args.seed, suffix="best"))
        save_checkpoint(path, model, meta={
            "experiment_name": exp_tag,
            "arch": "gru4rec",
            "scale": spec["name"],
            "seed": int(args.seed),
            "epoch": result.best_epoch,
            "metrics": report["metrics"],
            "best_metric_name": result.metric_name,
            "best_metric": result.best_metric,
            "content_fusion": None,
            "model_config": model.config_snapshot(),
        })
        report["checkpoint"] = os.path.relpath(path, ROOT)
        logger.info("GRU4Rec 最优权重已保存：%s", report["checkpoint"])
    return [report]


# =====================================================================
# 主流程
# =====================================================================
def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging()
    t_start = time.time()

    model_yaml = load_yaml(args.config)
    scale_yaml = load_yaml(args.scale_config)
    scale_name = args.scale or model_yaml.get("scale") \
        or scale_yaml.get("default_scale") or "dev"
    scales = scale_yaml.get("scales") or {}
    if scale_name not in scales:
        raise SystemExit(f"档位 {scale_name!r} 不存在，可用：{sorted(scales)}")
    spec = dict(scales[scale_name])
    spec["name"] = scale_name
    spec.setdefault("max_epochs", 30)
    spec.setdefault("eval_every_n_epochs", 5)

    train_yaml = dict(model_yaml.get("train") or {})
    eval_yaml = dict(model_yaml.get("eval") or {})
    ks = list(eval_yaml.get("ks") or [5, 10])

    seq_path = to_abs(os.path.join("data", "processed", "seq_dataset.pkl"))
    if not os.path.exists(seq_path):
        raise SystemExit(
            f"缺少 {seq_path}\n请先跑 `python scripts/run_stage1.py` 生成阶段一产物。")
    logger.info("加载 %s（约 360MB，需要十几秒）...",
                os.path.relpath(seq_path, ROOT))
    t0 = time.time()
    with open(seq_path, "rb") as f:
        ds = pickle.load(f)
    logger.info("  用户 %s / 物品 %s / 耗时 %.1fs",
                f"{len(ds['train']):,}", f"{len(ds['smap']):,}", time.time() - t0)

    ctx = build_shared_eval_data(
        ds, spec, args, max_seq_len=int((model_yaml.get("model") or {}).get(
            "max_seq_len", 50)))

    device = torch.device(args.device or train_yaml.get("device") or "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("配置要求 cuda 但当前不可用，回退到 cpu（GRU4Rec 会非常慢）")
        device = torch.device("cpu")

    ckpt_dir = to_abs(args.out_dir
                      or (model_yaml.get("checkpoint") or {}).get("save_dir")
                      or "./data/checkpoints")
    log_dir = to_abs(args.log_dir)

    todo = (list(BASELINE_CHOICES) if args.baseline == "all" else [args.baseline])
    all_reports: list = []
    for name in todo:
        logger.info("=" * 70)
        logger.info("基线 %s", name)
        logger.info("=" * 70)
        if name == "popularity":
            reports = run_popularity(ctx, args, ks)
        elif name == "itemcf":
            reports = run_itemcf(ctx, args, ks)
        else:
            reports = run_gru4rec(ctx, args, ks, model_yaml, train_yaml,
                                  spec, device, ckpt_dir)
        for r in reports:
            r.update({
                "scale": scale_name,
                "split": args.split,
                "seed": int(args.seed),
                "n_items": ctx["n_items"],
                "n_train_users": int(ctx["train_rows"].size),
                "n_eval_users": int(ctx["eval_rows"].size),
                "n_eval_samples": int(ctx["eval_data"].n_samples),
                "n_eval_candidates": int(ctx["eval_data"].n_candidates),
                "n_leaked_dropped": int(ctx["n_leaked"]),
                "subset_seed": SUBSET_SEED,
                "neg_seed": DEFAULT_NEG_SEED,
                "protocol": "leave_one_out_neg100",
            })
            all_reports.append(r)
            if not args.no_save:
                os.makedirs(log_dir, exist_ok=True)
                tag = f"_{args.tag}" if args.tag else ""
                path = os.path.join(
                    log_dir,
                    f"baseline_{r['baseline']}_{scale_name}_{args.split}"
                    f"_seed{args.seed}{tag}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(r, f, ensure_ascii=False, indent=2)
                logger.info("报告已保存：%s", os.path.relpath(path, ROOT))

    logger.info("=" * 70)
    logger.info("汇总（档位 %s，%s 集，%d 组）", scale_name, args.split,
                len(all_reports))
    logger.info("=" * 70)
    for r in all_reports:
        m = r.get("metrics") or {}
        logger.info("  %-16s hr@10=%.4f  ndcg@10=%.4f  mrr=%.4f  参数 %s",
                    r["baseline"], m.get("hr@10", float("nan")),
                    m.get("ndcg@10", float("nan")), m.get("mrr", float("nan")),
                    f"{r['n_params']:,}")
    logger.info("总耗时 %.1fs", time.time() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
