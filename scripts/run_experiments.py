# -*- coding: utf-8 -*-
"""实验编排入口 —— `scripts/run_experiments.py`（M2.8）

用法
----
    # 打印执行计划（不训练、不评估，零 GPU 占用）
    python scripts/run_experiments.py --group E1 --dry-run

    # 跑单个实验组（可复用已有 checkpoint / 基线报告，跳过已完成步骤）
    python scripts/run_experiments.py --group E1
    python scripts/run_experiments.py --group E2
    python scripts/run_experiments.py --group E4

    # 全部（E3 会被拒绝，见下）
    python scripts/run_experiments.py --group all

设计
----
本脚本是**编排器**，不实现任何训练/评估逻辑本身：

| 环节 | 复用的唯一实现 |
|---|---|
| 训练 | `scripts/train.py`（子进程调用，产物命名可寻址） |
| 基线 | `scripts/run_baselines.py`（子进程调用） |
| 用户抽样 | `models/data/user_subset.py::resolve_scale_users` |
| 评估集拼装 | `models/eval/evaluator.py::build_eval_data` |
| 指标 | `models/eval/evaluator.py::evaluate / evaluate_grouped` |
| 权重加载 | `models/checkpoint/io.py::load_model_from_checkpoint`（按 meta 自动重建，含内容融合） |

输出目录结构与 `docs/evaluation-plan.md` §8.2 一致：
`experiments/{run_id}/E*/{model}/metrics.json` + `summary.csv` +
`config.yaml` / `git_commit.txt` / `requirements_freeze.txt` 快照。

范围声明（v1）
--------------
* **E3 冷启动拒绝执行**：`configs/experiment.yaml` 的 `E3_coldstart` 段仍是
  早期占位开关（`use_multi_interest` / `w_content_cold` 会被 `from_dict`
  静默忽略），直接跑会得到一组"全默认配置"的假结果 —— 见 gpu-queue §4.1。
  先重写配置与剥离逻辑，再解锁本组的执行路径。
* **HPO 未编排**：粗搜/细搜有自己的两阶段逻辑（先 smoke 网格、再 dev 细搜），
  等 E1/E2 主线跑完再接进来，避免半成品先占住口径。

⚠️ 工作目录无关：只用 `ROOT` 锚定项目根，禁止 `os.getcwd()`。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logger = logging.getLogger("run_experiments")

IMPLEMENTED_GROUPS = ("E1_overall", "E2_ablation", "E4_by_genre")
BLOCKED_GROUPS = {
    "E3_coldstart": "E3 配置仍是早期占位开关（gpu-queue §4.1），直接跑出假结果，已拒绝",
    "HPO": "HPO 两阶段编排未实现（等 E1/E2 主线跑完再接）",
}

# 排期估算用的实测常数（来源：gpu-queue.md §三，2026-09-15 校准）
MS_PER_STEP = {"sasrec": 30.1, "multi_interest": 39.0, "gru4rec": 20.7}
STEPS_PER_EPOCH = {"smoke": 522, "debug": 2129, "dev": 5315, "main": 10671}

# 用户抽样种子固定 42，与 scripts/train.py、run_baselines.py 同一条约定：
# 换训练种子时评估子集不许跟着换，否则"3 种子均值"不是同一测试集上的重跑。
SUBSET_SEED = 42


# =====================================================================
# 工具
# =====================================================================
def to_abs(p: str) -> str:
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(ROOT, p))


def load_yaml(path: str) -> dict:
    import yaml
    with open(to_abs(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _run(cmd: list) -> None:
    """子进程跑一条命令，输出直通本进程（训练日志不落盘会丢早停现场）。"""
    logger.info("$ %s", " ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) else c
                                 for c in cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"子进程失败（exit {proc.returncode}）：{' '.join(cmd)}")


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=15)
        return (out.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def _ckpt_path(exp_name: str, tag: str, scale: str, seed: int) -> str:
    """checkpoint 命名与 scripts/train.py 完全一致（checkpoint_name 的约定）。"""
    exp_tag = f"{exp_name}_{tag}"
    return to_abs(os.path.join("data", "checkpoints",
                               f"{exp_tag}_{scale}_seed{int(seed)}_best.pt"))


def _baseline_report(name: str, scale: str, split: str, seed: int) -> dict | None:
    """已有基线报告则读回（复用检测），否则 None。"""
    path = to_abs(os.path.join(
        "logs", f"baseline_{name}_{scale}_{split}_seed{int(seed)}.json"))
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# =====================================================================
# 计划构建（纯函数，--dry-run 与单测共用）
# =====================================================================
def resolve_seeds(cfg: dict, scale_spec: dict, override) -> list:
    if override:
        return [int(s) for s in override.split(",")]
    if cfg.get("seeds"):
        return [int(s) for s in cfg["seeds"]]
    return [int(s) for s in (scale_spec.get("seeds") or [42])]


def build_plan(cfg: dict, scale_yaml: dict, groups: list, *,
               seeds_override=None, scale_override=None,
               existing_baselines=None) -> list:
    """把实验组展开成步骤列表。

    步骤形如 {"kind", "desc", "hours"(估算), "reuse"(bool)}；
    kind ∈ {baselines, train, eval, eval_grouped}。
    本函数只读配置、不碰数据集 —— dry-run 与单元测试都走这里。

    `existing_baselines`：已有基线报告的 (name, scale, split, seed) 集合；
    None 表示实际查 logs/ 目录（复用检测），单测传 set() 保持封闭。
    """
    plan = []
    for g in groups:
        if g in BLOCKED_GROUPS:
            plan.append({"kind": "blocked", "group": g,
                         "desc": BLOCKED_GROUPS[g], "hours": 0.0})
            continue
        gcfg = cfg[g] or {}
        scale = scale_override or str(gcfg.get("scale") or cfg.get("default_scale"))
        spec = dict(scale_yaml.get("scales") or {}).get(scale) or {}
        seeds = resolve_seeds(gcfg, spec, seeds_override)
        steps_per_ep = STEPS_PER_EPOCH.get(scale, 0)
        epochs = int(spec.get("max_epochs") or 30)

        def _have(name: str, split: str, seed: int) -> bool:
            if existing_baselines is not None:
                return (name, scale, split, int(seed)) in existing_baselines
            return _baseline_report(name, scale, split, seed) is not None

        if g == "E1_overall":
            for m in gcfg.get("models") or []:
                name, impl = m["name"], m.get("impl", "")
                params = dict(m.get("params") or {})
                if "/baselines/" in impl.replace("\\", "/"):
                    if name == "gru4rec":
                        for s in seeds:
                            plan.append({
                                "kind": "baselines", "group": g, "model": name,
                                "desc": f"gru4rec main/test seed{s}（run_baselines）",
                                "hours": MS_PER_STEP["gru4rec"] * steps_per_ep * epochs
                                         / 3.6e6, "reuse": _have(name, "test", s)})
                    else:
                        plan.append({
                            "kind": "baselines", "group": g, "model": name,
                            "desc": f"{name} 零训练基线（run_baselines）",
                            "hours": 0.05, "reuse": _have(name, "test", 42)})
                    continue
                # 神经模型：每种子一次训练 + 一次 test 评估
                ms = MS_PER_STEP["multi_interest"] if params.get("arch") == "multi_interest" \
                    else MS_PER_STEP["sasrec"]
                for s in seeds:
                    plan.append({
                        "kind": "train", "group": g, "model": name, "seed": s,
                        "desc": f"{name} seed{s}（train.py --scale {scale}）",
                        "hours": ms * steps_per_ep * epochs / 3.6e6})
                plan.append({
                    "kind": "eval", "group": g, "model": name,
                    "desc": f"{name} ×{len(seeds)} 种子 → main/test 评估（进程内）",
                    "hours": 0.3})

        elif g == "E2_ablation":
            reuse = dict(gcfg.get("reuse_from") or {})
            for gname, gparams in (gcfg.get("groups") or {}).items():
                src = reuse.get(gname)
                if src:
                    plan.append({
                        "kind": "eval", "group": g, "model": gname,
                        "desc": f"{gname} 复用 {src} 的权重，仅按 dev 口径重评",
                        "hours": 0.1, "reuse": True})
                    continue
                ms = MS_PER_STEP["multi_interest"] \
                    if gparams.get("arch") == "multi_interest" else MS_PER_STEP["sasrec"]
                plan.append({
                    "kind": "train", "group": g, "model": gname, "seed": seeds[0],
                    "desc": f"{gname}（train.py --scale {scale}）",
                    "hours": ms * steps_per_ep * epochs / 3.6e6})
                plan.append({
                    "kind": "eval", "group": g, "model": gname,
                    "desc": f"{gname} dev/val 评估（进程内）", "hours": 0.1})

        elif g == "E4_by_genre":
            n_variants = len((cfg.get("E2_ablation") or {}).get("groups") or {})
            plan.append({
                "kind": "eval_grouped", "group": g,
                "desc": f"{n_variants} 个消融变体 × 12 类题材分组评估（零训练）",
                "hours": 0.2 * max(1, n_variants)})
    return plan


def print_plan(plan: list) -> None:
    total = 0.0
    logger.info("=" * 78)
    for st in plan:
        if st["kind"] == "blocked":
            logger.info("⛔ [%s] %s", st["group"], st["desc"])
            continue
        flag = "♻️ 复用" if st.get("reuse") else "跑"
        hours = st.get("hours") or 0.0
        total += 0 if st.get("reuse") else hours
        logger.info("▸ %-13s %s  %s（≈%.2f h）", st["kind"], flag, st["desc"], hours)
    logger.info("-" * 78)
    logger.info("预计新增 GPU 时间（不计复用）：≈ %.1f h", total)


# =====================================================================
# 进程内评估（复用唯一实现；数据集由调用方加载一次传入）
# =====================================================================
def build_scale_eval(ds: dict, scale_spec: dict, split: str,
                     max_seq_len: int, n_negatives: int = 100):
    """按档位构造评估集（与 train.py / run_baselines.py 同一套唯一实现）。"""
    from models.data.negatives import DEFAULT_NEG_SEED
    from models.data.user_subset import resolve_scale_users
    from models.eval.evaluator import build_eval_data, drop_leaked_samples

    train_seqs, val_seqs, test_seqs = ds["train"], ds["val"], ds["test"]
    n_users, n_items = len(train_seqs), len(ds["smap"])
    uids = np.empty(n_users, dtype=np.int64)
    for uid, row in ds["umap"].items():
        uids[int(row)] = int(uid)
    _, eval_rows = resolve_scale_users(
        uids, float(scale_spec["user_ratio"]), float(scale_spec["eval_user_ratio"]),
        seed=SUBSET_SEED)
    val_items = np.array([int(val_seqs[int(r)][0]) for r in eval_rows], dtype=np.int64)
    test_items = (np.array([int(test_seqs[int(r)][0]) for r in eval_rows],
                           dtype=np.int64) if split == "test" else None)
    data = build_eval_data(
        train_seqs, eval_rows, n_items, split=split,
        val_items=val_items, test_items=test_items, max_seq_len=max_seq_len,
        n_negatives=n_negatives, seed=DEFAULT_NEG_SEED, label=split)
    data, n_leaked = drop_leaked_samples(data)
    return data, n_leaked


def eval_checkpoint(ckpt: str, ds: dict, scale_spec: dict, split: str,
                    ks: list, max_seq_len: int, device, genres_of_item=None,
                    grouped: bool = False, min_samples: int = 50) -> dict:
    """加载 checkpoint → 重建模型 → 在指定档位/切分上评估。

    `load_model_from_checkpoint` 会按 meta 自动重建结构（含 M2.6b 内容融合），
    训练与评估的装配逻辑若各写一套迟早分叉 —— 这里坚持不自己建模型。
    """
    import torch
    from models.checkpoint.io import load_model_from_checkpoint
    from models.eval.evaluator import evaluate, evaluate_grouped

    model, meta = load_model_from_checkpoint(ckpt)
    model.to(device)
    model.eval()
    data, n_leaked = build_scale_eval(ds, scale_spec, split, max_seq_len)
    if genres_of_item is not None and data.n_samples:
        # 目标物品的题材逐样本贴回（E4 分组用；EvalData 已通过 init 校验形状）
        data.genres = genres_of_item[data.positives]

    if grouped:
        out = evaluate_grouped(model.score, data, ks=tuple(ks),
                               batch_size=1024, device=device,
                               min_samples=min_samples)
        out["n_leaked_dropped"] = int(n_leaked)
        out["n_params"] = int(model.n_params)
        return out
    metrics = evaluate(model.score, data, ks=tuple(ks), batch_size=1024,
                       device=device)
    return {
        "metrics": metrics, "n_params": int(model.n_params),
        "n_leaked_dropped": int(n_leaked),
        "checkpoint": os.path.relpath(ckpt, ROOT),
        "best_epoch": meta.get("epoch"),
        "weight_scale": meta.get("scale"), "weight_seed": meta.get("seed"),
        "arch": meta.get("arch"),
    }


# =====================================================================
# 实验组执行
# =====================================================================
def run_e1(cfg, scale_yaml, model_yaml, args, out_dir, ds, ks, device) -> dict:
    """E1：热度 / ItemCF / GRU4Rec / SASRec / ours —— main 档 3 种子，test 报数。"""
    from models.eval.evaluator import evaluate

    gcfg = cfg["E1_overall"]
    scale = args.scale_override or str(gcfg.get("scale") or "main")
    spec = dict(scale_yaml["scales"][scale])
    spec["name"] = scale
    seeds = resolve_seeds(gcfg, spec, args.seeds)
    split = "test"                     # E1 正式数字口径：val 早停 + test 报数
    exp_name = model_yaml.get("experiment_name") or "sasrec"
    max_seq_len = int((model_yaml.get("model") or {}).get("max_seq_len", 50))
    py = sys.executable

    results: dict[str, list] = {}
    for m in gcfg.get("models") or []:
        name, impl = m["name"], m.get("impl", "")
        params = dict(m.get("params") or {})
        logger.info("---- E1 / %s ----", name)

        if "/baselines/" in impl.replace("\\", "/"):
            per_seed = []
            todo_seeds = seeds if name == "gru4rec" else [42]
            for s in todo_seeds:
                rep = _baseline_report(name, scale, split, s)
                if rep is None or args.force:
                    cmd = [py, to_abs("scripts/run_baselines.py"),
                           "--baseline", name, "--scale", scale,
                           "--split", split, "--seed", str(s)]
                    if args.epochs:
                        cmd += ["--epochs", str(args.epochs)]
                    _run(cmd)
                    rep = _baseline_report(name, scale, split, s)
                    if rep is None:
                        raise SystemExit(f"基线跑完但找不到报告：{name} seed{s}")
                per_seed.append({"seed": s, "metrics": rep["metrics"],
                                 "n_params": rep.get("n_params", 0),
                                 "report": os.path.relpath(to_abs(os.path.join(
                                     "logs", f"baseline_{name}_{scale}_{split}"
                                     f"_seed{s}.json")), ROOT)})
            results[name] = per_seed
            continue

        # 神经模型：每种子训练（已有 checkpoint 则复用）→ test 评估
        arch = str(params.get("arch") or "sasrec")
        tag = f"e1_{name}"
        per_seed = []
        for s in seeds:
            ckpt = _ckpt_path(exp_name, tag, scale, s)
            if not os.path.exists(ckpt) or args.force:
                cmd = [py, to_abs("scripts/train.py"),
                       "--scale", scale, "--model", arch,
                       "--seed", str(s), "--tag", tag]
                if params.get("content_fusion"):
                    cmd += ["--content-fusion",
                            "--content-mode", str(params.get("content_mode") or "add")]
                else:
                    cmd += ["--no-content-fusion"]
                if args.epochs:
                    cmd += ["--epochs", str(args.epochs)]
                if args.limit_train_batches:
                    cmd += ["--limit-train-batches", str(args.limit_train_batches)]
                _run(cmd)
            per_seed.append(eval_checkpoint(
                ckpt, ds, spec, split, ks, max_seq_len, device))
        results[name] = per_seed

    _write_seed_summary(out_dir, "E1_overall", results, ks)
    return results


def _e2_variant_checkpoints(cfg, model_yaml, args) -> dict:
    """E2 的四个变体 → checkpoint 路径（E2_2/E2_3 自训，其余复用 E1 seed42）。"""
    exp_name = model_yaml.get("experiment_name") or "sasrec"
    reuse = dict((cfg.get("E2_ablation") or {}).get("reuse_from") or {})
    e1_tags = {"E1_overall.sasrec": "e1_sasrec", "E1_overall.ours": "e1_ours"}
    e1_scale = str((cfg.get("E1_overall") or {}).get("scale") or "main")
    out = {}
    for gname in (cfg.get("E2_ablation") or {}).get("groups") or {}:
        src = reuse.get(gname)
        if src:
            out[gname] = _ckpt_path(exp_name, e1_tags[src], e1_scale, 42)
        else:
            out[gname] = _ckpt_path(exp_name, f"e2_{gname}",
                                    str((cfg.get("E2_ablation") or {}).get("scale")
                                        or "dev"), 42)
    return out


def run_e2(cfg, scale_yaml, model_yaml, args, out_dir, ds, ks, device) -> dict:
    """E2：4 组消融。E2_1/E2_4 复用 E1 权重（按 dev 口径重评），E2_2/E2_3 新训。"""
    gcfg = cfg["E2_ablation"]
    scale = args.scale_override or str(gcfg.get("scale") or "dev")
    spec = dict(scale_yaml["scales"][scale])
    spec["name"] = scale
    seeds = resolve_seeds(gcfg, spec, args.seeds)
    seed = seeds[0]                    # 消融单种子（configs/experiment.yaml 头部口径）
    exp_name = model_yaml.get("experiment_name") or "sasrec"
    max_seq_len = int((model_yaml.get("model") or {}).get("max_seq_len", 50))
    py = sys.executable

    results: dict[str, dict] = {}
    for gname, gparams in (gcfg.get("groups") or {}).items():
        gparams = dict(gparams or {})
        logger.info("---- E2 / %s ----", gname)
        reuse = dict(gcfg.get("reuse_from") or {})
        if gname in reuse:
            # 复用 = 不重训，但必须在**同一档位口径**下重评（混着报才不可比）
            src = reuse[gname]
            e1_tags = {"E1_overall.sasrec": "e1_sasrec", "E1_overall.ours": "e1_ours"}
            ckpt = _ckpt_path(exp_name, e1_tags[src],
                              str((cfg.get("E1_overall") or {}).get("scale") or "main"), 42)
            if not os.path.exists(ckpt):
                raise SystemExit(f"E2 复用源缺失：{src} 的 checkpoint {ckpt}；先跑 --group E1")
            rep = eval_checkpoint(ckpt, ds, spec, "val", ks, max_seq_len, device)
            rep["reused_from"] = src
            results[gname] = rep
            continue

        arch = str(gparams.get("arch") or "sasrec")
        ckpt = _ckpt_path(exp_name, f"e2_{gname}", scale, seed)
        if not os.path.exists(ckpt) or args.force:
            cmd = [py, to_abs("scripts/train.py"),
                   "--scale", scale, "--model", arch,
                   "--seed", str(seed), "--tag", f"e2_{gname}"]
            if gparams.get("content_fusion"):
                cmd += ["--content-fusion",
                        "--content-mode", str(gparams.get("content_mode") or "add")]
            else:
                cmd += ["--no-content-fusion"]
            if int(gparams.get("num_interests") or 0):
                cmd += ["--num-interests", str(gparams["num_interests"])]
            if args.epochs:
                cmd += ["--epochs", str(args.epochs)]
            if args.limit_train_batches:
                cmd += ["--limit-train-batches", str(args.limit_train_batches)]
            _run(cmd)
        results[gname] = eval_checkpoint(ckpt, ds, spec, "val", ks, max_seq_len, device)

    _write_ablation_summary(out_dir, "E2_ablation", results, ks,
                            baseline="E2_1_pure_sasrec")
    return results


def run_e4(cfg, scale_yaml, model_yaml, args, out_dir, ds, ks, device) -> dict:
    """E4：复用 E2 的四个变体，在 main 档 test 集上按 primary_genre 分组评估。"""
    import pandas as pd

    gcfg = cfg["E4_by_genre"]
    scale = args.scale_override or str(
        (cfg.get("E1_overall") or {}).get("scale") or "main")
    spec = dict(scale_yaml["scales"][scale])
    spec["name"] = scale
    max_seq_len = int((model_yaml.get("model") or {}).get("max_seq_len", 50))

    # 题材标签：item_stats.parquet 的 primary_genre（池内 idx → anime_id → genre）
    item_stats = pd.read_parquet(to_abs(os.path.join("data", "processed",
                                                     "item_stats.parquet")))
    inv_smap = {int(v): int(k) for k, v in ds["smap"].items()}
    genre_of = dict(zip(item_stats["anime_id"].astype(int),
                        item_stats["primary_genre"]))
    genres_of_item = np.zeros(len(ds["smap"]) + 1, dtype=np.int64)   # idx 从 1 起
    for idx, aid in inv_smap.items():
        genres_of_item[idx] = int(genre_of.get(aid, 0))
    n_known = int((genres_of_item[1:] > 0).sum())
    logger.info("题材覆盖：%d / %d 个物品有 primary_genre", n_known, len(ds["smap"]))

    variants = _e2_variant_checkpoints(cfg, model_yaml, args)
    matrix, details = {}, {}
    for gname, ckpt in variants.items():
        if not os.path.exists(ckpt):
            raise SystemExit(f"E4 变体 checkpoint 缺失：{gname} → {ckpt}；"
                             "先跑 --group E1 / --group E2")
        logger.info("---- E4 / %s ----", gname)
        rep = eval_checkpoint(ckpt, ds, spec, "test", ks, max_seq_len, device,
                              genres_of_item=genres_of_item, grouped=True,
                              min_samples=int(gcfg.get("min_samples") or 50))
        if rep.get("weight_scale") not in (None, scale):
            # 消融配置的既定口径（experiment.yaml: reuse E2）：E2_2/E2_3 是
            # dev 档权重。论文报告时必须注明权重档位，别让读者误读成全量训练。
            logger.warning("⚠️ %s 的权重来自 %s 档训练（复用口径），"
                           "本组在 %s 档 test 集上评估", gname,
                           rep.get("weight_scale"), scale)
        details[gname] = rep
        groups = rep.get("groups") or {}
        for g, gm in groups.items():
            matrix.setdefault(int(g), {})[gname] = gm

    # recall10_matrix.csv：行 = 题材，列 = 变体（值 + 样本量）
    names = list(variants)
    csv_path = os.path.join(out_dir, "E4_by_genre", "recall10_matrix.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["primary_genre"] + [f"{n}_recall@10" for n in names]
                   + [f"{n}_n_samples" for n in names])
        for g in sorted(matrix):
            row = [g]
            for n in names:
                row.append(matrix[g].get(n, {}).get("recall@10", ""))
            for n in names:
                row.append(matrix[g].get(n, {}).get("n_samples", ""))
            w.writerow(row)
    with open(os.path.join(out_dir, "E4_by_genre", "metrics.json"),
              "w", encoding="utf-8") as f:
        json.dump({"variants": details, "matrix_csv": os.path.relpath(csv_path, ROOT),
                   "scale": scale, "split": "test",
                   "weight_scale_note": "E2_2/E2_3 权重来自 dev 档（experiment.yaml 复用口径）"},
                  f, ensure_ascii=False, indent=2)
    logger.info("E4 分题材矩阵已写入：%s", os.path.relpath(csv_path, ROOT))
    return details


# =====================================================================
# 汇总落盘
# =====================================================================
def _mean_std(per_seed: list, key: str) -> tuple:
    vals = [p["metrics"][key] for p in per_seed
            if isinstance(p.get("metrics"), dict) and key in p["metrics"]]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _write_seed_summary(out_dir: str, group: str, results: dict, ks: list) -> None:
    gdir = os.path.join(out_dir, group)
    os.makedirs(gdir, exist_ok=True)
    keys = [f"hr@{k}" for k in ks] + [f"ndcg@{k}" for k in ks] + ["mrr"]
    rows = []
    for name, per_seed in results.items():
        for p in per_seed:
            gpath = os.path.join(gdir, name)
            os.makedirs(gpath, exist_ok=True)
            with open(os.path.join(gpath, f"metrics_seed{p['seed']}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(p, f, ensure_ascii=False, indent=2)
        row = {"model": name, "n_seeds": len(per_seed)}
        for k in keys:
            mean, std = _mean_std(per_seed, k)
            row[k] = f"{mean:.4f}±{std:.4f}"
        row["n_params"] = per_seed[0].get("n_params")
        rows.append(row)
    with open(os.path.join(gdir, "summary.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    logger.info("%s 汇总：%s", group, os.path.relpath(gdir, ROOT))


def _write_ablation_summary(out_dir: str, group: str, results: dict,
                            ks: list, baseline: str) -> None:
    gdir = os.path.join(out_dir, group)
    os.makedirs(gdir, exist_ok=True)
    keys = [f"hr@{k}" for k in ks] + [f"ndcg@{k}" for k in ks]
    base = (results.get(baseline) or {}).get("metrics") or {}
    rows = []
    for name, rep in results.items():
        with open(os.path.join(gdir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        m = rep.get("metrics") or {}
        row = {"group": name, "n_params": rep.get("n_params"),
               "reused_from": rep.get("reused_from", ""),
               "weight_scale": rep.get("weight_scale", "")}
        for k in keys:
            row[k] = f"{m.get(k, float('nan')):.4f}"
            b = base.get(k)
            if b and name != baseline:
                row[f"{k}_vs_base"] = f"{m[k] - b:+.4f}"
        rows.append(row)
    with open(os.path.join(gdir, "summary.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    logger.info("%s 汇总：%s", group, os.path.relpath(gdir, ROOT))


# =====================================================================
# 主流程
# =====================================================================
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="实验编排（M2.8）：按 configs/experiment.yaml 串起 E1~E4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--scale-config", default="configs/scale.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--group", default="E1",
                    help="E1 / E2 / E4 / all（逗号分隔可多组）；E3、HPO 已知会被拒绝")
    ap.add_argument("--out", default=None, help="输出目录，默认 experiments/{时间}_{组}")
    ap.add_argument("--seeds", default=None, help="覆盖种子，如 42 或 42,2024,2025")
    ap.add_argument("--scale-override", default=None,
                    help="覆盖各组的档位（探查用；正式数字禁用）")
    ap.add_argument("--epochs", type=int, default=None, help="传给 train.py / run_baselines.py")
    ap.add_argument("--limit-train-batches", type=int, default=None,
                    help="传给 train.py（冒烟验证编排逻辑用）")
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="不复用已有产物，全部重跑")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    return ap.parse_args(argv)


def _normalize_groups(raw: str) -> list:
    alias = {"E1": "E1_overall", "E2": "E2_ablation", "E3": "E3_coldstart",
             "E4": "E4_by_genre"}
    if raw.strip().lower() == "all":
        return ["E1_overall", "E2_ablation", "E3_coldstart", "E4_by_genre"]
    out = []
    for part in raw.split(","):
        part = part.strip()
        out.append(alias.get(part, part))
    for g in out:
        if g not in IMPLEMENTED_GROUPS and g not in BLOCKED_GROUPS:
            raise SystemExit(f"未知实验组 {g!r}；可用：{sorted(IMPLEMENTED_GROUPS)}（E3/HPO 已知阻塞）")
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging()
    t0 = time.time()

    cfg = load_yaml(args.config)
    scale_yaml = load_yaml(args.scale_config)
    model_yaml = load_yaml(args.model_config)
    groups = _normalize_groups(args.group)

    plan = build_plan(cfg, scale_yaml, groups, seeds_override=args.seeds,
                      scale_override=args.scale_override)
    print_plan(plan)
    if args.dry_run:
        return 0

    # 执行前确认没有阻塞组混进来（E3/HPO 拒绝执行，绝不产出假结果）
    blocked = [p for p in plan if p["kind"] == "blocked"]
    if blocked:
        raise SystemExit("计划里有被拒绝的实验组：\n" +
                         "\n".join(f"  ⛔ [{p['group']}] {p['desc']}" for p in blocked)
                         + "\n先按 gpu-queue §4.1 解锁，再单独跑这些组。")

    out_dir = to_abs(args.out or os.path.join(
        "experiments", f"{time.strftime('%Y%m%d_%H%M')}_"
                       f"{'_'.join(g.split('_')[0] for g in groups)}"))
    os.makedirs(out_dir, exist_ok=True)
    import shutil
    shutil.copyfile(to_abs(args.config), os.path.join(out_dir, "config.yaml"))
    with open(os.path.join(out_dir, "git_commit.txt"), "w", encoding="utf-8") as f:
        f.write(_git_commit() + "\n")
    lock = to_abs("requirements.lock")
    if os.path.exists(lock):
        shutil.copyfile(lock, os.path.join(out_dir, "requirements_freeze.txt"))

    import torch
    device = torch.device(args.device or "cuda" if torch.cuda.is_available() else "cpu")
    ks = list((model_yaml.get("eval") or {}).get("ks") or [5, 10])

    seq_path = to_abs(os.path.join("data", "processed", "seq_dataset.pkl"))
    if not os.path.exists(seq_path):
        raise SystemExit(f"缺少 {seq_path}；先跑 python scripts/run_stage1.py")
    logger.info("加载数据集 %s ...", os.path.relpath(seq_path, ROOT))
    with open(seq_path, "rb") as f:
        ds = pickle.load(f)

    need_ds = {"E1_overall", "E2_ablation", "E4_by_genre"}
    results = {}
    for g in groups:
        if g not in need_ds:
            continue
        if g == "E1_overall":
            results[g] = run_e1(cfg, scale_yaml, model_yaml, args, out_dir, ds,
                                ks, device)
        elif g == "E2_ablation":
            results[g] = run_e2(cfg, scale_yaml, model_yaml, args, out_dir, ds,
                                ks, device)
        elif g == "E4_by_genre":
            results[g] = run_e4(cfg, scale_yaml, model_yaml, args, out_dir, ds,
                                ks, device)

    logger.info("全部完成，耗时 %.1f min；产物在 %s",
                (time.time() - t0) / 60, os.path.relpath(out_dir, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
