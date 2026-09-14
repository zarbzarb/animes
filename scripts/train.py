# -*- coding: utf-8 -*-
"""SASRec 训练入口 —— `scripts/train.py`

用法
----
    # 用配置里的默认档位（configs/model.yaml 的 scale 字段 = dev）
    python scripts/train.py

    # 指定档位与配置
    python scripts/train.py --scale smoke

    # 只跑一个种子（main 档默认会跑 3 个种子，很慢）
    python scripts/train.py --scale main --seed 42

    # 学习率重标定（M2.4 的待办：batch 256→1024 后 lr 没跟着调）
    python scripts/train.py --scale dev --lr 0.002 --warmup-ratio 0.05

    # 快速自检：只跑 20 个 batch、200 个评估用户
    python scripts/train.py --scale smoke --limit-train-batches 20 --eval-users 200

    ⚠️ 工作目录无关：在任意目录执行结果一致。这也是本脚本的硬性要求
    （`docs/dev-conventions.md` 3.0）：脚本里只用
    `ROOT = dirname(dirname(abspath(__file__)))` 定位项目根，
    **不出现 `os.getcwd()`**。

本脚本负责全部 IO
-----------------
`models/sasrec/train.py` 是无 IO 的（不读 pkl、不写权重、不读 yaml），
所以下面这些事都在本脚本完成：

    读 configs/*.yaml          → 构造 dataclass
    读 data/processed/seq_dataset.pkl → 构造 Dataset
    按档位抽用户（nested_user_subset） → user_rows / eval_rows
    组装验证集（build_eval_data）      → eval_fn 回调
    写 data/checkpoints/*.pt          → 最优权重
    写 logs/train_*.json              → 训练历史与指标

抽样口径（与 `configs/scale.yaml` 的三条不变式一致，勿改）
----------------------------------------------------------
1. **只抽用户，不抽物品** —— 物品池恒为全量 15,687，模型输出层维度不随档位变。
2. **确定性嵌套** —— 各档位基于同一个 `subset_order(uids, seed=42)` 的前缀，
   所以 5% ⊂ 10% ⊂ 100%。
3. **评估子集从本档位用户中抽**（不是从全量用户池）——
   评估用户数 = 本档位用户数 × `eval_user_ratio`；同档位内固定不变，
   因此跨 epoch 的指标可比、best model 不会被选错。
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
import yaml
from torch.utils.data import DataLoader

# 脚本独立运行时，项目根不在 sys.path 里（pytest 有 pythonpath=.，脚本没有）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.checkpoint.io import checkpoint_name, save_checkpoint  # noqa: E402
from models.content_encoder.model import (  # noqa: E402
    build_model,
    load_content_matrix_for_fusion,
)
from models.data.negatives import DEFAULT_NEG_SEED  # noqa: E402
from models.data.user_subset import subset_order  # noqa: E402
from models.eval.evaluator import build_eval_data, drop_leaked_samples, evaluate  # noqa: E402
from models.multi_interest.config import MultiInterestConfig  # noqa: E402
from models.multi_interest.model import MultiInterestSASRec  # noqa: E402
from models.sasrec.config import OptimConfig, SASRecConfig, TrainConfig  # noqa: E402
from models.sasrec.dataset import SlidingWindowDataset, WindowBatchIterator  # noqa: E402
from models.sasrec.model import SASRec  # noqa: E402
from models.sasrec.train import fit, set_seed  # noqa: E402
# 用户抽样的种子：固定为 42，**不随训练种子变化**。
# 理由：若它跟着训练种子变，换种子时评估子集也跟着换，"3 个种子的均值"
# 就不是"同一个测试集上的 3 次重跑"，而混进了数据差异。
SUBSET_SEED = 42

logger = logging.getLogger("train")


# =====================================================================
# 工具
# =====================================================================
def to_abs(p: str) -> str:
    """把路径按「相对项目根」解释后转绝对路径。

    与 `scripts/preprocess.py::to_abs_path` 是同一条约定：
    配置里写 `./data/checkpoints` 时，无论从哪里执行脚本都指向
    `<项目根>/data/checkpoints`。
    """
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(ROOT, p))


def load_yaml(path: str) -> dict:
    with open(to_abs(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logging(level: int = logging.INFO) -> None:
    """把日志同时打到 stdout。`logging` 而非 print，便于阶段三接日志系统。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def resolve_scale(scale_cfg: dict, name: str) -> dict:
    """取出一个档位的配置，并把全局不变式合并进去。"""
    scales = scale_cfg.get("scales") or {}
    if name not in scales:
        raise SystemExit(
            f"档位 {name!r} 不存在。可用档位：{sorted(scales)}\n"
            f"（定义见 configs/scale.yaml）")
    spec = dict(scales[name])
    spec["name"] = name
    # 档位没写全时用全局默认兜底，避免 KeyError
    spec.setdefault("user_ratio", 0.05)
    spec.setdefault("seeds", [42])
    spec.setdefault("max_epochs", 30)
    spec.setdefault("eval_every_n_epochs", 5)
    spec.setdefault("eval_user_ratio", 0.1)
    return spec


def select_rows(n_users: int, ratio: float, order: np.ndarray) -> np.ndarray:
    """从哈希全序 `order` 中取前 `ratio` 比例的位置，返回**升序**行号。

    用 `order` 的前缀而不是另抽一次，是"嵌套"这条不变式的全部实现所在
    （见 `configs/scale.yaml` 的 invariants）。返回升序是为了让下游
    （负采样、评估输入构造）的遍历顺序稳定、缓存友好。
    """
    if ratio >= 1.0:
        return np.arange(n_users, dtype=np.int64)
    n_keep = max(1, min(n_users, int(round(n_users * float(ratio)))))
    return np.sort(order[:n_keep].astype(np.int64))


class LimitedLoader:
    """只产出前 `limit` 个 batch 的 DataLoader 包装。

    为什么需要它：冒烟自检时不想等一整个 epoch（main 档一轮 4.5 分钟，
    但一轮至少要跑完才能看到 loss 趋势）。`fit()` 需要 `len(loader)` 来
    按比例算 warmup 步数，`itertools.islice` 没有 `__len__`，所以这里
    自己包一个。
    """

    def __init__(self, loader, limit: int):
        self.loader = loader
        self.limit = int(limit)

    def __len__(self) -> int:
        return min(int(self.limit), len(self.loader))

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.limit:
                break
            yield batch


# =====================================================================
# 参数
# =====================================================================
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="SASRec 训练（M2.4）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", default="configs/model.yaml",
                    help="模型/训练配置（相对项目根）")
    ap.add_argument("--scale-config", default="configs/scale.yaml",
                    help="档位定义文件（相对项目根）")
    ap.add_argument("--scale", default=None,
                    help="训练档位：smoke/debug/dev/main/full；默认取配置里的 scale")
    ap.add_argument("--model", default=None, choices=["sasrec", "multi_interest"],
                    help="模型结构：sasrec（M2.4 基座）/ multi_interest（M2.5 多兴趣）。"
                         "默认取配置里的 arch 字段，再默认 sasrec")
    ap.add_argument("--content-fusion", dest="content_fusion",
                    action="store_true", default=None,
                    help="启用候选侧内容融合（M2.6b）：内容向量拼进物品表示、"
                         "端到端训练。默认取配置 content_fusion.candidate_side")
    ap.add_argument("--no-content-fusion", dest="content_fusion",
                    action="store_false",
                    help="显式关闭候选侧内容融合（覆盖配置）")
    ap.add_argument("--content-mode", default=None, choices=["concat", "add"],
                    help="候选侧融合方式：add（零初始化残差，默认）/ "
                         "concat（拼接后降维，对照臂）")
    ap.add_argument("--seed", type=int, default=None,
                    help="只跑这一个种子；默认跑档位定义的全部种子")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖档位的 max_epochs")
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖 batch_size")
    ap.add_argument("--lr", type=float, default=None, help="覆盖学习率（lr 重标定用）")
    ap.add_argument("--warmup-ratio", type=float, default=None,
                    help="线性 warmup 占总步数的比例，如 0.05")
    ap.add_argument("--lr-decay", default=None, choices=["constant", "cosine"],
                    help="warmup 之后的衰减方式")
    ap.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    ap.add_argument("--limit-train-batches", type=int, default=None,
                    help="每轮只跑前 N 个 batch（快速自检用）")
    ap.add_argument("--eval-users", type=int, default=None,
                    help="只评估前 N 个评估用户（快速自检用）")
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader 进程数")
    ap.add_argument("--eval-batch-size", type=int, default=4096,
                    help="评估时的批大小（纯前向，可以比训练大）")
    ap.add_argument("--no-final-eval", action="store_true",
                    help="跳过「用最优权重再评估一次」的收尾步骤")
    ap.add_argument("--no-save", action="store_true", help="不写 checkpoint / 日志")
    ap.add_argument("--tag", default=None,
                    help="产物文件名后缀标记（例如 lr 扫描时区分不同组）")
    ap.add_argument("--out-dir", default=None,
                    help="checkpoint 目录，默认取配置的 checkpoint.save_dir")
    ap.add_argument("--log-dir", default="logs", help="训练日志目录")
    ap.add_argument("--device", default=None, help="cuda / cpu，默认取配置")
    return ap.parse_args(argv)


# =====================================================================
# 主流程
# =====================================================================
def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging()
    t_start = time.time()

    # ---------- 配置 ----------
    model_yaml = load_yaml(args.config)
    scale_yaml = load_yaml(args.scale_config)

    scale_name = args.scale or model_yaml.get("scale") \
        or scale_yaml.get("default_scale") or "dev"
    spec = resolve_scale(scale_yaml, scale_name)

    m_cfg = dict(model_yaml.get("model") or {})
    train_yaml = dict(model_yaml.get("train") or {})
    eval_yaml = dict(model_yaml.get("eval") or {})
    exp_name = model_yaml.get("experiment_name") or "sasrec"

    # ---------- 数据集 ----------
    seq_path = to_abs(os.path.join("data", "processed", "seq_dataset.pkl"))
    if not os.path.exists(seq_path):
        raise SystemExit(
            f"缺少 {seq_path}\n请先跑 `python scripts/run_stage1.py` 生成阶段一产物。")
    logger.info("加载 %s（约 360MB，需要十几秒）...",
                os.path.relpath(seq_path, ROOT))
    t0 = time.time()
    with open(seq_path, "rb") as f:
        ds = pickle.load(f)
    train_seqs, val_seqs = ds["train"], ds["val"]
    n_users = len(train_seqs)
    n_items = len(ds["smap"])
    logger.info("  用户 %s / 物品 %s / 耗时 %.1fs",
                f"{n_users:,}", f"{n_items:,}", time.time() - t0)

    max_seq_len = int(m_cfg.get("max_seq_len", 50))

    # ---------- 用户抽样（三条不变式，见模块 docstring）----------
    # ⚠️ 抽样依据必须是 **user_id** 而不是数组行号：
    # `models/data/user_subset.py` 的整个设计前提就是「按 uid 哈希，
    # 与数组顺序无关」。若改用行号，一旦 stage1 重跑导致 umap 顺序变化，
    # 同一个 ratio 会选中完全不同的一批用户，历史实验立刻失去可比性。
    # 这里先把 umap 的 (uid -> row) 反转成「行号 -> uid」的稠密数组。
    t0 = time.time()
    uids = np.empty(n_users, dtype=np.int64)
    for uid, row in ds["umap"].items():
        uids[int(row)] = int(uid)
    order = subset_order(uids, seed=SUBSET_SEED)
    train_rows = select_rows(n_users, float(spec["user_ratio"]), order)
    # 评估用户 = 本档位用户的子集（比例相对**本档位**，不是全量用户池）
    n_eval_cap = max(1, int(round(train_rows.size * float(spec["eval_user_ratio"]))))
    # train_rows 已按行号升序，这里要的是"同一把哈希尺子的前缀"，
    # 所以从 order 里取，而不是从升序的 train_rows 里取。
    train_row_set = np.zeros(n_users, dtype=bool)
    train_row_set[train_rows] = True
    eval_rows = np.sort(order[train_row_set[order]][:n_eval_cap].astype(np.int64))
    if args.eval_users is not None:
        eval_rows = eval_rows[:int(args.eval_users)]
    logger.info(
        "档位 %s：user_ratio=%g -> 训练用户 %s；eval_user_ratio=%g -> 评估用户 %s"
        "（用时 %.1fs）",
        scale_name, spec["user_ratio"], f"{train_rows.size:,}",
        spec["eval_user_ratio"], f"{eval_rows.size:,}", time.time() - t0)

    # ---------- 数据集与 DataLoader ----------
    train_ds = SlidingWindowDataset(
        train_seqs, max_seq_len=max_seq_len, user_rows=train_rows)
    summary = train_ds.summary()
    logger.info("训练窗口 %s（均值 %.1f / 用户），输入上限 %d，截断用户 %s",
                f"{summary['n_windows']:,}", summary["mean_train_len"],
                summary["input_cap"], f"{summary['truncated_users']:,}")
    if summary["n_windows"] == 0:
        raise SystemExit("训练窗口数为 0，请检查档位的 user_ratio")

    device = torch.device(args.device or train_yaml.get("device") or "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("配置要求 cuda 但当前不可用，回退到 cpu（会非常慢）")
        device = torch.device("cpu")

    # ---------- 模型结构（不随档位变化）----------
    # arch 决定"在基座之上加什么"。多兴趣与基座共用同一份 model: 段的
    # 基座超参（hidden/layers/heads/dropout…），E2 消融的"其余超参一致"
    # 是结构上保证的，不是靠人肉对齐两份配置。
    arch = str(args.model or model_yaml.get("arch") or "sasrec")
    mi_cfg = None
    if arch == "multi_interest":
        mi_cfg = MultiInterestConfig.from_dict(m_cfg, n_items=n_items)
        model_cfg = mi_cfg.sasrec
        logger.info(
            "模型：multi_interest K=%d routing_iters=%d | 基座 hidden=%d "
            "layers=%d heads=%d dropout=%g，输入上限 %d",
            mi_cfg.num_interests, mi_cfg.routing_iters,
            model_cfg.hidden_size, model_cfg.num_layers,
            model_cfg.num_heads, model_cfg.dropout, model_cfg.input_cap)
    else:
        model_cfg = SASRecConfig.from_dict(m_cfg, n_items=n_items)
        logger.info("模型：hidden=%d layers=%d heads=%d dropout=%g，输入上限 %d",
                    model_cfg.hidden_size, model_cfg.num_layers,
                    model_cfg.num_heads, model_cfg.dropout, model_cfg.input_cap)

    # ---------- 候选侧内容融合（M2.6b）----------
    # 三层优先级：命令行 > 配置 content_fusion.candidate_side > 默认关。
    # 关闭时 content_matrix=None，模型装配路径与 M2.4 完全一致（可比性）。
    fusion_yaml = dict(model_yaml.get("content_fusion") or {})
    use_content = (bool(args.content_fusion) if args.content_fusion is not None
                   else bool(fusion_yaml.get("candidate_side", False)))
    content_mode = str(args.content_mode or fusion_yaml.get("mode") or "add")
    content_matrix = None
    content_file = None
    if use_content:
        content_file = to_abs(fusion_yaml.get("content_file")
                              or os.path.join("data", "features",
                                              "content_vec_512.npy"))
        if not os.path.exists(content_file):
            raise SystemExit(
                f"启用内容融合但缺少内容向量：{content_file}\n"
                "请先跑 `python scripts/build_content_vectors.py`。")
        content_matrix = load_content_matrix_for_fusion(content_file, n_items)
        logger.info("候选侧内容融合已启用：%s（%s）",
                    os.path.relpath(content_file, ROOT), content_mode)

    # ---------- 验证集（只构建一次，跨 epoch 复用 —— 硬约束）----------
    t0 = time.time()
    val_items = np.array([int(val_seqs[int(r)][0]) for r in eval_rows], dtype=np.int64)
    eval_data = build_eval_data(
        train_seqs, eval_rows, n_items,
        split="val", val_items=val_items, max_seq_len=max_seq_len,
        n_negatives=int(eval_yaml.get("neg_sample_num", 100)),
        seed=DEFAULT_NEG_SEED, label="val",
    )
    eval_data, n_leaked = drop_leaked_samples(eval_data)
    logger.info(
        "验证集：%s 样本 × %d 候选（构建 %.1fs）；剔除答案泄漏 %s 条",
        f"{eval_data.n_samples:,}", eval_data.n_candidates,
        time.time() - t0, f"{n_leaked:,}")
    if n_leaked:
        # 不静默：泄漏率变化会直接改变指标的可比性
        logger.warning(
            "  泄漏率 %.3f%%（成因：用户重复消费同一部番，属数据固有现象，"
            "论文中须注明）", 100.0 * n_leaked / max(1, n_leaked + eval_data.n_samples))

    ks = list(eval_yaml.get("ks") or [5, 10])

    # ---------- 产物目录（只解析一次，供所有种子复用）----------
    ckpt_dir = to_abs(args.out_dir
                      or (model_yaml.get("checkpoint") or {}).get("save_dir")
                      or "./data/checkpoints")
    log_dir = to_abs(args.log_dir)

    # ---------- 逐个种子训练 ----------
    seeds = [int(args.seed)] if args.seed is not None else \
        [int(s) for s in (spec.get("seeds") or [42])]
    reports = []
    for i, seed in enumerate(seeds, start=1):
        logger.info("=" * 70)
        logger.info("种子 %d（%d/%d）", seed, i, len(seeds))
        logger.info("=" * 70)
        report = run_one_seed(
            args=args, seed=seed, spec=spec, scale_name=scale_name,
            model_cfg=model_cfg, train_yaml=train_yaml, eval_yaml=eval_yaml,
            exp_name=exp_name, train_ds=train_ds,
            eval_data=eval_data, ks=ks, device=device, n_items=n_items,
            n_leaked=n_leaked, ckpt_dir=ckpt_dir, log_dir=log_dir,
            arch=arch, mi_cfg=mi_cfg,
            content_matrix=content_matrix, content_mode=content_mode,
            content_file=content_file,
        )
        reports.append(report)

    logger.info("全部完成，总耗时 %.1fs", time.time() - t_start)
    if len(reports) > 1:
        _log_seed_summary(reports, ks, spec)
    return 0


def run_one_seed(
    args, seed: int, spec: dict, scale_name: str, model_cfg: SASRecConfig,
    train_yaml: dict, eval_yaml: dict, exp_name: str, train_ds,
    eval_data, ks, device, n_items: int, n_leaked: int,
    ckpt_dir: str, log_dir: str,
    arch: str = "sasrec", mi_cfg: MultiInterestConfig = None,
    content_matrix=None, content_mode: str = "add",
    content_file: str = None,
) -> dict:
    """跑一个种子：建模型 → 训练 → 存最优权重 → 记录报告。"""
    set_seed(seed)

    # ---------- 训练控制（档位决定规模，命令行决定超参覆盖）----------
    tc = dict(train_yaml)
    tc["epochs"] = int(args.epochs if args.epochs is not None
                       else spec["max_epochs"])
    tc["seed"] = seed
    tc["device"] = str(device)
    if args.batch_size is not None:
        tc["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        tc["num_workers"] = int(args.num_workers)
    if args.no_amp:
        tc["amp"] = False
    train_cfg = TrainConfig.from_dict(
        tc, eval_cfg={"every_n_epochs": int(spec["eval_every_n_epochs"])})

    oc = OptimConfig.from_dict(train_yaml)
    if args.lr is not None or args.warmup_ratio is not None or args.lr_decay is not None:
        oc = replace(
            oc,
            lr=float(args.lr) if args.lr is not None else oc.lr,
            warmup_ratio=(float(args.warmup_ratio) if args.warmup_ratio is not None
                          else oc.warmup_ratio),
            lr_decay=(args.lr_decay or oc.lr_decay),
        )
    if args.lr is not None:
        logger.info("学习率被命令行覆盖为 %g（warmup=%.0f%%，decay=%s）",
                    oc.lr, oc.warmup_ratio * 100, oc.lr_decay)

    # ---------- 批迭代器 ----------
    # 默认走 WindowBatchIterator（单进程、无 collate 开销）。
    # 实测对比（bs=1024，6,533 用户档位）：
    #   DataLoader(num_workers=0) → 17.1 ms/batch 取样
    #   WindowBatchIterator       →  约 5 ms/batch
    #   DataLoader(num_workers=4) → 慢 15 倍（Windows spawn 复制整个序列数据集）
    # 所以 num_workers>0 只在明确需要对比时手动开启。
    nw = int(train_cfg.num_workers)
    if nw > 0:
        logger.warning(
            "启用 torch DataLoader 多进程（num_workers=%d）。⚠️ Windows 下会把"
            "整个序列数据集 pickle 到每个 worker，实测比单进程慢 15 倍；"
            "仅在 Linux 或做对比实验时使用。", nw)
        g = torch.Generator()
        g.manual_seed(seed)      # 固定 shuffle 顺序，让同种子可复现
        loader = DataLoader(
            train_ds,
            batch_size=int(train_cfg.batch_size),
            shuffle=True,
            num_workers=nw,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=4,
            generator=g,
        )
    else:
        loader = WindowBatchIterator(
            train_ds, batch_size=int(train_cfg.batch_size),
            shuffle=True, seed=seed)
    if args.limit_train_batches:
        loader = LimitedLoader(loader, int(args.limit_train_batches))
        logger.info("⚠️ 快速自检模式：每轮只跑 %d 个 batch", args.limit_train_batches)

    # ---------- 验证回调（模型用当前参数，不额外占据显存）----------
    eval_batch = int(args.eval_batch_size)

    def eval_fn(model) -> dict:
        res = evaluate(model.score, eval_data, ks=tuple(ks),
                       batch_size=eval_batch, device=device)
        return res

    # ---------- 模型 ----------
    # 统一走 models/content_encoder/model.py 的工厂：训练与 checkpoint 重建
    # 必须共用同一条装配路径（否则「训练用 concat、加载建成 add」会静默发生）
    model = build_model(
        arch, model_cfg, n_items, content_matrix=content_matrix,
        content_mode=content_mode, mi_cfg=mi_cfg).to(device)
    if content_matrix is not None:
        logger.info(
            "内容融合：候选侧 %s（内容 %d 维 → hidden %d），投影层参数 +%s，"
            "内容矩阵 %d 维固定特征（不可学）",
            content_mode, int(content_matrix.shape[1]), int(model_cfg.hidden_size),
            f"{model.repr_provider.n_params:,}",
            int(content_matrix.shape[1]))
    logger.info("参数量 %s（arch=%s）；训练配置 batch=%d lr=%g amp=%s "
                "workers=%d epochs=%d",
                f"{model.n_params:,}", arch, train_cfg.batch_size, oc.lr,
                train_cfg.amp_enabled(), nw, train_cfg.epochs)

    # ---------- 训练 ----------
    result = fit(
        model=model, train_loader=loader, cfg=train_cfg, optim_cfg=oc,
        eval_fn=eval_fn, n_items=n_items,
        grad_accum_steps=int(train_yaml.get("grad_accum_steps", 1)),
    )

    # ---------- 用最优权重再评估一次 ----------
    # 这一步是"必须有效"的：它证明 best_state_dict 真的能 load 回模型、
    # 且其指标与训练过程中记录的最优值一致（不一致说明快照逻辑有 bug）。
    final_metrics = None
    if not args.no_final_eval and result.best_state_dict is not None:
        model.load_state_dict(result.best_state_dict)
        # ⚠️ 必须显式切 eval：`fit()` 在每次评估后会调回 `model.train()`
        # 以便继续训练，所以这里如果不切，dropout 会重新打开 ——
        # 指标会比真实值低且每次不同（实测差 3e-3，且不可复现）。
        # 评估器本身不持有模型、不负责切模式，这个责任在调用方。
        model.eval()
        final_metrics = eval_fn(model)
        best_key = result.metric_name
        if best_key in final_metrics:
            delta = abs(float(final_metrics[best_key]) - float(result.best_metric))
            logger.info("最优权重复评：%s=%.6f（训练中记录 %.6f，差 %.2e）",
                        best_key, float(final_metrics[best_key]),
                        float(result.best_metric), delta)
            if delta > 1e-6:
                logger.warning(
                    "  ⚠️ 复评与记录不一致，请检查 best_state_dict 的快照逻辑")

    # ---------- 落盘 ----------
    report = {
        "experiment_name": exp_name,
        "arch": arch,
        "content_fusion": (None if content_matrix is None else {
            **model.repr_provider.config_snapshot(),
            "file": os.path.relpath(content_file, ROOT),
            "n_params_total": int(model.n_params),
        }),
        "scale": scale_name,
        "seed": int(seed),
        "device": str(device),
        "n_items": int(n_items),
        "n_train_users": int(train_ds.lens.size),
        "n_train_windows": int(train_ds.n_windows),
        "n_eval_users": int(eval_data.n_samples),
        "n_eval_candidates": int(eval_data.n_candidates),
        "n_leaked_dropped": int(n_leaked),
        "optim": {
            "lr": float(oc.lr), "weight_decay": float(oc.weight_decay),
            "warmup_ratio": float(oc.warmup_ratio), "lr_decay": str(oc.lr_decay),
        },
        "train_config": {
            "batch_size": int(train_cfg.batch_size),
            "epochs": int(train_cfg.epochs),
            "eval_every_n_epochs": int(train_cfg.eval_every_n_epochs),
            "early_stop_patience": int(train_cfg.early_stop_patience),
            "grad_clip_norm": float(train_cfg.grad_clip_norm),
            "amp": bool(train_cfg.amp_enabled()),
            "num_workers": int(nw),
        },
        "result": result.as_report(),
        "final_metrics": final_metrics,
    }

    if not args.no_save:
        exp_tag = f"{exp_name}{('_' + args.tag) if args.tag else ''}"
        path = os.path.join(
            ckpt_dir, checkpoint_name(exp_tag, scale_name, seed, suffix="best"))
        # 内容融合的信息必须进 meta：`load_model_from_checkpoint` 靠它
        # 决定"要不要装 provider、装哪种 mode、读哪个文件"，否则重建出来的
        # 模型能加载权重却语义不同（形状一样，静默出错）。
        fusion_meta = None
        if content_matrix is not None:
            fusion_meta = dict(model.repr_provider.config_snapshot())
            fusion_meta["file"] = os.path.relpath(content_file, ROOT)
        save_checkpoint(path, model, meta={
            "experiment_name": exp_tag,
            "arch": arch,
            "scale": scale_name,
            "seed": int(seed),
            "epoch": result.best_epoch,
            "metrics": (final_metrics if final_metrics is not None
                        else (result.records[-1].metrics if result.records else {})),
            "best_metric_name": result.metric_name,
            "best_metric": result.best_metric,
            "content_fusion": fusion_meta,     # None = 无融合（M2.4 基线口径）
            "model_config": model.config_snapshot(),
        })
        report["checkpoint"] = os.path.relpath(path, ROOT)
        logger.info("最优权重已保存：%s", report["checkpoint"])

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir, f"train_{exp_tag}_{scale_name}_seed{seed}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["log_path"] = os.path.relpath(log_path, ROOT)
        logger.info("训练日志已保存：%s", report["log_path"])

    return report


def _log_seed_summary(reports: list, ks: list, spec: dict) -> None:
    """多种子时打印均值±标准差 —— 这是论文里 main 档要报的数字。"""
    logger.info("=" * 70)
    logger.info("多种子汇总（档位 %s，%d 个种子）", reports[0]["scale"], len(reports))
    logger.info("=" * 70)
    for key in [f"hr@{k}" for k in ks] + [f"ndcg@{k}" for k in ks] + ["mrr"]:
        vals = []
        for r in reports:
            m = r.get("final_metrics") or {}
            if key in m and m[key] is not None:
                vals.append(float(m[key]))
        if not vals:
            continue
        arr = np.asarray(vals)
        logger.info("  %-8s 均值 %.4f ± %.4f（n=%d）%s",
                    key, arr.mean(), arr.std(ddof=1) if arr.size > 1 else 0.0,
                    arr.size, "" if arr.size > 1 else "  ← 单种子无法算标准差")


if __name__ == "__main__":
    sys.exit(main())
