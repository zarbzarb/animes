# -*- coding: utf-8 -*-
"""阶段 2 / M2.0：训练吞吐基准 —— 把「训练到底要多久」从估算变成实测。

为什么要写这个脚本
------------------
项目原来的排期依据是 docs/evaluation-plan.md 6.5 的「~77 h」，但那是按 **RTX 3090**
估的，本机是 **RTX 2060 6GB**，数字差一个量级；而且当时连「一个 epoch 有多少样本」
都是推的。阶段一实测给出硬数字：

    每 epoch 全量训练样本 = Σ train_len = 109,081,471（均值 83.5 / 用户）

于是「总耗时」的算法是：

    总耗时 = Σ_各实验 ( 单次训练耗时 ) ，单次训练耗时 = epoch 数 × 单 epoch 耗时
    单 epoch 耗时 = 样本数 ÷ 吞吐(samples/s)

样本数已经有了（本脚本会从 user_stats.parquet 精确算出各档位，不再线性外推），
**唯一缺的就是吞吐**。本脚本的任务就是实测它，并把整条预算链一次算清楚。

用法
----
    # 默认：测 AMP 开/关 + batch 扫描 + 全档位预算外推（推荐）
    python scripts/bench_throughput.py

    # 快速冒烟（少跑几步，只看量级）
    python scripts/bench_throughput.py --steps 10 --warmup 3 --batch-sizes 256

    # 只想要各档位样本数/显存上限，不想等基准跑完
    python scripts/bench_throughput.py --no-bench

    # 真实模型落地后（M2.4+）用它复测，替换参考计算图
    python scripts/bench_throughput.py --mode model \
        --model-factory models.multi_interest.model:build_model

    ⚠ 工作目录无关：以上命令在任意目录下执行结果一致（路径一律以项目根为锚点）。

产物
----
    logs/bench_throughput.json   机器可读：环境、实测吞吐、各档位预算、实验矩阵总耗时
    控制台表格                    人工阅读：直接贴进 progress.md 的那张表

阅读导引（函数 → 作用）
-----------------------
    load_yaml()              读 configs/*.yaml（相对项目根锚定）
    probe_environment()      采集 GPU / 驱动 / torch / CPU 实况，写进产物
    load_item_vocab()        物品池规模（15,687）与内容向量维度（512）
    load_scale_profiles()    各档位【精确】用户数与样本数（Σ train_len，非外推）
    RefMultiInterestSASRec   参考计算图：真实模型落地前用来标定 GPU 吞吐
    bench_train_steps()      实测前向+反向吞吐（AMP 开/关）
    bench_infer_steps()      实测纯前向吞吐（评估用，no_grad）
    micro_matmul_tflops()    峰值算力对拍，判断「离算力上限还有多远」
    build_budget()           外推各档位单次训练耗时 + 整个实验矩阵总耗时
    main()                   按上面顺序串起来并落盘

设计说明（重要，别当成"跑分玩具"）
----------------------------------
1. **参考计算图 ≠ 真实模型，但算力结构一致。**
   真实的 MultiInterestSASRec 要到 M2.4 才落地，本脚本先用一个结构等价的参考实现
   （物品嵌入 → 因果自注意力 ×num_layers → K=4 胶囊动态路由 ×routing_iters
   → 候选侧内容拼接 → 101 候选打分 → BCE）把 GPU 吞吐标定出来。
   由于 step 耗时由**计算图形状**决定，与具体权重初始化无关，这个数字对排期是有效的。
   ⚠️ 一旦 M2.4 落地，**必须用 `--mode model` 复测一次**并把本脚本产物替换掉，
   论文里写的必须是真实模型的数字。
2. **序列长度取 max_seq_len=50 是真实值，不是上界。**
   实测 train_len 均值 83.5 > 50，即绝大多数样本被截断到恰好 50（脚本会打印该比例）。
   所以用 L=50 基准不会高估计算量。
3. **训练与评估的吞吐分开测。**
   评估虽然只有前向，但次数多（每 eval_every_n_epochs 轮一次），本脚本单独测
   infer 吞吐再乘评估用户数——因为 6GB 显存下评估和训练往往不是同一个瓶颈。
4. **AMP 是重点对比项，不是可选项。**
   RTX 2060 是 Turing 架构，有 FP16 Tensor Core。若实测 AMP 提速不明显，
   说明瓶颈在数据加载或 kernel 启动开销，而不是算力——这会改变优化方向，
   所以两个数都要实测、都要记录。
5. **本脚本不写任何实验结论**，只产出实测数字。所有数字必须来自真实运行，
   禁止手填（与 docs/result-analysis.md 第 8 节的红线一致）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# scripts/ 不在任何包里，`from models... import` 只有在项目根位于 sys.path 时才成立。
# 显式插一次，保证「在任意目录执行本脚本」都能导入（与 dev-conventions 3.0 一致）。
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 阶段一实测的权威样本量（Σ train_len）。脚本会现场重算并与它对账，
# 对不上说明抽样口径或 user_stats 变了，必须停下来查，不能默默继续。
EXPECTED_FULL_SAMPLES = 109_081_471
EXPECTED_FULL_USERS = 1_306_691

# 各档位样本量的重算口径：用户级 Σ train_len。
# 说明：train_len 是「留一法切分后训练段长度」，阶段一用它求和恰好 = 109,081,471，
# 与文档口径逐位吻合，故这里沿用同一口径。滑窗真正的 (输入, 目标) 对数比它少
# 约 1.2%（每用户少 1 条），对排期无影响，但知道这件事以避免日后对不上账时误判。
SAMPLE_FORMULA = "sum(train_len)"


# =====================================================================
# 工具
# =====================================================================
def log(msg: str) -> None:
    """带时间戳打印进度（flush 保证长任务时能实时看到输出）。"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_yaml(rel_path: str) -> dict[str, Any]:
    """读 configs/ 下的 yaml。rel_path 相对项目根，避免依赖 CWD。"""
    abs_path = os.path.join(ROOT, rel_path)
    with open(abs_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def human_seconds(sec: float) -> str:
    """把秒数转成人能读的字符串（排期表里全是 5 位数秒没人看得懂）。"""
    if sec < 90:
        return f"{sec:.1f}s"
    if sec < 5400:
        return f"{sec / 60:.1f}min"
    if sec < 72000:
        return f"{sec / 3600:.2f}h"
    return f"{sec / 86400:.2f}d"


def human_int(n: int | float) -> str:
    """千分位整数，方便和 109,081,471 这种数字对照。"""
    return f"{int(n):,}"


# =====================================================================
# 参考计算图
# =====================================================================
class _RefBlock(nn.Module):
    """一个 Pre-LN 的因果自注意力块（SASRec 的标准结构）。

    为什么自己写而不用 nn.TransformerEncoderLayer：
    标准 Layer 是 Post-LN 且要传 attn_mask，这里需要精确控制「Pre-LN + 因果掩码」，
    否则算力口径会和真实实现有偏差（Post-LN 多两次 LayerNorm 的同步开销）。
    """

    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        # FFN 扩张比 4，与 SASRec / Transformer 原文一致
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class RefMultiInterestSASRec(nn.Module):
    """参考计算图：与 MultiInterestSASRec 算力结构一致，但不是最终模型。

    复刻的结构（参数全部来自 configs/model.yaml，不是硬编码）：
        item_emb(d) + pos_emb(L) -> Dropout
        -> num_layers × Pre-LN 因果自注意力（FFN 4d）
        -> LayerNorm
        -> K 个胶囊查询做 routing_iters 次动态路由（软分配 + 加权聚合）
        -> 候选集打分：行为路「胶囊·物品嵌入」取 max + 内容路「胶囊均值·内容投影」
        -> BCEWithLogits（1 正 + neg_sample_num 负）

    ⚠️ 明确不包含：early stop、checkpoint、指标计算、真实数据读盘。
       它只负责回答「这一步要多少秒」，不负责正确性。
    """

    def __init__(
        self,
        n_items: int,
        seq_len: int,
        d: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        num_interests: int,
        routing_iters: int,
        content_dim: int,
        w_behavior: float,
        w_content: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.routing_iters = routing_iters
        self.w_behavior = w_behavior
        self.w_content = w_content

        # 物品嵌入同时充当「胶囊的候选打分表」，对应 shared_item_emb: true
        self.item_emb = nn.Embedding(n_items, d)
        self.pos_emb = nn.Embedding(seq_len, d)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [_RefBlock(d, num_heads, dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(d)

        # 胶囊查询：K 个可学习向量，路由时被反复修正
        self.capsule_q = nn.Parameter(torch.randn(num_interests, d) * 0.02)
        # 内容侧：512 维语义向量压回 hidden_size，对应 candidate_side 内容融合
        self.content_proj = nn.Linear(content_dim, d)

    # ---------- 前向的三段，拆开是为了让基准脚本能单独测"只编码"的成本 ----------
    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """编码序列，返回每个位置的隐状态 [B, L, d]。"""
        b, length = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0)
        x = self.emb_drop(self.item_emb(seq) + self.pos_emb(pos))
        # 因果掩码：位置 i 只能看到 <= i，上三角置 -inf（float 掩码配合 MultiheadAttention）
        causal = torch.triu(
            torch.ones(length, length, device=seq.device, dtype=torch.bool), diagonal=1
        )
        attn_mask = torch.zeros(length, length, device=seq.device)
        attn_mask.masked_fill_(causal, float("-inf"))
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.out_norm(x)

    def capsules(self, hidden: torch.Tensor) -> torch.Tensor:
        """动态路由得到 K 个兴趣胶囊 [B, K, d]。

        这里用「软分配 + 加权聚合」迭代 routing_iters 次来近似原版的
        squash 动态路由：耗时结构（K×L 的注意力矩阵反复重算）与真实实现一致，
        是算力口径的代理，不代表真实路由的数值行为。
        """
        b, length, d = hidden.shape
        q = self.capsule_q.unsqueeze(0).expand(b, -1, -1).contiguous()
        scale = d ** -0.5
        for _ in range(self.routing_iters):
            logits = torch.matmul(q, hidden.transpose(1, 2)) * scale
            weights = torch.softmax(logits, dim=-1)
            q = torch.matmul(weights, hidden)
        return q

    def forward(
        self,
        seq: torch.Tensor,
        cand_ids: torch.Tensor,
        cand_content: torch.Tensor,
    ) -> torch.Tensor:
        """返回候选集上的 logits [B, C]。C = 1 正 + neg_sample_num 负。"""
        hidden = self.encode(seq)
        caps = self.capsules(hidden)                                  # [B, K, d]

        cand_emb = self.item_emb(cand_ids)                            # [B, C, d]
        behavior = torch.einsum("bkd,bcd->bkc", caps, cand_emb).max(dim=1).values

        # 内容路：胶囊均值（用户全局兴趣）与候选内容投影的内积
        user_vec = caps.mean(dim=1)                                   # [B, d]
        content_vec = self.content_proj(cand_content)                 # [B, C, d]
        content = torch.einsum("bd,bcd->bc", user_vec, content_vec)

        return self.w_behavior * behavior + self.w_content * content


# =====================================================================
# 环境与数据事实
# =====================================================================
def probe_environment(device: torch.device) -> dict[str, Any]:
    """采集环境实况 —— 这些数要写进论文的「实验环境」小节，必须自动采集而非手写。"""
    env: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cpu_count_logical": os.cpu_count(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        env.update(
            {
                "gpu_name": props.name,
                "gpu_total_mem_gb": round(props.total_memory / 1024**3, 2),
                "gpu_compute_capability": f"{props.major}.{props.minor}",
                "gpu_multi_processor_count": props.multi_processor_count,
                "gpu_is_turing_fp16_tc": props.major == 7 and props.minor == 5,
                "amp_supported": props.major >= 7,
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            }
        )
    return env


def load_item_vocab() -> dict[str, int]:
    """物品池规模与内容向量维度 —— 直接读阶段一产物，不写常量（改了要能自动跟上）。"""
    item_path = os.path.join(ROOT, "data", "processed", "item_stats.parquet")
    vec_path = os.path.join(ROOT, "data", "features", "content_vec_512.npy")
    if not os.path.exists(item_path):
        raise FileNotFoundError(
            f"缺少 {item_path}，请先跑阶段一：python scripts/run_stage1.py"
        )
    if not os.path.exists(vec_path):
        raise FileNotFoundError(
            f"缺少 {vec_path}，请先跑阶段一：python scripts/run_stage1.py"
        )
    import pandas as pd

    n_items = len(pd.read_parquet(item_path, columns=["anime_id"]))
    # mmap 只读形状，不把 512 维向量整块读进内存
    vec = np.load(vec_path, mmap_mode="r")
    return {"n_items": int(n_items), "content_dim": int(vec.shape[1])}


def load_scale_profiles(
    scale_cfg: dict[str, Any], max_seq_len: int
) -> dict[str, dict[str, Any]]:
    """算出各档位的【精确】用户数与样本数。

    为什么不用「全量 × ratio」外推：抽样是按 user_id 哈希取的，各用户序列长度
    并不相同，实测 5% 档样本占比是 4.87% 而不是 5.00%（差 2.6%）。
    既然 user_stats.parquet 就在手边，能精确算就别估。
    """
    import pandas as pd

    from models.data.user_subset import nested_user_subset

    stats = pd.read_parquet(
        os.path.join(ROOT, "data", "processed", "user_stats.parquet"),
        columns=["user_id", "train_len"],
    )
    uids = stats["user_id"].to_numpy()
    lens = stats["train_len"].to_numpy()

    # 对账：全量必须等于阶段一实测值，对不上就停，别拿错口径去排期
    total = int(lens.sum())
    if total != EXPECTED_FULL_SAMPLES or len(lens) != EXPECTED_FULL_USERS:
        raise RuntimeError(
            "user_stats 与阶段一实测不符，拒绝据此排期："
            f"样本 {total}（期望 {EXPECTED_FULL_SAMPLES}）、"
            f"用户 {len(lens)}（期望 {EXPECTED_FULL_USERS}）。"
            "请先确认 data/processed/user_stats.parquet 是否被重新生成过。"
        )

    seq_cap = max_seq_len
    profiles: dict[str, dict[str, Any]] = {}
    for name in scale_cfg["scales"]:
        cfg = scale_cfg["scales"][name]
        ratio = float(cfg["user_ratio"])
        mask = nested_user_subset(uids, ratio)
        sel_len = lens[mask]
        n_users = int(mask.sum())
        # 截断到 max_seq_len 后的有效长度：用来判断"固定填充到 50"与
        # "按批动态填充"两种实现之间差多少算力（见下方打印的说明）。
        eff_len = np.minimum(sel_len, seq_cap)
        profiles[name] = {
            "user_ratio": ratio,
            "n_users": n_users,
            "n_samples": int(sel_len.sum()),
            "sample_share": float(sel_len.sum() / total),
            "mean_train_len": float(sel_len.mean()),
            "median_train_len": float(np.median(sel_len)),
            "mean_seq_len_effective": float(eff_len.mean()),
            "frac_seq_at_cap": float((sel_len >= seq_cap).mean()),
            "max_epochs": int(cfg["max_epochs"]),
            "eval_every_n_epochs": int(cfg["eval_every_n_epochs"]),
            "eval_user_ratio": float(cfg["eval_user_ratio"]),
            # 评估用户数相对【本档位用户】计算，不是相对全量用户池 ——
            # 否则会出现"冒烟档训练 6 千人、评估 6 万人"的荒谬配比。
            "n_eval_users": int(round(float(cfg["eval_user_ratio"]) * n_users)),
            "seeds": list(cfg["seeds"]),
            "n_seeds": len(cfg["seeds"]),
        }
    return profiles


# =====================================================================
# 基准测量
# =====================================================================
def make_batches(
    n_batches: int,
    batch_size: int,
    seq_len: int,
    n_cand: int,
    n_items: int,
    content_dim: int,
    device: torch.device,
    rng: torch.Generator,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """预生成若干批数据常驻显存。

    为什么预生成：我们要测的是**计算**吞吐。如果每步现场造数据，测出来的
    是 randint 的速度，混进了不属于 GPU 计算的开销。真实数据加载吞吐
    由 M2.3 的 Dataset + DataLoader 单独测（见 docs/progress.md 的 M2.0 清单）。
    """
    batches = []
    for _ in range(n_batches):
        seq = torch.randint(0, n_items, (batch_size, seq_len), device=device, generator=rng)
        cand = torch.randint(0, n_items, (batch_size, n_cand), device=device, generator=rng)
        cont = torch.randn(batch_size, n_cand, content_dim, device=device, generator=rng)
        batches.append((seq, cand, cont))
    return batches


def _cuda_timer(device: torch.device) -> Callable[[], float]:
    """返回一个「从起点到现在」的累计秒表。

    CUDA 必须用 event 计时并 synchronize，否则测到的是"把 kernel 丢进队列"的时间，
    会给出虚高的吞吐（这是跑分脚本最常见的假快）。返回**累计**值而不是单步差值，
    因为单步差值在最后一次调用时只覆盖最后一步，会被误当成整体耗时。
    """
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        start.record()

        def tick() -> float:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / 1000.0

        return tick

    last = [time.perf_counter()]

    def tick_cpu() -> float:
        now = time.perf_counter()
        dt = now - last[0]
        last[0] = now
        return dt

    return tick_cpu


def bench_train_steps(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    steps: int,
    warmup: int,
    amp: bool,
    device: torch.device,
    labels: torch.Tensor,
) -> dict[str, Any]:
    """实测「前向 + 反向 + 优化器步」的吞吐。返回秒/步与样本/秒。"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()

    def one_step(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        seq, cand, cont = batch
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(seq, cand, cont)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    batch_size = batches[0][0].shape[0]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for i in range(warmup):
        one_step(batches[i % len(batches)])
    if device.type == "cuda":
        torch.cuda.synchronize()

    tick = _cuda_timer(device)
    for i in range(steps):
        one_step(batches[i % len(batches)])
    elapsed = tick()

    result: dict[str, Any] = {
        "amp": bool(amp),
        "batch_size": batch_size,
        "steps": steps,
        "seconds_per_step": elapsed / steps,
        "samples_per_second": (steps * batch_size) / elapsed,
    }
    if device.type == "cuda":
        result["peak_mem_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        result["peak_mem_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    return result


@torch.no_grad()
def bench_infer_steps(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    steps: int,
    warmup: int,
    amp: bool,
    device: torch.device,
) -> dict[str, Any]:
    """实测纯前向吞吐（评估阶段用）。

    评估次数多（每 eval_every_n_epochs 轮一次 × 整个训练过程），
    全量评估 = 1,306,691 用户 × 101 候选，所以它必须单独有一行数字。
    """
    model.eval()

    def one_step(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        seq, cand, cont = batch
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            model(seq, cand, cont)

    for i in range(warmup):
        one_step(batches[i % len(batches)])
    if device.type == "cuda":
        torch.cuda.synchronize()

    tick = _cuda_timer(device)
    for i in range(steps):
        one_step(batches[i % len(batches)])
    elapsed = tick()

    batch_size = batches[0][0].shape[0]
    return {
        "amp": bool(amp),
        "batch_size": batch_size,
        "steps": steps,
        "seconds_per_step": elapsed / steps,
        "samples_per_second": (steps * batch_size) / elapsed,
    }


def estimate_flops_per_sample(
    seq_len: int,
    d: int,
    num_layers: int,
    num_interests: int,
    routing_iters: int,
    n_cand: int,
    content_dim: int,
) -> float:
    """解析估算单个样本「前向 + 反向」的浮点运算量（单位 FLOP）。

    为什么要算这个：单看 samples/s 无法判断"还有多少优化空间"。
    把它换算成实际达到的 TFLOPS，再和微基准的峰值一比（MFU），
    才能区分两种情况：
        低 MFU  -> 瓶颈是 kernel 启动/小矩阵效率，调大 batch 就能改善
        高 MFU  -> 已经贴着硬件跑，只剩换精度/换卡
    这直接决定后续优化的方向，所以值得算清楚而不是拍脑袋。

    口径（与 RefMultiInterestSASRec 的结构一一对应，系数 2 表示乘加各算一次）：
        自注意力 QKV / 输出投影 : 3 次和 1 次 [L,d]x[d,d]
        注意力打分与加权        : 2 次 [L,d]x[d,L]
        FFN（扩张 4 倍）        : 2 次 [L,d]x[d,4d]
        胶囊路由                : routing_iters 次「打分 + 聚合」，各 2 次 [K,d]x[d,L]
        行为路打分              : [K,d]x[d,C]
        内容路                  : 投影 [C,content_dim]x[content_dim,d] + 内积 [C,d]
    反向按前向的 2 倍计（业界通用近似）。
    """
    per_layer = (
        3 * 2 * seq_len * d * d          # QKV
        + 2 * 2 * seq_len * seq_len * d  # scores + weighted sum
        + 2 * seq_len * d * d            # output projection
        + 2 * (2 * seq_len * d * 4 * d)  # FFN 两层
    )
    fwd = num_layers * per_layer
    fwd += routing_iters * 2 * (2 * num_interests * seq_len * d)
    fwd += 2 * num_interests * n_cand * d                       # 行为路
    fwd += 2 * n_cand * content_dim * d + 2 * n_cand * d        # 内容路
    return 3.0 * fwd


def micro_matmul_tflops(device: torch.device, size: int = 4096, iters: int = 20) -> dict[str, Any]:
    """测一次大矩阵乘的峰值算力，用来判断模型步「离硬件上限还有多远」。

    为什么需要它：如果模型步只有峰值的 3%，说明瓶颈是 kernel 启动/小矩阵效率，
    调大 batch 才有用；如果已经到了 40%，那就只剩换卡/换精度可选。
    这个判断直接决定优化方向，所以不能只看一个"步耗时"。
    """
    if device.type != "cuda":
        return {"note": "CPU 上不做峰值对拍"}
    a = torch.randn(size, size, device=device, dtype=torch.float16)
    b = torch.randn(size, size, device=device, dtype=torch.float16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    tick = _cuda_timer(device)
    for _ in range(iters):
        a @ b
    dt = tick()
    flops = 2.0 * size**3 * iters
    return {
        "dtype": "fp16",
        "size": size,
        "tflops": flops / dt / 1e12,
    }


# =====================================================================
# 预算外推
# =====================================================================
def build_budget(
    profiles: dict[str, dict[str, Any]],
    train_sps: float,
    infer_sps: float,
    scenarios: list[int],
) -> dict[str, Any]:
    """把实测吞吐换算成「单次训练耗时」与「整个实验矩阵耗时」。

    单次训练耗时 = 跑满 max_epochs 的训练 + 期间所有评估
                  = max_epochs × 样本数/train_sps
                  + n_eval_passes × 评估用户数/infer_sps

    注意评估是**独立于训练**的一笔开销，且量级常常被忽略：
    全量评估 130.7 万用户 × 101 候选，与一个 epoch 同量级（见 model.yaml 备注）。
    """
    out: dict[str, Any] = {"per_scale": {}, "train_sps": train_sps, "infer_sps": infer_sps}

    def scale_cost(name: str, epochs: int) -> dict[str, float]:
        p = profiles[name]
        train_s = epochs * p["n_samples"] / train_sps
        n_eval_passes = math.floor(epochs / p["eval_every_n_epochs"]) + 1
        eval_s = n_eval_passes * p["n_eval_users"] / infer_sps
        return {
            "epochs": epochs,
            "train_hours": train_s / 3600,
            "n_eval_passes": n_eval_passes,
            "eval_hours": eval_s / 3600,
            "total_hours": (train_s + eval_s) / 3600,
        }

    for name, p in profiles.items():
        upper = scale_cost(name, p["max_epochs"])
        out["per_scale"][name] = {
            "n_users": p["n_users"],
            "n_samples": p["n_samples"],
            "sample_share": p["sample_share"],
            "n_eval_users": p["n_eval_users"],
            "seconds_per_epoch": p["n_samples"] / train_sps,
            "upper_bound": upper,
            "early_stop_scenarios": {f"stop_at_{s}": scale_cost(name, s) for s in scenarios},
        }
    return out


def load_experiment_matrix(profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """从 configs/experiment.yaml 推出「到底要训多少次、每次训多少」的清单。

    这里刻意保留显式计数而不是全自动推导：实验组的复用规则（reuse_from）
    是省时间的核心设计，写清楚"哪几组要真训、哪几组直接复用"比自动算更不容易错。
    任何一次实验设计变更都必须回来改这里，并在 progress.md 留痕。
    """
    exp = load_yaml("configs/experiment.yaml")
    hpo = exp.get("HPO", {})
    extra = exp.get("extra", {})

    e1 = exp["E1_overall"]
    e1_models = [m for m in e1["models"] if m["name"] != "itemcf"]  # itemcf 无需训练
    e1_scale = e1["scale"]
    e1_n = len(e1_models) * profiles[e1_scale]["n_seeds"]

    e2 = exp["E2_ablation"]
    e2_groups = list(e2["groups"])
    e2_reuse = set(e2.get("reuse_from", {}))
    e2_n = len([g for g in e2_groups if g not in e2_reuse])
    e2_scale = e2["scale"]

    e3 = exp["E3_coldstart"]
    e3_models = list(e3["models"])
    e3_reuse = set(e3.get("reuse_from", {}))
    e3_n = len([m["name"] for m in e3_models if m["name"] not in e3_reuse])
    e3_scale = e3["scale"]

    hpo_coarse = int(hpo.get("coarse_grid_size", 0)) * len(hpo.get("seeds", [42]))
    hpo_refine = int(hpo.get("refine_top_k", 0))
    hpo_coarse_scale = hpo.get("scale", "smoke")
    hpo_refine_scale = hpo.get("refine_scale", "dev")

    extra_scale = extra.get("scale", "dev")
    extra_n = len(extra.get("k_sensitivity", {}).get("num_interests", [])) + len(
        extra.get("fusion_weight_sensitivity", {}).get("w_content", [])
    )

    matrix = [
        {"group": "E1 整体对比", "scale": e1_scale, "n_trainings": e1_n,
         "note": f"{len(e1_models)} 个神经模型 × {profiles[e1_scale]['n_seeds']} 种子"},
        {"group": "E2 消融（仅 2/3 组需新训）", "scale": e2_scale, "n_trainings": e2_n,
         "note": "E2_1 复用 E1.sasrec、E2_4 复用 E1.ours"},
        {"group": "E3 冷启动（仅 2 组需新训）", "scale": e3_scale, "n_trainings": e3_n,
         "note": "pure_sasrec 复用 E2_1"},
        {"group": "E4 分题材", "scale": e2_scale, "n_trainings": 0,
         "note": "完全复用 E2 训练结果，只有评估开销"},
        {"group": "HPO 粗搜", "scale": hpo_coarse_scale, "n_trainings": hpo_coarse,
         "note": f"coarse_grid_size={hpo.get('coarse_grid_size')} × 单种子"},
        {"group": "HPO 细搜", "scale": hpo_refine_scale, "n_trainings": hpo_refine,
         "note": f"粗搜前 {hpo.get('refine_top_k')} 组到开发档"},
        {"group": "补充实验", "scale": extra_scale, "n_trainings": extra_n,
         "note": "K 敏感性 + 融合权重敏感性"},
    ]
    return matrix


# =====================================================================
# 主流程
# =====================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="训练吞吐基准：实测 samples/s 并外推各档位与实验矩阵耗时",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：python scripts/bench_throughput.py --steps 30 --batch-sizes 128 256 512",
    )
    p.add_argument("--model-config", default="configs/model.yaml", help="模型配置（相对项目根）")
    p.add_argument("--scale-config", default="configs/scale.yaml", help="档位配置（相对项目根）")
    p.add_argument("--out", default="logs/bench_throughput.json", help="结果落盘路径（相对项目根）")
    p.add_argument("--mode", choices=["ref", "model"], default="ref",
                   help="ref=参考计算图（默认，真实模型未落地时用）；model=导入真实模型")
    p.add_argument("--model-factory", default="models.multi_interest.model:build_model",
                   help="--mode model 时的模块:函数，形如 pkg.mod:fn(cfg)->nn.Module")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 256, 512, 1024],
                   help="batch 扫描列表，用来找 6GB 显存上限")
    p.add_argument("--train-negatives", type=int, default=None,
                   help="训练侧负样本数；不指定则等于评估口径（100）。"
                        "评估候选集恒为 1 正 + 100 负，不受本参数影响")
    p.add_argument("--steps", type=int, default=30, help="每个配置的计时步数（不含 warmup）")
    p.add_argument("--warmup", type=int, default=5, help="预热步数")
    p.add_argument("--no-amp", action="store_true", help="跳过 AMP 对比，只测 fp32")
    p.add_argument("--no-bench", action="store_true", help="跳过基准，只打印数据事实与档位表")
    p.add_argument("--device", default=None, help="cuda / cuda:0 / cpu，默认自动选择")
    return p.parse_args()


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    log("⚠️  torch.cuda.is_available() 为 False —— 将只在 CPU 上跑，数字不能用于排期")
    return torch.device("cpu")


def build_model(
    args: argparse.Namespace,
    model_cfg: dict[str, Any],
    vocab: dict[str, int],
    device: torch.device,
    neg_sample_num: int,
) -> tuple[nn.Module, int]:
    """按 --mode 构造待测模型。返回 (model, n_cand)。

    neg_sample_num 由调用方从 configs/experiment.yaml 的 base 段读出后传入，
    不在本函数里兜底取值 —— 负样本数直接决定候选集大小和 step 耗时，
    用错默认值会静默给出偏快的吞吐。
    """
    m = model_cfg["model"]
    fusion = model_cfg.get("content_fusion", {})
    n_cand = 1 + int(neg_sample_num)

    if args.mode == "ref":
        model = RefMultiInterestSASRec(
            n_items=vocab["n_items"],
            seq_len=int(m["max_seq_len"]),
            d=int(m["hidden_size"]),
            num_layers=int(m["num_layers"]),
            num_heads=int(m["num_heads"]),
            dropout=float(m["dropout"]),
            num_interests=int(m["num_interests"]),
            routing_iters=int(m["routing_iters"]),
            content_dim=vocab["content_dim"],
            w_behavior=float(fusion.get("w_behavior", 0.7)),
            w_content=float(fusion.get("w_content", 0.3)),
        )
    else:
        module_name, _, fn_name = args.model_factory.partition(":")
        if not module_name or not fn_name:
            raise ValueError(f"--model-factory 格式应为 'pkg.mod:fn'，收到 {args.model_factory!r}")
        try:
            module = __import__(module_name, fromlist=[fn_name])
        except ImportError as exc:
            raise SystemExit(
                f"无法导入真实模型 {module_name}：{exc}\n"
                "真实模型要到 M2.4 才落地。在此之前请用默认的 --mode ref 标定 GPU 吞吐；"
                "M2.4 完成后必须用 --mode model 复测并替换 logs/bench_throughput.json。"
            ) from exc
        model = getattr(module, fn_name)(model_cfg)

    return model.to(device), n_cand


def main() -> int:
    args = parse_args()
    t_start = time.time()

    model_cfg = load_yaml(args.model_config)
    scale_cfg = load_yaml(args.scale_config)
    exp_cfg = load_yaml("configs/experiment.yaml")

    # 负样本数只在 experiment.yaml 的 base 段里（model.yaml 没有），显式取一次并落进产物，
    # 避免"悄悄用错候选集大小"。它直接决定 step 耗时，错了吞吐就是假的。
    #
    # ⚠️ 训练与评估的候选集大小是【两件事】，必须分开：
    #   · 评估侧恒为 1 正 + neg_sample_num 负（口径固定，改则指标不可比）；
    #   · 训练侧由 --train-negatives 决定，SASRec 原论文为「每位置 1 负样本」。
    #   先前版本把评估口径直接套到训练侧，会让训练耗时被高估数倍。
    neg_sample_num = int(exp_cfg.get("base", {}).get("neg_sample_num", 100))
    eval_neg = neg_sample_num
    train_neg = eval_neg if args.train_negatives is None else int(args.train_negatives)
    if train_neg < 1:
        raise SystemExit(f"--train-negatives 必须 >= 1，收到 {train_neg}")
    max_seq_len = int(model_cfg["model"]["max_seq_len"])

    device = resolve_device(args.device)
    env = probe_environment(device)
    vocab = load_item_vocab()

    log(f"设备：{env.get('gpu_name', env['device'])} | torch {env['torch']} | "
        f"cuda_runtime {env['torch_cuda_runtime']} | amp={env.get('amp_supported')}")
    log(f"物品池 {human_int(vocab['n_items'])} | 内容向量 {vocab['content_dim']} 维 | "
        f"训练候选 {1 + train_neg}（1 正 + {train_neg} 负）| "
        f"评估候选 {1 + eval_neg}（1 正 + {eval_neg} 负）")

    log("正在从 user_stats.parquet 精确重算各档位样本量 ...")
    profiles = load_scale_profiles(scale_cfg, max_seq_len)

    print()
    print("=" * 96)
    print("① 各档位数据规模（精确值，非「全量 × 比例」外推）")
    print("=" * 96)
    print(f"{'档位':<8}{'用户数':>12}{'样本数':>16}{'样本占比':>10}"
          f"{'评估用户数':>12}{'epoch上限':>10}{'评估频率':>10}")
    for name, p in profiles.items():
        print(f"{name:<8}{human_int(p['n_users']):>12}{human_int(p['n_samples']):>16}"
              f"{p['sample_share'] * 100:>9.2f}%{human_int(p['n_eval_users']):>12}"
              f"{p['max_epochs']:>10}{p['eval_every_n_epochs']:>10}")
    print()
    full_p = profiles["full"]
    print(f"  序列长度口径：train_len 均值 {full_p['mean_train_len']:.1f}（中位数 "
          f"{full_p['median_train_len']:.0f}），全量档 {full_p['frac_seq_at_cap'] * 100:.1f}% 的用户"
          f"序列 ≥ {max_seq_len} 会被截断到恰好满长。")
    print(f"  截断后的有效平均长度 = {full_p['mean_seq_len_effective']:.1f}。基准固定按 L="
          f"{max_seq_len} 测：")
    print("    · 若 M2.3 采用「统一填充到 max_seq_len」（当前配置的默认预期）→ L=50 就是真实口径；")
    print("    · 若采用「按批动态填充」→ 约一半样本更短，L=50 是保守上界（只是偏慢，不会偏快）。")
    print("  两种实现的差异在 M2.3 落地时确认；本脚本的耗时预算一律按保守（偏慢）口径给出。")

    result: dict[str, Any] = {
        "environment": env,
        "vocab": vocab,
        "neg_sample_num": neg_sample_num,
        "train_negatives": train_neg,
        "eval_negatives": eval_neg,
        "max_seq_len": max_seq_len,
        "sample_formula": SAMPLE_FORMULA,
        "scale_profiles": profiles,
        "model_info": {"mode": args.mode},
        "benches": {},
        "budget": None,
        "experiment_matrix": None,
        "experiment_matrix_total_hours": None,
    }

    if not args.no_bench:
        model, train_n_cand = build_model(args, model_cfg, vocab, device, train_neg)
        eval_n_cand = 1 + eval_neg
        n_params = sum(p.numel() for p in model.parameters())
        log(f"待测模型：{type(model).__name__} | 参数量 {human_int(n_params)} | "
            f"训练候选 C={train_n_cand}（1 正 + {train_n_cand - 1} 负）| "
            f"评估候选 C={eval_n_cand}")
        result["model_info"].update(
            {"class": type(model).__name__, "n_params": n_params,
             "train_n_cand": train_n_cand, "eval_n_cand": eval_n_cand}
        )

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True   # 形状固定，让 cuDNN 自动选最快 kernel

        amp_modes = [False] if args.no_amp else [False, True]
        for bs in args.batch_sizes:
            seed = int(model_cfg["train"].get("seed", 42))
            # 训练与评估各生成一套 batch：候选维度分别是 train_n_cand / eval_n_cand。
            # 必须分开测，否则改动训练口径会连带污染评估吞吐的测量值。
            batches = make_batches(
                4, bs, max_seq_len, train_n_cand, vocab["n_items"], vocab["content_dim"],
                device, torch.Generator(device=device).manual_seed(seed),
            )
            batches_eval = make_batches(
                4, bs, max_seq_len, eval_n_cand, vocab["n_items"], vocab["content_dim"],
                device, torch.Generator(device=device).manual_seed(seed),
            )
            labels = torch.zeros(bs, train_n_cand, device=device)
            labels[:, 0] = 1.0   # 第 0 列固定放正样本（与评估口径一致，见 evaluation-plan 5.2）

            for amp in amp_modes:
                tag = f"bs{bs}_{'amp' if amp else 'fp32'}"
                try:
                    tr = bench_train_steps(
                        model, batches, args.steps, args.warmup, amp, device, labels
                    )
                except torch.cuda.OutOfMemoryError:
                    log(f"  {tag:<14} 显存不足（OOM）—— 该 batch 不可用，跳过")
                    result["benches"][tag] = {"oom": True, "batch_size": bs, "amp": amp}
                    torch.cuda.empty_cache()
                    continue
                inf = bench_infer_steps(
                    model, batches_eval, args.steps, args.warmup, amp, device
                )
                result["benches"][tag] = {"train": tr, "infer": inf}
                mem = tr.get("peak_mem_alloc_gb", float("nan"))
                log(f"  {tag:<14} train {tr['samples_per_second']:>10,.0f} samples/s "
                    f"({tr['seconds_per_step'] * 1000:>7.1f} ms/step) | "
                    f"infer {inf['samples_per_second']:>10,.0f} | peak {mem:.2f} GB")
            del batches, batches_eval, labels
            if device.type == "cuda":
                torch.cuda.empty_cache()

        micro = micro_matmul_tflops(device)
        result["model_info"]["micro_fp16_matmul"] = micro
        result["model_info"]["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # ---------- ② 实测吞吐表 ----------
        done = {t: v for t, v in result["benches"].items() if "train" in v}
        print()
        print("=" * 96)
        print("② 实测吞吐（samples/s）与显存")
        print("=" * 96)
        print(f"{'配置':<14}{'训练':>14}{'评估(纯前向)':>16}{'ms/step':>12}"
              f"{'峰值显存':>12}{'AMP提速':>10}")
        for tag, val in done.items():
            bs = val["train"]["batch_size"]
            fp32 = done.get(f"bs{bs}_fp32", {}).get("train", {})
            speed = val["train"]["samples_per_second"]
            gain = ""
            if val["train"]["amp"] and fp32:
                gain = f"{speed / fp32['samples_per_second']:.2f}x"
            print(f"{tag:<14}{speed:>14,.0f}{val['infer']['samples_per_second']:>16,.0f}"
                  f"{val['train']['seconds_per_step'] * 1000:>12.1f}"
                  f"{val['train'].get('peak_mem_alloc_gb', 0):>11.2f}G{gain:>10}")

        if "tflops" in micro:
            # 与峰值对拍：MFU 低说明瓶颈在小矩阵/kernel 启动而非算力，
            # 这个判断直接决定下一步是"调 batch"还是"只能换卡"。
            flops_per_sample = estimate_flops_per_sample(
                max_seq_len,
                int(model_cfg["model"]["hidden_size"]),
                int(model_cfg["model"]["num_layers"]),
                int(model_cfg["model"]["num_interests"]),
                int(model_cfg["model"]["routing_iters"]),
                train_n_cand,
                vocab["content_dim"],
            )
            result["model_info"]["flops_per_sample"] = flops_per_sample
            print(f"\n  单样本算力需求（解析估算，含反向）：{flops_per_sample / 1e9:.3f} GFLOP")
            print(f"  对照：fp16 大矩阵乘峰值实测 {micro['tflops']:.1f} TFLOPS —— 各配置的 MFU：")
            for tag, val in done.items():
                eff = val["train"]["samples_per_second"] * flops_per_sample / 1e12
                mfu = 100.0 * eff / micro["tflops"]
                hint = "   ← 瓶颈在 kernel 开销/小矩阵，调大 batch 可能继续改善" if mfu < 20 else ""
                print(f"      {tag:<14} 有效 {eff:>7.2f} TFLOPS  MFU {mfu:>5.1f}%{hint}")

        # ---------- ③④ 预算外推 ----------
        if done:
            best_tag = max(done, key=lambda t: done[t]["train"]["samples_per_second"])
            train_sps = done[best_tag]["train"]["samples_per_second"]
            infer_sps = done[best_tag]["infer"]["samples_per_second"]
            log(f"预算外推基准：{best_tag}（train {train_sps:,.0f} / infer {infer_sps:,.0f} samples/s）")

            budget = build_budget(profiles, train_sps, infer_sps, scenarios=[10, 20, 30])
            budget["based_on"] = best_tag
            result["budget"] = budget

            print()
            print("=" * 96)
            print(f"③ 各档位单次训练耗时预算（基于实测 {best_tag}）")
            print("=" * 96)
            print(f"{'档位':<8}{'sec/epoch':>12}{'训练':>11}{'评估':>11}{'合计':>11}"
                  f"{'10ep早停':>12}{'20ep早停':>12}")
            for name, b in budget["per_scale"].items():
                up = b["upper_bound"]
                sc = b["early_stop_scenarios"]
                print(f"{name:<8}{b['seconds_per_epoch']:>11.0f}s{up['train_hours']:>10.2f}h"
                      f"{up['eval_hours']:>10.2f}h{up['total_hours']:>10.2f}h"
                      f"{human_seconds(sc['stop_at_10']['total_hours'] * 3600):>12}"
                      f"{human_seconds(sc['stop_at_20']['total_hours'] * 3600):>12}")

            matrix = load_experiment_matrix(profiles)
            result["experiment_matrix"] = matrix
            total_hours = 0.0
            print()
            print("=" * 96)
            print("④ 整个实验矩阵总耗时（这才是真正决定「能不能做完」的数字）")
            print("=" * 96)
            print(f"{'实验组':<32}{'档位':<8}{'训练次数':>10}{'单次':>11}{'小计':>12}  说明")
            for row in matrix:
                unit = (
                    budget["per_scale"][row["scale"]]["upper_bound"]["total_hours"]
                    if row["n_trainings"]
                    else 0.0
                )
                sub = unit * row["n_trainings"]
                total_hours += sub
                print(f"{row['group']:<32}{row['scale']:<8}{row['n_trainings']:>10}"
                      f"{(f'{unit:.2f}h' if row['n_trainings'] else '-'):>11}"
                      f"{(human_seconds(sub * 3600) if row['n_trainings'] else '-'):>12}"
                      f"  {row['note']}")
            print("-" * 96)
            print(f"合计（跑满 max_epochs 的保守口径，早停会显著更低）："
                  f"{human_seconds(total_hours * 3600)}")
            result["experiment_matrix_total_hours"] = total_hours
            result["total_hours_note"] = "上限口径（跑满 max_epochs），早停会显著低于此值"

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    log(f"结果已写入 {os.path.relpath(out_path, ROOT)}（耗时 {time.time() - t_start:.1f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
