# -*- coding: utf-8 -*-
"""M2.2 / M2.3 验收：独立重算评估栈的关键口径，逐条 PASS / FAIL。

为什么单独写一个验收器
----------------------
M2.2（评估器）与 M2.3（滑动窗口 Dataset）是**主链路上最容易"静默跑偏"的两块**：

* Dataset 的滑窗索引错一位，指标会小幅变化，看不出是 bug 还是正常波动；
* 负采样不小心把训练物品放进去，指标会**明显变好**——而"变好"没人会去查；
* 评估批分块若把批次间状态串起来，换个 batch_size 结果就变，实验不可复现。

所以这里的做法和 ``verify_stage1.py`` 一致：**不信任任何报告**，
回到 ``data/processed/seq_dataset.pkl`` 与 ``configs/`` 重新推导每一条结论，
再与阶段一定稿的数字对账。

检查分五组
----------
    A 数据集结构   用户/物品数、窗口总数、截断用户数是否与阶段一逐位吻合。
    B 因果性       真实序列上「只改未来」，前面窗口的输入必须逐字节不变。
    C 负采样       不含训练物品、不含正样本、无重复、确定性、与顺序/子集无关。
    D 评估输入     val 用 train；test 用 train+val 且**不含 test 目标**。
    E 端到端       随机打分器的 HR@10 应≈ 10/101（评估链路无偏的硬证据）+ 吞吐。

用法
----
    python scripts/verify_eval_stack.py                    # 快速档（约 1 分钟）
    python scripts/verify_eval_stack.py --sample-users 2000   # 更稳的负采样校验
    python scripts/verify_eval_stack.py --eval-users 20000    # 更大的端到端评估

退出码
------
    0 = 全部通过；1 = 存在 FAIL（可直接接 CI）
"""

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")

# ---------- M2.2 / M2.3 定稿口径（改这里等于改验收标准，务必同步 docs）----------
EXPECT = dict(
    n_users=1_306_691,
    n_items=15_687,
    total_windows=109_081_471,   # Σ train_len，滑动窗口展开后的训练样本总数
    truncated_users=592_753,     # len(train) > 48 的用户数（被输入上限截断）
    max_seq_len=50,
    input_cap=48,                # 50 - 2（val / test 各占 1 个位置）
    neg_sample_num=100,
    neg_seed=98765,
)

_RESULTS = []


# =====================================================================
# 脚手架
# =====================================================================
def head(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def check(name: str, ok: bool, detail: str = "") -> bool:
    _RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(ok)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="M2.2 / M2.3 评估栈验收")
    ap.add_argument("--sample-users", type=int, default=20000,
                    help="负采样与因果性检查抽样的用户数")
    ap.add_argument("--eval-users", type=int, default=20000,
                    help="端到端评估使用的用户数")
    ap.add_argument("--seed", type=int, default=42, help="抽样种子")
    ap.add_argument("--json", default=os.path.join(PROC, "eval_stack_acceptance.json"),
                    help="结构化结果落盘路径（相对项目根）")
    return ap.parse_args()


def load_seq_dataset() -> dict:
    path = os.path.join(PROC, "seq_dataset.pkl")
    if not os.path.exists(path):
        raise SystemExit(
            f"缺少 {path}。请先跑 scripts/run_stage1.py 生成阶段一产物。")
    log(f"加载 {os.path.relpath(path, ROOT)} ...")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_cfg() -> dict:
    with open(os.path.join(ROOT, "configs", "model.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


# =====================================================================
# 主流程
# =====================================================================
def main() -> int:
    args = parse_args()
    t_start = time.time()
    sys.path.insert(0, ROOT)

    from models.data.negatives import sample_negatives
    from models.eval.evaluator import (
        EvalData, collect_ranks, drop_leaked_samples, evaluate, find_leaked_mask,
    )
    from models.sasrec.dataset import (
        TARGET_SLOTS, SlidingWindowDataset, build_eval_inputs, count_windows,
    )

    ds = load_seq_dataset()
    train, val, test = ds["train"], ds["val"], ds["test"]
    n_users = len(train)
    n_items = len(ds["smap"])
    model_cfg = load_model_cfg()
    max_seq_len = int(model_cfg["model"]["max_seq_len"])

    # =================================================================
    # A 数据集结构
    # =================================================================
    head("A 数据集结构（与阶段一定稿口径对账）")
    check("A1 用户数 == 1,306,691", n_users == EXPECT["n_users"], f"实际 {n_users:,}")
    check("A2 物品数 == 15,687", n_items == EXPECT["n_items"], f"实际 {n_items:,}")

    t0 = time.time()
    stats = count_windows(train)
    log(f"  窗口统计耗时 {time.time() - t0:.1f}s")
    check("A3 窗口总数 == Σ train_len == 109,081,471",
          stats["n_windows"] == EXPECT["total_windows"],
          f"实际 {stats['n_windows']:,}（均值 {stats['mean_windows']:.2f}）")

    ds_all = SlidingWindowDataset(train, max_seq_len=max_seq_len)
    s = ds_all.summary()
    check("A4 输入上限 = max_seq_len - 2 = 48", s["input_cap"] == EXPECT["input_cap"],
          f"实际 {s['input_cap']}")
    check("A5 Dataset 窗口数与 count_windows 一致",
          s["n_windows"] == stats["n_windows"],
          f"{s['n_windows']:,} vs {stats['n_windows']:,}")
    check("A6 被截断用户数 == 592,753",
          s["truncated_users"] == EXPECT["truncated_users"],
          f"实际 {s['truncated_users']:,}")
    check("A7 每个用户至少有 1 个窗口（train 序列非空）",
          int((ds_all.lens == 0).sum()) == 0,
          f"空序列用户 {(ds_all.lens == 0).sum()}")

    # =================================================================
    # B 因果性（真实数据）
    # =================================================================
    head("B 因果性：第 t 个样本不得看到 t 及其后的任何物品")

    # 取一个长序列（窗口多、能暴露越界），把它的后半段整体换掉，
    # 前半段对应的所有窗口输入必须逐字节不变。
    cand_rows = np.argsort(ds_all.lens)[::-1][:1]
    row = int(cand_rows[0])
    seq = list(train[row])
    cut = len(seq) // 2
    mutated = seq[:cut] + [(x % n_items) + 1 for x in seq[cut:]]

    ds_mut = SlidingWindowDataset({row: mutated}, max_seq_len=max_seq_len,
                                  user_rows=[row])
    ds_one = SlidingWindowDataset({row: seq}, max_seq_len=max_seq_len, user_rows=[row])

    # 该行在全局窗口编号中的位置 -> 用独立 dataset 逐窗口比
    same, diff = 0, 0
    for t in range(cut):
        a = ds_one[t][0].tolist()
        b = ds_mut[t][0].tolist()
        if a == b:
            same += 1
        else:
            diff += 1
    check("B1 改动未来物品后，前半段窗口输入逐字节不变",
          diff == 0 and same == cut,
          f"用户行 {row}（序列长 {len(seq)}）比对 {cut} 个窗口，不一致 {diff} 个")

    # 后半段必须真的变了，否则说明上面的比较是无意义的（比如两序列其实相同）
    changed = sum(1 for t in range(cut, len(seq))
                  if ds_one[t][1].item() != ds_mut[t][1].item())
    check("B2 被改动的后半段 target 确实不同（证明 B1 不是空比）",
          changed > 0, f"后半段 {len(seq) - cut} 个窗口中 target 变化 {changed} 个")

    # 逐个窗口断言：输入必须**严格等于** `seq[t-48:t]` 的左填充。
    # 这里刻意用「位置口径」而不是「值口径」：本数据集里用户会重复消费同一部番
    # （实测 2 万抽样用户中 1.7% 的 test 目标在更早的历史里已出现过），
    # 按值比较会把这种正常现象误判成越界。
    mismatch = 0
    n_probe = min(len(seq), 200)
    for t in range(n_probe):
        x, _ = ds_one[t]
        lo = max(0, t - EXPECT["input_cap"])
        expect = np.full(EXPECT["input_cap"], 0, dtype=np.int64)
        piece = seq[lo:t]
        if piece:
            expect[EXPECT["input_cap"] - len(piece):] = piece
        if x.numpy().tolist() != expect.tolist():
            mismatch += 1
    check("B3 输入严格等于 seq[t-48:t] 的左填充（滑窗切片无越界）",
          mismatch == 0, f"比对 {n_probe} 个窗口，不一致 {mismatch} 个")

    # 用户重复消费导致「target 值出现在历史里」——这是数据性质，不是 bug。
    # 单独报告比例，因为 evaluation-plan 5.2 要求这类样本必须剔除并记录数量。
    rep = sum(1 for t in range(n_probe) if int(seq[t]) in set(seq[:t]))
    check("B4 记录「target 值已在历史出现过」的重复消费比例（信息项，不作 FAIL）",
          True, f"{n_probe} 个窗口中 {rep} 个（{100 * rep / max(n_probe, 1):.1f}%）")

    # =================================================================
    # C 负采样
    # =================================================================
    head("C 负采样：无泄漏、无重复、确定性、与顺序/子集无关")

    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(n_users, size=min(args.sample_users, n_users),
                              replace=False)).astype(np.int64)
    pos = np.array([int(test[int(r)][0]) for r in rows], dtype=np.int64)

    t0 = time.time()
    neg = sample_negatives(rows, pos, n_items, n_negatives=EXPECT["neg_sample_num"],
                           seed=EXPECT["neg_seed"], train_seqs=train)
    dt_neg = time.time() - t0
    log(f"  负采样 {rows.size:,} 用户耗时 {dt_neg:.2f}s"
        f"（{1e6 * dt_neg / rows.size:.0f} us/用户）")

    check("C1 形状 == (N, 100)", neg.negatives.shape == (rows.size, EXPECT["neg_sample_num"]),
          f"实际 {neg.negatives.shape}")
    check("C2 索引落在 [1, 15,687]",
          int(neg.negatives.min()) >= 1 and int(neg.negatives.max()) <= n_items,
          f"[{neg.negatives.min()}, {neg.negatives.max()}]")

    dup = sum(1 for i in range(rows.size)
              if len(set(neg.negatives[i].tolist())) != EXPECT["neg_sample_num"])
    check("C3 每行 100 个负样本互不相同（无放回）", dup == 0, f"有重复的行 {dup}")
    check("C4 候选充足，无需有放回补齐", neg.n_shortfall == 0, f"shortfall={neg.n_shortfall}")

    hit_train = 0
    for i in range(rows.size):
        if set(neg.negatives[i].tolist()) & set(train[int(rows[i])]):
            hit_train += 1
    check("C5 负样本不含该用户的训练物品", hit_train == 0, f"命中 {hit_train} 行")

    hit_pos = int((neg.negatives == pos[:, None]).any(axis=1).sum())
    check("C6 负样本不含正样本本身", hit_pos == 0, f"命中 {hit_pos} 行")

    neg2 = sample_negatives(rows, pos, n_items, n_negatives=EXPECT["neg_sample_num"],
                            seed=EXPECT["neg_seed"], train_seqs=train)
    check("C7 同 seed 重复调用结果逐位一致",
          np.array_equal(neg.negatives, neg2.negatives))

    perm = rng.permutation(rows.size)
    neg3 = sample_negatives(rows[perm], pos[perm], n_items,
                            n_negatives=EXPECT["neg_sample_num"],
                            seed=EXPECT["neg_seed"], train_seqs=train)
    order_ok = all(np.array_equal(neg3.negatives[k], neg.negatives[int(perm[k])])
                   for k in range(rows.size))
    check("C8 打乱用户顺序后逐行结果不变（顺序无关）", order_ok)

    sub = np.array([0, rows.size // 2, rows.size - 1], dtype=np.int64)
    neg4 = sample_negatives(rows[sub], pos[sub], n_items,
                            n_negatives=EXPECT["neg_sample_num"],
                            seed=EXPECT["neg_seed"], train_seqs=train)
    subset_ok = all(np.array_equal(neg4.negatives[k], neg.negatives[int(sub[k])])
                    for k in range(sub.size))
    check("C9 只评子集时与该用户在全量中的负样本一致（子集无关）", subset_ok)

    check("C10 dtype 为 int32（1.31 亿条也不撑爆缓存）",
          neg.negatives.dtype == np.int32, str(neg.negatives.dtype))

    # =================================================================
    # D 评估输入
    # =================================================================
    head("D 评估输入构造：val 用 train；test 用 train+val 且不含 test 目标")

    # D 组用「均匀铺开」的 200 个用户，而不是 rows 的前 200 个：
    # rows 是按行号升序的，前 200 个全落在行号最小的一段（序列短、无重复消费），
    # 会让泄漏率恒显示为 0，掩盖真实情况。均匀铺开才能覆盖长短序列两类用户。
    ev_rows = rows[np.linspace(0, rows.size - 1, min(200, rows.size)).astype(np.int64)]
    x_val = build_eval_inputs(train, ev_rows, max_seq_len=max_seq_len, split="val")
    x_test = build_eval_inputs(train, ev_rows, max_seq_len=max_seq_len, split="test",
                               val_items=np.array([int(val[int(r)][0]) for r in ev_rows]),
                               n_items=n_items)
    check("D1 形状 == (N, 48)",
          x_val.shape == (ev_rows.size, EXPECT["input_cap"])
          and x_test.shape == (ev_rows.size, EXPECT["input_cap"]),
          f"{x_val.shape} / {x_test.shape}")

    # 说明：本数据集里用户会重复消费同一部番，因此「test 目标在更早历史里出现过」
    # 属于数据固有现象，不是构造错误。evaluation-plan 5.2 规定这类样本必须
    # 【剔除并记录数量】，故这里的判据是"比例在合理范围内"，
    # 真正的硬性要求由 E 组的"剔除后必须为 0"来保证。
    #
    # 判据用**模型实际可见的输入矩阵**（已截断到 48）而不是全历史：
    # 超长序列里位置很靠前的重复，截断后模型根本看不到，按全历史算会高估泄漏。
    tgt_test = np.array([int(test[int(r)][0]) for r in ev_rows], dtype=np.int64)
    visible_test = (x_test == tgt_test[:, None]).any(axis=1)
    rate_test = float(visible_test.mean()) if ev_rows.size else 0.0
    check("D2 test 目标可见泄漏率在数据固有水平内（< 5%，将按协议剔除）",
          rate_test < 0.05,
          f"可见泄漏 {int(visible_test.sum())}/{ev_rows.size}（{100 * rate_test:.2f}%）")

    # val 目标也不该出现在 val 输入里（val 输入只有 train）
    tgt_val = np.array([int(val[int(r)][0]) for r in ev_rows], dtype=np.int64)
    visible_val = (x_val == tgt_val[:, None]).any(axis=1)
    rate_val = float(visible_val.mean()) if ev_rows.size else 0.0
    check("D3 val 目标可见泄漏率在数据固有水平内（< 5%，将按协议剔除）",
          rate_val < 0.05,
          f"可见泄漏 {int(visible_val.sum())}/{ev_rows.size}（{100 * rate_val:.2f}%）")

    # test 输入应当比 val 输入多"一个"有效位置（多了 val 这条历史），
    # 用 PAD 数量之差来验证，而不是直接比数组相等。
    pad_val = int((x_val == 0).sum(axis=1).sum())
    pad_test = int((x_test == 0).sum(axis=1).sum())
    check("D4 test 输入的 PAD 不多于 val 输入（历史更长）", pad_test <= pad_val,
          f"PAD 数 val={pad_val:,} test={pad_test:,}")

    # =================================================================
    # E 端到端评估
    # =================================================================
    head("E 端到端：随机打分器的 HR@10 应≈ 10/101（评估链路无偏的硬证据）")

    n_eval = min(args.eval_users, rows.size)
    e_rows = rows[:n_eval]
    e_pos = pos[:n_eval]
    e_neg = neg.negatives[:n_eval]
    e_x = build_eval_inputs(train, e_rows, max_seq_len=max_seq_len, split="test",
                            val_items=np.array([int(val[int(r)][0]) for r in e_rows]),
                            n_items=n_items)

    data = EvalData(input_ids=e_x, positives=e_pos, negatives=e_neg,
                    user_rows=e_rows, label="random_baseline")

    leaked_mask = find_leaked_mask(data)
    n_leaked = int(leaked_mask.sum())
    rate_leaked = n_leaked / max(data.n_samples, 1)
    check("E1 可见答案泄漏率与 D 组一致（数据固有，按协议剔除）",
          rate_leaked < 0.05,
          f"泄漏 {n_leaked}/{data.n_samples}（{100 * rate_leaked:.2f}%）")
    data_clean, n_drop = drop_leaked_samples(data)
    check("E2 剔除后不再有任何泄漏样本（硬性要求）",
          n_drop == n_leaked and int(find_leaked_mask(data_clean).sum()) == 0,
          f"剔除 {n_drop} 条，剩余泄漏 {int(find_leaked_mask(data_clean).sum())} 条")

    # 预生成一张「物品 -> 随机分数」表：
    #   * 分数对物品**独立同分布**（这才是"随机排序"该有的样子）；
    #   * 表本身固定，所以完全确定性，与批大小/遍历顺序无关。
    #
    # ⚠️ 不要用 `(id * C) % P` 这类线性哈希冒充随机：相邻 id 的哈希差恒定，
    # 于是"正样本 id 分布"与"负样本 id 分布"的任何差异都会变成**系统性偏移**，
    # 表现为 HR 稳定偏离 10/101 且样本量再大也不收敛。本项目踩过这个坑。
    score_table = np.random.default_rng(20260912).random(n_items + 1).astype(np.float32)

    def random_score_fn(input_ids, candidate_ids):
        """确定性"随机"打分器，用于验证评估链路无偏。"""
        idx = candidate_ids.detach().cpu().numpy()
        return torch.from_numpy(score_table[idx]).to(candidate_ids.device)

    t0 = time.time()
    res = evaluate(random_score_fn, data_clean, ks=(5, 10), batch_size=1024)
    dt_eval = time.time() - t0
    n_scored = data_clean.n_samples * data_clean.n_candidates
    log(f"  评估 {data_clean.n_samples:,} 用户 × {data_clean.n_candidates} 候选 "
        f"= {n_scored:,} 次打分，耗时 {dt_eval:.2f}s "
        f"({n_scored / dt_eval / 1e4:.1f} 万次/秒)")

    # 容忍带用 3σ 而不是拍一个固定值：样本量一变，同样大小的偏差
    # 统计显著性完全不同。固定 tol 会在小样本时误报、大样本时漏报。
    def ber_tol(p: float, n: int, k: float = 3.0) -> float:
        if n <= 0:
            return 1.0
        return k * math.sqrt(p * (1 - p) / n)

    expected_hr10 = 10 / 101
    tol10 = ber_tol(expected_hr10, data_clean.n_samples)
    check("E3 随机打分器 HR@10 ≈ 10/101（3σ 容忍带）",
          abs(res["hr@10"] - expected_hr10) < tol10,
          f"实测 {res['hr@10']:.4f}，期望 {expected_hr10:.4f} ± {tol10:.4f}"
          f"（n={data_clean.n_samples}）")
    expected_hr5 = 5 / 101
    tol5 = ber_tol(expected_hr5, data_clean.n_samples)
    check("E4 随机打分器 HR@5 ≈ 5/101（3σ 容忍带）",
          abs(res["hr@5"] - expected_hr5) < tol5,
          f"实测 {res['hr@5']:.4f}，期望 {expected_hr5:.4f} ± {tol5:.4f}")
    check("E5 n_samples 随指标一起返回",
          res["n_samples"] == data_clean.n_samples,
          f"{res['n_samples']:,}")

    ranks = collect_ranks(random_score_fn, data_clean, batch_size=1024)
    check("E6 rank 落在 [1, 101]（评估链路无越界）",
          int(ranks.min()) >= 1 and int(ranks.max()) <= 101,
          f"[{int(ranks.min())}, {int(ranks.max())}]")

    res_b1 = evaluate(random_score_fn, data_clean, ks=(5, 10), batch_size=1)
    same_res = all(abs(res_b1[k] - res[k]) < 1e-12
                   for k in ("hr@5", "hr@10", "ndcg@5", "ndcg@10", "mrr"))
    check("E7 batch_size=1 与 batch_size=1024 结果一致（分块无损）", same_res)

    # =================================================================
    # 汇总
    # =================================================================
    n_total = len(_RESULTS)
    n_pass = sum(1 for r in _RESULTS if r["ok"])
    elapsed = time.time() - t_start

    head(f"汇总：{n_pass}/{n_total} 通过，耗时 {elapsed:.1f}s")
    if n_pass != n_total:
        for r in _RESULTS:
            if not r["ok"]:
                print(f"  FAIL  {r['name']}  {r['detail']}")
    else:
        print("  全部通过。评估栈口径与阶段一产物一致。")

    out_path = os.path.join(ROOT, args.json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(elapsed, 2),
        "expect": EXPECT,
        "sample_users": int(rows.size),
        "eval_users": int(data_clean.n_samples),
        "neg_sampling_us_per_user": round(1e6 * dt_neg / max(rows.size, 1), 2),
        "scoring_per_sec": round(n_scored / max(dt_eval, 1e-9)),
        "random_baseline": {k: res[k] for k in
                            ("hr@5", "hr@10", "ndcg@5", "ndcg@10", "mrr", "n_samples")},
        "n_checks": n_total,
        "n_passed": n_pass,
        "results": _RESULTS,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"结构化结果已写入 {os.path.relpath(out_path, ROOT)}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
