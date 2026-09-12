# -*- coding: utf-8 -*-
"""阶段一验收：独立重算关键结论，逐条 PASS / FAIL。

与 ``data/processed/*_report.json`` 的区别
------------------------------------------
报告是**产物**，本脚本是**检验器**。它不把任何报告当作依据，而是回到
``dataset/`` 与 ``configs/`` 原始文件重新推导每一条结论，再与产物比对。
因此报告写错、或者产物与代码不一致，都会在这里暴露。

阅读导引（A~G 各段到底在证明什么）
-----------------------------------
    A 产物完整性      15 个产物是否存在且非空。不齐就直接退出——后面所有检查都依赖它们。
    B 序列数据集结构  用户/物品数对不对、pkl 结构对不对、是否**原样保留**了官方切分。
    C 交互口径        一次全量扫 ratings.npy，独立复现「正样本=7」「过滤>=10」「重复评分口径」。
    D 题材体系与合规  12 类题材是否每类有物品、is_forbidden 标记对不对、Hentai/Erotica 确实没被剔。
    E 内容向量        形状、L2 归一化、池内矩阵(15687)与全量矩阵(20237)是否按 anime_id 精确对齐。
    F 时序口径与冷启动 确认无时间戳、holdout 新番集合正确、负样本池限定新番、池内无严格冷启动。
    G 数据源口径      仅 --full：npy 集合复现 100% 而 csv 明显偏低，用来证明「必须用 npy」。

    实现上是「线性脚本」而不是函数堆：每段用 head() 打标题，用 check() 记录一条结论，
    最后统一汇总。想加一条检查，就照抄一行 check(...) 即可。

用法
----
    python scripts/verify_stage1.py                      # 快速档（约 40 秒，跑 A~F）
    python scripts/verify_stage1.py --full               # 追加 G 段权威源比对（共约 2 分钟）
    python scripts/verify_stage1.py --sample-users 500   # 加大抽样用户数（更稳但更慢）

退出码
------
    0 = 全部通过；1 = 存在 FAIL（可直接接 CI）
"""

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "dataset")
PROC = os.path.join(ROOT, "data", "processed")
FEAT = os.path.join(ROOT, "data", "features")

# ---------- 阶段一定稿口径（改这里等于改验收标准，务必同步 docs）----------
# 这些数字是「验收标准」：脚本会拿它们和磁盘上的产物、原始数据对照。
# 任何一项调整都意味着阶段一无产物作废，必须同步 docs/evaluation-plan.md 并重跑全流程。
EXPECT = dict(
    n_ratings=148_170_496,   # ratings.npy 行数（全量评分条数）
    n_anime=20_237,          # animes.csv 动漫元数据条数
    n_users=1_306_691,       # 过滤后用户数（== len(umap)）
    n_items=15_687,          # 过滤后物品数（== len(smap)）
    pos_thr=7,               # 正样本 = rating >= 7（实测：1~6 全丢、7~10 全留）
    min_pos=10,              # 用户 / 物品过滤：正样本交互数 >= 10
    max_len=50,              # 序列截断长度
    dim=512,                 # 内容向量维度（DistilBERT 768 -> PCA 512）
    holdout_year=2021,       # 冷启动 holdout 口径的年份阈值
)

# ---- 命令行参数 ----
# 注意：本脚本是「线性脚本」，参数在模块顶层解析并直接使用，没有 main() 包装。
ap = argparse.ArgumentParser(description="阶段一验收")
ap.add_argument("--full", action="store_true", help="追加 ratings.npy vs csv 权威源比对（慢）")
ap.add_argument("--sample-users", type=int, default=300,
                help="抽样用户数：同时用于正样本阈值复现与权威源比对")
ap.add_argument("--out", default=os.path.join(PROC, "stage1_acceptance.json"),
                help="验收结果 JSON 落盘路径")
args = ap.parse_args()

# ---- 全局状态 ----
T0 = time.time()   # 计时起点，最后打印总耗时
ROWS = []          # 每条 check 的 {check, ok, detail}，最后汇总成 JSON


def head(title: str) -> None:
    """打印一个分段的标题（A~G），仅用于人眼浏览。"""
    print(f"\n{title}")
    print("-" * 66)


def check(name: str, ok: bool, detail: str = "") -> bool:
    """记录并打印一条验收结论。

    这是整个脚本唯一的「结论出口」——所有断言都必须走这里，
    这样 ROWS 才是完整的、汇总计数才不会漏项。
    """
    ROWS.append(dict(check=name, ok=bool(ok), detail=str(detail)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")
    return bool(ok)


def log(msg: str) -> None:
    """打印一条过程信息（非结论），缩进两级以区分于 PASS/FAIL。"""
    print(f"  · {msg}", flush=True)


def max_rating_by_item(rows) -> dict:
    """把同一 (用户, 物品) 的多行评分折叠为「最大评分」。

    ratings.npy 中存在同一 (userID, animeID) 出现多行的情况（抽样 300 用户中
    16 户、541 条冗余行）。dataset.pkl 的口径是**逐行判定**——只要存在任一行
    rating >= 7，该物品即计入该用户的正样本。因此这里必须取 max 而不是取最后一行，
    否则会把「先评 7 后改为 6」这类冲突项误判为负样本。
    """
    best = {}
    for a, r in rows:
        a, r = int(a), int(r)
        if r > best.get(a, -1):
            best[a] = r
    return best


# =====================================================================
# A. 产物完整性
# ---------------------------------------------------------------------
# 先确认「该有的文件都在」。这里只查「存在且非空」，不查内容对错——
# 内容由 B~G 各段负责。产物不齐就没必要往下跑，直接 sys.exit(1)。
# =====================================================================
head("A. 产物完整性")

# (目录, 文件名) 清单：11 个在 data/processed/，4 个在 data/features/
FILES = [
    (PROC, "seq_dataset.pkl"),              # 序列数据集（361MB，核心产物）
    (PROC, "anime_meta.parquet"),           # 清洗后的动漫元数据 + 12 类题材
    (PROC, "anime_detailed_tags.parquet"),  # 细分标签（内容向量用）
    (PROC, "item_stats.parquet"),           # 物品统计（交互数/年份/主题材）
    (PROC, "user_stats.parquet"),           # 用户序列长度
    (PROC, "preprocess_report.json"),       # 全量统计报告
    (PROC, "validation_report.json"),       # 与官方 dataset.pkl 的交叉校验
    (PROC, "cold_start_subset.pkl"),        # 冷启动子集
    (PROC, "cold_start_negatives.npy"),     # 冷启动专用负样本
    (PROC, "cold_start_report.json"),       # 冷启动统计报告
    (PROC, "dataset_source_verdict.json"),  # 数据源取证结论
    (FEAT, "content_vec_512.npy"),          # 内容向量（池内，15687×512）
    (FEAT, "content_vec_512_all.npy"),      # 内容向量（全量，20237×512）
    (FEAT, "content_ids.npy"),              # 全量矩阵每行对应的 anime_id
    (FEAT, "content_meta.json"),            # 内容向量构建快照
]
missing = [f for d, f in FILES if not (os.path.isfile(os.path.join(d, f))
                                       and os.path.getsize(os.path.join(d, f)) > 0)]
total_mb = sum(os.path.getsize(os.path.join(d, f)) for d, f in FILES
               if os.path.isfile(os.path.join(d, f))) / 1e6
check(f"{len(FILES)} 个产物文件存在且非空", not missing,
      f"合计 {total_mb:.0f} MB" + (f"；缺失 {missing}" if missing else ""))
if missing:
    print("\n产物不齐，后续检查无意义，提前退出。")
    sys.exit(1)

# =====================================================================
# B. 序列数据集结构（含「未污染官方切分」回归校验）
# ---------------------------------------------------------------------
# 本段回答三个问题：
#   1) 规模对不对（用户 1,306,691 / 物品 15,687）；
#   2) 结构能不能按位置索引取出（0..N-1 稠密，不是 dict 稀疏）；
#   3) **有没有擅自改动官方 dataset.pkl 的切分**（这是最关键的回归项）。
# =====================================================================
head("B. 序列数据集结构")

with open(os.path.join(PROC, "seq_dataset.pkl"), "rb") as f:
    out = pickle.load(f)
train, val, test = out["train"], out["val"], out["test"]
umap, smap, meta_seq = out["umap"], out["smap"], out.get("meta")

check("输出 pkl 含 train/val/test/umap/smap/meta 六项",
      all(k in out for k in ("train", "val", "test", "umap", "smap", "meta")),
      f"keys = {sorted(out.keys())}")

n_u, n_i = len(umap), len(smap)
check("用户数 == 1,306,691", n_u == EXPECT["n_users"], f"实际 {n_u:,}")
check("物品数 == 15,687", n_i == EXPECT["n_items"], f"实际 {n_i:,}")

# 「稠密」= 索引 0 / 中间 / 末尾三个位置都能直接取到，说明是 list 而非 dict。
# 官方 dataset.pkl 是稠密结构，我们的产物必须保持一致，否则下游按位置取会 KeyError。
dense = all(all(i in d for i in (0, n_u // 2, n_u - 1)) for d in (train, val, test))
check("train/val/test 以 0..N-1 位置索引可访问（稠密）", dense,
      f"train[0]={len(train[0])} train[-1]={len(train[n_u - 1])}")

# meta 是我们自己加的字典，记录切分口径；用它可追溯「这份数据是怎么来的」。
check("meta 口径：order_source=dataset.pkl / split=leave_one_out / max_len=50",
      isinstance(meta_seq, dict)
      and meta_seq.get("order_source") == "dataset.pkl"
      and meta_seq.get("split") == "leave_one_out"
      and int(meta_seq.get("max_len", 0)) == EXPECT["max_len"],
      f"meta = {meta_seq}")

# ---- 回归校验：产物 vs 官方 dataset.pkl ----
# 抽 2000 个用户，逐个比 train/val/test 的内容是否**完全相同**。
# 这一项如果 FAIL，说明我们偷偷改了切分（比如剔了 Hentai），那所有实验都不可比。
with open(os.path.join(DATASET, "dataset.pkl"), "rb") as f:
    src = pickle.load(f)
probe_idx = random.Random(42).sample(range(n_u), 2000)   # 固定种子，可复现
same = sum(1 for i in probe_idx
           if list(train[i]) == list(src["train"][i])
           and list(val[i]) == list(src["val"][i])
           and list(test[i]) == list(src["test"][i]))
check("输出 pkl 的 train/val/test 与官方切分逐键一致（抽 2000 用户）",
      same == len(probe_idx), f"一致 {same}/{len(probe_idx)}")
# umap/smap 是 id ↔ index 的映射，被改写会导致索引错位，单独校验。
check("umap / smap 与官方切分完全一致",
      umap == src["umap"] and smap == src["smap"], "id 映射未被改写")
del src   # dataset.pkl 很大，用完立刻释放

# 每个用户所有正样本总数的最小值应当 >= 10（这正是过滤阈值）。
# 注意这是「从产物反推阈值」的独立验证，不是读配置。
tot = np.fromiter((len(train[i]) + len(val[i]) + len(test[i]) for i in range(n_u)),
                  dtype=np.int64, count=n_u)
check(f"每用户正样本总数 >= {EXPECT['min_pos']}（独立验证过滤阈值）",
      int(tot.min()) >= EXPECT["min_pos"],
      f"最小值 {int(tot.min())}，中位数 {int(np.median(tot))}，合计 {int(tot.sum()):,}")
del tot

# =====================================================================
# C. 交互口径：一次全量扫描 ratings.npy
# ---------------------------------------------------------------------
# 这是全脚本最重的一段，目标是用**原始评分文件**独立推出三条口径：
#   C1  正样本阈值 = rating >= 7，且重复评分按「逐行」而非「最后一行」判定
#   C2  用户过滤阈值 = 正样本交互数 >= 10（复现出 pkl 的用户数）
#   C3  物品过滤阈值 = 正样本交互数 >= 10（精确复现出 len(smap)）
# 做法：只扫一遍文件，边扫边用 bincount 累计计数（向量化，避免 1.5 亿次循环）。
# =====================================================================
head(f"C. 交互口径（全量扫描 ratings.npy + {args.sample_users} 用户逐项复现，约 40 秒）")

npy_path = os.path.join(DATASET, "ratings.npy")
# mmap_mode="r"：把 6.9GB 文件当「磁盘上的数组」用，只把当前切片读进内存。
# 全量 148,170,496×3 的 int32 约 1.7GB，直接 np.load 会吃掉大量内存，没必要。
arr = np.load(npy_path, mmap_mode="r")
check("ratings.npy 形状 == (148,170,496, 3)",
      arr.shape == (EXPECT["n_ratings"], 3), f"实际 {arr.shape}")

inv_smap = {int(v): int(k) for k, v in smap.items()}   # index -> anime_id

# 抽样用户：固定种子 0，保证「快速档」和「--full 档」抽的是同一批人，结果可比。
# 抽样用户要两用：C1 逐项复现正样本集合；G 段比对 npy / csv。
sample_uids = random.Random(0).sample(sorted(umap.keys()), args.sample_users)
sample_set = np.array(sorted(int(u) for u in sample_uids), dtype=np.int64)

# 计数数组。下标 = 原始 userID / animeID（不是连续索引），所以要按 id 上界开。
#   用户 id 实测最大 1,774,522 -> 开 2,000,000 足够（约 16MB，可接受）
#   动漫 id 实测最大 20,237 -> 开 n_anime+1
# bincount 的 minlength 必须传这个长度，否则短数组会加到超出长度的下标而报错。
cnt_u = np.zeros(2_000_000, dtype=np.int64)
cnt_a = np.zeros(EXPECT["n_anime"] + 1, dtype=np.int64)
sample_rows = {int(u): [] for u in sample_uids}   # 抽样用户的逐行原始评分

# ---- 分块扫描：每块 2000 万行 ----
for s in range(0, arr.shape[0], 20_000_000):
    b = np.asarray(arr[s:s + 20_000_000])          # 把 mmap 切片真正读进内存
    m = b[:, 2] >= EXPECT["pos_thr"]               # 正样本掩码
    if m.any():
        bb = b[m]
        # 累计「每用户 / 每物品的正样本数」——这就是过滤阈值要用的量
        cnt_u += np.bincount(bb[:, 0], minlength=2_000_000)
        cnt_a += np.bincount(bb[:, 1], minlength=EXPECT["n_anime"] + 1)
    # 顺手把抽样用户的行原样存下来（保留重复行，C1 需要判断重复口径）
    pm = np.isin(b[:, 0], sample_set)
    if pm.any():
        for u, a, r in b[pm]:
            sample_rows[int(u)].append((int(a), int(r)))
log(f"扫描完成：正样本 {int(cnt_a.sum()):,} 条 "
    f"（占全部评分 {cnt_a.sum() / EXPECT['n_ratings']:.2%}）")

# ---- C1. 正样本阈值 = 7，且按「逐行」口径（max 而非最后一行）----
# 对每个抽样用户：用「原始评分 >= 7 且属于 smap」重建物品集合，与 pkl 对比。
# 同时统计重复 (用户,物品) 行的规模，以及是否存在跨 7 分冲突（用来暴露 last-value 陷阱）。
ok_thr, n_dup_users, n_dup_rows, n_conflict, conflict_ok = True, 0, 0, 0, True
bad_detail = []
for u in sample_uids:
    idx = umap[u]
    # pkl 里该用户的全部物品（train+val+test 三条序列拼起来）
    pkl_items = {inv_smap[int(x)] for x in
                 (list(train[idx]) + list(val[idx]) + list(test[idx]))}
    rows_u = sample_rows[int(u)]
    by_item = defaultdict(list)
    for a, r in rows_u:
        by_item[int(a)].append(int(r))
    if len(rows_u) != len(by_item):
        # 同一个 (用户,物品) 出现多行 -> 统计重复规模
        n_dup_users += 1
        n_dup_rows += len(rows_u) - len(by_item)
        # 冲突 = 取 max 和取最后一行会得出不同的正/负判定
        if any((max(rs) >= EXPECT["pos_thr"]) != (rs[-1] >= EXPECT["pos_thr"])
               for rs in by_item.values()):
            n_conflict += 1
            m = max_rating_by_item(rows_u)
            row_wise = {a for a, r in m.items() if r >= EXPECT["pos_thr"] and a in smap}
            if row_wise != pkl_items:
                conflict_ok = False
    pos = {a for a, r in max_rating_by_item(rows_u).items()
           if r >= EXPECT["pos_thr"] and a in smap}
    if pos != pkl_items:
        ok_thr = False
        if len(bad_detail) < 3:
            bad_detail.append(f"uid={u} 对称差={len(pos ^ pkl_items)}")

check(f"正样本阈值 = {EXPECT['pos_thr']}（{len(sample_uids)} 用户逐项复现）", ok_thr,
      "; ".join(bad_detail) if bad_detail else
      f"{len(sample_uids)}/{len(sample_uids)} 集合完全一致")
check("重复评分按「任一行 >= 7 即正样本」的逐行口径处理", conflict_ok,
      f"{n_dup_users}/{len(sample_uids)} 户存在同一 (用户,物品) 多行评分"
      f"（{n_dup_rows} 条冗余），其中 {n_conflict} 户存在跨 7 分冲突，均按逐行口径复现")

# ---- C2. 用户过滤阈值 ----
# cnt_u[0] 是「userID=0」这个不存在的用户，切片 [1:] 跳过它。
# 独立算出的合格用户数应当和 pkl 用户数几乎相等，容差 200 是给边界情形留的余量
# （实测差 14，见 preprocess.py docstring 第 2 条偏差）。
n_u_ge10 = int((cnt_u[1:] >= EXPECT["min_pos"]).sum())
check(f"正样本>={EXPECT['min_pos']} 的用户数 ≈ pkl 用户数（容差 200）",
      abs(n_u_ge10 - n_u) <= 200,
      f"独立重算 {n_u_ge10:,} vs pkl {n_u:,}（差 {n_u_ge10 - n_u}）")

# ---- C3. 物品过滤阈值 ----
# 这一项要求**精确相等**：独立算出的物品数必须正好是 15,687 == len(smap)。
# 它是「过滤规则被正确识别」最硬的证据（比 C2 更硬，因为物品侧没有边界歧义）。
n_i_ge10 = int((cnt_a[1:] >= EXPECT["min_pos"]).sum())
check(f"正样本>={EXPECT['min_pos']} 的物品数 == 15,687（精确）",
      n_i_ge10 == n_i, f"独立重算 {n_i_ge10:,} vs len(smap) {n_i:,}")
del arr, cnt_u, cnt_a   # 释放 mmap 与计数数组

# =====================================================================
# D. 题材体系与合规
# ---------------------------------------------------------------------
# 这一段把「题材映射」与「合规标记」都从**原始 json + 配置文件**重新推一遍：
#   · 12 类中文题材是不是每类都真的有物品（别出现空类）
#   · is_forbidden 标记与我们独立判定的 Hentai/Erotica 集合是否完全一致
#   · 顺便证伪一个早期文档结论：Hentai/Erotica **并没有**从推荐池里被剔除
# =====================================================================
head("D. 题材体系与合规")

meta = pd.read_parquet(os.path.join(PROC, "anime_meta.parquet"))
check("anime_meta.parquet 行数 == 20,237", len(meta) == EXPECT["n_anime"],
      f"实际 {len(meta):,}")

# 用 MAL 官方题材映射 id_to_genreids.json 独立判定「哪些动漫含 Hentai(20)/Erotica(21)」
gmap = json.load(open(os.path.join(DATASET, "id_to_genreids.json"), encoding="utf-8"))
forbidden_ids = {20, 21}   # MAL genre id：20=Erotica, 21=Hentai
forbid_indep = {int(k) for k, v in gmap.items()
                if any(int(x) in forbidden_ids for x in v)}
# 再看这些不合规动漫有多少**仍在推荐池内**——如果为 0，说明确实被剔过
forbid_in_pool = forbid_indep & set(int(k) for k in smap.keys())
check("Hentai/Erotica 未被剔除：池内仍含 > 1000 个此类物品",
      len(forbid_in_pool) > 1000,
      f"独立重算池内 {len(forbid_in_pool):,} 个（占物品池 {len(forbid_in_pool) / n_i:.2%}）")

# 产物的 is_forbidden 标记必须与独立判定**完全相等**（不能多标也不能漏标）
flag = set(meta.loc[meta["is_forbidden"], "anime_id"].astype(int))
check("anime_meta.is_forbidden 与独立判定一致",
      flag == forbid_indep, f"标记 {len(flag):,} 个 vs 独立 {len(forbid_indep):,} 个")

# 12 类题材的覆盖检查：每类的主题材物品数都应当 > 0
tx = yaml.safe_load(open(os.path.join(ROOT, "configs", "genre_taxonomy.yaml"),
                         encoding="utf-8"))
names = {int(c): s["name_cn"] for c, s in tx["genres_12"].items()}
cnt_primary = meta["primary_genre"].value_counts().to_dict()
present = {names[c] for c in names if cnt_primary.get(names[c], 0) > 0}
n_null = int(meta["primary_genre"].isna().sum())
check("12 类中文题材每类都有物品", len(present) == 12,
      f"命中 {len(present)}/12；无 12 类题材物品 {n_null} 条")
# 守恒式：有题材的 + 无题材的 == 总数。防止映射时静默丢条目。
check("题材覆盖数 + 无题材数 == 20,237",
      sum(cnt_primary.values()) + n_null == EXPECT["n_anime"],
      f"{sum(cnt_primary.values())} + {n_null} = {sum(cnt_primary.values()) + n_null}")

# =====================================================================
# E. 内容向量
# ---------------------------------------------------------------------
# 两个矩阵必须能对上：
#   content_vec_512.npy     (15687, 512) —— 按 smap 的 idx 排，第 i-1 行 = 物品 idx i
#   content_vec_512_all.npy (20237, 512) —— 按 anime_id 升序排
# 这里用 anime_id 当桥梁，验证两者指向同一个物品时向量完全一致（对齐检查）。
# =====================================================================
head("E. 内容向量")

vec_pool = np.load(os.path.join(FEAT, "content_vec_512.npy"))
vec_all = np.load(os.path.join(FEAT, "content_vec_512_all.npy"))
ids_all = np.load(os.path.join(FEAT, "content_ids.npy"))

check(f"content_vec_512.npy 形状 == ({EXPECT['n_items']}, {EXPECT['dim']})",
      vec_pool.shape == (EXPECT["n_items"], EXPECT["dim"]), f"实际 {vec_pool.shape}")
check(f"content_vec_512_all.npy 形状 == ({EXPECT['n_anime']}, {EXPECT['dim']})",
      vec_all.shape == (EXPECT["n_anime"], EXPECT["dim"]), f"实际 {vec_all.shape}")

# L2 归一化：FAISS 用内积检索时，内积 == 余弦相似度。若没归一化，相似度会被向量模长污染。
nrm = np.linalg.norm(vec_pool, axis=1)
check("池内矩阵每行已 L2 归一化", abs(nrm.max() - 1) < 1e-3 and abs(nrm.min() - 1) < 1e-3,
      f"范数范围 [{nrm.min():.6f}, {nrm.max():.6f}]")

# content_ids 必须是全量 anime_id 升序 —— 这是 all 矩阵的行序约定，被打破就全错位
check("content_ids 为全量 anime_id 升序",
      ids_all.shape[0] == EXPECT["n_anime"]
      and np.array_equal(ids_all, np.sort(meta["anime_id"].to_numpy(np.int32))),
      f"长度 {ids_all.shape[0]:,}，是否升序={bool(np.all(np.diff(ids_all) > 0))}")

# ---- 对齐检查：把「池内矩阵的行」和「全量矩阵里同一 anime_id 的行」逐条比对 ----
# searchsorted 在升序的 ids_all 里找到每个 anime_id 的位置，等价于一次字典查表（更快）。
a_ids = np.array(sorted(int(k) for k in smap.keys()), dtype=np.int32)
idxs = np.array([int(smap[int(a)]) for a in a_ids])       # 该 anime_id 在池内的 idx
pos = np.searchsorted(ids_all, a_ids)                     # 该 anime_id 在全量矩阵的行号
aligned = bool(np.array_equal(ids_all[pos], a_ids)
               and np.allclose(vec_pool[idxs - 1], vec_all[pos], atol=1e-5))
check("池内矩阵第 i-1 行 == 全量矩阵中同一 anime_id 的行（按 id 对齐）", aligned,
      f"比对 {len(a_ids):,} 个物品")
# 空行 = 某个池内物品没有内容向量（会让它的召回永远为 0），必须为 0 个
check("池内矩阵无空行（smap 中每个物品都有内容向量）",
      not bool(np.all(vec_pool == 0, axis=1).any()),
      f"空行数 {int(np.all(vec_pool == 0, axis=1).sum())}")

# =====================================================================
# F. 时序口径与冷启动
# ---------------------------------------------------------------------
#   · 证伪「按时间戳排序」：ratings.csv 只有 3 列，压根没有时间戳字段
#   · 独立重算 holdout 新番集合 = {year >= 2021} ∩ 推荐池，与产物比对
#   · 负样本必须只来自新番池（否则模型能靠热度先验作弊，内容融合贡献被高估）
#   · 证明池内**不存在**严格物品冷启动（每个物品训练集至少出现 6 次）
# =====================================================================
head("F. 时序口径与冷启动")

# 只读表头一行即可判定有没有时间戳列（不解析全文件，省时间）
hdr = open(os.path.join(DATASET, "ratings.csv"), encoding="utf-8").readline().strip()
check("ratings.csv 只有 userID/animeID/rating 三列（无时间戳）",
      hdr.split(",") == ["userID", "animeID", "rating"], f"表头 = {hdr}")

# 独立重算新番集合：item_stats 里的 year 作为年份来源，fillna(0) 让缺失年份一律不算新番
item_stats = pd.read_parquet(os.path.join(PROC, "item_stats.parquet"))
year_of = dict(zip(item_stats["anime_id"].astype(int), item_stats["year"].fillna(0)))
new_indep = {int(a) for a in smap.keys()
             if year_of.get(int(a), 0) >= EXPECT["holdout_year"]}

with open(os.path.join(PROC, "cold_start_subset.pkl"), "rb") as f:
    cs = pickle.load(f)
check(f"holdout 新番集合 == {{year >= {EXPECT['holdout_year']}}} ∩ 推荐池",
      {int(a) for a in cs["new_item_ids"]} == new_indep,
      f"子集 {len(cs['new_item_ids']):,} 个 vs 独立重算 {len(new_indep):,} 个")

# 负样本矩阵形状：(测试样本数, 每样本 100 个负例)
negs = np.load(os.path.join(PROC, "cold_start_negatives.npy"))
check("负样本池形状 == (样本数, 100)",
      negs.ndim == 2 and negs.shape[1] == 100, f"实际 {negs.shape}")
# 样本与负样本必须一一对应，否则评估时索引会错位
check("每条测试样本都有对应负样本", len(negs) == len(cs["test_sample_user_indices"]),
      f"负样本 {len(negs):,} 行 / 样本 {len(cs['test_sample_user_indices']):,} 条")

# 关键：负样本取值必须落在新番池的 smap 索引集合内（不是 anime_id，是池内索引！）
neg_ids = set(int(x) for x in np.unique(negs))
check("负样本只取自新番集合（同分布）",
      neg_ids <= set(int(a) for a in cs["new_item_pool_indices"]),
      f"负样本涉及 {len(neg_ids):,} 个不同物品（smap 索引），均属新番池")

# 统计每个物品在训练集里出现了几次：最小值 >= 1 就说明「池内没有被完全遮住的物品」，
# 也就意味着不存在严格物品冷启动 —— 这是改用 holdout 模拟口径的直接依据。
# 分批 concat 是为了避免一次性把 1 亿多个索引拼成超长数组。
cnt_train = np.zeros(EXPECT["n_items"] + 1, dtype=np.int64)
for s in range(0, n_u, 50_000):
    e = min(s + 50_000, n_u)
    flat = np.concatenate([np.asarray(train[i], dtype=np.int64) for i in range(s, e)])
    cnt_train += np.bincount(flat, minlength=EXPECT["n_items"] + 1)
mn = int(cnt_train[1:].min())
check("推荐池内不存在 0 次物品 → 无严格物品冷启动（故须用 holdout 模拟）",
      mn >= 1,
      f"训练集中物品最少出现 {mn} 次；出现 0 次的物品 {int((cnt_train[1:] == 0).sum())} 个")

# =====================================================================
# G. 数据源口径（可选，慢）
# ---------------------------------------------------------------------
# 只做一件事：分别用 ratings.npy 和 ratings.csv 重建物品集合，看谁与 dataset.pkl 一致。
# 预期 npy == 100%，csv 明显偏低（实测 95%），从而证明「必须用 npy」。
# 之所以慢：csv 没有索引，只能逐行读完整 148,170,496 行（约 1 分钟）。
# =====================================================================
if args.full:
    head(f"G. 权威源比对：ratings.npy vs ratings.csv（{len(sample_uids)} 用户，约 2 分钟）")
    uidset = set(int(u) for u in sample_uids)
    csv_rows = defaultdict(list)
    with open(os.path.join(DATASET, "ratings.csv"), encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd)                       # 跳过表头
        for row in rd:
            try:
                u = int(row[0])
            except Exception:          # 个别脏行直接跳过，不让它中断整段比对
                continue
            if u in uidset:            # 只留抽样用户的行（流式过滤，省内存）
                csv_rows[u].append((int(row[1]), int(row[2])))
    log(f"csv 收集完成：{sum(len(v) for v in csv_rows.values()):,} 行")

    # 对每个抽样用户，分别用 npy / csv 重建集合并与 pkl 比对（同样走 max 折叠，见 max_rating_by_item）
    npy_ok = csv_ok = 0
    for u in uidset:
        idx = umap[u]
        pkl_items = {inv_smap[int(x)] for x in
                     (list(train[idx]) + list(val[idx]) + list(test[idx]))}
        npy_set = {a for a, r in max_rating_by_item(sample_rows.get(int(u), [])).items()
                   if r >= EXPECT["pos_thr"] and a in smap}
        csv_set = {a for a, r in max_rating_by_item(csv_rows.get(u, [])).items()
                   if r >= EXPECT["pos_thr"] and a in smap}
        npy_ok += int(npy_set == pkl_items)
        csv_ok += int(csv_set == pkl_items)

    r_csv = csv_ok / len(uidset)
    check("ratings.npy 集合复现率 == 100%", npy_ok == len(uidset),
          f"{npy_ok}/{len(uidset)} = {npy_ok / len(uidset):.1%}")
    # 阈值 0.99 而不是「<100%」：csv 本就不是权威源，这里只要证明「明显更差」即可
    check("ratings.csv 集合复现率明显低于 npy（证明二者口径不同）",
          r_csv < 0.99, f"{csv_ok}/{len(uidset)} = {r_csv:.1%}")
else:
    print("\n（未启用 --full：跳过 ratings.npy vs csv 的权威源比对）")

# =====================================================================
# 汇总：统计 PASS/FAIL、打印失败明细、写 JSON、用退出码表达结果
# =====================================================================
n_pass = sum(1 for r in ROWS if r["ok"])
n_fail = len(ROWS) - n_pass
print("\n" + "=" * 66)
print(f"验收结果：{n_pass}/{len(ROWS)} 项通过，{n_fail} 项失败    耗时 {time.time() - T0:.0f}s")
print("=" * 66)
if n_fail:   # 有失败就把它们单独列出来，方便直接定位
    print("未通过项：")
    for r in ROWS:
        if not r["ok"]:
            print(f"  - {r['check']}  {r['detail']}")

summary = dict(
    passed=n_pass, failed=n_fail, total=len(ROWS),
    elapsed_sec=round(time.time() - T0, 1),
    expect=EXPECT, full_mode=bool(args.full),
    sample_users=len(sample_uids), checks=ROWS,
)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"结果已写入 {os.path.relpath(args.out, ROOT)}")
# 退出码：0=全过，1=有失败。这样 `verify_stage1.py && echo OK` 之类可以脚本化判断。
sys.exit(0 if n_fail == 0 else 1)
