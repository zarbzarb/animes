# -*- coding: utf-8 -*-
"""阶段 1-A：原始数据清洗 + 序列数据集构建 + 题材体系 + 新番子集。

用法
----
    # 全流程（推荐）
    python scripts/preprocess.py

    # 指定配置 / 只跑清洗不建新番子集
    python scripts/preprocess.py --config configs/data.yaml --skip-coldstart

    # 抽样校验规模（快速自检用，不产出正式数据集）
    python scripts/preprocess.py --validate-only --validate-users 200

产物（默认写入 data/processed/）
--------------------------------
    anime_meta.parquet      清洗后的动漫元数据 + 12 类题材标签 + is_primary
    seq_dataset.pkl         序列数据集 {train,val,test,umap,smap,meta}
    item_stats.parquet      每个物品的交互数 / 是否新番 / 主题材
    user_stats.parquet      每个用户的序列长度
    preprocess_report.json  全量统计报告（论文「数据集」节素材）
    validation_report.json  与 dataset.pkl 的交叉校验结果

设计说明（重要，与 rq.md 的偏差已在此更正）
-------------------------------------------
0. **权威数据源 = `dataset/ratings.npy`**。三个交互文件并非同一口径：
   `ratings.csv` 与 `ratings.dat` 完全相同，但相对 `ratings.npy` 存在整体 +1 偏移。
   用两者分别重建 `dataset.pkl` 的物品集合，1500 个抽样用户的一致率为
   **100.0%（npy）** 与 **90.7%（csv）**。全量抽样明细：可比评分单元 166,567 对，
   其中 8,896 对（5.34%）不同，且 99.2% 为 `csv = npy + 1`；
   落在 6/7 正样本边界、会造成判定翻转的有 1,494 对（0.90%）。
   故 pkl 由 npy 口径构建，本管线以 npy 为准。
   复现证据：`python scripts/diagnose_dataset_source.py`
   → `data/processed/dataset_source_verdict.json`
1. **正样本定义**：`rating >= 7`。抽样用户的评分保留/丢弃分布完全二分——
   rating 1~6 全部丢弃（0 保留），rating 7~10 全部保留（0 丢弃），无例外。
2. **过滤阈值**：用户与物品均按「正样本交互数 >= 10」过滤（非早期文档写的 <5 / <10）。
   复核：正样本>=10 的用户 1,306,705，pkl 为 1,306,691（差 14，属边界情形）；
   正样本>=10 的物品恰为 15,687 个 == len(smap)。
3. **序列顺序**：原始交互文件均无时间戳，pkl 顺序不可复现（npy 下集合 100% 可复现、
   顺序仅约 60%）。因此**以 dataset.pkl 的顺序为唯一时序口径**。
4. **合规题材**：实测 dataset.pkl **未**剔除 Hentai/Erotica，有 1,551 个此类物品
   在推荐池内。本项目在候选池与评估阶段屏蔽它们，但不改动提供的切分。
5. **无剧情简介**：animes.csv 无简介字段，内容文本用 标题+12类题材+细分标签 拼接，
   详见 scripts/build_content_vectors.py。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ID 上界取自全量实测（logs/probe_stage1i.txt）：userID <= 1,774,522；animeID <= 20,237
MAX_USER_ID = 1_774_522
MAX_ANIME_ID = 20_237

# ---------- 正样本阈值（实测得出，改动会使数据集不可比）----------
POSITIVE_RATING_THRESHOLD = 7


# =====================================================================
# 工具
# =====================================================================
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_taxonomy(path: str) -> tuple[dict, dict, list, list, set]:
    """返回 (genre_id->name, mal_id->中文类id, primary优先级, 音乐关键词, 禁用genre_id)"""
    tx = load_cfg(path)
    id2name = {int(k): v for k, v in tx["mal_genres"].items()}
    mal2cls = {}
    for cid, spec in tx["genres_12"].items():
        for mid in spec.get("mal_ids") or []:
            mal2cls[int(mid)] = int(cid)
    priority = list(tx["primary_priority"])
    music_kw = [k.lower() for k in tx["music_keywords"]]
    forbidden = set(tx["forbidden_genre_ids"])
    return id2name, mal2cls, priority, music_kw, forbidden


def parse_list_field(v) -> list:
    """animes.csv 的 genres / genres_detailed 是字符串形式的 python list。"""
    if not isinstance(v, str):
        return []
    try:
        r = ast.literal_eval(v)
        return list(r) if isinstance(r, (list, tuple)) else []
    except Exception:
        return []


def to_int_year(v):
    try:
        y = int(float(str(v).strip()))
        return y if 1900 <= y <= 2100 else None
    except Exception:
        return None


def to_float_score(v):
    """score 列含占位符 '?'，需显式转 NaN。"""
    s = str(v).strip()
    if s in {"", "?", "nan", "None"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


# =====================================================================
# 1. 动漫元数据清洗 + 12 类题材体系
# =====================================================================
def build_anime_meta(cfg: dict, id2name, mal2cls, priority, music_kw, forbidden) -> pd.DataFrame:
    log("读取 animes.csv ...")
    df = pd.read_csv(cfg["input"]["anime_meta"])
    raw_n = len(df)
    log(f"  原始 {raw_n} 条, 列: {list(df.columns)}")

    out = pd.DataFrame()
    out["anime_id"] = df["animeID"].astype(int)
    out["title"] = df["title"].fillna("").astype(str).str.strip()
    out["alternative_title"] = df["alternative_title"].fillna("").astype(str).str.strip()
    out["type"] = df["type"].fillna("UNKNOWN").astype(str).str.strip()
    out["year"] = df["year"].map(to_int_year)
    out["score"] = df["score"].map(to_float_score)
    out["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
    out["mal_url"] = df["mal_url"].fillna("").astype(str)
    out["image_url"] = df["image_url"].fillna("").astype(str)
    out["sequel"] = df["sequel"].fillna(False).astype(bool)

    # --- genres / genres_detailed ---
    out["genres"] = df["genres"].map(parse_list_field)
    out["genres_detailed"] = df["genres_detailed"].map(parse_list_field)

    # --- genre_id 映射（来自 id_to_genreids.json，覆盖全部 20,237 条）---
    log("读取 id_to_genreids.json ...")
    with open(cfg["input"]["genre_map"], encoding="utf-8") as f:
        gmap = json.load(f)
    gid_of = {int(k): sorted(set(int(x) for x in v)) for k, v in gmap.items()}
    out["genre_ids"] = out["anime_id"].map(lambda a: gid_of.get(a, []))

    # --- 12 类中文题材 ---
    # 「青春音乐」用词边界正则匹配细分标签：子串匹配会让 contraband 命中 band、
    # crossdressing 命中 sing，产生大量误判。
    kw_pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in music_kw) + r")\b") if music_kw else None

    def is_music(det: list) -> bool:
        if kw_pattern is None:
            return False
        return any(kw_pattern.search(str(t).lower()) for t in det)

    def to_12(gids: list, det: list) -> list:
        cls = []
        for g in gids:
            if g in forbidden:
                continue
            c = mal2cls.get(g)
            if c and c not in cls:
                cls.append(c)
        if is_music(det) and 11 not in cls:
            cls.append(11)
        return sorted(cls)

    out["genres_cn_ids"] = [to_12(g, d) for g, d in zip(out["genre_ids"], out["genres_detailed"])]
    cid2name = {cid: spec["name_cn"] for cid, spec in load_cfg(
        os.path.join(ROOT, "configs", "genre_taxonomy.yaml"))["genres_12"].items()}
    out["genres_cn"] = out["genres_cn_ids"].map(lambda ids: [cid2name[i] for i in ids])

    # --- is_primary：按优先级取第一个命中的题材 ---
    rank = {cid: i for i, cid in enumerate(priority)}

    def pick_primary(ids: list):
        if not ids:
            return None
        return sorted(ids, key=lambda c: rank.get(c, 999))[0]

    out["primary_genre_id"] = out["genres_cn_ids"].map(pick_primary)
    out["primary_genre"] = out["primary_genre_id"].map(lambda i: cid2name.get(i) if i else None)

    # --- 合规标记 ---
    out["is_forbidden"] = out["genre_ids"].map(lambda gs: any(g in forbidden for g in gs))

    # --- 异常值清理 ---
    n_bad_year = int(out["year"].isna().sum())
    n_bad_score = int(out["score"].isna().sum())
    log(f"  清洗: year 异常 {n_bad_year} 条, score 缺失/占位 {n_bad_score} 条, "
        f"合规剔除 {int(out['is_forbidden'].sum())} 条")
    return out


# =====================================================================
# 2. 扫描交互数据统计交互数（正样本口径）
# =====================================================================
def scan_ratings(cfg: dict, sample_uids: set) -> dict:
    """以 ratings.npy 为准统计（权威源判定见下方 AUDIT 说明）。

    权威源判定（实测，data/processed/dataset_source_verdict.json）：
      用 ratings.csv 重建物品集合，1500 抽样用户一致率仅 90.7%；
      用 ratings.npy 重建，一致率 100.0%（1500/1500）。
      可比评分单元 166,567 对，其中 8,896 对（5.34%）不同，且 99.2% 为 csv = npy + 1；
      落在 6/7 边界、会造成正样本判定翻转的有 1,494 对（0.90%）。
      ratings.csv 与 ratings.dat 完全相同（前 200 万行逐字节比对一致）。
      => dataset.pkl 由 ratings.npy 口径构建，故以其为准。
    """
    npy_path = os.path.join(ROOT, "dataset", "ratings.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"缺少权威源 {npy_path}；如需改用 csv，请修改本函数")
    log(f"扫描权威源 {npy_path}（rating>={POSITIVE_RATING_THRESHOLD} 记为正样本）...")
    arr = np.load(npy_path, mmap_mode="r")
    N = arr.shape[0]

    cnt_u_pos = np.zeros(MAX_USER_ID + 2, dtype=np.int64)
    cnt_a_pos = np.zeros(MAX_ANIME_ID + 2, dtype=np.int64)
    cnt_u_all = np.zeros(MAX_USER_ID + 2, dtype=np.int64)
    cnt_a_all = np.zeros(MAX_ANIME_ID + 2, dtype=np.int64)
    rating_hist = np.zeros(12, dtype=np.int64)

    sample_mask = np.zeros(MAX_USER_ID + 2, dtype=bool)
    if sample_uids:
        sample_mask[np.fromiter(sample_uids, dtype=np.int64)] = True
    sample_rows = defaultdict(list)

    CH = 10_000_000
    for s in range(0, N, CH):
        b = np.asarray(arr[s:s + CH])
        u, a, r = b[:, 0].astype(np.int64), b[:, 1].astype(np.int64), b[:, 2].astype(np.int64)
        cnt_u_all += np.bincount(u, minlength=MAX_USER_ID + 2)
        cnt_a_all += np.bincount(a, minlength=MAX_ANIME_ID + 2)
        m = r >= POSITIVE_RATING_THRESHOLD
        cnt_u_pos += np.bincount(u[m], minlength=MAX_USER_ID + 2)
        cnt_a_pos += np.bincount(a[m], minlength=MAX_ANIME_ID + 2)
        rating_hist += np.bincount(np.clip(r, 0, 11), minlength=12)
        if sample_uids:
            sel = b[sample_mask[b[:, 0]]]
            if len(sel):
                for uu, aa, rr in sel:
                    sample_rows[int(uu)].append((int(aa), int(rr)))
        if s % 50_000_000 == 0:
            log(f"  ... {s:,}/{N:,}")
    log(f"  扫描完成，共 {N:,} 行；抽样用户 {len(sample_rows):,} 个")
    return dict(cnt_u_pos=cnt_u_pos, cnt_a_pos=cnt_a_pos, cnt_u_all=cnt_u_all,
                cnt_a_all=cnt_a_all, rating_hist=rating_hist,
                sample_rows=dict(sample_rows), n_rows=int(N))


def audit_rating_sources(cfg: dict, n_rows: int = 2_000_000) -> dict:
    """对照 ratings.csv 与 ratings.npy 的评分差异，形成可申报的书面结论。

    注意：只在文件头部连续取样会低估差异（头部老用户评分极少落在 6/7 边界），
    因此这里额外取文件尾部窗口，避免得出「翻转 0 次」的错误结论。
    """
    arr = np.load(os.path.join(ROOT, "dataset", "ratings.npy"), mmap_mode="r")
    total = arr.shape[0]
    head = np.asarray(arr[:n_rows])
    tail = np.asarray(arr[total - n_rows:])

    def read_tail_lines(n: int) -> np.ndarray:
        """从 csv 末尾读取 n 行（跳过首行残行）。"""
        p = cfg["input"]["ratings"]
        size = os.path.getsize(p)
        with open(p, "rb") as fh:
            fh.seek(max(0, size - n * 14))
            fh.readline()                      # 丢弃残行
            lines = []
            while len(lines) < n:
                line = fh.readline()
                if not line:
                    break
                lines.append(line)
        rows = []
        for ln in lines:
            p3 = ln.decode("utf-8", "replace").strip().split(",")
            if len(p3) >= 3:
                try:
                    rows.append((int(p3[0]), int(p3[1]), int(p3[2])))
                except Exception:
                    pass
        return np.array(rows, dtype=np.int64)

    csv_head = pd.read_csv(cfg["input"]["ratings"], nrows=n_rows,
                           usecols=["userID", "animeID", "rating"]).to_numpy(np.int64)
    csv_tail = read_tail_lines(n_rows)
    m = min(len(tail), len(csv_tail))
    csv_tail, tail = csv_tail[-m:], tail[-m:]

    n_all = len(csv_head) + len(csv_tail)
    npy_all = np.vstack([head, tail])
    csv_all = np.vstack([csv_head, csv_tail])

    sel = npy_all[:, 2] != csv_all[:, 2]
    from collections import Counter
    deltas = Counter((csv_all[sel, 2] - npy_all[sel, 2]).tolist())
    flip = int(((npy_all[:, 2] >= POSITIVE_RATING_THRESHOLD) !=
                (csv_all[:, 2] >= POSITIVE_RATING_THRESHOLD)).sum())
    out = dict(
        compared_rows=int(n_all),
        sample_scope="文件头部 2,000,000 行 + 尾部 2,000,000 行（窗口粗检，非全量）",
        mismatched_rating_cells=int(sel.sum()),
        mismatch_ratio=round(float(sel.sum()) / max(n_all, 1), 5),
        delta_distribution={int(k): int(v) for k, v in sorted(deltas.items())},
        boundary_flips=flip,
        boundary_flip_ratio=round(flip / max(n_all, 1), 6),
        verdict="窗口粗检，不代表全量；权威结论由 scripts/diagnose_dataset_source.py 产出，"
                "落盘 data/processed/dataset_source_verdict.json。"
                "结论：dataset.pkl 由 ratings.npy 口径构建（集合复现率 100% vs csv 90.7%），"
                "本管线一律以 ratings.npy 为权威交互源。",
    )
    log(f"  数据源审计: 比对 {n_all:,} 行，评分不同 {int(sel.sum()):,} "
        f"({out['mismatch_ratio']:.3%})，跨 7 边界翻转 {flip:,}")
    return out




# =====================================================================
# 3. 序列数据集（复用 dataset.pkl 的时序口径）
# =====================================================================
def build_sequences(cfg: dict, max_len: int, min_seq_len: int) -> dict:
    path = cfg["input"]["dataset_pkl"]
    log(f"读取 {path} 并以其时序口径重建序列 ...")
    with open(path, "rb") as f:
        ds = pickle.load(f)

    train, val, test = ds["train"], ds["val"], ds["test"]
    umap, smap = ds["umap"], ds["smap"]
    n_users = len(train)

    lens = np.empty(n_users, dtype=np.int32)
    trunc = 0
    for i in range(n_users):
        n = len(train[i])
        lens[i] = n
        if n + 2 > max_len:
            trunc += 1

    meta = dict(
        n_users=n_users,
        n_items=len(smap),
        max_len=max_len,
        min_seq_len=min_seq_len,
        split="leave_one_out",
        order_source="dataset.pkl",
        truncated_users=int(trunc),
    )
    log(f"  用户 {n_users:,} / 物品 {len(smap):,} / 训练序列超长(>max_len-2) 用户 {trunc:,}")
    return dict(train=train, val=val, test=test, umap=umap, smap=smap, meta=meta,
                train_lens=lens)


# =====================================================================
# 4. 交叉校验：用原始 csv + rating>=7 复现物品集合
# =====================================================================
def cross_validate(scan: dict, seq: dict, meta_by_id: pd.DataFrame,
                   sample_users: int, seed: int) -> dict:
    log("交叉校验：从权威源 ratings.npy 复现每个用户的物品集合 ...")
    smap, umap = seq["smap"], seq["umap"]
    inv_smap = {v: k for k, v in smap.items()}

    sample_rows = scan["sample_rows"]
    uids = [u for u in sample_rows if u in umap]
    random.Random(seed).shuffle(uids)
    uids = uids[:sample_users]

    # 主校验：不做合规过滤（实测 dataset.pkl 未剔除合规题材，见 compliance 段）
    exact_order, exact_set, total, bad = 0, 0, 0, []
    # 对照：若额外剔除合规题材，会破坏多少用户
    set_with_forbid_filter = 0
    forbidden_items = set(meta_by_id.loc[meta_by_id["is_forbidden"], "anime_id"].tolist())
    for u in uids:
        idx = umap[u]
        pkl_items = [inv_smap[x] for x in (seq["train"][idx] + seq["val"][idx] + seq["test"][idx])]
        cand = [a for a, r in sample_rows[u]
                if r >= POSITIVE_RATING_THRESHOLD and a in smap]
        cand_c = [a for a in cand if a not in forbidden_items]
        total += 1
        if set(cand) == set(pkl_items):
            exact_set += 1
        if cand == pkl_items:
            exact_order += 1
        elif len(bad) < 5:
            bad.append(dict(user_id=int(u), n_raw=len(cand), n_pkl=len(pkl_items),
                            set_equal=set(cand) == set(pkl_items)))
        if set(cand_c) == set(pkl_items):
            set_with_forbid_filter += 1

    cu_pos, ca_pos = scan["cnt_u_pos"], scan["cnt_a_pos"]
    n_user_ge10 = int((cu_pos[1:MAX_USER_ID + 1] >= 10).sum())
    n_item_ge10 = int((ca_pos[1:MAX_ANIME_ID + 1] >= 10).sum())

    report = dict(
        n_users_ge10_positive=n_user_ge10,
        n_users_in_pkl=len(umap),
        n_items_ge10_positive=n_item_ge10,
        n_items_in_smap=len(smap),
        sampled_users=total,
        set_match=exact_set,
        order_match=exact_order,
        set_match_rate=round(exact_set / total, 4) if total else None,
        order_match_rate=round(exact_order / total, 4) if total else None,
        set_match_with_forbidden_filter=set_with_forbid_filter,
        mismatches=bad,
        conclusion=(
            f"物品集合可由 rating>={POSITIVE_RATING_THRESHOLD} + 用户/物品正样本>=10 完整复现"
            f"（{exact_set}/{total}）；顺序仅 {exact_order}/{total} 可复现，"
            f"原始文件无时间戳，故序列顺序以 dataset.pkl 为准。"
        ),
    )
    log(f"  抽样 {total} 用户: 集合一致 {exact_set}/{total}, 顺序一致 {exact_order}/{total}")
    log(f"  对照：若额外剔除合规题材，集合一致率降至 {set_with_forbid_filter}/{total}")
    return report


# =====================================================================
# 4b. 合规专项统计（实测 dataset.pkl 未剔除 Hentai/Erotica）
# =====================================================================
def compliance_audit(meta: pd.DataFrame, seq: dict) -> dict:
    smap = seq["smap"]
    inv_smap = {v: k for k, v in smap.items()}
    forb_meta = set(meta.loc[meta["is_forbidden"], "anime_id"].tolist())
    forb_in_pool = sorted(forb_meta & set(smap.keys()))

    n_test_forb = 0
    for i in range(len(seq["test"])):
        t = seq["test"][i]
        if len(t) == 1 and t[0] in smap and inv_smap[t[0]] in forb_meta:
            n_test_forb += 1
    n_train_forb_sample = 0
    n_train_items_sample = 0
    step = max(1, len(seq["train"]) // 30_000)
    sampled = 0
    for i in range(0, len(seq["train"]), step):
        tr = seq["train"][i]
        n_train_items_sample += len(tr)
        n_train_forb_sample += sum(1 for x in tr if inv_smap.get(x) in forb_meta)
        sampled += 1
    rate = n_train_forb_sample / max(n_train_items_sample, 1)

    out = dict(
        forbidden_anime_in_metadata=len(forb_meta),
        forbidden_items_in_pool=len(forb_in_pool),
        forbidden_pool_ratio=round(len(forb_in_pool) / len(smap), 4),
        forbidden_test_samples=n_test_forb,
        forbidden_test_ratio=round(n_test_forb / len(seq["test"]), 5),
        forbidden_train_interaction_ratio=round(rate, 5),
        forbidden_train_interaction_ratio_basis=f"{sampled:,} 用户抽样（步长 {step}）",
        action=("保持 dataset.pkl 原始划分不变；合规题材在候选池与评估阶段屏蔽"
                "（打分置 -inf），不修改提供的切分，保证可复现与可审计"),
        note="docs 早期版本称已剔除 1,675 条，与所提供的 dataset.pkl 实际不符，已更正",
        forbidden_item_indices=sorted(int(smap[a]) for a in forb_in_pool),
    )
    log(f"  合规审计: 推荐池内含 Hentai/Erotica {len(forb_in_pool):,} 个 "
        f"({out['forbidden_pool_ratio']:.2%}), 受影响 test 样本 {n_test_forb:,} "
        f"({out['forbidden_test_ratio']:.3%}), 训练交互占比约 {rate:.2%}")
    return out



# =====================================================================
# 5. 新番（长尾）判定
# =====================================================================
def build_item_stats(seq: dict, scan: dict, meta: pd.DataFrame, cs_cfg: dict) -> pd.DataFrame:
    log("构建物品统计与新番标记 ...")
    smap = seq["smap"]
    ids = np.array(sorted(smap.keys()))
    cnt = scan["cnt_a_pos"][ids]

    df = pd.DataFrame({"anime_id": ids, "n_positive": cnt})
    q = float(cs_cfg.get("quantile", 0.10))
    thr = int(np.percentile(cnt, q * 100))
    df["is_tail"] = df["n_positive"] <= thr
    df["threshold_used"] = thr
    log(f"  长尾阈值: 交互数 <= P{int(q * 100)} = {thr}  -> 新番候选 {int(df['is_tail'].sum()):,} 个")

    m = meta[["anime_id", "title", "year", "type", "score", "primary_genre_id",
              "primary_genre", "genres_cn", "genres_cn_ids", "is_forbidden"]]
    df = df.merge(m, on="anime_id", how="left")

    if cs_cfg.get("min_year") is not None:
        df["is_tail_recent"] = df["is_tail"] & (df["year"].fillna(0) >= int(cs_cfg["min_year"]))
    else:
        df["is_tail_recent"] = df["is_tail"]
    log(f"  叠加新番年份条件(>= {cs_cfg.get('min_year')}) 后: {int(df['is_tail_recent'].sum()):,} 个")
    return df


# =====================================================================
# main
# =====================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="阶段1 数据预处理")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "data.yaml"))
    ap.add_argument("--taxonomy", default=os.path.join(ROOT, "configs", "genre_taxonomy.yaml"))
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--validate-users", type=int, default=200)
    ap.add_argument("--skip-coldstart", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    tx = load_cfg(args.taxonomy)
    id2name, mal2cls, priority, music_kw, forbidden = load_taxonomy(args.taxonomy)
    cs_cfg = dict(cfg.get("cold_start") or {})
    cs_cfg.setdefault("quantile", 0.10)
    cs_cfg.setdefault("min_year", 2015)

    outdir = os.path.join(ROOT, cfg["output"]["processed_dir"].lstrip("./"))
    featdir = os.path.join(ROOT, cfg["output"]["feature_dir"].lstrip("./"))
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(featdir, exist_ok=True)

    # --- 1) 元数据 ---
    meta = build_anime_meta(cfg, id2name, mal2cls, priority, music_kw, forbidden)

    # --- 2) 抽样用户（用于顺序校验）---
    with open(cfg["input"]["dataset_pkl"], "rb") as f:
        _ds = pickle.load(f)
    pool = list(_ds["umap"].keys())
    sample_uids = set(random.Random(args.seed).sample(pool, min(600, len(pool))))
    del _ds

    # --- 3) 扫描交互（权威源 ratings.npy）---
    scan = scan_ratings(cfg, sample_uids)
    audit = audit_rating_sources(cfg)

    # --- 4) 序列 ---
    seq = build_sequences(cfg, int(cfg["sequence"]["max_len"]), int(cfg["sequence"]["min_seq_len"]))

    # --- 5) 校验 + 合规审计 ---
    val = cross_validate(scan, seq, meta, args.validate_users, args.seed)
    comp = compliance_audit(meta, seq)

    # --- 6) 物品统计 ---
    item_stats = build_item_stats(seq, scan, meta, cs_cfg)

    # --- 7) 滑动窗口样本量（口径：每用户 len(train)）---
    n_train_items = int(seq["train_lens"].sum())
    win = dict(
        definition="对长度为 n 的训练序列构造 n 个 (prefix -> next_item) 样本",
        total_windows=n_train_items,
        mean_per_user=float(seq["train_lens"].mean()),
        max_per_user=int(seq["train_lens"].max()),
        note="实际训练时由 models/sasrec/dataset.py 在线生成，不落盘；此处仅统计规模",
    )

    # --- 8) 用户统计 ---
    user_stats = pd.DataFrame({
        "user_id": list(seq["umap"].keys()),
        "user_index": list(seq["umap"].values()),
        "train_len": seq["train_lens"],
    })

    # --- 9) 报告 ---
    hist = scan["rating_hist"]
    report = dict(
        input_rows=int(scan["n_rows"]),
        authoritative_source="dataset/ratings.npy",
        source_audit=audit,
        input_files=dict(
            ratings_npy="权威源：物品集合复现率 100%（1500 抽样用户）",
            ratings_csv=("dataset/ratings.csv 与 ratings.dat 完全相同；"
                         "相对 npy 存在整体 +1 评分偏移，单独使用集合复现率仅 90.7%"
                         "（明细见 dataset_source_verdict.json）"),
        ),
        positive_rating_threshold=POSITIVE_RATING_THRESHOLD,
        rating_hist={int(i): int(hist[i]) for i in range(12) if hist[i]},
        positive_ratio=float(hist[POSITIVE_RATING_THRESHOLD:].sum() / max(scan["n_rows"], 1)),
        anime_meta_rows=int(len(meta)),
        forbidden_in_metadata=int(meta["is_forbidden"].sum()),
        n_users=int(len(seq["umap"])),
        n_items=int(len(seq["smap"])),
        sequence_order_source="dataset.pkl（原始文件无时间戳，顺序不可复现）",
        sliding_window=win,
        genre_distribution={
            row["primary_genre"]: int(row["n"])
            for row in item_stats.groupby("primary_genre").size().reset_index(name="n").to_dict("records")
        },
        genre_coverage_any={
            str(cid): int(item_stats["genres_cn_ids"].map(lambda x: cid in (x or [])).sum())
            for cid in range(1, 13)
        },
        meta_missing=dict(
            year=int(meta["year"].isna().sum()),
            score=int(meta["score"].isna().sum()),
            primary_genre=int(meta["primary_genre_id"].isna().sum()),
        ),
        compliance=comp,
        deviations_from_rq=[],
    )
    if report["meta_missing"]["primary_genre"]:
        report["deviations_from_rq"].append(
            f"{report['meta_missing']['primary_genre']} 条动漫无 12 类题材标签"
            f"（其 MAL 题材仅为合规剔除项 Hentai/Erotica）"
        )
    report["deviations_from_rq"].extend([
        "三个交互文件口径不一致：以 ratings.npy 为权威源（集合复现率 100%），"
        "ratings.csv/.dat 存在整体 +1 偏移（集合复现率仅 90.7%）；"
        "明细见 data/processed/dataset_source_verdict.json",
        "原始数据无时间戳：rq.md 的『按时间戳升序』无法执行，改用 dataset.pkl 的既有序列顺序",
        "animes.csv 无剧情简介字段：内容特征改用 标题+12类题材+细分标签",
        "dataset.pkl 实际未剔除 Hentai/Erotica，共 1,551 个物品在推荐池内；"
        "本项目在候选池与评估阶段屏蔽，不改动原始切分",
        "过滤阈值实测为「正样本交互数 >= 10」（用户与物品），非早期文档所写的 <5 / <10",
    ])

    if args.validate_only:
        with open(os.path.join(ROOT, "logs", "validate_only.json"), "w", encoding="utf-8") as f:
            json.dump(dict(report=report, validation=val), f, ensure_ascii=False, indent=2)
        log("validate-only 完成 -> logs/validate_only.json")
        return 0

    # --- 10) 落盘 ---
    log("写出产物 ...")
    meta.drop(columns=["genres", "genres_detailed"]).to_parquet(
        os.path.join(outdir, "anime_meta.parquet"), index=False)
    pd.DataFrame({"anime_id": meta["anime_id"], "genres_detailed": meta["genres_detailed"]}
                 ).to_parquet(os.path.join(outdir, "anime_detailed_tags.parquet"), index=False)
    item_stats.to_parquet(os.path.join(outdir, "item_stats.parquet"), index=False)
    user_stats.to_parquet(os.path.join(outdir, "user_stats.parquet"), index=False)

    with open(os.path.join(outdir, cfg["output"]["sequence_file"]), "wb") as f:
        pickle.dump(dict(train=seq["train"], val=seq["val"], test=seq["test"],
                         umap=seq["umap"], smap=seq["smap"], meta=seq["meta"]), f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # 新番子集由 scripts/build_cold_start_subset.py 单独负责（避免双重实现）
    if args.skip_coldstart:
        log("已跳过新番子集（--skip-coldstart）")
    else:
        log("提示：新番子集请运行 scripts/build_cold_start_subset.py")

    with open(os.path.join(outdir, "preprocess_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    log("=" * 60)
    log(f"完成。用户 {report['n_users']:,} / 物品 {report['n_items']:,} / "
        f"正样本率 {report['positive_ratio']:.2%}")
    log(f"输出目录: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
