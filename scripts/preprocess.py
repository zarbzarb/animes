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

    ⚠ 工作目录无关：以上命令在任意目录下执行结果一致。
    配置里的路径写作 './dataset/animes.csv'（相对项目根），脚本会在读配置时
    统一锚定到项目根（见 anchor_paths）。所以下面两种写法等价：
        cd F:/pj && python scripts/preprocess.py
        python F:/pj/scripts/preprocess.py          # 在别处执行也没问题
    命令行传的 --config / --taxonomy 同样按「相对项目根」解释。

产物（默认写入 data/processed/）
--------------------------------
    anime_meta.parquet      清洗后的动漫元数据 + 12 类题材标签 + is_primary
    anime_detailed_tags.parquet  细分标签（内容向量构建时要用，单独存一份省内存）
    seq_dataset.pkl         序列数据集 {train,val,test,umap,smap,meta}
    item_stats.parquet      每个物品的交互数 / 是否新番 / 主题材
    user_stats.parquet      每个用户的序列长度
    preprocess_report.json  全量统计报告（论文「数据集」节素材）
    validation_report.json  与 dataset.pkl 的交叉校验结果

阅读导引（函数 → 流水线步骤，按 main 里的调用顺序）
---------------------------------------------------
    load_taxonomy()          读 configs/genre_taxonomy.yaml，得到 21→12 的映射与禁用题材
    build_anime_meta()    ① 清洗 animes.csv，产出 12 类中文题材 + 主题材 + 合规标记
    scan_ratings()        ② 全量扫 ratings.npy，按 rating>=7 统计每用户/每物品交互数
    audit_rating_sources() ②' 抽头尾窗口粗检 npy vs csv 的评分差异（仅供自查）
    build_sequences()     ③ 复用 dataset.pkl 的时序口径重建序列数据集
    cross_validate()      ④ 用原始评分独立复现物品集合，验证口径正确
    compliance_audit()    ④' 统计 Hentai/Erotica 在池内/训练/测试中的实际占比
    build_item_stats()    ⑤ 生成物品统计表与新番（长尾）标记
    main()                  按 ①~⑤ 串起来，最后统一落盘

    注意：新番子集**不在本脚本**产出，由 scripts/build_cold_start_subset.py 单独负责，
    避免同一套逻辑写两遍（见 main 末尾的提示）。

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
# ID 上界取自 ratings.npy 全量实测（可用 scripts/verify_stage1.py 复核）：
# userID <= 1,774,522；animeID <= 20,237
MAX_USER_ID = 1_774_522
MAX_ANIME_ID = 20_237

# ---------- 正样本阈值（实测得出，改动会使数据集不可比）----------
POSITIVE_RATING_THRESHOLD = 7


# =====================================================================
# 工具
# =====================================================================
def log(msg: str) -> None:
    """带时间戳打印一条进度信息（flush 保证长任务时能实时看到输出）。"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# 配置里路径的书写约定：一律**相对项目根目录**（如 './dataset/animes.csv'）。
# 下面这张表声明了哪些字段是「路径」，从而必须锚定到 ROOT。
# 不在这里的字段（如 output.sequence_file）是**文件名**，之后还要和
# processed_dir / feature_dir 再做一次 join，锚定反而会破坏它们。
_PATH_SECTIONS = (
    ("input", None),                                # input 段每个键都是路径
    ("output", ("processed_dir", "feature_dir")),    # 其余键是文件名
    ("content_feature", ("model_dir",)),
)


def anchor_paths(cfg: dict) -> dict:
    """把配置中的「路径型字段」锚定到项目根目录 ROOT（原地修改并返回）。

    为什么需要（实际踩过的坑）
    ------------------------
    configs/data.yaml 里路径写作 './dataset/animes.csv'，只有当前工作目录
    恰好是项目根时才能打开。于是：

        用绝对路径调用（IDE「运行」按钮 / 任务计划 / 换个终端）
            python F:/pj/scripts/preprocess.py
        会直接抛 FileNotFoundError: './dataset/animes.csv'

    本函数把相对路径统一转成绝对路径，脚本从此与「当前工作目录」解耦。
    已经是绝对路径的值原样保留，因此重复调用是幂等的。
    """
    for sec, keys in _PATH_SECTIONS:
        d = cfg.get(sec)
        if not isinstance(d, dict):
            continue
        for k in (list(d) if keys is None else keys):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                d[k] = os.path.normpath(
                    v if os.path.isabs(v) else os.path.join(ROOT, v))
    return cfg


def load_cfg(path: str, anchor: bool = True) -> dict:
    """读一个 YAML 配置。

    anchor=True（默认）时顺带调用 anchor_paths()，把相对路径锚定到项目根，
    使脚本与「当前工作目录」无关。读 genre_taxonomy.yaml 这类不含路径字段
    的配置时，锚定是空操作，因此可以放心地共用同一个入口。
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return anchor_paths(cfg) if anchor else cfg


def to_abs_path(p: str) -> str:
    """把命令行传进来的路径按「相对项目根」解释（命令行参数与配置同一约定）。"""
    if not p:
        return p
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(ROOT, p))


def load_taxonomy(path: str) -> tuple[dict, dict, list, list, set]:
    """读 configs/genre_taxonomy.yaml，展开成便于查表的结构。

    返回五元组：
        id2name     MAL genre_id -> 英文名（21 类）
        mal2cls     MAL genre_id -> 12 类中文题材 id（由 genres_12 的 mal_ids 展平）
        priority    12 类题材的优先级顺序（越靠前越"具体"，用于选主题材）
        music_kw    「青春音乐」关键词表（MAL 无对应 genre，靠细分标签兜底）
        forbidden   禁用题材 id 集合 {20, 21} = Erotica / Hentai
    """
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
    """animes.csv 的 genres / genres_detailed 是字符串形式的 python list。

    例："['Action', 'Comedy']" -> ['Action', 'Comedy']。
    用 ast.literal_eval 而不是 json.loads：csv 里是单引号，不是合法 JSON。
    解析失败一律返回空列表，不让个别脏数据中断整条流水线。
    """
    if not isinstance(v, str):
        return []
    try:
        r = ast.literal_eval(v)
        return list(r) if isinstance(r, (list, tuple)) else []
    except Exception:
        return []


def to_int_year(v):
    """年份规范化：非数字或超出 1900~2100 的一律返回 None（视为缺失）。"""
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
    """① 清洗 animes.csv，产出动漫元数据表。

    输入参数里除 cfg 外都来自 load_taxonomy()：
        id2name   MAL genre_id -> 英文名（保留备用）
        mal2cls   21 类 MAL 主题材 -> 12 类中文题材的映射
        priority  12 类题材的优先级顺序（决定「主题材」取哪一个）
        music_kw  「青春音乐」的关键词表（MAL 没有对应 genre id，只能靠细分标签兜底）
        forbidden {20, 21} = Erotica / Hentai，参与映射时需要跳过

    输出的关键列：
        genre_ids        该动漫的 MAL 原始题材 id 列表
        genres_cn_ids    映射到 12 类后的 id 列表（可多类）
        genres_cn        对应的中文名列表
        primary_genre_id 主题材 id（用于题材分布统计）
        is_forbidden     是否含 Hentai/Erotica（注意：语义是「评估时屏蔽」而非「已删除」）
    """
    log("读取 animes.csv ...")
    df = pd.read_csv(cfg["input"]["anime_meta"])
    raw_n = len(df)
    log(f"  原始 {raw_n} 条, 列: {list(df.columns)}")

    # ---- 逐列搬运 + 类型规范化 ----
    # 原则：字符串列统一 fillna("") 后 strip，数值列用专门函数转（因为存在 '?' 占位符）
    out = pd.DataFrame()
    out["anime_id"] = df["animeID"].astype(int)
    out["title"] = df["title"].fillna("").astype(str).str.strip()
    out["alternative_title"] = df["alternative_title"].fillna("").astype(str).str.strip()
    out["type"] = df["type"].fillna("UNKNOWN").astype(str).str.strip()
    out["year"] = df["year"].map(to_int_year)              # 不合理年份 -> None
    out["score"] = df["score"].map(to_float_score)         # '?' -> NaN
    out["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
    out["mal_url"] = df["mal_url"].fillna("").astype(str)
    out["image_url"] = df["image_url"].fillna("").astype(str)
    out["sequel"] = df["sequel"].fillna(False).astype(bool)

    # --- genres / genres_detailed ---
    # 这两列在 csv 里是「字符串形式的 Python list」，如 "['Action', 'Comedy']"，
    # 不能直接当 JSON 解析，必须走 ast.literal_eval（见 parse_list_field）。
    out["genres"] = df["genres"].map(parse_list_field)
    out["genres_detailed"] = df["genres_detailed"].map(parse_list_field)

    # --- genre_id 映射（来自 id_to_genreids.json，覆盖全部 20,237 条）---
    # 这个 json 是数据集自带的 MAL 官方题材映射，比解析 csv 的 genres 字符串更可靠。
    # 用 sorted(set(...)) 去重并排序，保证同一份输入每次产出完全一致（可复现）。
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
        """判断细分标签里是否出现音乐类关键词（词边界匹配，避免子串误判）。"""
        if kw_pattern is None:
            return False
        return any(kw_pattern.search(str(t).lower()) for t in det)

    def to_12(gids: list, det: list) -> list:
        """把 MAL 题材 id 列表映射成 12 类中文题材 id 列表。

        规则：
          1) 跳过禁用题材（Hentai/Erotica 不参与分类）
          2) 逐个查 mal2cls 表；查不到就丢弃（有些冷门 id 不在 12 类体系内）
          3) 若细分标签命中音乐关键词，则额外补上 11 号「青春音乐」
          4) 去重 + 升序，保证可复现
        """
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
    # priority 是「具体题材优先」的顺序（如 运动竞技 排在 剧情 前面），
    # 目的是保护小众题材不被大类（剧情/喜剧）吞掉 —— 论文要专门验证小众题材效果。
    rank = {cid: i for i, cid in enumerate(priority)}

    def pick_primary(ids: list):
        """取优先级最高（rank 最小）的那个题材作为主题材；无题材返回 None。"""
        if not ids:
            return None
        return sorted(ids, key=lambda c: rank.get(c, 999))[0]

    out["primary_genre_id"] = out["genres_cn_ids"].map(pick_primary)
    out["primary_genre"] = out["primary_genre_id"].map(lambda i: cid2name.get(i) if i else None)

    # --- 合规标记 ---
    # 只要 MAL 题材里含任一禁用 id 就标记。实测有 1,675 条，
    # 其中 1,551 条仍在 dataset.pkl 的推荐池内 —— 所以标记的目的是「评估时屏蔽」。
    out["is_forbidden"] = out["genre_ids"].map(lambda gs: any(g in forbidden for g in gs))

    # --- 异常值清理统计（只记录数量，不删行：删行会让索引与官方切分对不上）---
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
    # mmap：6.9GB 文件不整读进屋，按块取；arr 形状 (148,170,496, 3)，三列 = 用户/物品/评分
    arr = np.load(npy_path, mmap_mode="r")
    N = arr.shape[0]

    # 计数数组按「原始 id」做下标（不是连续索引），所以长度取 id 上界 + 2 留余量。
    #   *_pos = 该用户/物品的正样本数（用于过滤阈值）
    #   *_all = 全部评分数（用于报告评分分布）
    cnt_u_pos = np.zeros(MAX_USER_ID + 2, dtype=np.int64)
    cnt_a_pos = np.zeros(MAX_ANIME_ID + 2, dtype=np.int64)
    cnt_u_all = np.zeros(MAX_USER_ID + 2, dtype=np.int64)
    cnt_a_all = np.zeros(MAX_ANIME_ID + 2, dtype=np.int64)
    rating_hist = np.zeros(12, dtype=np.int64)   # 评分直方图（下标 0..11）

    # 抽样用户掩码：用 bool 数组做下标比 set 快得多（逐行 list 判断会拖慢全量扫描）
    sample_mask = np.zeros(MAX_USER_ID + 2, dtype=bool)
    if sample_uids:
        sample_mask[np.fromiter(sample_uids, dtype=np.int64)] = True
    sample_rows = defaultdict(list)   # 抽样用户的逐行原始评分，供顺序/集合校验

    # ---- 分块扫描 ----
    # 每块 1000 万行；bincount 是向量化的「按值计数」，避免 1.5 亿次 Python 循环。
    # minlength 必须等于计数数组长度，否则 id 超长时会报错。
    CH = 10_000_000
    for s in range(0, N, CH):
        b = np.asarray(arr[s:s + CH])
        u, a, r = b[:, 0].astype(np.int64), b[:, 1].astype(np.int64), b[:, 2].astype(np.int64)
        cnt_u_all += np.bincount(u, minlength=MAX_USER_ID + 2)
        cnt_a_all += np.bincount(a, minlength=MAX_ANIME_ID + 2)
        m = r >= POSITIVE_RATING_THRESHOLD          # 正样本掩码
        cnt_u_pos += np.bincount(u[m], minlength=MAX_USER_ID + 2)
        cnt_a_pos += np.bincount(a[m], minlength=MAX_ANIME_ID + 2)
        rating_hist += np.bincount(np.clip(r, 0, 11), minlength=12)   # clip 防越界
        if sample_uids:
            sel = b[sample_mask[b[:, 0]]]
            if len(sel):
                for uu, aa, rr in sel:
                    sample_rows[int(uu)].append((int(aa), int(rr)))
        if s % 50_000_000 == 0:                     # 每 5000 万行打一次进度
            log(f"  ... {s:,}/{N:,}")
    log(f"  扫描完成，共 {N:,} 行；抽样用户 {len(sample_rows):,} 个")
    return dict(cnt_u_pos=cnt_u_pos, cnt_a_pos=cnt_a_pos, cnt_u_all=cnt_u_all,
                cnt_a_all=cnt_a_all, rating_hist=rating_hist,
                sample_rows=dict(sample_rows), n_rows=int(N))


def audit_rating_sources(cfg: dict, n_rows: int = 2_000_000) -> dict:
    """对照 ratings.csv 与 ratings.npy 的评分差异，形成可申报的书面结论。

    ⚠ 这是**窗口粗检**，不是全量结论：它只比文件头尾各 200 万行，
    因为 csv 与 npy 按 userID 聚簇排列，窗口内覆盖不到全部用户段，
    得到的差异率会系统性偏小。本函数产出的数字只用于「冒烟自检」，
    写进论文的权威结论请引用 scripts/diagnose_dataset_source.py 的产物
    （data/processed/dataset_source_verdict.json）。

    注意：只在文件头部连续取样会低估差异（头部老用户评分极少落在 6/7 边界），
    因此这里额外取文件尾部窗口，避免得出「翻转 0 次」的错误结论。
    """
    arr = np.load(os.path.join(ROOT, "dataset", "ratings.npy"), mmap_mode="r")
    total = arr.shape[0]
    head = np.asarray(arr[:n_rows])              # npy 头部窗口
    tail = np.asarray(arr[total - n_rows:])      # npy 尾部窗口

    def read_tail_lines(n: int) -> np.ndarray:
        """从 csv 末尾读取 n 行（跳过首行残行）。

        csv 没有随机访问能力，只能 seek 到文件末尾往前读。
        每行约 14 字节（"1774522,20237,10\n" 量级），用 14 估一个大致起点再丢弃残行。
        """
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
    # 头尾两段行数可能因为估算起点而略不等，尾部对齐到较短的尾巴上再比
    m = min(len(tail), len(csv_tail))
    csv_tail, tail = csv_tail[-m:], tail[-m:]

    n_all = len(csv_head) + len(csv_tail)
    npy_all = np.vstack([head, tail])
    csv_all = np.vstack([csv_head, csv_tail])

    # 评分不同 = csv 与 npy 在同一 (用户,物品) 上的打分不一致
    sel = npy_all[:, 2] != csv_all[:, 2]
    from collections import Counter
    deltas = Counter((csv_all[sel, 2] - npy_all[sel, 2]).tolist())   # 差值分布（应为 +1 为主）
    # 翻转 = 该单元格在 csv 里 >=7 但在 npy 里 <7（或反之），会改变正样本判定
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
    """③ 重建序列数据集：**直接沿用 dataset.pkl 的既有切分与顺序**。

    为什么不自己切分？
        rq.md 要求「按时间戳升序排列后留一法切分」，但原始交互文件
        （ratings.csv/.dat/.npy）**都没有时间戳字段**，只有 (userID, animeID, rating)。
        实测：物品集合能用 rating>=7 100% 复现，但顺序只能复现约 60%。
        既然顺序不可复现，就一律以 dataset.pkl 提供的顺序为唯一时序口径，
        这样本项目所有实验与官方切分完全可比。

    切分口径（官方已切好，本函数只做搬运 + 统计）：
        test = 序列最后 1 条；val = 倒数第 2 条；train = 其余
        train 超过 max_len-2 的用户在训练时由 dataset.py 在线截断（此处只统计数量）
    """
    path = cfg["input"]["dataset_pkl"]
    log(f"读取 {path} 并以其时序口径重建序列 ...")
    with open(path, "rb") as f:
        ds = pickle.load(f)

    train, val, test = ds["train"], ds["val"], ds["test"]
    umap, smap = ds["umap"], ds["smap"]
    n_users = len(train)

    # 统计训练序列长度：>max_len-2 的用户在训练阶段会被截断，数量记进 meta 备查
    lens = np.empty(n_users, dtype=np.int32)
    trunc = 0
    for i in range(n_users):
        n = len(train[i])
        lens[i] = n
        if n + 2 > max_len:      # +2 是 val 和 test 两条也要占位
            trunc += 1

    # meta 记录口径来源，下游（以及验收脚本）靠它判断切分是否被改动过
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
    """④ 交叉校验：用原始评分独立复现每个用户的物品集合，验证口径推断正确。

    校验的是「同一批用户，用 ratings.npy + rating>=7 重建出的物品集合，
    是否等于 dataset.pkl 里该用户的 train+val+test 并集」。
      · 集合一致率高 → 说明「正样本阈值 = 7」的推断正确
      · 顺序一致率低 → 说明顺序确实不可复现，佐证「以 dataset.pkl 为准」的决定

    同时做一个反事实对照：如果额外剔除 Hentai/Erotica，一致率会掉多少
    —— 用来量化「擅自剔除合规题材」的代价，支撑「不改动官方切分」的决策。
    """
    log("交叉校验：从权威源 ratings.npy 复现每个用户的物品集合 ...")
    smap, umap = seq["smap"], seq["umap"]
    inv_smap = {v: k for k, v in smap.items()}   # 池内索引 -> anime_id

    # 从上一步的抽样用户里再取一批来做校验（固定种子，保证可复现）
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
        # pkl 侧：该用户三条序列拼起来，再翻译回 anime_id
        pkl_items = [inv_smap[x] for x in (seq["train"][idx] + seq["val"][idx] + seq["test"][idx])]
        # 原始侧：该用户所有 rating>=7 且属于推荐池的物品
        cand = [a for a, r in sample_rows[u]
                if r >= POSITIVE_RATING_THRESHOLD and a in smap]
        cand_c = [a for a in cand if a not in forbidden_items]   # 反事实：再剔除合规题材
        total += 1
        if set(cand) == set(pkl_items):
            exact_set += 1
        if cand == pkl_items:          # 顺序也完全一致（更难达到）
            exact_order += 1
        elif len(bad) < 5:             # 只留 5 条不一致样例供排查，避免报告过大
            bad.append(dict(user_id=int(u), n_raw=len(cand), n_pkl=len(pkl_items),
                            set_equal=set(cand) == set(pkl_items)))
        if set(cand_c) == set(pkl_items):
            set_with_forbid_filter += 1

    # 顺带从全量计数里独立算一遍过滤阈值下的用户/物品数，供与 pkl 对照
    cu_pos, ca_pos = scan["cnt_u_pos"], scan["cnt_a_pos"]
    n_user_ge10 = int((cu_pos[1:MAX_USER_ID + 1] >= 10).sum())
    n_item_ge10 = int((ca_pos[1:MAX_ANIME_ID + 1] >= 10).sum())

    report = dict(
        n_users_ge10_positive=n_user_ge10,      # 独立重算的合格用户数
        n_users_in_pkl=len(umap),               # pkl 里的用户数
        n_items_ge10_positive=n_item_ge10,      # 独立重算的合格物品数（应精确 == 15687）
        n_items_in_smap=len(smap),
        sampled_users=total,
        set_match=exact_set,
        order_match=exact_order,
        set_match_rate=round(exact_set / total, 4) if total else None,
        order_match_rate=round(exact_order / total, 4) if total else None,
        set_match_with_forbidden_filter=set_with_forbid_filter,   # 反事实一致率
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
    """④' 合规专项审计：量化 Hentai/Erotica 在数据集里的实际存在感。

    背景：docs 早期版本写「已剔除 Hentai 1,602 + Erotica 73 = 1,675 条」，
    但实测 dataset.pkl **并没有剔除**，池内仍有 1,551 个此类物品。
    本函数把「池内数量 / test 正样本受影响数 / 训练交互占比」都算出来，
    作为「不改动官方切分、改在候选池与评估阶段屏蔽」这一决策的依据。

    处理动作（action 字段）：
        保持 dataset.pkl 原始划分不变；打分时把这些物品置 -inf。
        这样既不破坏与官方切分的可比性，又能保证不向用户推荐不合规内容。
    """
    smap = seq["smap"]
    inv_smap = {v: k for k, v in smap.items()}
    forb_meta = set(meta.loc[meta["is_forbidden"], "anime_id"].tolist())   # 元数据里的不合规动漫
    forb_in_pool = sorted(forb_meta & set(smap.keys()))                    # 其中仍在推荐池内的

    # test 侧：留一法下每个用户 test 恰好 1 条，统计这条正样本本身是不是不合规物品
    n_test_forb = 0
    for i in range(len(seq["test"])):
        t = seq["test"][i]
        if len(t) == 1 and t[0] in smap and inv_smap[t[0]] in forb_meta:
            n_test_forb += 1
    # train 侧：总量上亿，逐条数太慢，用固定步长抽样估计占比（步长会写进报告，保证可复现）
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
        forbidden_anime_in_metadata=len(forb_meta),              # 元数据侧共 1,675 条
        forbidden_items_in_pool=len(forb_in_pool),               # 池内 1,551 个
        forbidden_pool_ratio=round(len(forb_in_pool) / len(smap), 4),
        forbidden_test_samples=n_test_forb,                      # 受影响的 test 正样本数
        forbidden_test_ratio=round(n_test_forb / len(seq["test"]), 5),
        forbidden_train_interaction_ratio=round(rate, 5),
        forbidden_train_interaction_ratio_basis=f"{sampled:,} 用户抽样（步长 {step}）",
        action=("保持 dataset.pkl 原始划分不变；合规题材在候选池与评估阶段屏蔽"
                "（打分置 -inf），不修改提供的切分，保证可复现与可审计"),
        note="docs 早期版本称已剔除 1,675 条，与所提供的 dataset.pkl 实际不符，已更正",
        forbidden_item_indices=sorted(int(smap[a]) for a in forb_in_pool),   # 供模型侧直接屏蔽
    )
    log(f"  合规审计: 推荐池内含 Hentai/Erotica {len(forb_in_pool):,} 个 "
        f"({out['forbidden_pool_ratio']:.2%}), 受影响 test 样本 {n_test_forb:,} "
        f"({out['forbidden_test_ratio']:.3%}), 训练交互占比约 {rate:.2%}")
    return out



# =====================================================================
# 5. 新番（长尾）判定
# =====================================================================
def build_item_stats(seq: dict, scan: dict, meta: pd.DataFrame, cs_cfg: dict) -> pd.DataFrame:
    """⑤ 生成物品统计表：每个推荐池物品的交互数、题材、年份、长尾/新番标记。

    注意 is_tail / is_tail_recent 只是**概览性标记**，真正给 E3 实验用的
    冷启动子集由 scripts/build_cold_start_subset.py 单独产出
    （那边会用「年份 >= min_year」的 holdout 口径，不以这里的长尾分位为准）。
    """
    log("构建物品统计与新番标记 ...")
    smap = seq["smap"]
    ids = np.array(sorted(smap.keys()))          # 推荐池内的 anime_id，升序
    cnt = scan["cnt_a_pos"][ids]                 # 对齐取各自的全局正样本交互数

    df = pd.DataFrame({"anime_id": ids, "n_positive": cnt})
    # 长尾阈值：取正样本交互数的 P{q} 分位（默认 P10）
    q = float(cs_cfg.get("quantile", 0.10))
    thr = int(np.percentile(cnt, q * 100))
    df["is_tail"] = df["n_positive"] <= thr
    df["threshold_used"] = thr
    log(f"  长尾阈值: 交互数 <= P{int(q * 100)} = {thr}  -> 新番候选 {int(df['is_tail'].sum()):,} 个")

    # 合并元数据里的展示字段（缺内容的物品 how="left" 会留下 NaN，下游自行处理）
    m = meta[["anime_id", "title", "year", "type", "score", "primary_genre_id",
              "primary_genre", "genres_cn", "genres_cn_ids", "is_forbidden"]]
    df = df.merge(m, on="anime_id", how="left")

    # 「长尾 且 年份够新」才算新番候选 —— 只有年份新才是真的「上线晚、训练时不存在」
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
    """按 ①~⑤ 串起整条流水线，最后统一落盘。返回 0 表示成功。"""
    ap = argparse.ArgumentParser(description="阶段1 数据预处理")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "data.yaml"))
    ap.add_argument("--taxonomy", default=os.path.join(ROOT, "configs", "genre_taxonomy.yaml"))
    ap.add_argument("--validate-only", action="store_true",
                    help="只跑校验不落盘产物（快速自检用）")
    ap.add_argument("--validate-users", type=int, default=200,
                    help="交叉校验的抽样用户数")
    ap.add_argument("--skip-coldstart", action="store_true",
                    help="跳过新番子集提示（子集本身由另一个脚本负责）")
    ap.add_argument("--seed", type=int, default=42, help="抽样与打乱的随机种子")
    args = ap.parse_args()

    # 命令行路径也按「相对项目根」解释，与配置里的约定保持一致。
    # 这样下面两种调用完全等价，脚本放到哪个目录下启动都不会错：
    #     cd F:/pj && python scripts/preprocess.py
    #     python F:/pj/scripts/preprocess.py
    args.config = to_abs_path(args.config)
    args.taxonomy = to_abs_path(args.taxonomy)

    cfg = load_cfg(args.config)          # 内部已把 input/output 的路径锚定到 ROOT
    tx = load_cfg(args.taxonomy)
    id2name, mal2cls, priority, music_kw, forbidden = load_taxonomy(args.taxonomy)
    cs_cfg = dict(cfg.get("cold_start") or {})
    cs_cfg.setdefault("quantile", 0.10)
    cs_cfg.setdefault("min_year", 2015)

    # 这两项已在 load_cfg() -> anchor_paths() 中变成绝对路径，直接用即可
    outdir = cfg["output"]["processed_dir"]
    featdir = cfg["output"]["feature_dir"]
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(featdir, exist_ok=True)

    # --- 1) 元数据清洗：animes.csv -> 12 类中文题材 + 主题材 + 合规标记 ---
    meta = build_anime_meta(cfg, id2name, mal2cls, priority, music_kw, forbidden)

    # --- 2) 抽样用户（样本量固定 600，用于第 ④ 步的集合/顺序校验）---
    # 只读 dataset.pkl 的 umap 取用户 id 池，读完立刻 del，避免长期占用内存。
    with open(cfg["input"]["dataset_pkl"], "rb") as f:
        _ds = pickle.load(f)
    pool = list(_ds["umap"].keys())
    sample_uids = set(random.Random(args.seed).sample(pool, min(600, len(pool))))
    del _ds

    # --- 3) 扫描交互（权威源 ratings.npy）：拿到每用户/每物品的交互数 ---
    scan = scan_ratings(cfg, sample_uids)
    audit = audit_rating_sources(cfg)   # 头尾窗口粗检 npy vs csv，仅自查用

    # --- 4) 序列数据集：沿用 dataset.pkl 的切分与顺序 ---
    seq = build_sequences(cfg, int(cfg["sequence"]["max_len"]), int(cfg["sequence"]["min_seq_len"]))

    # --- 5) 交叉校验 + 合规审计 ---
    val = cross_validate(scan, seq, meta, args.validate_users, args.seed)
    comp = compliance_audit(meta, seq)

    # --- 6) 物品统计表（交互数 / 长尾 / 新番标记）---
    item_stats = build_item_stats(seq, scan, meta, cs_cfg)

    # --- 7) 滑动窗口样本量（口径：每用户 len(train)）---
    # 训练样本是在训练时由 models/sasrec/dataset.py 在线生成的，不落盘；
    # 这里只统计「如果全展开会有多少样本」，用于报告与显存估算。
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

    # --- 9) 汇总报告 ---
    # 这份 report 是论文「数据集」一节的主要素材来源，字段尽量自解释。
    hist = scan["rating_hist"]
    report = dict(
        input_rows=int(scan["n_rows"]),
        authoritative_source="dataset/ratings.npy",
        source_audit=audit,                        # 头尾窗口粗检结果（非全量）
        input_files=dict(
            ratings_npy="权威源：物品集合复现率 100%（1500 抽样用户）",
            ratings_csv=("dataset/ratings.csv 与 ratings.dat 完全相同；"
                         "相对 npy 存在整体 +1 评分偏移，单独使用集合复现率仅 90.7%"
                         "（明细见 dataset_source_verdict.json）"),
        ),
        positive_rating_threshold=POSITIVE_RATING_THRESHOLD,
        rating_hist={int(i): int(hist[i]) for i in range(12) if hist[i]},   # 只留有值的档位
        positive_ratio=float(hist[POSITIVE_RATING_THRESHOLD:].sum() / max(scan["n_rows"], 1)),
        anime_meta_rows=int(len(meta)),
        forbidden_in_metadata=int(meta["is_forbidden"].sum()),
        n_users=int(len(seq["umap"])),
        n_items=int(len(seq["smap"])),
        sequence_order_source="dataset.pkl（原始文件无时间戳，顺序不可复现）",
        sliding_window=win,
        # 题材分布（按主题材计数）与题材覆盖（一个动漫可命中多类，故两套数不相等）
        genre_distribution={
            row["primary_genre"]: int(row["n"])
            for row in item_stats.groupby("primary_genre").size().reset_index(name="n").to_dict("records")
        },
        genre_coverage_any={
            str(cid): int(item_stats["genres_cn_ids"].map(lambda x: cid in (x or [])).sum())
            for cid in range(1, 13)
        },
        # 缺失值统计：year/score 是原始占位符，primary_genre 为空说明该动漫只有合禁题材
        meta_missing=dict(
            year=int(meta["year"].isna().sum()),
            score=int(meta["score"].isna().sum()),
            primary_genre=int(meta["primary_genre_id"].isna().sum()),
        ),
        compliance=comp,
        deviations_from_rq=[],      # 与 rq.md 的实测偏差清单，逐条追加在下面
    )
    if report["meta_missing"]["primary_genre"]:
        report["deviations_from_rq"].append(
            f"{report['meta_missing']['primary_genre']} 条动漫无 12 类题材标签"
            f"（其 MAL 题材仅为合规剔除项 Hentai/Erotica）"
        )
    # 以下 5 条是实测发现、与 rq.md 原文不符的地方，写进产物以便追溯
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

    # --validate-only：只跑校验、不落正式产物（改代码后自检用，避免覆盖已有产物）
    if args.validate_only:
        # logs/ 被 .gitignore 排除，全新克隆的仓库里并不存在，必须先建目录
        vdir = os.path.join(ROOT, "logs")
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, "validate_only.json"), "w", encoding="utf-8") as f:
            json.dump(dict(report=report, validation=val), f, ensure_ascii=False, indent=2)
        log("validate-only 完成 -> logs/validate_only.json")
        return 0

    # --- 10) 落盘 ---
    log("写出产物 ...")
    # 元数据落两处：主表去掉两个 list 列（parquet 存 list 列读写都慢），
    # 细分标签单独存一份给 build_content_vectors.py 用。
    meta.drop(columns=["genres", "genres_detailed"]).to_parquet(
        os.path.join(outdir, "anime_meta.parquet"), index=False)
    pd.DataFrame({"anime_id": meta["anime_id"], "genres_detailed": meta["genres_detailed"]}
                 ).to_parquet(os.path.join(outdir, "anime_detailed_tags.parquet"), index=False)
    item_stats.to_parquet(os.path.join(outdir, "item_stats.parquet"), index=False)
    user_stats.to_parquet(os.path.join(outdir, "user_stats.parquet"), index=False)

    # 核心产物：序列数据集（约 361MB）。protocol=HIGHEST_PROTOCOL 加载更快。
    # 注意只写 train/val/test/umap/smap/meta 六项，结构与官方 dataset.pkl 保持一致。
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
