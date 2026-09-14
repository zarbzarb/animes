# -*- coding: utf-8 -*-
"""导出「物品索引 ↔ 数据集 animeID」的双向映射（线上召回的必备产物）。

用法
----
    python scripts/export_item_index.py

依赖：先跑 scripts/preprocess.py（需要 data/processed/seq_dataset.pkl）

产物
----
    data/features/item_index.json
        {
          "n_items": 15687,
          "src_to_index": {"1": 1, "2": 2, ...},
          "index_to_src": {"1": "1", ...},
          "source": ".../seq_dataset.pkl#smap",
          "checksum_sha1": "<对 forward 映射的稳定摘要>"
        }

为什么必须单独落一份，而不能"用 src_anime_id 当索引"
--------------------------------------------------
`seq_dataset.pkl` 的 `smap`（数据集 animeID → 池内索引）**不是恒等映射**。
实测：前 7,581 项恰好相等（`smap[k] == k`），从 7,582 起出现一个缺口
（该 animeID 未进推荐池），此后**整体错位 1**，累计到末尾错位 291。
于是

    item_index != src_anime_id      对约 8,100 个物品成立

如果线上直接用 `src_anime_id` 去查物品嵌入表，这些物品**全部取错向量**，
召回结果会变得很差、**且不会有任何报错** —— 这是本项目最容易静默出错的地方，
与内容矩阵「第 i-1 行 = 物品 idx i」的行序约定同源，必须显式落盘 + 校验。

校验
----
`tests/test_retrieval/test_item_index.py` 会断言：
① `index_to_src` 与 `src_to_index` 严格互逆；
② 索引集合恰为 `1..n_items` 无缺口；
③ `n_items` 与 checkpoint 的 `model_config.n_items` 一致。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys

# 路径锚定：与工作目录无关（项目约定，禁止 os.getcwd()）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SEQ = os.path.join(ROOT, "data", "processed", "seq_dataset.pkl")
DEFAULT_OUT = os.path.join(ROOT, "data", "features", "item_index.json")


def _log(msg: str) -> None:
    print(f"[export_item_index] {msg}", flush=True)


def build_mapping(seq_path: str) -> dict:
    with open(seq_path, "rb") as f:
        seq = pickle.load(f)
    smap = seq.get("smap")
    if not isinstance(smap, dict) or not smap:
        raise SystemExit(f"{seq_path} 里没有可用的 smap")

    src_to_index = {int(src): int(idx) for src, idx in smap.items()}
    index_to_src = {idx: src for src, idx in src_to_index.items()}
    if len(index_to_src) != len(src_to_index):
        dup = len(src_to_index) - len(index_to_src)
        raise SystemExit(f"smap 存在重复索引（{dup} 个），索引空间不合法")

    n_items = len(src_to_index)
    idxs = sorted(index_to_src)
    if idxs != list(range(1, n_items + 1)):
        missing = sorted(set(range(1, n_items + 1)) - set(idxs))[:10]
        raise SystemExit(f"索引不是 1..{n_items} 的连续区间，缺口示例：{missing}")

    checksum = hashlib.sha1(
        ",".join(f"{k}:{src_to_index[k]}" for k in sorted(src_to_index))
        .encode("utf-8")).hexdigest()

    return {
        "n_items": n_items,
        "src_to_index": {str(k): src_to_index[k] for k in sorted(src_to_index)},
        "index_to_src": {str(k): index_to_src[k] for k in idxs},
        "source": seq_path,
        "checksum_sha1": checksum,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="导出物品索引双向映射")
    ap.add_argument("--seq", default=DEFAULT_SEQ)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not os.path.exists(args.seq):
        raise SystemExit(f"找不到 {args.seq}，请先跑 scripts/preprocess.py")
    payload = build_mapping(args.seq)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    nonident = sum(1 for k, v in payload["src_to_index"].items() if int(k) != v)
    _log(f"已写出 {args.out}")
    _log(f"  n_items={payload['n_items']}  非恒等映射条目={nonident} "
         f"({nonident / payload['n_items']:.1%})")
    _log(f"  checksum_sha1={payload['checksum_sha1'][:12]}")
    if nonident:
        _log("  ⚠️ smap 非恒等 —— 线上绝不可用 src_anime_id 直接当物品索引")
    return 0


if __name__ == "__main__":
    sys.exit(main())
