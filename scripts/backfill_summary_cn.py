# -*- coding: utf-8 -*-
"""回填 anime.summary_cn（中文简介）与补齐 title_cn。

数据链（全部免代理可直连）：
  MAL id --(data/raw_cn/bangumi_data.json 的 site:mal -> site:bangumi)--> bgm subject id
         --(data/raw_cn/bgmd_full.json 的 subjects[].summary)--> 中文简介

用法：
  python scripts/backfill_summary_cn.py                 # 全量（缺啥补啥，DB 即检查点）
  python scripts/backfill_summary_cn.py --limit 2000    # 只填头部 N 条（按 n_interactions 降序）

无 LLM 依赖：bgmd 是 Bangumi（番组计划）网页爬取快照，简体/繁体中文人工简介，
质量高于机翻。简介仍缺失的（bgmd 未收录）保持 NULL，前端已有英文回退。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from dotenv import dotenv_values  # noqa: E402
import pymysql  # noqa: E402

RAW = ROOT / "data" / "raw_cn"


def load_bangumi_data() -> dict[int, int]:
    """bangumi_data.json：MAL id -> bgm subject id。"""
    d = json.load(open(RAW / "bangumi_data.json", encoding="utf-8"))
    out: dict[int, int] = {}
    for it in d["items"]:
        mal = next((s.get("id") for s in it.get("sites", []) if s.get("site") == "mal"), None)
        bgm = next((s.get("id") for s in it.get("sites", []) if s.get("site") == "bangumi"), None)
        if mal and bgm:
            try:
                out[int(mal)] = int(bgm)
            except (TypeError, ValueError):
                continue
    return out


def load_bgmd() -> dict[int, dict]:
    """bgmd full.json：bgm subject id -> {summary, title_zh}。"""
    p = RAW / "bgmd_full.json"
    if not p.exists():
        return {}
    d = json.load(open(p, encoding="utf-8"))
    out: dict[int, dict] = {}
    for s in d.get("subjects") or []:
        zh = ((s.get("alias") or {}).get("zh") or [None])[0]
        out[int(s["id"])] = {
            "summary": (s.get("summary") or "").strip(),
            "title_zh": (zh or "").strip(),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理 n_interactions 降序的前 N 条（0=全量）")
    args = ap.parse_args()

    mal2bgm = load_bangumi_data()
    bgmd = load_bgmd()
    print(f"bangumi-data 映射 {len(mal2bgm)} 条 | bgmd {len(bgmd)} 条")

    cfg = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    conn = pymysql.connect(host=cfg["MYSQL_HOST"], port=int(cfg["MYSQL_PORT"]),
                           user=cfg["MYSQL_USER"], password=cfg["MYSQL_PASSWORD"],
                           database=cfg["MYSQL_DB"], charset="utf8mb4",
                           autocommit=False)
    cur = conn.cursor()

    order = "ORDER BY n_interactions DESC" if args.limit else ""
    limit = f"LIMIT {int(args.limit)}" if args.limit else ""
    cur.execute(f"SELECT id, mal_url, title_cn FROM anime {order} {limit}")
    rows = cur.fetchall()

    n_sum = n_title = n_miss = 0
    updates: list[tuple[str, str | None, str | None, int]] = []
    for rid, url, title_cn in rows:
        m = re.search(r"/anime/(\d+)", url or "")
        if not m:
            continue
        bgm = mal2bgm.get(int(m.group(1)))
        info = bgmd.get(bgm) if bgm else None
        if not info:
            n_miss += 1
            continue
        new_sum = info["summary"] or None
        new_title = title_cn or info["title_zh"] or None
        if new_sum:
            n_sum += 1
        if new_title and new_title != title_cn:
            n_title += 1
        if new_sum or (new_title and new_title != title_cn):
            updates.append((new_title, new_sum, rid))

    cur.executemany(
        "UPDATE anime SET title_cn = %s, summary_cn = %s WHERE id = %s", updates)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM anime WHERE summary_cn IS NOT NULL")
    total_cn = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM anime WHERE title_cn IS NOT NULL")
    total_title = cur.fetchone()[0]
    print(f"本轮：简介 {n_sum} 条 | 标题补齐 {n_title} 条 | bgmd 未覆盖 {n_miss} 条")
    print(f"全表：summary_cn {total_cn} | title_cn {total_title}")
    conn.close()


if __name__ == "__main__":
    main()
