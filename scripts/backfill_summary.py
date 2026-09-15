# -*- coding: utf-8 -*-
"""anime.summary（剧情简介）回填脚本 —— AniList GraphQL 批量方案。

背景
----
数据集 animes.csv 无简介字段（preprocess.py §"无剧情简介"），库里 14,136 条
anime.summary 全空。用户要"追番前先看介绍"，需外部回填。

选型（2026-09-15 实测）
--------------------
- Jikan（MAL 镜像）：逐条接口当时持续 504，且 1 req/条 × 限速 60/min ≈ 4 小时，弃。
- manami-project/anime-offline-database：schema 无 synopsis 字段，弃。
- **AniList GraphQL**（graphql.anilist.co）：直连可用，`idMal_in` 一批查 50 条
  全带 description，限速 90 req/min → 全量 ≈ 283 请求 ≈ 5 分钟。

要点
----
- **键**：`anime.mal_url`（myanimelist.net/anime/431）提取 MAL ID，与
  `src_anime_id` 是两个数域；AniList 的 `idMal` 即 MAL ID。
- **顺序**：按 `n_interactions` 降序，头部番优先覆盖推荐位。
- **断点续传**：DB 即检查点（只处理 summary 为空的行）；AniList 收录外的
  idMal 记入 logs/summary_missing.json，不反复重试。
- **清洗**：description 是 HTML（<br>、<i>…），转纯文本 + 实体解码。
- **CWD 无关**：.env 锚定 `Path(__file__).parents[1]`，不碰 os.getcwd()。

用法
----
    python scripts/backfill_summary.py                 # 全量（可多次续跑）
    python scripts/backfill_summary.py --limit 2000    # 只填头部 2000
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"

ANILIST_URL = "https://graphql.anilist.co"
QUERY = """query ($ids: [Int]) {
  Page(page: 1, perPage: 50) {
    media(idMal_in: $ids, type: ANIME) {
      idMal
      description(asHtml: false)
    }
  }
}"""
BATCH = 50
REQUEST_INTERVAL = 1.05        # ≈57 req/min，低于 AniList 90/min 上限
RETRY_BACKOFF = [3, 8, 20]     # 429/5xx/网络错误退避，用尽则本批跳过
MAL_ID_RE = re.compile(r"myanimelist\.net/anime/(\d+)")
TAG_RE = re.compile(r"<[^>]+>")


def _clean_desc(text: str) -> str:
    """AniList description 是 HTML：标签转换行/剔除，实体解码，压掉多余空行。"""
    s = (text or "").strip()
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = TAG_RE.sub("", s)
    s = html.unescape(s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _load_db_cfg() -> dict:
    from dotenv import dotenv_values
    cfg = dotenv_values(ROOT / ".env")
    return {k: cfg.get(k, "") for k in
            ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DB")}


def _connect(cfg: dict):
    import pymysql
    return pymysql.connect(
        host=cfg["MYSQL_HOST"], port=int(cfg["MYSQL_PORT"]),
        user=cfg["MYSQL_USER"], password=cfg["MYSQL_PASSWORD"],
        database=cfg["MYSQL_DB"], charset="utf8mb4", autocommit=False)


def _fetch_batch(mal_ids: list[int]) -> dict[int, str] | None:
    """一批 50 条；失败重试用尽返回 None（调用方跳过本批）。"""
    body = json.dumps({"query": QUERY, "variables": {"ids": mal_ids}}).encode("utf-8")
    for backoff in [*RETRY_BACKOFF, None]:
        try:
            req = urllib.request.Request(
                ANILIST_URL, data=body, method="POST",
                headers={"Content-Type": "application/json",
                         "Accept": "application/json",
                         # Cloudflare 1010 会拦 python-urllib 的默认指纹，需伪装浏览器 UA
                         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                                       "Chrome/126.0.0.0 Safari/537.36"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("errors") and not payload.get("data"):
                raise RuntimeError(f"GraphQL errors: {payload['errors'][:1]}")
            out = {}
            for m in payload["data"]["Page"]["media"]:
                if m.get("idMal") is not None:
                    out[int(m["idMal"])] = _clean_desc(m.get("description") or "")
            return out
        except Exception as exc:  # noqa: BLE001 —— 网络层错误统一退避重试
            if backoff is None:
                print(f"  ! 批量请求失败（{type(exc).__name__}）重试用尽，本批跳过", flush=True)
                return None
            print(f"  ! 请求失败（{type(exc).__name__}），{backoff}s 后重试", flush=True)
            time.sleep(backoff)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="回填 anime.summary（AniList GraphQL）")
    ap.add_argument("--limit", type=int, default=0,
                    help="本次最多回填条数；0=不限制（填到没有空行为止）")
    args = ap.parse_args(argv)

    LOG_DIR.mkdir(exist_ok=True)
    conn = _connect(_load_db_cfg())
    with conn.cursor() as cur:
        cur.execute(
            "SELECT src_anime_id, mal_url FROM anime "
            "WHERE (summary IS NULL OR summary = '') "
            "AND mal_url LIKE '%%myanimelist.net/anime/%%' "
            "ORDER BY n_interactions DESC, src_anime_id ASC")
        rows = cur.fetchall()

    todo = [(int(sid), int(MAL_ID_RE.search(url).group(1)))
            for sid, url in rows if MAL_ID_RE.search(url or "")]
    print(f"待回填 {len(todo)} 条（无 mal_url 跳过 {len(rows) - len(todo)} 条）", flush=True)
    if args.limit > 0:
        todo = todo[: args.limit]

    done = missing = 0
    still_missing: list[int] = []
    t0 = time.time()
    for start in range(0, len(todo), BATCH):
        chunk = todo[start:start + BATCH]
        got = _fetch_batch([mal_id for _, mal_id in chunk])
        if got is None:
            still_missing.extend(mal_id for _, mal_id in chunk)
            continue
        updates = []
        for sid, mal_id in chunk:
            desc = got.get(mal_id)
            if desc is None:
                missing += 1
                still_missing.append(mal_id)   # AniList 未收录该 idMal
                continue
            updates.append((desc, sid))
            done += 1
        if updates:
            with conn.cursor() as cur:
                cur.executemany("UPDATE anime SET summary = %s WHERE src_anime_id = %s",
                                updates)
            conn.commit()

        n = min(start + BATCH, len(todo))
        rate = n / max(time.time() - t0, 1)
        print(f"  进度 {n}/{len(todo)}  成功 {done}  AniList 缺失 {missing}  "
              f"速率 {rate:.1f} 条/s", flush=True)
        time.sleep(REQUEST_INTERVAL)

    (LOG_DIR / "summary_missing.json").write_text(
        json.dumps(sorted(set(still_missing))), encoding="utf-8")
    print(f"完成：成功 {done}，AniList 未收录 {missing}，"
          f"耗时 {(time.time() - t0) / 60:.1f} 分钟；缺失清单 logs/summary_missing.json",
          flush=True)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
