# -*- coding: utf-8 -*-
"""anime.title_cn / summary_cn 中文化脚本。

两层数据源（按优先级）
--------------------
1. **bangumi-data 映射**（data/raw_cn/bangumi_data.json，npm CDN 可下，
   jsdelivr 直连）：人工维护的通行中文译名，覆盖 2016 后新番 → 实测
   可填 7,131/14,136（头部热门全中）。零成本、零外部请求。
2. **LLM 翻译兜底**（.env 的 OpenAI 兼容接口）：补 bangumi-data 缺的
   老番标题 + 简介中文。⚠️ 需要**有效的** LLM_API_KEY —— 2026-09-15 时
   .env 里是占位符（sk-xxx…），LLM 步骤会自动跳过并提示，不影响第 1 层。

范围与成本决策
----
- 标题：先 bangumi-data 全量映射，缺的交给 LLM（一批 40 条）。
- 简介：bangumi-data 无简介字段；只翻**头部**（--summary-limit，默认
  2000，按 n_interactions 降序）—— 推荐位几乎全是头部番，全量 13.5k 条
  是数百万 token，不值。

健壮性
----
- DB 即检查点：只处理 `title_cn IS NULL` / `summary_cn IS NULL` 的行。
- LLM 输出强制 JSON，解析失败整批重试一次，再失败跳过下轮再补。
- 限速 1 req/s，429 退避。

用法
----
    python scripts/translate_cn.py                # 全流程（含 LLM，key 无效自动跳过）
    python scripts/translate_cn.py --titles-only
    python scripts/translate_cn.py --summary-limit 5000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BGD_PATH = ROOT / "data" / "raw_cn" / "bangumi_data.json"
BGD_URL = "https://cdn.jsdelivr.net/npm/bangumi-data@0.3/dist/data.json"
TITLE_BATCH = 40
SUMMARY_BATCH = 5
REQUEST_INTERVAL = 1.05
RETRY_BACKOFF = [5, 15, 40]

TITLE_PROMPT = (
    "把下面的动漫英文/日文罗马音标题翻译成中文。规则：\n"
    "1. 优先使用中文圈最通行的官方/通行译名（如 Attack on Titan → 进击的巨人）；\n"
    "2. 没有通行译名的按含义直译；无实义的假名串保留原文；\n"
    "3. 只输出 JSON：{\"items\": [{\"i\": 序号, \"t\": \"中文译名\"}]}，序号用输入给的 i。\n\n"
    "标题列表：\n{titles}")

SUMMARY_PROMPT = (
    "把下面的动漫剧情简介（英文）翻译成简体中文。规则：\n"
    "1. 忠实原意、语句通顺，人名/作品名用中文圈通行译名；\n"
    "2. 保留换行；控制在原文长度以内，不要扩写；\n"
    "3. 只输出 JSON：{\"items\": [{\"i\": 序号, \"t\": \"中文简介\"}]}。\n\n"
    "简介列表：\n{sums}")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
MAL_ID_RE = re.compile(r"/anime/(\d+)")

cfg: dict = {}


# ---------------------------------------------------------------- LLM
def _llm_cfg() -> dict:
    from dotenv import dotenv_values
    c = dotenv_values(ROOT / ".env")
    return {"base_url": c.get("LLM_BASE_URL", ""), "key": c.get("LLM_API_KEY", ""),
            "model": c.get("LLM_MODEL", "qwen-plus")}


def _chat(prompt: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1, "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    for backoff in [*RETRY_BACKOFF, None]:
        try:
            req = urllib.request.Request(
                cfg["base_url"].rstrip("/") + "/chat/completions", data=body,
                method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {cfg['key']}", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            if backoff is None:
                raise RuntimeError(f"LLM 请求失败: {exc}") from exc
            print(f"  ! LLM 请求失败（{type(exc).__name__}），{backoff}s 重试", flush=True)
            time.sleep(backoff)
    raise RuntimeError("unreachable")


def _parse_items(text: str) -> dict[int, str]:
    data = json.loads(text)
    return {int(it["i"]): str(it.get("t") or "").strip()
            for it in (data.get("items") or [])}


# ---------------------------------------------------------------- 层 1：bangumi-data
def _bgd_mapping() -> dict[int, str]:
    if not BGD_PATH.exists():
        print(f"bangumi-data 文件不存在（{BGD_PATH}），跳过第 1 层", flush=True)
        return {}
    d = json.loads(BGD_PATH.read_text(encoding="utf-8"))
    mal2cn: dict[int, str] = {}
    for it in d["items"]:
        mal = next((s.get("id") for s in it.get("sites", [])
                    if s.get("site") == "mal"), None)
        cn = (it.get("titleTranslate") or {}).get("zh-Hans") or []
        if mal and cn and int(mal) not in mal2cn:
            mal2cn[int(mal)] = str(cn[0]).strip()
    return mal2cn


def _fill_by_bgd(conn, mal2cn: dict[int, str]) -> tuple[int, int]:
    """按 mal_url 映射填 title_cn；返回 (本次填入, 剩余未覆盖)。"""
    with conn.cursor() as cur:
        cur.execute("SELECT id, mal_url FROM anime WHERE title_cn IS NULL")
        rows = cur.fetchall()
    updates = []
    for rid, url in rows:
        m = MAL_ID_RE.search(url or "")
        if m and int(m.group(1)) in mal2cn:
            updates.append((mal2cn[int(m.group(1))], rid))
    if updates:
        with conn.cursor() as cur:
            cur.executemany("UPDATE anime SET title_cn = %s WHERE id = %s", updates)
        conn.commit()
    return len(updates), len(rows) - len(updates)


# ---------------------------------------------------------------- 层 2：LLM
def _translate_rows(conn, rows: list[tuple], *, kind: str, batch: int,
                    max_tokens: int, label: str, col: str) -> None:
    done = 0
    t0 = time.time()
    tpl = TITLE_PROMPT if kind == "title" else SUMMARY_PROMPT
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        listing = "\n".join(f"{i}. {str(txt)[:1500]}"
                            for i, (_, txt) in enumerate(chunk, start=1))
        try:
            out = _parse_items(_chat(tpl.format(titles=listing, sums=listing),
                                     max_tokens))
        except Exception as exc:
            print(f"  ! 批次 {start}~{start + len(chunk)} 失败，跳过：{exc}",
                  flush=True)
            time.sleep(REQUEST_INTERVAL)
            continue
        updates = [(out[i], rid) for i, (rid, _) in enumerate(chunk, start=1)
                   if out.get(i)]
        if updates:
            with conn.cursor() as cur:
                cur.executemany(f"UPDATE anime SET {col} = %s WHERE id = %s", updates)
            conn.commit()
        done += len(updates)
        n = min(start + batch, len(rows))
        rate = n / max(time.time() - t0, 1)
        print(f"  [{label}] {n}/{len(rows)}  已写 {done}  速率 {rate:.1f} 条/s",
              flush=True)
        time.sleep(REQUEST_INTERVAL)


def _llm_available() -> bool:
    if not cfg["base_url"] or not cfg["key"] or cfg["key"].startswith("sk-xxx"):
        print("⚠️ LLM_API_KEY 无效（占位符或为空），跳过 LLM 层；"
              "配好 key 后重跑本脚本可补齐长尾标题与简介中文", flush=True)
        return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="翻译 anime 标题/简介为中文")
    ap.add_argument("--titles-only", action="store_true", help="只翻标题")
    ap.add_argument("--summary-limit", type=int, default=2000,
                    help="简介 LLM 翻译条数上限（热度降序）；0=不翻简介")
    args = ap.parse_args(argv)

    from dotenv import dotenv_values
    import pymysql
    db = dotenv_values(ROOT / ".env")
    conn = pymysql.connect(host=db["MYSQL_HOST"], port=int(db["MYSQL_PORT"]),
                           user=db["MYSQL_USER"], password=db["MYSQL_PASSWORD"],
                           database=db["MYSQL_DB"], charset="utf8mb4",
                           autocommit=False)

    # ---------- 层 1：bangumi-data 映射（无外部请求） ----------
    filled, remain = _fill_by_bgd(conn, _bgd_mapping())
    print(f"[bangumi-data] 标题新填 {filled} 条，剩余未覆盖 {remain} 条", flush=True)

    # ---------- 层 2：LLM 兜底（key 无效自动跳过） ----------
    if _llm_available():
        cur = conn.cursor()
        cur.execute("SELECT id, title FROM anime "
                    "WHERE title_cn IS NULL ORDER BY n_interactions DESC, id ASC")
        rows = cur.fetchall()
        print(f"[LLM] 待翻标题 {len(rows)} 条", flush=True)
        _translate_rows(conn, rows, kind="title", batch=TITLE_BATCH,
                        max_tokens=4096, label="标题", col="title_cn")

        if args.summary_limit > 0 and not args.titles_only:
            cur.execute(
                "SELECT id, summary FROM anime "
                "WHERE summary_cn IS NULL AND summary IS NOT NULL AND summary <> '' "
                "ORDER BY n_interactions DESC, id ASC LIMIT %s",
                (args.summary_limit,))
            rows = cur.fetchall()
            print(f"[LLM] 待翻简介 {len(rows)} 条", flush=True)
            _translate_rows(conn, rows, kind="summary", batch=SUMMARY_BATCH,
                            max_tokens=4096, label="简介", col="summary_cn")

    conn.close()
    print("全部完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
