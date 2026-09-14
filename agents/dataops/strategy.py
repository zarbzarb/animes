# -*- coding: utf-8 -*-
"""A8 的数据质检与题材归类策略 —— **纯函数**。

质检的 5 类检查（`data_quality_log.check_type` 的取值域）
------------------------------------------------------
| 类型 | 判据 | 为什么必要 |
|---|---|---|
| `missing` | 必填字段为空 | 标题为空会让前端卡片空白 |
| `duplicate` | `src_anime_id` 重复 | 重复会让同一部番占两个名额 |
| `outlier` | 年份/评分越界 | 年份 9999 会破坏"新番"判定 |
| `forbidden` | 含 Hentai/Erotica | **合规要求**，必须拦截 |
| `genre_unmapped` | 21 类无法归并到 12 类 | A3/A4 的题材重合度会算错 |

⚠️ 一个刻意的设计：**质检不自动修数据**。
`docs/agents.md` §A8 的能力边界写着"不直接删除动漫（只标记下架）"。
所以质检只产出报告与建议，修复是独立的 `fix_data` 任务且需要人工确认
（`dry_run` 默认行为在 agent 层）。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

__all__ = [
    "CHECK_MISSING", "CHECK_DUPLICATE", "CHECK_OUTLIER", "CHECK_FORBIDDEN",
    "CHECK_GENRE_UNMAPPED", "ALERT_NORMAL", "ALERT_WARN", "ALERT_CRITICAL",
    "FORBIDDEN_MAL_GENRES", "check_rows", "classify_genres", "summarize",
]

CHECK_MISSING = "missing"
CHECK_DUPLICATE = "duplicate"
CHECK_OUTLIER = "outlier"
CHECK_FORBIDDEN = "forbidden"
CHECK_GENRE_UNMAPPED = "genre_unmapped"

ALERT_NORMAL, ALERT_WARN, ALERT_CRITICAL = 0, 1, 2

# MAL 的 20/21 号题材（Erotica / Hentai）—— 合规红线，见 configs/genre_taxonomy.yaml
FORBIDDEN_MAL_GENRES = (20, 21)

YEAR_MIN, YEAR_MAX = 1900, 2100
SCORE_MIN, SCORE_MAX = 0.0, 10.0

# 21 类 MAL → 12 类中文（与 configs/genre_taxonomy.yaml 的 genres_12.mal_ids 同源）
MAL_TO_12: dict[int, int] = {}
for _gid, _mals in {
    1: (8,), 2: (1, 4), 3: (12, 19), 4: (5, 17, 18), 5: (11, 7), 6: (10,),
    7: (9,), 8: (16,), 9: (6, 13), 10: (3, 2, 15), 12: (14,),
}.items():
    for _m in _mals:
        MAL_TO_12[_m] = _gid


def classify_genres(mal_ids: Sequence[int]) -> tuple[list[int], list[int]]:
    """21 类 MAL 题材 → `(12 类 id 列表, 未归并的 id 列表)`。

    未归并的是"合法但本体系不收录"的题材（如 11=Music 需要靠细分标签判定，
    见 `configs/genre_taxonomy.yaml` 的 `music_keywords`）。它们**不是错误**，
    但要报出来 —— 否则"某题材召回率恒为 0"时没人知道是数据没归类。
    """
    mapped: list[int] = []
    unmapped: list[int] = []
    for m in mal_ids:
        gid = MAL_TO_12.get(int(m))
        if gid is None:
            unmapped.append(int(m))
        elif gid not in mapped:
            mapped.append(gid)
    return sorted(mapped), sorted(unmapped)


def check_rows(rows: Sequence[dict], *, required: Sequence[str] = ("title",)
               ) -> list[dict]:
    """对一批动漫元数据跑 5 类检查，返回问题明细（不是汇总）。

    返回 `[{check_type, anime_id, field, value, message}]`。
    汇总在 `summarize()` 里做 —— 分开是为了让"哪一条数据有问题"
    可以直接给运营看，而不只是一个百分比。
    """
    problems: list[dict] = []
    seen: dict[int, int] = {}

    for r in rows:
        aid = int(r.get("src_anime_id") or r.get("anime_id") or 0)

        for f in required:
            if r.get(f) in (None, "", []):
                problems.append({"check_type": CHECK_MISSING, "anime_id": aid,
                                 "field": f, "value": r.get(f),
                                 "message": f"必填字段 {f} 为空"})

        if aid:
            if aid in seen:
                problems.append({"check_type": CHECK_DUPLICATE, "anime_id": aid,
                                 "field": "src_anime_id", "value": aid,
                                 "message": f"src_anime_id={aid} 重复"})
            seen[aid] = 1

        year = r.get("year")
        if year is not None:
            try:
                y = int(year)
                if not (YEAR_MIN < y < YEAR_MAX):
                    problems.append({"check_type": CHECK_OUTLIER, "anime_id": aid,
                                     "field": "year", "value": year,
                                     "message": f"年份越界：{year}"})
            except (TypeError, ValueError):
                problems.append({"check_type": CHECK_OUTLIER, "anime_id": aid,
                                 "field": "year", "value": year,
                                 "message": f"年份非数值：{year!r}"})

        score = r.get("score")
        if score is not None:
            try:
                s = float(score)
                if not (SCORE_MIN <= s <= SCORE_MAX):
                    problems.append({"check_type": CHECK_OUTLIER, "anime_id": aid,
                                     "field": "score", "value": score,
                                     "message": f"评分越界：{score}"})
            except (TypeError, ValueError):
                pass

        raw_mals = r.get("genre_raw") or r.get("mal_genres") or []
        mals = [int(x) for x in raw_mals if str(x).strip().lstrip("-").isdigit()]
        bad = [m for m in mals if m in FORBIDDEN_MAL_GENRES]
        if bad:
            problems.append({"check_type": CHECK_FORBIDDEN, "anime_id": aid,
                             "field": "genre_raw", "value": bad,
                             "message": f"含违规题材 MAL id {bad}，应下架"})
        if mals:
            _mapped, unmapped = classify_genres(mals)
            if unmapped:
                problems.append({"check_type": CHECK_GENRE_UNMAPPED, "anime_id": aid,
                                 "field": "genre_raw", "value": unmapped,
                                 "message": f"MAL id {unmapped} 未归并到 12 类体系"})
    return problems


def summarize(problems: Sequence[dict], *, target_table: str = "anime",
              total_rows: int = 0, critical_rate: float = 0.05,
              warn_rate: float = 0.01) -> list[dict]:
    """把问题明细汇总成 `data_quality_log` 的行。

    告警分级只看 `problem_rate`：`>= critical_rate` 严重，`>= warn_rate` 警告。
    违规题材（`forbidden`）**不受阈值约束，只要有就是严重** —— 它是合规问题，
    不是数据质量问题，1 条也要报。
    """
    by_type: dict[str, list[dict]] = {}
    for p in problems:
        by_type.setdefault(str(p["check_type"]), []).append(p)

    out: list[dict] = []
    for ctype in (CHECK_MISSING, CHECK_DUPLICATE, CHECK_OUTLIER,
                  CHECK_FORBIDDEN, CHECK_GENRE_UNMAPPED):
        items = by_type.get(ctype, [])
        n = len(items)
        rate = round(n / total_rows, 6) if total_rows > 0 else 0.0
        if ctype == CHECK_FORBIDDEN:
            level = ALERT_CRITICAL if n else ALERT_NORMAL
        elif rate >= critical_rate:
            level = ALERT_CRITICAL
        elif rate >= warn_rate:
            level = ALERT_WARN
        else:
            level = ALERT_NORMAL
        out.append({
            "check_type": ctype, "target_table": target_table,
            "total_rows": int(total_rows), "problem_rows": n,
            "problem_rate": rate,
            "detail": {"samples": items[:20]}, "alert_level": level,
        })
    return out
