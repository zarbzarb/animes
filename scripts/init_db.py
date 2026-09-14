# -*- coding: utf-8 -*-
"""初始化数据库：建表 + 基础数据 + 真实番剧 + 演示用户。

用法
----
    # 建表 + 导入番剧 + 造 5 个演示用户（默认）
    python scripts/init_db.py

    # 只建表，不导入数据
    python scripts/init_db.py --no-anime --no-demo

    # 换库（覆盖 .env 里的 MYSQL_*）
    python scripts/init_db.py --url "mysql+pymysql://root:pw@127.0.0.1:3306/anirec?charset=utf8mb4"

    # 清空后重建（**会 DROP 全部业务表**，仅限开发环境）
    python scripts/init_db.py --reset

数据来源（都是既有产物，不重新计算）
----------------------------------
* 番剧元数据：`data/processed/anime_meta.parquet`
  由 `scripts/preprocess.py` 产出，含标题/年份/评分/集数/图片与 12 类中文题材
  （`genres_cn_ids` / `primary_genre_id`）与合规标记 `is_forbidden`。**直接复用，不在此重算**。
* 物品交互数：`data/processed/item_stats.parquet` 的 `n_positive`
  ⚠️ `anime.n_interactions` 必须来自它，**不能**由演示用户的追番记录统计得出 ——
  库里只有 5 个演示用户，统计出来每部番最多 5 次交互，全部会被
  `COLD_START_THRESHOLD=10` 判成冷启动，A3/A4 的冷启动分支会集体走错。
* 物品池：`data/features/item_index.json`
  只导入「模型池内」的 15,687 部。为什么：A2/A4 产出的是**池内索引**，
  网关要靠 `src_anime_id` 反查元信息；若库里存在池外番剧，浏览能看到、
  却永远召不回，演示时容易误判成"召回漏了"。
* 演示用户：`data/processed/seq_dataset.pkl` 的 `train` 序列
  （`train[uid]` 是**池内索引**列表，经 `item_index` 反解成 `src_anime_id` 落库）。

幂等性
------
本脚本可反复执行：番剧按 `src_anime_id` upsert，演示用户按 `username` 复用，
追番记录按 `(user_id, anime_id)` upsert。重复跑不会产生重复行。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import date, datetime, timedelta

# 路径锚定：与工作目录无关（项目约定，禁止 os.getcwd()）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_META = os.path.join(ROOT, "data", "processed", "anime_meta.parquet")
DEFAULT_STATS = os.path.join(ROOT, "data", "processed", "item_stats.parquet")
DEFAULT_INDEX = os.path.join(ROOT, "data", "features", "item_index.json")
DEFAULT_SEQ = os.path.join(ROOT, "data", "processed", "seq_dataset.pkl")

# 演示用户：内部用户序号（seq_dataset 的 0-based uid）→ (用户名, 昵称, 画像)
DEMO_USERS: list[tuple[int, str, str, str]] = [
    (0, "anifan", "番剧迷", "偏好明确的高活跃用户"),
    (1, "casual", "随便看看", "低活跃、口味分散"),
    (2, "binger", "一口气看完", "追番量大、评分积极"),
    (3, "critic", "挑剔的观众", "评分严格、弃番多"),
    (4, "newbie", "新来的", "冷启动用户（记录很少）"),
]
# 管理员账号：**不绑定数据集用户**（`src_user_id=None`，无追番记录）。
# 为什么要单独给一个：`/admin/*` 需要 `role=1`，而数据集里的 5 个演示用户
# 全是普通用户 —— 没有这个账号，管理端 14 个接口在演示时**一个都点不开**，
# 只能靠改数据库临时提权。
DEMO_ADMIN: tuple[str, str] = ("admin", "系统管理员")
DEMO_PASSWORD = "P@ssw0rd"


def _log(msg: str) -> None:
    print(f"[init_db] {msg}", flush=True)


# ---------------------------------------------------------------- 建表


def _create_schema(reset: bool, apply_extras: bool) -> None:
    from sqlalchemy import text

    from server.core.config import settings
    from server.db.base import Base
    from server.db.session import get_engine, init_db

    if settings.is_sqlite:
        _log("检测到 SQLite，跳过 CREATE DATABASE")
    else:
        _create_mysql_database()

    engine = get_engine()
    if reset:
        _log("--reset：删除全部已注册的表")
        Base.metadata.drop_all(engine)

    init_db(engine=engine, with_seed=True, apply_mysql_extras=apply_extras)
    with engine.connect() as conn:
        n = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE()"
            if not settings.is_sqlite
            else "SELECT COUNT(*) FROM sqlite_master WHERE type='table'")).scalar()
    _log(f"建表完成，当前表数量：{n}")


def _create_mysql_database() -> None:
    """MySQL 下先建库（连接串里若库不存在会直接连不上）。"""
    import pymysql

    from server.core.config import settings

    conn = pymysql.connect(host=settings.MYSQL_HOST, port=settings.MYSQL_PORT,
                           user=settings.MYSQL_USER, password=settings.MYSQL_PASSWORD,
                           charset="utf8mb4", connect_timeout=5)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{settings.MYSQL_DB}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.commit()
        _log(f"数据库 `{settings.MYSQL_DB}` 就绪")
    finally:
        conn.close()


# ---------------------------------------------------------------- 番剧


def _import_anime(batch: int = 2000) -> int:
    """把「池内」番剧写入 `anime` / `anime_genre`（meta ∪ stats 合并）。"""
    import pandas as pd
    from sqlalchemy import delete, func, insert, select
    from sqlalchemy.dialects.mysql import insert as mysql_insert
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    from server.core.config import settings
    from server.db.models.anime import Anime, AnimeContent, AnimeGenre, Genre
    from server.db.session import session_scope

    if not os.path.exists(DEFAULT_META):
        _log(f"⚠️  找不到 {DEFAULT_META}，跳过番剧导入")
        return 0
    if not os.path.exists(DEFAULT_INDEX):
        _log(f"⚠️  找不到 {DEFAULT_INDEX}，跳过番剧导入"
             "（先跑 python scripts/export_item_index.py）")
        return 0

    with open(DEFAULT_INDEX, "r", encoding="utf-8") as f:
        pool = {int(k) for k in (json.load(f).get("src_to_index") or {})}
    _log(f"物品池 {len(pool)} 部，开始读取 parquet")

    meta = pd.read_parquet(DEFAULT_META)
    keep = ["anime_id", "title", "alternative_title", "type", "year", "score",
            "episodes", "mal_url", "image_url", "genres_cn_ids", "primary_genre_id",
            "is_forbidden"]
    meta = meta[[c for c in keep if c in meta.columns]]

    # 交互数来自 item_stats（见模块 docstring 的 ⚠️）
    if os.path.exists(DEFAULT_STATS):
        stats = pd.read_parquet(DEFAULT_STATS)
        cols = [c for c in ("anime_id", "n_positive", "is_tail") if c in stats.columns]
        df = meta.merge(stats[cols], on="anime_id", how="left")
        _log(f"已合并 item_stats：{len(df)} 行")
    else:
        _log(f"⚠️  找不到 {DEFAULT_STATS}，n_interactions 置 0")
        df = meta.copy()
        df["n_positive"] = 0
        df["is_tail"] = False

    df = df[df["anime_id"].astype(int).isin(pool)]
    df = df[~df["is_forbidden"].astype(bool)]        # 合规：Erotica / Hentai 不入库
    df["n_positive"] = df["n_positive"].fillna(0).astype(int)
    _log(f"待导入 {len(df)} 部（池内 ∩ 非违禁）")

    def _clean(v):
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        s = str(v)
        return None if s in ("", "nan", "None") else s

    def _rows(chunk):
        for r in chunk.itertuples(index=False):
            yield {
                "src_anime_id": int(r.anime_id),
                "title": str(r.title),
                "alt_title": _clean(getattr(r, "alternative_title", None)),
                "type": _clean(getattr(r, "type", None)),
                "year": (int(r.year) if getattr(r, "year", None) is not None
                         and not pd.isna(r.year) else None),
                "score": (float(r.score) if getattr(r, "score", None) is not None
                          and not pd.isna(r.score) else None),
                "episodes": (int(r.episodes) if getattr(r, "episodes", None) is not None
                             and not pd.isna(r.episodes) else None),
                "image_url": _clean(getattr(r, "image_url", None)),
                "mal_url": _clean(getattr(r, "mal_url", None)),
                # 注意：`anime` 表**没有** primary_genre_id 列 ——
                # 主标签存在 `anime_genre.is_primary`（见下方 links 构造），
                # 这样一部番的多题材关系与主标签只有一个事实来源。
                "n_interactions": int(r.n_positive),
                "is_forbidden": 0,
                "is_online": 1,
            }

    # 用 `__table__` 而不是映射类：ORM 实体的批量 values() 会把
    # 关系属性（`genres`）也拉进解析路径，报 `_bulk_update_tuples` 之类的怪错。
    # 种子脚本只关心表，用 Core 的表对象最直接。
    table = Anime.__table__
    is_mysql = not settings.is_sqlite
    total = 0
    for start in range(0, len(df), batch):
        rows = list(_rows(df.iloc[start:start + batch]))
        if not rows:
            continue
        stmt = (mysql_insert(table) if is_mysql else sqlite_insert(table)).values(rows)
        update_cols = {c: getattr(stmt.inserted, c) for c in (
            "title", "alt_title", "type", "year", "score", "episodes",
            "image_url", "mal_url", "n_interactions", "is_online")}
        stmt = stmt.on_duplicate_key_update(**update_cols) if is_mysql \
            else stmt.on_conflict_do_update(index_elements=["src_anime_id"],
                                            set_=update_cols)
        with session_scope() as s:
            s.execute(stmt)
        total += len(rows)
        _log(f"  已写入 {total}/{len(df)}")

    # ---- 题材关联：先清后建，避免重复（anime_genre 无业务字段，重建最干净）
    with session_scope() as s:
        src_to_pk = {int(src): int(pk) for pk, src in s.execute(
            select(Anime.id, Anime.src_anime_id)).all()}
        genre_ids = {int(g) for (g,) in s.execute(select(Genre.id)).all()}
    _log(f"库内番剧 {len(src_to_pk)} 部 / 题材 {len(genre_ids)} 类")

    links: list[dict] = []
    for r in df.itertuples(index=False):
        pk = src_to_pk.get(int(r.anime_id))
        if pk is None:
            continue
        # ⚠️ parquet 里的 list 列读回来是 **numpy.ndarray**，
        # `isinstance(x, (list, tuple))` 会判 False —— 初版就因此静默产出 0 行关联。
        raw_g = getattr(r, "genres_cn_ids", None)
        if isinstance(raw_g, (str, bytes)) or raw_g is None:
            continue
        try:
            gids = [int(g) for g in list(raw_g)]
        except (TypeError, ValueError):
            continue
        prim_raw = getattr(r, "primary_genre_id", None)
        prim = (int(prim_raw) if prim_raw is not None
                and not pd.isna(prim_raw) else None)
        for g in gids:
            if g not in genre_ids:
                continue
            links.append({"anime_id": pk, "genre_id": g,
                          "is_primary": 1 if g == prim else 0})
    if links:
        with session_scope() as s:
            s.execute(delete(AnimeGenre))
            for i in range(0, len(links), batch):
                s.execute(insert(AnimeGenre), links[i:i + batch])
    _log(f"题材关联 {len(links)} 行")

    # ---- 冷启动池（`cold_start_pool`）
    # ⚠️ 本数据集 **没有** 交互数 <10 的番（`n_positive` 最小值恰好 = 10 =
    # COLD_START_THRESHOLD），所以 `list_anime(cold_only=True)` 恒为空。
    # 这是项目已知结论（H4：主协议无真冷启动物品）。
    # 演示要跑通「新番」链路，只能走 A3 的 ① 号来源 —— `cold_start_pool` 表，
    # 它本来就是 A8 维护的权威口径（见 agents/coldstart/adapter.py）。
    # 这里把**交互数最少的一批**（长尾）登记为新番池，A3 会给它们打
    # `is_cold_start=True`，A4 会沿用该标记（fusion/agent.py:133）。
    n_cold = _import_cold_start_pool(df, src_to_pk, limit=60)

    with session_scope() as s:
        n_content = s.scalar(select(func.count()).select_from(AnimeContent)) or 0
    if not n_content:
        _log("提示：anime_content 为空（A3 走 content_vec_512.npy 直读，不依赖该表）")

    return total


def _import_cold_start_pool(df, src_to_pk: dict[int, int], *,
                            limit: int = 60, batch: int = 500) -> int:
    """把交互数最少的一批番登记进 `cold_start_pool`（A3 新番候选的 ① 号来源）。"""
    from sqlalchemy import delete, insert
    from sqlalchemy.dialects.mysql import insert as mysql_insert
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    from server.core.config import settings
    from server.db.models.recommend import ColdStartPool
    from server.db.session import session_scope

    sub = df.nsmallest(int(limit), "n_positive")
    rows: list[dict] = []
    for r in sub.itertuples(index=False):
        pk = src_to_pk.get(int(r.anime_id))
        if pk is None:
            continue
        try:
            year = int(r.year)
        except (TypeError, ValueError):
            year = 0
        # `season` 在设计里是 `2026Q3` 这种季度键，但数据源只有年份、没有月份
        # （`anime_meta.parquet` 不带季度）。**不编造季度**，直接用年份字符串：
        # 它在后台看板里只是个分组键，语义清晰优于虚假精度。
        season = str(year) if year > 0 else "unknown"
        rows.append({"anime_id": pk, "season": season[:6],
                     "n_interactions": int(r.n_positive),
                     "content_vec_ready": 1, "exposure_cnt": 0,
                     "click_cnt": 0, "fav_cnt": 0, "is_active": 1})
    if not rows:
        _log("冷启动池：无可登记条目")
        return 0

    table = ColdStartPool.__table__
    is_mysql = not settings.is_sqlite
    with session_scope() as s:
        s.execute(delete(table))
        for i in range(0, len(rows), batch):
            stmt = (mysql_insert(table) if is_mysql
                    else sqlite_insert(table)).values(rows[i:i + batch])
            if is_mysql:
                stmt = stmt.on_duplicate_key_update(
                    season=stmt.inserted.season,
                    n_interactions=stmt.inserted.n_interactions,
                    is_active=stmt.inserted.is_active)
            else:
                stmt = stmt.on_conflict_do_update(
                    index_elements=["anime_id"],
                    set_={"season": stmt.excluded.season,
                          "n_interactions": stmt.excluded.n_interactions,
                          "is_active": stmt.excluded.is_active})
            s.execute(stmt)
    _log(f"冷启动池：登记 {len(rows)} 部（交互数最少的一批，交互数区间 "
         f"{rows[0]['n_interactions']}–{rows[-1]['n_interactions']}）")
    _log("  ⚠️ 本数据集无 n_interactions<10 的番，'新番'由本表定义而非交互阈值 —— "
         "与 A3 adapter 的注释口径一致（H4：主协议无真冷启动物品）")
    return len(rows)


# ---------------------------------------------------------------- 演示用户


def _import_demo_users(with_records: bool = True) -> dict:
    """造演示用户；`with_records=True` 时用真实序列铺追番记录。"""
    import random

    from sqlalchemy import func, select

    from server.core.security import hash_password
    from server.db.models.record import WatchRecord
    from server.db.models.user import User
    from server.db.session import session_scope

    if not os.path.exists(DEFAULT_SEQ):
        _log(f"⚠️  找不到 {DEFAULT_SEQ}，演示用户不铺追番记录")

    seq = None
    index_to_src: dict[int, int] = {}
    need_records = with_records and os.path.exists(DEFAULT_SEQ)
    if need_records:
        with open(DEFAULT_SEQ, "rb") as f:
            seq = pickle.load(f)
        with open(DEFAULT_INDEX, "r", encoding="utf-8") as f:
            index_to_src = {int(k): int(v)
                            for k, v in (json.load(f).get("index_to_src") or {}).items()}
        _log(f"序列数据就绪：{len(seq.get('train') or {})} 个用户")

    pwd_hash = hash_password(DEMO_PASSWORD)
    created: dict[str, int] = {}
    today = date.today()

    from server.db.models.anime import Anime
    with session_scope() as s:
        src_to_pk = {int(src): int(pk) for pk, src in s.execute(
            select(Anime.id, Anime.src_anime_id)).all()}

    for uid, username, nickname, note in DEMO_USERS:
        with session_scope() as s:
            u = s.scalars(select(User).where(User.username == username)).first()
            if u is None:
                u = User(username=username, password_hash=pwd_hash,
                         nickname=nickname, role=0, status=1,
                         email=f"{username}@example.com")
                s.add(u)
                s.flush()
            created[username] = int(u.id)
            user_pk = int(u.id)

        if not need_records or seq is None:
            continue

        seq_all: list[int] = list(seq["train"].get(uid) or [])
        # 冷启动用户只留 3 条；其余保留最近 60 条
        keep = 3 if username == "newbie" else 60
        idxs = seq_all[-keep:]
        rng = random.Random(1000 + uid)
        rows: list[dict] = []
        for pos, idx in enumerate(idxs):
            src = index_to_src.get(int(idx))
            pk = src_to_pk.get(int(src)) if src is not None else None
            if pk is None:
                continue
            # 最近的几条设为"在看"，倒数第 3 条设为"弃番"，其余"已看"
            if pos >= len(idxs) - 2:
                status = 1
            elif pos == max(0, len(idxs) - 3):
                status = 3
            else:
                status = 2
            rating = rng.choice([7, 8, 8, 9, 9, 10]) if status == 2 else None
            watched = today - timedelta(days=(len(idxs) - pos) * 12)
            rows.append({"user_id": user_pk, "anime_id": pk, "status": status,
                         "rating": rating,
                         "progress": (rng.randint(1, 24) if status == 1 else None),
                         "watched_at": watched.isoformat() if status == 2 else None,
                         "updated_at": watched.isoformat(),
                         "tags": None})

        with session_scope() as s:
            for r in rows:
                wr = s.scalars(select(WatchRecord).where(
                    WatchRecord.user_id == r["user_id"],
                    WatchRecord.anime_id == r["anime_id"])).first()
                if wr is None:
                    wr = WatchRecord(user_id=r["user_id"], anime_id=r["anime_id"])
                    s.add(wr)
                wr.status = r["status"]
                wr.rating = r["rating"]
                wr.progress = r["progress"] or 0
                # ⚠️ 必须显式写 updated_at：A2 的输入序列按 updated_at 升序构造，
                # 若交给 onupdate 默认值，所有记录的时间会挤成"现在"，排序失效。
                if r["updated_at"]:
                    wr.updated_at = datetime.fromisoformat(r["updated_at"])
                if r["watched_at"]:
                    wr.watched_at = date.fromisoformat(r["watched_at"])
        _log(f"  用户 {username}(id={user_pk}) 追番 {len(rows)} 条 —— {note}")

    # ⚠️ 故意**不**在这里用 watch_record 重算 `anime.n_interactions`。
    # 线上该字段确实由追番记录派生（gateway._refresh_anime_popularity），
    # 但演示库里只有 5 个用户，重算会把每部番的交互数压到 ≤5，
    # 全部低于 COLD_START_THRESHOLD=10 → A3/A4 的冷启动分支集体走错。
    # 演示库的 n_interactions 以 item_stats.parquet 的 n_positive 为准（见 _import_anime）。

    # ---- 管理员账号（role=1），让 /admin/* 的 14 个接口能真的点开
    admin_name, admin_nick = DEMO_ADMIN
    with session_scope() as s:
        u = s.scalars(select(User).where(User.username == admin_name)).first()
        if u is None:
            u = User(username=admin_name, password_hash=pwd_hash,
                     nickname=admin_nick, role=1, status=1,
                     email=f"{admin_name}@example.com")
            s.add(u)
            s.flush()
        else:
            u.role = 1                     # 幂等：既有账号也提权
        created[admin_name] = int(u.id)
    _log(f"  管理员 {admin_name}(id={created[admin_name]}) role=1 —— {admin_nick}")

    return created


def _warm_profiles(user_ids: dict) -> None:
    """给演示用户预跑一次 A1 画像，让首页/雷达图首次打开就有内容。"""
    import asyncio

    from agents.common.data import bind_gateway
    from agents.common.envelope import Envelope
    from agents.common.registry import AgentRegistry
    from server.db.gateway import SqlGateway
    from agents.bootstrap import register_all

    register_all()
    bind_gateway(SqlGateway())
    a1 = AgentRegistry.get("A1")

    async def _run():
        for name, pk in user_ids.items():
            env = Envelope.request(from_agent="INIT", to_agent="A1",
                                   action="profile.get",
                                   payload={"user_id": pk, "scope": "both"},
                                   trace_id="tr_1a17e0000001", timeout_ms=8000)
            out = await a1.handle(env)
            tag = (out.payload or {}).get("user_tag") if not out.is_error else out.payload
            _log(f"  画像 {name}(id={pk}) → {tag}")

    asyncio.run(_run())


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description="初始化 AniRec 数据库")
    ap.add_argument("--url", default=None, help="覆盖 DATABASE_URL")
    ap.add_argument("--reset", action="store_true", help="先删表再建（开发环境）")
    ap.add_argument("--no-anime", action="store_true", help="不导入番剧")
    ap.add_argument("--no-demo", action="store_true", help="不造演示用户")
    ap.add_argument("--no-seed", action="store_true", help="不跑 A1 预热")
    ap.add_argument("--no-extras", action="store_true", help="跳过 MySQL 分区/全文索引")
    args = ap.parse_args()

    if args.url:
        os.environ["DATABASE_URL"] = args.url

    from server.core.logging import setup_logging
    setup_logging()

    from server.core.config import settings
    from server.db.session import get_engine

    _log(f"目标库：{settings.database_url.split('@')[-1]}")
    _create_schema(args.reset, apply_extras=not args.no_extras)

    if not args.no_anime:
        n = _import_anime()
        _log(f"番剧导入完成：{n} 部")

    users: dict[str, int] = {}
    if not args.no_demo:
        users = _import_demo_users()
        _log(f"演示用户：{users}（密码统一 {DEMO_PASSWORD}）")
        if users and not args.no_seed:
            _log("预热 A1 画像 …")
            try:
                _warm_profiles(users)
            except Exception as exc:            # 预热失败不影响库可用
                _log(f"⚠️  画像预热失败（不阻塞）：{type(exc).__name__}: {exc}")

    with get_engine().connect() as conn:
        from sqlalchemy import func, select

        from server.db.models.anime import Anime, Genre
        from server.db.models.record import WatchRecord
        from server.db.models.user import User
        stats = {
            "users": conn.execute(select(func.count()).select_from(User)).scalar(),
            "animes": conn.execute(select(func.count()).select_from(Anime)).scalar(),
            "genres": conn.execute(select(func.count()).select_from(Genre)).scalar(),
            "records": conn.execute(select(func.count()).select_from(WatchRecord)).scalar(),
        }
    _log(f"✅ 完成：{stats}")
    _log("启动服务：python -m uvicorn server.main:app --reload --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
