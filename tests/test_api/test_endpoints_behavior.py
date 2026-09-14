# -*- coding: utf-8 -*-
"""关键接口的行为回归（在真应用 + SQLite 内存库上，见 conftest.py）。

范围界定
--------
`test_envelope.py` 锁信封形状、`test_auth_guard.py` 锁鉴权矩阵，
本文件锁**业务行为**：参数校验、状态码、数据形状、以及几条曾经坏过的路径。

回归记录（2026-09-14）
--------------------
* `GET /records/timeline`：给网关传了不存在的 `limit` 参数，
  未捕获 TypeError → 500（冒烟只测 feed/chat/admin，没覆盖到）。
  修复：网关 `user_genre_time_series()` 增加 `limit`。
  本文件的 `TestRecordsTimeline` 锁住它 —— 注意断言的是 **HTTP 200 + code=0**，
  不能只看"信封完整"：统一异常处理器会把 500 包成合法信封（code=50001），
  光看信封对 500 是盲的。
* `POST /auth/login`：用户名不存在与密码错误必须返回**同一个** 40101，
  否则接口变成账号枚举器（auth.py 有此注释，这里落成断言）。
"""

from __future__ import annotations

from server.core.security import hash_password, verify_password


class TestHealth:
    def test_reports_component_status(self, client):
        body = client.get("/api/v1/health").json()
        assert body["code"] == 0
        d = body["data"]
        assert d["db"] == "up"          # sqlite 内存库是活的
        assert d["cache"] == "memory"   # conftest 强制走内存缓存
        assert d["status"] == "ok"
        assert isinstance(d["degraded"], list)
        # 测试态不跑 lifespan，Agent 运行时未装配 —— 如实报告，而不是谎报 ready
        assert d["agents_ready"] is False


class TestGenres:
    def test_lists_12_genres(self, client):
        body = client.get("/api/v1/genres").json()
        assert body["code"] == 0
        d = body["data"]
        assert d["n"] == 12
        assert {int(g["genre_id"]) for g in d["list"]} == set(range(1, 13))


class TestAnimes:
    def test_list_pagination_shape(self, client, seeded):
        body = client.get("/api/v1/animes", params={"page": 1, "size": 2}).json()
        assert body["code"] == 0
        d = body["data"]
        assert d["pagination"]["total"] == 3
        assert d["pagination"]["pages"] == 2          # 3 条 / 每页 2
        assert len(d["list"]) == 2

    def test_list_second_page_has_the_rest(self, client, seeded):
        body = client.get("/api/v1/animes", params={"page": 2, "size": 2}).json()
        assert len(body["data"]["list"]) == 1

    def test_genre_filter_shrinks_total(self, client, seeded):
        body = client.get("/api/v1/animes", params={"genre_id": 1}).json()
        titles = [a["title"] for a in body["data"]["list"]]
        assert body["data"]["pagination"]["total"] == 1
        assert titles == ["测试番·热血机战"]

    def test_detail_with_genres(self, client, seeded):
        body = client.get("/api/v1/animes/9001").json()
        assert body["code"] == 0
        d = body["data"]
        assert d["src_anime_id"] == 9001
        assert d["is_cold_start"] is False
        # 详情的 genres 形状：[{genre_id, genre}]（不是 list_anime 的并串）
        assert {g["genre_id"] for g in d["genres"]} == {1, 6}
        assert {"热血战斗", "科幻机战"} <= {g["genre"] for g in d["genres"]}

    def test_detail_unknown_is_40401(self, client):
        resp = client.get("/api/v1/animes/999999")
        assert resp.status_code == 404
        assert resp.json()["code"] == 40401

    def test_similar_unknown_is_40401(self, client):
        resp = client.get("/api/v1/animes/999999/similar")
        assert resp.status_code == 404


class TestRecordsCrud:
    def test_full_lifecycle(self, client, auth):
        # 新增
        r = client.post("/api/v1/records", headers=auth,
                        json={"anime_id": 9001, "status": 1, "rating": 8}).json()
        assert r["code"] == 0 and r["data"]["is_new"] is True
        rid = r["data"]["record_id"]
        assert r["data"]["recompute_scheduled"] is True

        # 列表带番剧装饰
        lst = client.get("/api/v1/records", headers=auth).json()["data"]
        assert lst["pagination"]["total"] == 1
        item = lst["list"][0]
        assert item["id"] == rid
        assert item["anime"]["src_anime_id"] == 9001
        assert item["anime"]["title"] == "测试番·热血机战"
        assert item["status_label"] == "在看"
        assert item["rating"] == 8

        # 修改
        p = client.put(f"/api/v1/records/{rid}", headers=auth,
                       json={"status": 2, "rating": 9}).json()
        assert p["code"] == 0 and p["data"]["is_new"] is False

        # 状态筛选能对上
        lst2 = client.get("/api/v1/records", headers=auth,
                          params={"status": 2}).json()["data"]
        assert lst2["pagination"]["total"] == 1
        assert lst2["list"][0]["status_label"] == "已看"

        # 删除后二次删除 404
        assert client.delete(f"/api/v1/records/{rid}", headers=auth).json()["code"] == 0
        resp = client.delete(f"/api/v1/records/{rid}", headers=auth)
        assert resp.status_code == 404 and resp.json()["code"] == 40401

    def test_unknown_anime_rejected(self, client, auth):
        resp = client.post("/api/v1/records", headers=auth,
                           json={"anime_id": 999999, "status": 1})
        assert resp.status_code == 404
        assert resp.json()["code"] == 40401

    def test_out_of_range_status_rejected(self, client, auth):
        resp = client.post("/api/v1/records", headers=auth,
                           json={"anime_id": 9001, "status": 9})
        assert resp.status_code == 400 and resp.json()["code"] == 40001


class TestRecordsTimeline:
    """回归：timeline 曾因给网关传不存在的 `limit` 而 500（见模块 docstring）。

    ⚠️ `bucket.count` 的语义是 **Σ(每个题材下的去重番剧数)**，不是去重番剧数：
    一部番挂在 N 个题材下就贡献 N。这是 A6 时间序列（按题材分行的
    `user_genre_time_series`）在展示层的直加和 —— 测试按此口径断言，
    改语义要连同 A6 一起改。
    """

    def test_returns_200_with_buckets(self, client, auth):
        made = client.post("/api/v1/records", headers=auth,
                           json={"anime_id": 9001, "status": 1}).json()
        assert made["code"] == 0                       # 修复前：这一步就已经 500

        body = client.get("/api/v1/records/timeline", headers=auth,
                          params={"granularity": "month", "limit": 12})
        assert body.status_code == 200, \
            f"timeline 返回 {body.status_code} —— limit 参数回归？"
        assert body.json()["code"] == 0                # 500 会被包成 50001，必须看 code
        d = body.json()["data"]
        assert d["granularity"] == "month"
        assert isinstance(d["timeline"], list) and len(d["timeline"]) == 1
        b = d["timeline"][0]
        # 9001 挂在题材 1(热血战斗) 与 6(科幻机战) 下 → 计数 2
        assert b["count"] == 2
        assert b["genres"] == [{"genre": "热血战斗", "count": 1},
                               {"genre": "科幻机战", "count": 1}]

    def test_limit_caps_the_number_of_periods(self, client, auth):
        """limit 截断的是**周期数**，且同一周期内的题材计数不能被截掉。"""
        for aid in (9001, 9002, 9003):
            assert client.post("/api/v1/records", headers=auth,
                               json={"anime_id": aid, "status": 2}).json()["code"] == 0
        d = client.get("/api/v1/records/timeline", headers=auth,
                       params={"granularity": "month", "limit": 1}).json()["data"]
        assert len(d["timeline"]) == 1                 # 全部落在同一个周期
        # 9001 贡献 2（两个题材）+ 9002/9003 各 1 = 4；limit 只裁周期，不裁题材
        assert d["timeline"][0]["count"] == 4
        assert {g["genre"] for g in d["timeline"][0]["genres"]} == {
            "热血战斗", "科幻机战", "日常治愈", "悬疑推理"}


class TestRecordsStats:
    def test_counts_by_status(self, client, auth):
        client.post("/api/v1/records", headers=auth,
                    json={"anime_id": 9001, "status": 1, "rating": 8})
        client.post("/api/v1/records", headers=auth,
                    json={"anime_id": 9002, "status": 3})
        d = client.get("/api/v1/records/stats", headers=auth).json()["data"]
        assert d["total"] == 2
        assert d["by_status"]["在看"] == 1 and d["by_status"]["弃番"] == 1
        assert d["n_rated"] == 1 and d["avg_rating"] == 8.0


class TestAuth:
    def test_register_rejects_weak_password(self, client):
        resp = client.post("/api/v1/auth/register",
                           json={"username": "newbie", "password": "abcdefgh"})
        assert resp.status_code == 400 and resp.json()["code"] == 40001

    def test_register_rejects_bad_username_charset(self, client):
        resp = client.post("/api/v1/auth/register",
                           json={"username": "坏名字", "password": "abc12345"})
        assert resp.status_code == 400 and resp.json()["code"] == 40001

    def test_register_then_login_roundtrip(self, client):
        r = client.post("/api/v1/auth/register",
                        json={"username": "newbie", "password": "abc12345",
                              "nickname": "新人"}).json()
        assert r["code"] == 0 and r["data"]["access_token"]
        # 重复用户名 40901
        dup = client.post("/api/v1/auth/register",
                          json={"username": "newbie", "password": "abc12345"})
        assert dup.status_code == 409 and dup.json()["code"] == 40901
        # 登录成功
        l = client.post("/api/v1/auth/login",
                        json={"username": "newbie", "password": "abc12345"}).json()
        assert l["code"] == 0 and l["data"]["user"]["nickname"] == "新人"

    def test_login_does_not_leak_username_existence(self, client, seeded):
        """用户名不存在与密码错误必须是**同一个**错误 —— 账号枚举防护。"""
        unknown = client.post("/api/v1/auth/login",
                              json={"username": "no_such_user",
                                    "password": "whatever123"}).json()
        wrong_pwd = client.post("/api/v1/auth/login",
                                json={"username": "demo",
                                      "password": "wrong-password"}).json()
        assert unknown["code"] == wrong_pwd["code"] == 40101
        assert unknown["message"] == wrong_pwd["message"]

    def test_password_hash_never_leaves_the_db(self, client, gw, seeded, password,
                                               auth, admin_auth):
        """哈希串只存库里：网关的所有 user dict（含 /users/me、登录返回）
        都不得携带它 —— `_user_dict` 是唯一的出口，漏一次就是全站泄漏。"""
        # 库里存的是加盐哈希，且不是明文
        from server.db.models.user import User
        from server.db.session import session_scope
        with session_scope() as s:
            row = s.query(User).filter(User.username == "demo").one()
            stored = row.password_hash
        assert stored != password and not verify_password(password + "x", stored)
        assert verify_password(password, stored)
        assert hash_password(password) != stored       # 加盐：同明文不同串
        # 网关出口不携带哈希。唯一例外是 `get_user_by_username` —— 它存在的
        # 目的就是把哈希交给 login 做校验（源码注释"仅供 server 鉴权"），
        # 真正的用户侧保证是：**HTTP 响应**里永远不出现它（下一断言）。
        for u in (gw.get_user(int(seeded["user"]["id"])),
                  *gw.list_users(limit=10)):
            assert "password_hash" not in u, "网关 dict 泄漏了哈希串"
        for resp in (client.get("/api/v1/users/me", headers=auth),
                     client.get("/api/v1/admin/users", headers=admin_auth)):
            assert "password_hash" not in resp.text, "HTTP 响应泄漏了哈希串"


class TestUsers:
    def test_me_roundtrip(self, client, auth):
        me = client.get("/api/v1/users/me", headers=auth).json()["data"]
        assert me["username"] == "demo"
        up = client.put("/api/v1/users/me", headers=auth,
                        json={"nickname": "新昵称"}).json()
        assert up["code"] == 0 and up["data"]["nickname"] == "新昵称"

    def test_change_password_invalidates_old(self, client, auth, password):
        resp = client.put("/api/v1/users/me/password", headers=auth,
                          json={"old_password": password,
                                "new_password": "brand-new-888"})
        assert resp.json()["code"] == 0
        old = client.post("/api/v1/auth/login",
                          json={"username": "demo", "password": password})
        assert old.status_code == 401                  # 旧密码立刻失效
        new = client.post("/api/v1/auth/login",
                          json={"username": "demo", "password": "brand-new-888"})
        assert new.json()["code"] == 0

    def test_clear_memory_is_idempotent(self, client, auth):
        for _ in range(2):
            r = client.delete("/api/v1/users/me/memory", headers=auth).json()
            assert r["code"] == 0


class TestAdmin:
    def test_create_and_offline_anime(self, client, admin_auth, auth):
        made = client.post("/api/v1/admin/animes", headers=admin_auth,
                           json={"src_anime_id": 9100, "title": "管理新建",
                                 "genre_ids": [7]}).json()
        assert made["code"] == 0
        src = int(made["data"]["src_anime_id"])
        assert src == 9100
        # 详情/下架路径参数都是**数据集 animeID**（src_anime_id），不是库主键
        assert client.get(f"/api/v1/animes/{src}").json()["code"] == 0

        # 回归（2026-09-14）：下架曾只影响列表/计数/热门四条读路，
        # 详情、相似、require_anime（加追番）三条单查读门照样放行 ——
        # 管理员点"下架"对用户等于没下架，且不报任何错。
        # 修复后：is_visible_anime() 收敛三处判定。
        off = client.put(f"/api/v1/admin/animes/{src}/offline", headers=admin_auth)
        assert off.json()["code"] == 0
        assert client.get(f"/api/v1/animes/{src}").status_code == 404
        assert client.get(f"/api/v1/animes/{src}/similar").status_code == 404
        resp = client.post("/api/v1/records", headers=auth,
                           json={"anime_id": src, "status": 1})
        assert resp.status_code == 404 and resp.json()["code"] == 40401
        # 但 admin 编辑路径仍要能看到下架番（否则无法重新上架/编辑）
        alive = client.put(f"/api/v1/admin/animes/{src}", headers=admin_auth,
                           json={"src_anime_id": src, "title": "改名仍可编辑"})
        assert alive.json()["code"] == 0

    def test_user_list(self, client, admin_auth):
        d = client.get("/api/v1/admin/users", headers=admin_auth).json()["data"]
        assert d["pagination"]["total"] == 2           # demo + root
        assert all("password_hash" not in u for u in d["list"]), \
            "用户列表绝不能把哈希串发给前端"
