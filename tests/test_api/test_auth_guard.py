# -*- coding: utf-8 -*-
"""鉴权不变量的**全端点**审计。

`server/deps.py` 的 docstring 把要害说得很清楚：同一份「取当前用户」的逻辑
被 20+ 个 handler 用到，**总有一条路由忘了校验 token 的风险不会报错、
只会静默放行**。逐个接口写"无 token 应该 401"的用例，写漏的那条恰好就是
出事的那条 —— 所以这里仍然走全端点探针。

探针怎么判定（不依赖 FastAPI 内部结构）
------------------------------------
* **无 token** 返回 `40101` ⇔ 这条路由要求登录。依赖在 handler 之前解析，
  所以 `current_user` 的 401 永远先于任何业务错误，判定不会歧义。
* **普通用户 token** 返回 `40301` ⇔ 这条路由要求管理员。
* 其余 ⇔ 公开（注意"公开"只表示**不需要用户登录**，不代表没有别的守卫，
  比如 `/internal/*` 有独立的内网 token 校验）。

为什么不遍历 `app.routes` 找 `dependant`
------------------------------------
fastapi 0.141.1 起 `include_router` 不再摊平子路由（包成 `_IncludedRouter`，
完整路径藏在 `original_router` 这种私有字段里）。端点清单改从
`app.openapi()["paths"]` 拿 —— 公开稳定接口，且与路由表同源。
见 `test_envelope.py::api_routes` 的详细记录（那里真实踩过收出 0 条的坑）。

回归记录（2026-09-14）
--------------------
首轮探针跑出来的不是鉴权问题，而是一个 **500**：
`GET /records/timeline` 一直在给 `user_genre_time_series()` 传不存在的
`limit` 参数，未捕获 TypeError 直接炸。冒烟只测了 feed/chat/admin 所以没发现。
由此多出一条全局断言：**任何端点在任何探测下都不得返回 5xx** ——
统一异常处理器会把未捕获异常包成合法信封（`code=50001`），
所以"信封完整"类测试对 500 是**盲**的，必须显式断状态码。
"""

from __future__ import annotations

import re

import pytest

_METHODS = ("get", "post", "put", "patch", "delete")

#: 测试态跳过的两条（SSE 不透传信封 / deep 会真的加载模型），与
#: `test_envelope.py::_AUDIT_SKIP` 保持同一份语义。
_AUDIT_SKIP = {("POST", "/api/v1/chat/message"), ("GET", "/api/v1/health/deep")}

#: ── 公开端点白名单（**显式固化**）────────────────────────────────────
#: 新增公开接口必须来这里登记一行 —— 这个摩擦是刻意的：
#: "忘了校验 token" 是静默放行，只有让名单变更过人眼才拦得住。
#: 注：列表里 40401/40001 只是探测请求恰好缺参数/缺数据，
#: 真正的判定是"无 token 时 code != 40101"。
PUBLIC_ENDPOINTS = frozenset({
    ("GET", "/api/v1/animes"),                  # 匿名可浏览
    ("GET", "/api/v1/animes/{anime_id}"),
    ("GET", "/api/v1/animes/{anime_id}/similar"),
    ("GET", "/api/v1/animes/{anime_id}/reviews"),  # 评价列表：匿名可浏览（与详情/相似同级）
    ("GET", "/api/v1/genres"),
    ("POST", "/api/v1/auth/login"),
    ("POST", "/api/v1/auth/register"),
    ("GET", "/api/v1/auth/captcha"),             # 登录验证码（未登录就要能取）
    ("GET", "/api/v1/health"),
    ("POST", "/api/v1/internal/agents/{name}"),  # 有独立内网 token 守卫（40301）
    ("GET", "/api/v1/internal/health"),
})

#: ── 仅管理员端点白名单 ─────────────────────────────────────────────
ADMIN_ENDPOINTS = frozenset({
    ("GET", "/api/v1/admin/agents/health"),
    ("GET", "/api/v1/admin/animes"),
    ("POST", "/api/v1/admin/animes"),
    ("PUT", "/api/v1/admin/animes/{anime_id}"),
    ("PUT", "/api/v1/admin/animes/{anime_id}/offline"),
    ("GET", "/api/v1/admin/cold-start"),
    ("GET", "/api/v1/admin/metrics"),
    ("POST", "/api/v1/admin/tasks/offline-recommend"),
    ("GET", "/api/v1/admin/traces/{trace_id}"),
    ("GET", "/api/v1/admin/users"),
})

#: 有**独立守卫**的公开端点：不吃用户 JWT，任何没有自己凭证的请求
#: （包括合法登录用户）都该被拒 —— 所以普通用户探测拿到 40301 是正确行为。
_SELF_GUARDED = frozenset({
    ("POST", "/api/v1/internal/agents/{name}"),   # 内网 token（40301）
})

_MIN_ENDPOINTS = 40


def _fill(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "1", path)


def all_endpoints() -> list[tuple[str, str]]:
    from server.main import app

    out: list[tuple[str, str]] = []
    for path, item in app.openapi()["paths"].items():
        if not path.startswith("/api/v1"):
            continue
        for method in _METHODS:
            if method in item and (method.upper(), path) not in _AUDIT_SKIP:
                out.append((method.upper(), path))
    return sorted(out)


ENDPOINTS = all_endpoints()


class TestWhitelistConsistency:
    """白名单必须与现实对得上，且必须覆盖全部端点。"""

    def test_covers_enough_endpoints(self):
        assert len(ENDPOINTS) >= _MIN_ENDPOINTS, (
            f"只收集到 {len(ENDPOINTS)} 个端点 —— openapi 枚举可能已失效，"
            f"此时本文件会假装全绿")

    def test_whitelists_are_disjoint(self):
        overlap = PUBLIC_ENDPOINTS & ADMIN_ENDPOINTS
        assert not overlap, f"同时出现在两个名单里：{sorted(overlap)}"

    def test_whitelists_only_name_real_endpoints(self):
        """名单里出现已删除/改名的端点时，测试必须提醒维护者。"""
        known = set(ENDPOINTS) | _AUDIT_SKIP
        stale = (PUBLIC_ENDPOINTS | ADMIN_ENDPOINTS) - known
        assert not stale, f"名单里的端点已不存在：{sorted(stale)}"


@pytest.mark.parametrize(("method", "path"), ENDPOINTS,
                         ids=[f"{m} {p}" for m, p in ENDPOINTS])
class TestAuthAudit:
    """每个端点三类身份各探一次，分类结果必须与白名单一致。"""

    def test_anonymous(self, client, method, path):
        """无 token：公开端点不能 401；其余必须 401 + 40101。"""
        resp = client.request(method, _fill(path))
        assert resp.status_code < 500, \
            f"{method} {path}：无 token 探测出现 5xx（{resp.status_code}），" \
            f"这是未捕获异常被兜底处理器吞掉的表现"
        body = resp.json()
        if (method, path) in PUBLIC_ENDPOINTS:
            assert body["code"] != 40101, \
                f"{method} {path} 已登记为公开，却开始要求登录 —— " \
                f"是接口变了还是白名单该更新？"
        else:
            assert resp.status_code == 401, \
                f"{method} {path} 未登记为公开，却没拦下匿名请求" \
                f"（http={resp.status_code} code={body['code']}）—— " \
                f"静默放行！请补 Depends(current_user) 或登记白名单"
            assert body["code"] == 40101

    def test_normal_user(self, client, auth, method, path):
        """普通用户 token：管理员端点必须 403；其余不能是鉴权错误。"""
        resp = client.request(method, _fill(path), headers=auth)
        assert resp.status_code < 500, \
            f"{method} {path}：普通用户探测出现 5xx（{resp.status_code}）"
        body = resp.json()
        if (method, path) in _SELF_GUARDED:
            # 合法用户 JWT 不等于有内网凭证 —— 必须仍然拒绝
            assert resp.status_code == 403 and body["code"] == 40301, \
                f"{method} {path} 有独立守卫，却放行了普通用户 " \
                f"(http={resp.status_code} code={body['code']})"
        elif (method, path) in ADMIN_ENDPOINTS:
            assert resp.status_code == 403 and body["code"] == 40301, \
                f"{method} {path} 登记为仅管理员，普通用户却拿到了 " \
                f"http={resp.status_code} code={body['code']}"
        else:
            assert body["code"] not in (40101, 40301), \
                f"{method} {path} 不在管理员名单里，却对普通用户返回鉴权错误"


@pytest.mark.parametrize(("method", "path"), sorted(ADMIN_ENDPOINTS),
                         ids=[f"{m} {p}" for m, p in sorted(ADMIN_ENDPOINTS)])
def test_admin_passes_auth_gate(client, admin_auth, method, path):
    """管理员 token：至少要过了鉴权门（不能 401/403），业务错误不算失败。"""
    resp = client.request(method, _fill(path), headers=admin_auth)
    assert resp.status_code < 500
    body = resp.json()
    assert body["code"] not in (40101, 40301), \
        f"{method} {path}：管理员也被 401/403 拦截（code={body['code']}）"
