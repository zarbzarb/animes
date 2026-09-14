# -*- coding: utf-8 -*-
"""统一响应体（`docs/api-specification.md` §1.1）的**全路由**审计。

为什么是"全路由"而不是挑几个接口测
----------------------------------
协议要求 **所有**接口返回 `{code, message, data, meta, trace_id, elapsed_ms}`。
历史上真正出过的错都不是"某个 handler 写错"，而是**某条路由没被包上**：

* 聚合路由 `/api/v1/health` 直接装饰在 `api_v1_router` 上，而聚合路由默认
  `route_class` 是 FastAPI 原版 —— 不显式换掉的话，这一条会退回裸 dict，
  响应里没有 `trace_id`/`elapsed_ms`，**其余 40 条都正常**。
* handler 标注 `-> Enveloped` 会被 FastAPI 当成 `response_model`，Pydantic 按
  字段裁剪，响应恰好剩 `{code, data, message, meta}` —— 也不报错。

这两类错误单测挑接口是抓不住的（挑到的那条恰好是对的）。所以这里**遍历
`app.routes`**，一条不漏地验。

`_AUDIT_SKIP` 之外的每一条 `/{path}` 都会被真实请求一次。
"""

from __future__ import annotations

import re

import pytest

#: 信封的六个字段，一个不多一个不少。
ENVELOPE_KEYS = {"code", "message", "data", "meta", "trace_id", "elapsed_ms"}

TRACE_RE = re.compile(r"^tr_[0-9a-f]{12}$")

#: 审计时跳过：这些路由的**正确**行为就不是 JSON 信封。
#: * `/api/v1/chat/message` —— SSE 流式，`EnvelopeRoute` 刻意原样透传
#:   （否则流式响应会被整体读进内存）。它的真实行为由 `scripts/smoke_api.py`
#:   在真服务上覆盖（含"有回复文本"与"cards"断言）。
#: * `/api/v1/health/deep` —— 深度连通性自检，会真的去注册召回模型、
#:   跑一遍 A1。测试态没有运行时，它必然走异常分支；本文件只关心信封形状，
#:   不值得为它付出加载模型的秒级开销。
_AUDIT_SKIP = {("POST", "/api/v1/chat/message"), ("GET", "/api/v1/health/deep")}

#: 审计到的路由条数下限。**这是测试自身的有效性守卫**：
#: 一旦路由收集逻辑写错（比如漏了 `include_router` 的子路由、
#: 或 `app.routes` 结构变了），参数化会收出空列表 —— 那时若没有这条断言，
#: 整个文件会"0 条用例，全绿"，比不写测试更糟。
_MIN_ROUTES = 40


def _fill(path: str) -> str:
    """把 `/animes/{anime_id}` 之类的路径参数填成 1（不关心取值，只要路由能匹配）。"""
    return re.sub(r"\{[^}]+\}", "1", path)


_METHODS = ("get", "post", "put", "patch", "delete")


def api_routes() -> list[tuple[str, str]]:
    """真应用上所有 `/api/v1` 端点，(METHOD, path) 列表。

    ⚠️ **不能遍历 `app.routes` 找子路由** —— 本项目用的 FastAPI 0.141.1 里，
    `include_router()` 不再把子路由摊平进 `app.routes`，而是包成一个
    `_IncludedRouter` 对象（它在 `app.routes` 里**没有 `path` 属性**，
    子路由藏在 `original_router` 这个内部字段里）。

    实测后果（2026-09-14）：`[r for r in app.routes if r.path.startswith("/api/v1")]`
    返回 **0 条** —— 于是参数化收出空列表，整个审计文件"0 条用例、全绿"。
    项目里 44 个端点一个都没被验，而且没有任何报错。

    所以这里改用 `app.openapi()["paths"]`：它是公开稳定接口，
    由同一张路由表生成，**跨 FastAPI 版本都不会变**，也不依赖任何内部字段。
    """
    from server.main import app

    out: list[tuple[str, str]] = []
    for path, item in app.openapi()["paths"].items():
        if not path.startswith("/api/v1"):
            continue
        for method in _METHODS:
            if method in item and (method.upper(), path) not in _AUDIT_SKIP:
                out.append((method.upper(), path))
    return sorted(out)


ROUTES = api_routes()
ROUTE_IDS = [f"{m} {p}" for m, p in ROUTES]


def _assert_envelope(resp, *, where: str) -> dict:
    """一份响应体必须是合法信封。返回 body 方便继续断言。"""
    body = resp.json()
    assert isinstance(body, dict), f"{where}：响应体不是 JSON 对象，而是 {type(body).__name__}"
    assert set(body) == ENVELOPE_KEYS, (
        f"{where}：信封字段不对。多={sorted(set(body) - ENVELOPE_KEYS)} "
        f"少={sorted(ENVELOPE_KEYS - set(body))}")
    assert isinstance(body["code"], int), f"{where}：code 不是 int"
    assert isinstance(body["message"], str) and body["message"], f"{where}：message 为空"
    assert isinstance(body["meta"], dict), f"{where}：meta 不是对象"
    assert TRACE_RE.match(body["trace_id"] or ""), \
        f"{where}：trace_id 不合规（应为 tr_+12 位 hex），实际 {body['trace_id']!r}"
    assert isinstance(body["elapsed_ms"], int) and body["elapsed_ms"] >= 0, \
        f"{where}：elapsed_ms 不是非负整数"
    return body


class TestRouteCount:
    """测试有效性守卫（见 `_MIN_ROUTES` 注释）。"""

    def test_audit_covers_enough_routes(self):
        assert len(ROUTES) >= _MIN_ROUTES, (
            f"只审计到 {len(ROUTES)} 条路由（下限 {_MIN_ROUTES}）—— "
            f"路由收集逻辑可能已经失效，此时全文件会假装全绿")

    def test_skip_list_entries_still_exist(self):
        """跳过的路由必须**真实存在**：否则改名后跳过名单会变成永久的静默黑洞。"""
        from server.main import app

        known = {(m.upper(), p) for p, item in app.openapi()["paths"].items()
                 for m in _METHODS if m in item}
        missing = _AUDIT_SKIP - known
        assert not missing, f"跳过名单里的这些路由已不存在：{sorted(missing)}"


@pytest.mark.parametrize(("method", "path"), ROUTES, ids=ROUTE_IDS)
def test_every_route_returns_envelope(client, method, path):
    """无凭据请求每条路由：**无论成败**响应体都必须是完整信封。

    这里刻意不断言状态码与 code —— 无凭据时各条路由的预期结果本来就不同
    （401 / 400 / 200）。本用例只锁"信封形状"这一个不变量，
    具体状态码由 `test_auth_guard.py` 与 `test_endpoints_behavior.py` 分别锁。
    """
    resp = client.request(method, _fill(path))
    body = _assert_envelope(resp, where=f"{method} {path}")
    # X-Trace-Id 响应头必须与响应体一致：前端报障时靠它对日志，
    # 两者不一致等于给了一条查不到的线索。
    assert resp.headers.get("X-Trace-Id") == body["trace_id"], \
        f"{method} {path}：响应头 X-Trace-Id 与响应体 trace_id 不一致"


class TestTraceId:
    def test_header_echoed_when_client_provides_one(self, client):
        """客户端传入的 `X-Trace-Id` 必须被沿用（全链路串联的前提）。"""
        tid = "tr_0123456789ab"
        resp = client.get("/api/v1/health", headers={"X-Trace-Id": tid})
        assert resp.json()["trace_id"] == tid
        assert resp.headers["X-Trace-Id"] == tid

    def test_generated_when_absent(self, client):
        """未传时必须自动生成一个合规的，不能是 `-` 或空。"""
        body = client.get("/api/v1/health").json()
        assert TRACE_RE.match(body["trace_id"])

    def test_two_requests_get_different_ids(self, client):
        """trace_id 不能是进程级常量（那样所有请求在日志里混成一条）。"""
        a = client.get("/api/v1/health").json()["trace_id"]
        b = client.get("/api/v1/health").json()["trace_id"]
        assert a != b


class TestErrorPathsAlsoEnveloped:
    """错误路径最容易漏包 —— FastAPI 默认会返回 `{"detail": ...}`。"""

    def test_unknown_path_404(self, client):
        resp = client.get("/api/v1/definitely-not-a-route")
        assert resp.status_code == 404
        body = _assert_envelope(resp, where="404 未匹配路径")
        assert body["code"] == 40401
        assert resp.headers.get("X-Trace-Id") == body["trace_id"]

    def test_method_not_allowed(self, client):
        """405 也要是信封（`DELETE /api/v1/health` 不存在）。"""
        resp = client.delete("/api/v1/health")
        assert resp.status_code == 405
        _assert_envelope(resp, where="405 方法不允许")

    def test_validation_error_maps_to_40001(self, client):
        """校验失败 -> HTTP 400 + `code=40001`（协议 §6 的映射，不是 422）。"""
        resp = client.post("/api/v1/auth/login", json={"username": "demo"})
        assert resp.status_code == 400
        body = _assert_envelope(resp, where="422 校验失败")
        assert body["code"] == 40001
        assert body["meta"].get("error_name") == "INVALID_PARAM"
        assert body["data"] is None

    def test_error_message_names_the_offending_field(self, client):
        """报错必须点名是哪个字段 —— 否则前端只能把整个表单标红。"""
        resp = client.post("/api/v1/auth/login", json={"username": "demo"})
        assert "password" in resp.json()["message"]
