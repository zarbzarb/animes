"""SPA 托管与回退语义（server/main.py 的 dist 挂载 + catch-all）。

生产部署形态：`npm run build` 产物由 FastAPI 直接托管 —— `/` 与前端深层
路由（/login 等）回 SPA 入口；`/assets/*` 由 StaticFiles 挂载提供。
**不能破坏的语义**：`/api/v1/*` 未匹配到的路径必须仍是信封 404
（曾有把 API 404 也回退成 index.html 的风险）。

两个踩过的坑（都有测试锁定）：
* catch-all 注册早于 `/assets` mount → mount 被 `/{full_path:path}` 截走，
  静态资源全部 404（实测踩过）；
* dist 未构建时（CI 常态）必须回退演示页 `web/index.html`，仍要 200。
"""

import pytest

pytestmark = pytest.mark.usefixtures("api_engine")


def test_root_serves_html(client):
    """/ 恒为 200 的 HTML：dist 优先，dist 缺席则回退演示页 —— 不允许白屏或 JSON。"""
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_deep_route_falls_back_to_spa(client):
    """前端深层路由刷新（/login 等）必须回 SPA 入口而不是 404。"""
    r = client.get("/login")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_unknown_api_path_is_envelope_404(client):
    """/api/* 未匹配路径必须是信封 404 —— 绝不能被 SPA 回退成 200 + HTML。"""
    r = client.get("/api/v1/definitely-not-a-route")
    assert r.status_code == 404
    body = r.json()
    assert body["code"] == 40401
    assert "text/html" not in r.headers["content-type"]


def test_docs_still_served(client):
    """/docs 不能被 catch-all 吞掉（Swagger 是开发期主入口）。"""
    r = client.get("/docs")
    assert r.status_code == 200
