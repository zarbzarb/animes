"""回归测试：/static/uploads 挂载必须存在且先于 /static。

历史教训：头像上传接口把文件写进 data/uploads 并把 /static/uploads/...
落库，但 main.py 的 `/static/uploads` mount 曾在并行编辑中被同文件的
后续 Edit 静默覆盖丢失 —— 上传"成功"却永远 404，前端 el-avatar 回退
显示昵称首字。而当时在跑的旧进程加载过丢失前的代码，验证全是假阳性，
直到用户用 PyCharm 全新启动才暴露。
"""

from pathlib import Path

from fastapi.testclient import TestClient

from server.main import UPLOAD_DIR, app


def test_uploads_mount_registered_before_static():
    """uploads 挂载必须存在，且注册顺序在 /static 之前（按序匹配）。"""
    paths = [getattr(r, "path", None) for r in app.router.routes]
    assert "/static/uploads" in paths, (
        "/static/uploads 挂载丢失 —— 头像上传会成功但永远 404"
    )
    assert paths.index("/static/uploads") < paths.index("/static")


def test_uploaded_avatar_is_served(tmp_path: Path):
    """端到端：data/uploads 下的文件必须能经 /static/uploads/ 取到。"""
    name = "test_avatar_regression.png"
    target = UPLOAD_DIR / name
    target.write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes")
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get(f"/static/uploads/{name}")
        assert resp.status_code == 200
        assert resp.content == b"\x89PNG\r\n\x1a\nfake-bytes"
    finally:
        target.unlink(missing_ok=True)


def test_unknown_static_path_still_404_envelope():
    """/static 下不存在的路径仍是信封 404，不能回退成 index.html。"""
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/static/no/such/file.js")
    assert resp.status_code == 404
    assert resp.json()["code"] != 0
