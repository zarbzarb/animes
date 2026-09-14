# -*- coding: utf-8 -*-
"""对**正在运行**的服务做端到端冒烟（HTTP 层，不 mock）。

    # 1) 先起服务
    python -m uvicorn server.main:app --port 8000

    # 2) 另一个终端跑冒烟
    python scripts/smoke_api.py
    python scripts/smoke_api.py --base http://127.0.0.1:8000 --size 6

为什么单独放一个脚本而不是写成 pytest
------------------------------------
这一层要验证的是「装配是否真的发生」—— 网关有没有注入、模型有没有预热、
统一响应体有没有生效。这些只有在**真实进程**里才成立（pytest 里 import
即装配，会掩盖掉 lifespan 的问题）。所以它是运维/验收脚本，不是单测。

设计要点
--------
* 绕过本机代理：`http.proxy=127.0.0.1:12450` 会把 localhost 请求也拦走，
  表现为莫名其妙的 502。这里显式用空 `ProxyHandler`。
* 只读为主：唯一的写操作是 `/recommend/feedback`（幂等窗口内会去重），
  反复跑不会污染数据。
* 每个用例打印**响应信封的完整性**（`code/message/trace_id/elapsed_ms`），
  因为「少了两个字段」恰恰是最容易漏、又最不容易被发现的 bug。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Optional

ENVELOPE_KEYS = {"code", "message", "data", "trace_id", "elapsed_ms"}

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

PASS: list[str] = []
FAIL: list[str] = []


def _call(method: str, base: str, path: str, body: Any = None,
          token: Optional[str] = None, timeout: float = 120.0
          ) -> tuple[int, dict]:
    req = urllib.request.Request(base + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(body).encode() if body is not None else None
    try:
        with _opener.open(req, data, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:
            return exc.code, {"_raw": "<非 JSON 响应>"}


def _call_sse(method: str, base: str, path: str, body: Any = None,
              token: Optional[str] = None, timeout: float = 180.0
              ) -> tuple[int, list[dict]]:
    """`/chat/message` 是 **SSE**（`text/event-stream`），不能按 JSON 解析。

    返回 `(status, events)`，`events` 是逐条 `data:` 解析出来的对象。
    早期冒烟脚本把它当 JSON 读，结果每次都在这里抛 `JSONDecodeError` ——
    接口其实完全正常，是测试写错了。
    """
    req = urllib.request.Request(base + path, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(body).encode() if body is not None else None
    try:
        with _opener.open(req, data, timeout=timeout) as r:
            status, raw = r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, [{"_error": exc.read().decode("utf-8", "replace")[:300]}]

    events: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            events.append(json.loads(chunk))
        except ValueError:
            events.append({"_raw": chunk[:200]})
    return status, events


def check(name: str, cond: bool, detail: str = "") -> bool:
    """`detail` **只在失败时打印** —— 它写的是诊断信息（"哪里不对"），
    成功时也打出来会让人误以为出了问题（例如 ✅ 后面跟一句
    "用户 admin 登录失败"）。"""
    (PASS if cond else FAIL).append(name)
    mark = "✅" if cond else "❌"
    if detail and not cond:
        print(f"  {mark} {name} — {detail}")
    else:
        print(f"  {mark} {name}")
    return cond


def envelope_ok(name: str, resp: dict) -> bool:
    """统一响应体必须五个键齐全 —— 少 trace_id/elapsed_ms 是静默 bug。"""
    missing = ENVELOPE_KEYS - set(resp)
    ok = check(f"{name} · 响应信封完整", not missing,
               f"缺 {sorted(missing)}")
    if ok:
        print(f"       code={resp.get('code')} trace={resp.get('trace_id')} "
              f"elapsed={resp.get('elapsed_ms')}ms")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--user", default="anifan")
    ap.add_argument("--password", default="P@ssw0rd")
    ap.add_argument("--admin", default="admin")
    ap.add_argument("--admin-password", default="P@ssw0rd")
    ap.add_argument("--size", type=int, default=6)
    args = ap.parse_args()
    B = args.base.rstrip("/") + "/api/v1"

    print("=" * 68)
    print(f"AniRec API 冒烟 → {B}")
    print("=" * 68)

    # ---------------------------------------------------------- 基础
    print("\n[1] 基础与元数据")
    st, d = _call("GET", B, "/health")
    envelope_ok("health", d)
    if st == 200 and d.get("code") == 0:
        data = d.get("data") or {}
        print(f"     env={data.get('app_env')} db={data.get('db')} "
              f"cache={data.get('cache')} agents={data.get('agents_ready')}")
        print(f"     degraded={data.get('degraded')}")

    st, d = _call("GET", B, "/genres")
    envelope_ok("genres", d)
    if d.get("code") == 0:
        gl = (d.get("data") or {}).get("list") or []
        check("genres · 12 类题材", len(gl) == 12, f"n={len(gl)}")

    st, d = _call("GET", B, f"/animes?size={args.size}")
    envelope_ok("animes", d)
    if d.get("code") == 0:
        dd = d.get("data") or {}
        total = (dd.get("pagination") or {}).get("total")
        check("animes · 有池内番剧", bool(total), f"total={total}")
        for a in (dd.get("list") or [])[:3]:
            print(f"     · {a.get('title')} ({a.get('year')}) "
                  f"genres={a.get('genres')} n={a.get('n_interactions')}")

    # ---------------------------------------------------------- 鉴权
    print("\n[2] 鉴权")
    st, d = _call("POST", B, "/auth/login",
                  {"username": args.user, "password": args.password})
    envelope_ok("login", d)
    tok = (d.get("data") or {}).get("access_token")
    if not check("login · 拿到 token", bool(tok), f"status={st} msg={d.get('message')}"):
        print("\n无法继续（登录失败）。先确认：python scripts/init_db.py")
        return _summary()

    st, d = _call("GET", B, "/users/me", token=tok)
    envelope_ok("users/me", d)
    if d.get("code") == 0:
        u = d.get("data") or {}
        check("users/me · 身份正确", u.get("username") == args.user,
              f"username={u.get('username')}")

    # 未带 token 必须 401
    st, d = _call("GET", B, "/users/me")
    check("users/me · 未鉴权返回 401", st == 401, f"status={st}")

    # ---------------------------------------------------------- 推荐主链路
    print("\n[3] 推荐主链路（A0 编排 A1→A2/A3→A4→A5）")
    st, d = _call("GET", B, f"/recommend/feed?size={args.size}", token=tok,
                  timeout=180)
    envelope_ok("feed", d)
    dd = d.get("data") or {}
    meta = dd.get("meta") or {}
    items = dd.get("items") or []
    check("feed · 有推荐结果", len(items) > 0, f"n={len(items)}")
    check("feed · 无降级步骤", not (meta.get("degraded") or []),
          f"degraded={meta.get('degraded')}")
    print(f"     intent={meta.get('intent')} method={meta.get('intent_method')} "
          f"chain={meta.get('agent_chain')} elapsed={d.get('elapsed_ms')}ms")
    if items:
        for it in items[:args.size]:
            a = it.get("anime") or {}
            ex = it.get("explain") or {}
            print(f"     #{it.get('rank')} {a.get('title')} ({a.get('year')}) "
                  f"genres={a.get('genres')} final={it.get('final_score')} "
                  f"b={it.get('behavior_score')} c={it.get('content_score')} "
                  f"cold={it.get('is_cold_start')}")
            if ex.get("reason"):
                print(f"        ↳ {str(ex.get('reason'))[:70]} [{ex.get('source')}]")
        # 排序必须单调不增
        fs = [float(x.get("final_score") or 0) for x in items]
        check("feed · final_score 单调不增",
              all(fs[i] >= fs[i + 1] - 1e-9 for i in range(len(fs) - 1)))
        # 排名连续
        check("feed · rank 从 1 连续",
              [x.get("rank") for x in items] == list(range(1, len(items) + 1)))
        # 去重
        ids = [(x.get("anime") or {}).get("src_anime_id") for x in items]
        check("feed · 无重复番剧", len(ids) == len(set(ids)))
        # 主链路**只解释 top1**（`orchestrator/agent.py::_aggregate` 的设计），
        # 逐条解释走 `GET /recommend/{id}/explain`。所以这里只断言第 1 条。
        check("feed · top1 带解释理由",
              bool((items[0].get("explain") or {}).get("reason")))
        check("feed · anime{} 已补全展示字段",
              all((x.get("anime") or {}).get("title") for x in items))

    # 逐条解释：Top5 里每条都应该能单独拿到解释（这是设计上的第二条路径）
    for it in items[: min(5, len(items))]:
        src = (it.get("anime") or {}).get("src_anime_id")
        st, d = _call("GET", B, f"/recommend/{src}/explain", token=tok, timeout=90)
        if d.get("code") in (0, 60401):
            reason = (d.get("data") or {}).get("reason")
            check(f"explain({src}) · 有条目解释", bool(reason),
                  f"[{(d.get('data') or {}).get('source')}] {str(reason)[:40]}")

    # 分类推荐
    st, d = _call("GET", B, f"/recommend/by-genre?genre_id=1&size={args.size}",
                  token=tok, timeout=180)
    envelope_ok("by-genre", d)
    if d.get("code") in (0, 60401):
        check("by-genre · 回填题材名",
              bool((d.get("data") or {}).get("genre")),
              f"genre={(d.get('data') or {}).get('genre')}")

    # 新番
    st, d = _call("GET", B, f"/recommend/new-anime?size={args.size}", token=tok,
                  timeout=180)
    envelope_ok("new-anime", d)

    # 反馈（幂等：第二次会被去重）
    if items:
        src = (items[0].get("anime") or {}).get("src_anime_id")
        st, d = _call("POST", B, "/recommend/feedback",
                      {"anime_id": int(src), "action": "click", "scene": 0},
                      token=tok)
        envelope_ok("feedback", d)
        if d.get("code") == 0:
            print(f"     recorded={(d.get('data') or {}).get('recorded')} "
                  f"deduped={(d.get('data') or {}).get('deduped')}")

    # ---------------------------------------------------------- 分析
    print("\n[4] 分析与对话")
    st, d = _call("GET", B, "/analysis/interest-radar", token=tok, timeout=120)
    envelope_ok("radar", d)
    if d.get("code") == 0:
        rd = (d.get("data") or {}).get("radar") or []
        check("radar · 12 个扇区", len(rd) == 12, f"n={len(rd)}")
        top = sorted(rd, key=lambda x: -float(x.get("value") or 0))[:3]
        print("     top3: " + ", ".join(
            f"{x.get('genre')}={x.get('value')}" for x in top))
        # 回归守卫：曾经因为 `user_profile` 里一行 `top_genres=[]` 的半成品
        # 被 A1 当成权威结果永久返回，导致雷达图**长期恒为全 0** 且不报错。
        nz = sum(1 for x in rd if float(x.get("value") or 0) > 0)
        check("radar · 强度非全 0（画像不是空壳）", nz > 0,
              f"12 个题材全为 0 → 检查 A1 是否返回了空的 top_genres（非零 {nz}/12）")
        check("radar · 兴趣胶囊非全 0",
              any(float(c.get("strength") or 0) > 0
                  for c in ((d.get("data") or {}).get("interest_capsules") or [])))
        print(f"     summary: {str((d.get('data') or {}).get('summary_text'))[:60]}")

    st, d = _call("GET", B, "/analysis/drift-trend", token=tok, timeout=120)
    envelope_ok("drift-trend", d)

    # `/chat/message` 是 **SSE**，必须按 event-stream 解析（见 `_call_sse`）。
    st, events = _call_sse("POST", B, "/chat/message",
                           {"message": "有没有类似进击的巨人但更轻松的番？",
                            "session_id": ""},
                           token=tok)
    check("chat/message · SSE 200", st == 200, f"status={st}")
    kinds = [e.get("type") for e in events]
    check("chat/message · 有 chunk 事件", "chunk" in kinds, f"events={kinds}")
    check("chat/message · 有 done 事件", "done" in kinds or bool(events),
          f"events={kinds}")
    text = "".join(str(e.get("content") or "") for e in events
                   if e.get("type") == "chunk")
    cards = next((e.get("cards") for e in events if e.get("type") == "cards"), [])
    print(f"     reply: {text[:90]}")
    print(f"     cards: {len(cards or [])} 首")
    check("chat/message · 有回复文本", bool(text.strip()))

    # ---------------------------------------------------------- 管理端
    print("\n[5] 管理端（需要 role=1）")
    st, d = _call("POST", B, "/auth/login",
                  {"username": args.admin, "password": args.admin_password})
    atok = (d.get("data") or {}).get("access_token")
    if not check("admin 登录", bool(atok),
                 f"用户 {args.admin} 登录失败；先跑 python scripts/init_db.py"):
        return _summary()

    # 普通用户必须被 403 拦住 —— 否则等于管理后台对所有人开放
    st, d = _call("GET", B, "/admin/agents/health", token=tok)
    check("admin · 普通用户被 403 拦截", st == 403, f"status={st}")

    st, d = _call("GET", B, "/admin/agents/health", token=atok)
    envelope_ok("agents-health", d)
    if d.get("code") == 0:
        dd = d.get("data") or {}
        rows = dd.get("list") or dd.get("agents") or []
        check("agents-health · 10 个 Agent 有健康度", len(rows) >= 10
              or len(dd.get("health") or {}) >= 10, f"n={len(rows)}")
        print(f"     payload keys={sorted(dd.keys())[:8]}")

    st, d = _call("GET", B, "/admin/metrics", token=atok)
    envelope_ok("admin-metrics", d)

    st, d = _call("GET", B, "/admin/cold-start", token=atok)
    envelope_ok("admin-cold-start", d)

    # 内部连通性自检（不鉴权，仅本机）
    st, d = _call("GET", B, "/internal/health")
    envelope_ok("internal-health", d)

    return _summary()


def _summary() -> int:
    print("\n" + "=" * 68)
    print(f"通过 {len(PASS)} · 失败 {len(FAIL)}")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 68)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
