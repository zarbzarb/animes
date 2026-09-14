# -*- coding: utf-8 -*-
"""分层依赖静态检查（CI 门禁）。

    python scripts/check_imports.py
    python scripts/check_imports.py --verbose     # 顺便打印每层的 import 统计
    python scripts/check_imports.py --only R3

规则来自 `docs/dev-conventions.md` §3.2，与 `docs/project-structure.md` 的
依赖方向图一一对应：

    R1  models/     不得 import agents/ 或 server/
    R2  models/     不得 import torch 之外的重依赖（redis / sqlalchemy / fastapi ...）
    R3  agents/     不得 import server/
    R4  server/api/ 不得 import models/（必须经 service / adapter）
    R5  任何层      不得 import tests/

为什么用 AST 而不是"先 import 再看 `sys.modules`"
----------------------------------------------
1. **不真正 import**：`server/core/config.py` 连数据库、`agents/recall/adapter.py`
   要 `torch.load` —— 在 CI 机器上（没有 GPU、没有 .env）import 就会炸，
   门禁脚本自己先挂了，等于没有门禁。
2. **能看到函数体内的惰性 import**：本项目大量使用"函数内延迟 import"来削
   启动耗时（如 `agents/recall/adapter.py` 里的 `from agents.common.data import ...`）。
   `ast.walk` 会遍历到嵌套节点，一个都不漏。
3. **能抓动态 import 的字符串常量**：`importlib.import_module("server.db.session")`
   和 `__import__("agents.x")` 在语法上不是 `Import` 节点，只按 AST 的 Import
   系列节点扫会**完全看不见**它们 —— 那这道闸门留一个后门就够了。
   所以下面额外扫这两类调用的字符串字面量参数。

相对 import 的处理
----------------
`from ..common import base` 这种写法必须先解析成绝对模块名才能判层。
这里用「文件路径 → 包名」反推（`agents/fusion/agent.py` → `agents.fusion.agent`），
再按 Python 的相对 import 规则（`level` 个点回退 level-1 层）拼接。
**不做这一步会把 `agents/` 里所有 `from ..` 都当成"看不见"，规则形同虚设。**
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LAYERS = ("models", "agents", "server", "tests")

# R2 的重依赖**黑名单**（而不是白名单）。
# 用黑名单的理由：`models/` 里合法地会用到 `numpy`、`yaml`、`tqdm`、`sklearn`
# 之类的东西，写白名单会在每次加一个工具库时误报；而"能连外部服务/Bytes IO 的
# 重依赖"是很小且稳定的一个集合，列它更准。
HEAVY_DEPS = {
    "sqlalchemy", "pymysql", "redis", "fastapi", "starlette", "uvicorn",
    "requests", "httpx", "aiohttp", "apscheduler", "celery",
    "openai", "dashscope", "transformers", "datasets",
}
# `pydantic` 特意不在上面：`models/` 的 config 用 dataclass 就够，
# 但 `models/eval/metrics.py` 这类模块一旦需要类型校验也不该被拦住 ——
# 它没有 IO 能力，不破坏"models 纯函数"的性质。


class Violation:
    __slots__ = ("rule", "path", "lineno", "detail")

    def __init__(self, rule: str, path: Path, lineno: int, detail: str) -> None:
        self.rule, self.path, self.lineno, self.detail = rule, path, lineno, detail

    def __str__(self) -> str:
        rel = self.path.relative_to(PROJECT_ROOT).as_posix()
        return f"[{self.rule}] {rel}:{self.lineno}  {self.detail}"


def _layer_of(path: Path):
    """文件的所属层（顶层包名），不在四层里则返回 None。"""
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    return rel.parts[0] if rel.parts and rel.parts[0] in LAYERS else None


def _module_of(path: Path) -> str:
    """`F:/pj/agents/fusion/agent.py` → `agents.fusion.agent`（用于解析相对 import）。"""
    rel = path.relative_to(PROJECT_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_relative(module: str, level: int, node_module: str | None) -> str:
    """把 `level` 个点的相对 import 解析成绝对模块名。

    规则：`level=1` 是当前包（`from . import x`），`level=2` 回退一层。
    例：`agents/fusion/agent.py`（模块 `agents.fusion.agent`）+ `level=2` + `common`
    → 当前包 `agents.fusion` 再回退 1 层 = `agents` → `agents.common`。
    """
    pkg_parts = module.split(".")[:-1]          # 去掉文件名，得到所在包
    if level <= 0:
        return node_module or ""
    back = level - 1
    if back > 0:
        pkg_parts = pkg_parts[:-back] if back <= len(pkg_parts) else []
    if node_module:
        pkg_parts = pkg_parts + node_module.split(".")
    return ".".join(pkg_parts)


def _imported_names(tree: ast.AST, module: str) -> list[tuple[str, int]]:
    """收集本文件"会 import 到的顶层模块名"→ `[(name, lineno)]`。

    同时覆盖三类写法：
    * `import a.b`            → `a.b`
    * `from a.b import c`     → `a.b`
    * `from . import x`       → 解析后的绝对名
    * `importlib.import_module("a.b")` / `__import__("a.b")` → `a.b`
    """
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_relative(module, node.level, node.module)
            if base:
                out.append((base, node.lineno))
        elif isinstance(node, ast.Call):
            fn = node.func
            is_dyn = (
                (isinstance(fn, ast.Attribute) and fn.attr == "import_module")
                or (isinstance(fn, ast.Name) and fn.id == "__import__")
            )
            if is_dyn and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    out.append((arg.value, node.lineno))
    return out


def _check_file(path: Path) -> list[Violation]:
    layer = _layer_of(path)
    if layer is None:
        return []
    module = _module_of(path)
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [Violation("SYNTAX", path, exc.lineno or 0, f"语法错误：{exc.msg}")]

    out: list[Violation] = []
    for name, lineno in _imported_names(tree, module):
        top = name.split(".")[0]

        if layer == "models" and top in ("agents", "server"):
            out.append(Violation("R1", path, lineno,
                                 f"models 反向依赖 {top}：import {name}"))
        if layer == "models" and top in HEAVY_DEPS:
            out.append(Violation("R2", path, lineno,
                                 f"models 引入重依赖 {top}（models 必须保持可离线重算）：import {name}"))
        if layer == "agents" and top == "server":
            out.append(Violation("R3", path, lineno,
                                 f"agents 反向依赖 server：import {name}"))
        if layer == "server" and rel.startswith("server/api/") and top == "models":
            out.append(Violation("R4", path, lineno,
                                 f"server/api 直接 import models（应经 server/services）：import {name}"))
        if top == "tests":
            out.append(Violation("R5", path, lineno,
                                 f"{layer} 层不得 import tests：import {name}"))
    return out


SELF_TEST_CASES: list[tuple[str, str, str | None, str]] = [
    # (相对路径, 文件内容, 期望被抓住的规则 / None = 期望干净, 说明)
    ("models/_selfcheck_probe.py", "import agents.common.base\n", "R1",
     "models 反向 import agents"),
    ("models/_selfcheck_probe.py", "from server.db import session\n", "R1",
     "models 反向 import server（from 形式）"),
    ("models/_selfcheck_probe.py", "import sqlalchemy\n", "R2",
     "models 引入重依赖 sqlalchemy"),
    ("models/_selfcheck_probe.py", "def f():\n    import redis\n", "R2",
     "函数体内的惰性 import 也必须被看到"),
    ("agents/_selfcheck_probe.py", "import server.core.config\n", "R3",
     "agents 反向 import server"),
    ("agents/_selfcheck_probe.py",
     "import importlib\n\n\ndef f():\n"
     "    return importlib.import_module('server.db.session')\n", "R3",
     "动态 import 的字符串常量也要被看到（否则留了后门）"),
    ("agents/_selfcheck_probe.py",
     "def f():\n    return __import__('server.core.config')\n", "R3",
     "`__import__('...')` 同样要被抓"),
    ("server/api/_selfcheck_probe.py",
     "from models.sasrec.model import SASRec\n", "R4",
     "server/api 直接 import models"),
    ("agents/_selfcheck_probe.py",
     "from tests.test_metrics import x\n", "R5",
     "任何层不得 import tests"),
    # ⚠️ 这条是**解析器的有效性证明**，不能用"期望干净"的用例代替：
    # 若 `_resolve_relative` 返回空串或乱码，期望干净的用例会**照样通过**
    # （假阴性），而这条用相对写法回到顶层再进 `tests` 的用例就会漏掉。
    ("agents/_selfcheck_probe.py",
     "from ..tests.test_metrics import x\n", "R5",
     "相对 import 必须正确解析成绝对名（这里回到顶层 tests）"),
    # 同上的"合法"对照：`agents/` 里 `from ..common import ...` → `agents.common`
    ("agents/_selfcheck_probe.py",
     "from ..common import base\n", None,
     "相对 import 解析正确时不误报"),
]

# 「合法写法也得过」的对照组 —— 没有它，一个"见谁都报错"的坏检查器
# 也能通过上面所有用例。
SELF_TEST_CLEAN: list[tuple[str, str, str]] = [
    ("models/_selfcheck_probe.py",
     "import torch\nimport numpy as np\nfrom models.eval import metrics\n", "models 正常 import"),
    ("agents/_selfcheck_probe.py",
     "import torch\nfrom agents.common.envelope import Envelope\n"
     "from models.sasrec.model import SASRec\n", "agents → models 是允许的方向"),
    ("server/api/_selfcheck_probe.py",
     "from server.services import agent_bridge\n", "server/api 经 services 间接用模型"),
    ("models/_selfcheck_probe.py",
     "from .eval import metrics\nimport yaml\n", "同包内相对 import + 轻依赖"),
]


PROBE_TAG = "_selfcheck_probe"


def _probe_path(rel: str) -> Path:
    """探针文件名带 **pid**，避免两个并发的自检（或上次崩溃留下的残骸）撞名。

    踩过的坑：早期用固定名 `agents/_selfcheck_probe.py`，遇到"路径已被占用"
    就直接抛异常，一次自检因此中断；而外层还得靠额外跑一遍 `find` 才知道
    有没有残留。带 pid + 自带清理与残留自检，才是能反复跑的形状。
    """
    p = Path(rel)
    # ⚠️ pid 用 `_` 分隔而不是 `.`：文件名里的点会混进模块名，
    # 而相对 import 的解析依赖"模块名去掉最后一段 = 所在包"——
    # `agents/_selfcheck_probe.123.py` 会算出所在包是 `agents._selfcheck_probe`，
    # 于是 `from ..tests import x` 解析成 `agents.tests.x`，规则静默失效。
    return PROJECT_ROOT / p.parent / f"{p.stem}_{os.getpid()}{p.suffix}"


def _unlink_retry(path: Path, tries: int = 5) -> None:
    """Windows 上刚写完的文件偶尔被杀软/索引器短暂占用，unlink 会抛
    PermissionError（`missing_ok=True` 只挡 FileNotFoundError）。重试几次。"""
    for i in range(tries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05)


def _probe(rel: str, body: str) -> list[Violation]:
    """在真实目录里落一个临时文件跑一遍规则，结束**无条件删除**。

    ⚠️ 必须放在真实目录下（而不是 tempfile）：R1–R4 的判定依赖
    「文件属于哪一层」，而层是从"相对项目根的路径"推出来的。
    放到系统临时目录里会让所有用例都判不出层，于是"全绿"—— 一个
    什么都没检查的检查器，正好是最危险的结果。
    """
    path = _probe_path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(body, encoding="utf-8")
        return _check_file(path)
    finally:
        _unlink_retry(path)


def _sweep_stale_probes() -> list[Path]:
    """扫描并清掉历史残骸（含旧版固定名），返回清掉的文件列表。"""
    killed: list[Path] = []
    for layer in LAYERS:
        for p in (PROJECT_ROOT / layer).rglob(f"{PROBE_TAG}*.py"):
            _unlink_retry(p)
            killed.append(p)
    return killed


def self_test() -> int:
    """证明这个检查器**抓得住**违规，而不只是"什么都不报"。"""
    print("=" * 70)
    print("check_imports 自检：注入已知违规 / 已知合法，验证判定是否有效")
    print("=" * 70)
    bad = 0

    stale = _sweep_stale_probes()
    if stale:
        print(f"  （清掉历史残骸 {len(stale)} 个："
              f"{', '.join(p.name for p in stale)}）")

    for rel, body, want, desc in SELF_TEST_CASES:
        got = [v.rule for v in _probe(rel, body)]
        ok = (not got) if want is None else (want in got)
        tag = "期望干净" if want is None else f"期望 {want}"
        print(f"  {'✅' if ok else '❌'} {tag:9s} {desc}  → 命中 {got or '无'}")
        bad += 0 if ok else 1

    print()
    for rel, body, desc in SELF_TEST_CLEAN:
        got = [v.rule for v in _probe(rel, body)]
        ok = not got
        print(f"  {'✅' if ok else '❌'} 合法写法不应报错：{desc}  → {got or '干净'}")
        bad += 0 if ok else 1

    # 清理必须由自检自己保证 —— 靠"跑完再手动 find 一下"不是保证，
    # 哪次忘了看就留一个 `agents/_selfcheck_probe.py` 进仓库。
    left = _sweep_stale_probes()
    if left:
        print(f"\n  ❌ 自检残留了 {len(left)} 个探针文件："
              f"{', '.join(p.name for p in left)}")
        bad += 1
    else:
        print("\n  ✅ 无探针残留")

    total = len(SELF_TEST_CASES) + len(SELF_TEST_CLEAN)
    print("=" * 70)
    if bad:
        print(f"❌ 自检失败 {bad}/{total} 例 —— 这个检查器的判定不可信，先修它本身")
        return 1
    print(f"✅ 自检通过（{total} 例）：该报的报、该放的放")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="分层依赖静态检查")
    ap.add_argument("--verbose", action="store_true", help="打印每层的 import 统计")
    ap.add_argument("--only", default=None, help="只跑某条规则，如 R3")
    ap.add_argument("--self-test", action="store_true",
                    help="注入已知违规验证检查器本身有效（改本脚本后必跑）")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    files: list[Path] = []
    for layer in LAYERS:
        files += sorted((PROJECT_ROOT / layer).rglob("*.py"))
    # `scripts/` 不在 R1–R4 的分层规则内（它是离线工具与 CI 脚本自身），
    # 但 R5（不得 import tests）对它同样适用。
    files += sorted((PROJECT_ROOT / "scripts").glob("*.py"))

    violations: list[Violation] = []
    per_layer: Counter = Counter()
    imports_of: dict[str, Counter] = defaultdict(Counter)

    for f in files:
        layer = _layer_of(f) or "scripts"
        per_layer[layer] += 1
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError as exc:
            violations.append(Violation("SYNTAX", f, exc.lineno or 0, exc.msg))
            continue
        if args.verbose:
            for name, _ in _imported_names(tree, _module_of(f)):
                imports_of[layer][name.split(".")[0]] += 1
        for v in _check_file(f):
            if args.only and v.rule != args.only:
                continue
            violations.append(v)

    print("=" * 70)
    print("分层依赖检查  " + "  ".join(f"{k}={v}" for k, v in sorted(per_layer.items())))
    print("=" * 70)

    if args.verbose:
        for layer in sorted(imports_of):
            top = ", ".join(f"{n}({c})" for n, c in imports_of[layer].most_common(12))
            print(f"\n{layer}/ 引用的顶层模块： {top}")

    if violations:
        print(f"\n❌ 发现 {len(violations)} 处违规：\n")
        for v in sorted(violations, key=lambda x: (x.rule, str(x.path), x.lineno)):
            print("  " + str(v))
        print("\n修法：`agents`/`models` 需要上层能力时，用注入（`bind_gateway` /\n"
              "`bind_runtime`）或把接口下沉到本层，**不要反向 import**。\n"
              "依据：docs/dev-conventions.md §3.2 / docs/project-structure.md 依赖方向图。")
        return 1

    print("\n✅ 无跨层反向依赖（R1–R5 全部通过）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
