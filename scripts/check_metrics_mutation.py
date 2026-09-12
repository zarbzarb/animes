# -*- coding: utf-8 -*-
"""指标层变异测试：验证 tests/test_metrics/ 的断言**真的能抓住错误实现**。

为什么需要它
------------
「测试全绿」不等于「测试有效」。阶段 2 的 M2.1 就踩过一次：
把 `metrics.py` 里的 `stable=True` 改成 `False`（并列口径失效），88 个测试**依然全绿**——
因为小规模、并列少的输入下，非稳定排序会"碰巧"按列序输出。
直到把输入放大到真实候选规模（101 个、分数大量并列）才发现差异。

变异测试的做法：故意把实现改坏，如果测试还是全绿，说明测试有洞。
本脚本内置 6 类典型改错，覆盖：
    1. HR 边界        `<=` 写成 `<`
    2. NDCG 折损      分母 `log2(rank+1)` 写成 `log2(rank)`
    3. 排序稳定性     `stable=True` 去掉（并列次序变得不可复现）
    4. 排名基准       忘记 +1，变成 0-based
    5. 聚合口径       Recall 的 micro 写成 macro
    6. 批量独立性     `positive_rank` 对整批做全局排序而不是按行排序

用法
----
    python scripts/check_metrics_mutation.py            # 跑全部变异
    python scripts/check_metrics_mutation.py --list      # 只列出变异项

退出码
------
    0 = 基线通过，且每个变异都被测试捕获
    1 = 有变异漏网（测试需要补强），或基线本身未通过
    2 = 恢复失败（原文件已损坏，需人工处理）

安全说明
--------
脚本会**临时改写** `models/eval/metrics.py`，并在 `finally` 中恢复：
    1. 变异前把原文备份到内存 + `metrics.py.mutation-backup`；
    2. 每轮结束立即恢复；
    3. 全部结束后用 sha256 校验恢复结果与原文件一致；
    4. 校验不通过则以退出码 2 退出，并保留备份文件供人工恢复。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，中文会乱码
except Exception:
    pass

# 与其它 scripts/ 同一约定：路径相对项目根，脚本与当前工作目录无关
ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "models" / "eval" / "metrics.py"
BACKUP = SRC.with_suffix(".py.mutation-backup")

# (说明, 原文片段, 变异后片段)
MUTATIONS = [
    ("HR 边界 <= 写成 <（rank == K 不再算命中）",
     "return float((t <= k).to(torch.float64).mean())",
     "return float((t < k).to(torch.float64).mean())"),
    ("NDCG 折损分母 log2(rank+1) 写成 log2(rank)",
     "1.0 / torch.log2(t + 1.0)",
     "1.0 / torch.log2(t)"),
    ("排序稳定性 stable=True 去掉（并列次序不可复现）",
     "return torch.argsort(scores, dim=-1, descending=True, stable=True)",
     "return torch.argsort(scores, dim=-1, descending=True, stable=False)"),
    ("正样本排名忘记 +1（变成 0-based）",
     "return hit.to(torch.long).argmax(dim=-1) + 1",
     "return hit.to(torch.long).argmax(dim=-1)"),
    ("Recall 的 micro 聚合写成 macro（逐用户平均）",
     "return hits / total",
     "return hits / scores.shape[0]"),
    ("positive_rank 对整批做全局排序而不是按行排序",
     "    order = stable_order(scores)",
     "    order = stable_order(scores.flatten()).expand(scores.shape[0], -1)"),
]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_pytest() -> tuple[int, str]:
    """跑一遍指标测试，返回 (退出码, 结果摘要)。"""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests" / "test_metrics"),
         "-q", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    return r.returncode, (lines[-1] if lines else "(无输出)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="指标层变异测试：确认 tests/test_metrics/ 能抓住错误实现",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="详见 docs/progress.md 的 M2.1 与 docs/dev-conventions.md 4.2 节",
    )
    ap.add_argument("--list", action="store_true", help="只列出变异项，不执行")
    args = ap.parse_args()

    if args.list:
        print("内置变异项：")
        for i, (name, _, _) in enumerate(MUTATIONS, 1):
            print(f"  {i}. {name}")
        return 0

    original = SRC.read_text(encoding="utf-8")
    orig_hash = sha256(original)
    BACKUP.write_text(original, encoding="utf-8")

    print("=" * 72)
    print("指标层变异测试")
    print("=" * 72)
    print(f"目标文件: {SRC.relative_to(ROOT)}")
    print(f"备份文件: {BACKUP.relative_to(ROOT)}")
    print()

    try:
        # ---------- 0. 基线必须先通过 ----------
        # 这一步是踩坑后加的：如果基线本身在失败，退出码恒非 0，
        # 每个变异都会"看起来被抓住"，整个变异测试结论作废（假阳性）。
        t0 = time.time()
        rc, last = run_pytest()
        print(f"[基线] {last}   ({time.time() - t0:.1f}s)")
        if rc != 0:
            print("\n❌ 基线未通过，变异测试结论无效。请先修好测试再跑本脚本。")
            return 1

        # ---------- 1. 逐个变异 ----------
        print()
        missed = []
        untargeted = []
        for i, (name, old, new) in enumerate(MUTATIONS, 1):
            if old not in original:
                print(f"⚠️  [{i}] 变异目标未找到（实现可能已改动），跳过：{name}")
                untargeted.append(name)
                continue
            SRC.write_text(original.replace(old, new, 1), encoding="utf-8")
            try:
                rc, last = run_pytest()
            finally:
                SRC.write_text(original, encoding="utf-8")

            caught = rc != 0
            if not caught:
                missed.append(name)
            print(f"{'✅ 被抓住' if caught else '❌ 漏网  '} [{i}] {name}")
            print(f"            {last}")

        # ---------- 2. 恢复校验 ----------
        print()
        now = SRC.read_text(encoding="utf-8")
        if sha256(now) != orig_hash:
            print("❌ 恢复失败：metrics.py 与原文件不一致！")
            print(f"   备份保留在 {BACKUP}，请人工恢复：")
            print(f"   copy \"{BACKUP}\" \"{SRC}\"")
            return 2
        BACKUP.unlink(missing_ok=True)

        rc, last = run_pytest()
        print(f"[还原后复测] {last}")

        print()
        print("=" * 72)
        if missed or untargeted:
            print(f"结论：{len(MUTATIONS) - len(missed) - len(untargeted)}/{len(MUTATIONS)} 个变异被捕获")
            for m in missed:
                print(f"  ❌ 漏网：{m}")
            for m in untargeted:
                print(f"  ⚠️  未生效（目标片段不存在）：{m}")
            print("  → 测试存在盲区，请补充针对性的断言后再提交。")
            return 1
        print(f"结论：{len(MUTATIONS)}/{len(MUTATIONS)} 个变异全部被捕获，测试有效 ✅")
        print(f"      文件已恢复，sha256 = {orig_hash[:16]}...")
        return 0
    finally:
        # 兜底：任何异常路径下都保证原文件被还原
        if SRC.read_text(encoding="utf-8") != original:
            SRC.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
