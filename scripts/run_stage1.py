#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段一（数据预处理与数据集构建）流水线一键执行。

为什么需要这个脚本
------------------
阶段一有 6 个脚本，彼此存在依赖关系。顺序跑错时有两种失败模式：

  * **立刻报错**（还算友好）：例如先跑 ``build_content_vectors.py``，
    会因缺少 ``data/processed/anime_meta.parquet`` 直接抛 FileNotFoundError。
  * **静默得到错误结果**（真正危险）：例如 ``preprocess.py`` 改了配置之后
    只重跑 ``build_content_vectors.py``，内容向量就会和新的物品集合对不上，
    但脚本本身不会报错。

本脚本把正确顺序固化成代码，并在每一步之前做「前置文件检查」、
之后做「产物检查」，让顺序错误无法悄悄发生。

正确的依赖顺序（本脚本硬编码，改这里等于改流程）
------------------------------------------------
    ① preprocess.py               必须最先跑 —— ②③④ 中的三步都依赖它的产物
    ② download_content_encoder.py 仅本地权重缺失时执行（已有权重则自动跳过）
    ③ build_content_vectors.py    依赖 ① 的 anime_meta / anime_detailed_tags
    ④ build_cold_start_subset.py  依赖 ① 的 item_stats / seq_dataset
    ⑤ diagnose_dataset_source.py  只读 dataset/ 原始文件，与 ① 无依赖，可随时跑
    ⑥ verify_stage1.py            必须最后跑 —— 依赖 ①③④⑤ 的全部产物

    ③ 与 ④ 彼此独立，可以并行；⑤ 也可以挪到最前面。
    这里按 ① → ⑥ 串行执行，是为了让终端日志顺序与论文「数据流」章节一致。

用法
----
    python scripts/run_stage1.py                  # 缺什么跑什么；产物已存在则跳过
    python scripts/run_stage1.py --dry-run        # 只打印将要执行的命令，不真的执行
    python scripts/run_stage1.py --force          # 忽略「产物已存在」，全部重跑
    python scripts/run_stage1.py --from content   # 从某一步开始（跳过它前面的步骤）
    python scripts/run_stage1.py --only verify --verify-full
    python scripts/run_stage1.py --list           # 只列出步骤、依赖与产物，不执行

退出码
------
    0 = 所选步骤全部成功（若包含 verify，则要求验收 0 项 FAIL）
    1 = 某一步失败，或验收存在 FAIL
    2 = 前置文件缺失（会明确提示应先跑哪一个步骤）
"""

import argparse
import json
import os
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，中文会乱码
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_yaml(path: str) -> dict:
    """读 YAML 配置。缺 PyYAML 时返回空字典，后面走默认路径兜底。"""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


CFG = load_yaml(os.path.join(ROOT, "configs", "data.yaml"))

# ---------- 从配置里取路径，取不到就用默认值 ----------
# 之所以不硬编码路径：万一以后 processed_dir 改名，这里能自动跟着变。
_PROC_CFG = (CFG.get("output") or {}).get("processed_dir", "./data/processed")
_FEAT_CFG = (CFG.get("output") or {}).get("feature_dir", "./data/features")
PROC = _PROC_CFG.lstrip("./").replace("\\", "/") or "data/processed"
FEAT = _FEAT_CFG.lstrip("./").replace("\\", "/") or "data/features"
SEQ_FILE = (CFG.get("output") or {}).get("sequence_file", "seq_dataset.pkl")
CONTENT_VEC = (CFG.get("output") or {}).get("content_vec_file", "content_vec_512.npy")
COLD_OUT = (CFG.get("cold_start") or {}).get("output", "cold_start_subset.pkl")
_MODEL_NAME = (CFG.get("content_feature") or {}).get(
    "model_name", "distilbert-base-multilingual-cased")
_MODEL_DIR = (CFG.get("content_feature") or {}).get(
    "model_dir", f"./models/content_encoder/pretrained/{_MODEL_NAME}")
MODEL_DIR = _MODEL_DIR.lstrip("./").replace("\\", "/")

# 权重是否"已就绪"的判据：config.json + 任一种权重文件。
# 用文件存在性而不是目录存在性——下到一半中断的目录也是存在的。
MODEL_MARKERS = [f"{MODEL_DIR}/config.json"]
MODEL_WEIGHT_NAMES = ["model.safetensors", "pytorch_model.bin"]


def model_ready() -> bool:
    """本地 DistilBERT 权重是否已经下载完整。"""
    if not all(os.path.exists(os.path.join(ROOT, m)) for m in MODEL_MARKERS):
        return False
    return any(os.path.exists(os.path.join(ROOT, MODEL_DIR, w))
               for w in MODEL_WEIGHT_NAMES)


# =====================================================================
# 步骤定义 —— 这是本脚本的核心，顺序、依赖、产物都在这里
# =====================================================================
# 每个步骤的字段含义：
#   key      命令行引用用的短名（--from / --only 用）
#   title    打印出来的中文标题
#   script   scripts/ 下的脚本文件名
#   args     透传给该脚本的参数
#   requires 前置文件（相对 ROOT）；缺失则判定为"顺序跑错了"
#   outputs  产物文件（相对 ROOT）；全部存在则默认跳过该步
#   enabled  可选：返回 False 表示这一步当前不需要执行（例如权重已就绪）
#   always   可选：True 表示即使产物存在也要重跑（验收属于这一类）
STEPS = [
    dict(
        key="preprocess",
        title="① 清洗 + 序列构造 + 交叉校验（1-A）",
        script="preprocess.py",
        args=["--config", "configs/data.yaml", "--taxonomy", "configs/genre_taxonomy.yaml"],
        requires=[
            "dataset/animes.csv",
            "dataset/ratings.npy",
            "dataset/ratings.csv",
            "dataset/id_to_genreids.json",
            "dataset/dataset.pkl",
            "configs/data.yaml",
            "configs/genre_taxonomy.yaml",
        ],
        outputs=[
            f"{PROC}/anime_meta.parquet",
            f"{PROC}/anime_detailed_tags.parquet",
            f"{PROC}/item_stats.parquet",
            f"{PROC}/user_stats.parquet",
            f"{PROC}/{SEQ_FILE}",
            f"{PROC}/preprocess_report.json",
            f"{PROC}/validation_report.json",
        ],
    ),
    dict(
        key="encoder",
        title="② 下载 DistilBERT 权重（一次性，约 520MB，不入库）",
        script="download_content_encoder.py",
        args=[],
        requires=[],
        outputs=[f"{MODEL_DIR}/config.json", f"{MODEL_DIR}/model.safetensors"],
        # 权重已就绪时整步跳过；这是唯一一个"缺了不影响别的步骤"的步骤
        enabled=lambda: not model_ready(),
    ),
    dict(
        key="content",
        title="③ 内容特征库：DistilBERT[CLS] -> PCA 512 维（1-B）",
        script="build_content_vectors.py",
        args=["--dim", "512", "--reduce", "pca", "--batch-size", "64"],
        requires=[
            f"{PROC}/anime_meta.parquet",
            f"{PROC}/anime_detailed_tags.parquet",
            f"{PROC}/{SEQ_FILE}",           # 仅用于对齐池内矩阵，缺了会跳过该产物
        ],
        outputs=[
            f"{FEAT}/{CONTENT_VEC}",
            f"{FEAT}/content_vec_512_all.npy",
            f"{FEAT}/content_ids.npy",
            f"{FEAT}/content_meta.json",
        ],
    ),
    dict(
        key="coldstart",
        title="④ 冷启动新番子集 + 专用负样本（1-D）",
        script="build_cold_start_subset.py",
        args=["--config", "configs/data.yaml", "--mode", "holdout", "--min-year", "2021"],
        requires=[
            f"{PROC}/item_stats.parquet",
            f"{PROC}/{SEQ_FILE}",
        ],
        outputs=[
            f"{PROC}/{COLD_OUT}",
            f"{PROC}/cold_start_negatives.npy",
            f"{PROC}/cold_start_report.json",
        ],
    ),
    dict(
        key="diagnose",
        title="⑤ 数据源取证（可复现证据，供论文引用）",
        script="diagnose_dataset_source.py",
        args=["--users", "1500"],
        requires=["dataset/dataset.pkl", "dataset/ratings.npy", "dataset/ratings.csv"],
        outputs=[
            f"{PROC}/dataset_source_verdict.json",
            "logs/diagnose_dataset_source.txt",
        ],
    ),
    dict(
        key="verify",
        title="⑥ 阶段一验收（独立重算，逐条 PASS/FAIL）",
        script="verify_stage1.py",
        args=[],                                  # --verify-full 时追加 --full
        requires=[
            f"{PROC}/{SEQ_FILE}",
            f"{PROC}/item_stats.parquet",
            f"{PROC}/anime_meta.parquet",
            f"{PROC}/{COLD_OUT}",
            f"{PROC}/cold_start_negatives.npy",
            f"{PROC}/dataset_source_verdict.json",
            f"{FEAT}/{CONTENT_VEC}",
            f"{FEAT}/content_vec_512_all.npy",
            f"{FEAT}/content_ids.npy",
            f"{FEAT}/content_meta.json",
        ],
        outputs=[f"{PROC}/stage1_acceptance.json"],
        always=True,        # 验收永远重跑：产物存在不代表结论仍然成立
    ),
]

KEYS = [s["key"] for s in STEPS]


# =====================================================================
# 辅助函数
# =====================================================================
def rel_exists(rel: str) -> bool:
    """判断相对 ROOT 的文件是否存在且非空（0 字节视为未生成）。"""
    p = os.path.join(ROOT, rel)
    return os.path.isfile(p) and os.path.getsize(p) > 0


def missing(rels) -> list:
    """返回一批相对路径中"不存在"的那些。"""
    return [r for r in rels if not rel_exists(r)]


def human(sec: float) -> str:
    """把秒数格式化成 1m23s 这种好读的形式。"""
    sec = int(round(sec))
    return f"{sec}s" if sec < 60 else f"{sec // 60}m{sec % 60:02d}s"


def hr(ch: str = "=") -> None:
    print(ch * 72, flush=True)


# =====================================================================
# 主流程
# =====================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="阶段一流水线一键执行（按依赖顺序调用 6 个脚本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="步骤 key：" + " / ".join(KEYS),
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令")
    ap.add_argument("--force", action="store_true", help="忽略「产物已存在」，全部重跑")
    ap.add_argument("--from", dest="from_key", metavar="KEY",
                    help="从该步骤开始执行（跳过它之前的步骤）")
    ap.add_argument("--only", metavar="KEY[,KEY...]", help="只执行指定的步骤（逗号分隔）")
    ap.add_argument("--verify-full", action="store_true",
                    help="给验收步骤追加 --full（多 2 分钟，跑权威源比对）")
    ap.add_argument("--list", action="store_true", help="只列出步骤与依赖，不执行")
    args = ap.parse_args()

    # ---------- 只列步骤 ----------
    if args.list:
        hr()
        print("阶段一流水线步骤（按依赖顺序）")
        hr()
        for i, s in enumerate(STEPS, 1):
            st = "跳过（权重已就绪）" if not s.get("enabled", lambda: True)() else "待执行"
            print(f"\n[{i}] {s['title']}")
            print(f"    key      : {s['key']}")
            print(f"    脚本     : scripts/{s['script']} {' '.join(s['args'])}")
            print(f"    前置     : {', '.join(s['requires']) or '（无）'}")
            print(f"    产物     : {', '.join(s['outputs'])}")
            print(f"    当前状态 : {st}")
        return 0

    # ---------- 解析选择范围 ----------
    # 默认全部；--only 优先，其次 --from 截取尾部
    if args.only:
        want = [k.strip() for k in args.only.split(",") if k.strip()]
        bad = [k for k in want if k not in KEYS]
        if bad:
            print(f"未知步骤 key: {bad}；可选: {KEYS}")
            return 2
        chosen = [s for s in STEPS if s["key"] in want]
    elif args.from_key:
        if args.from_key not in KEYS:
            print(f"未知步骤 key: {args.from_key}；可选: {KEYS}")
            return 2
        chosen = STEPS[KEYS.index(args.from_key):]
    else:
        chosen = list(STEPS)

    # ---------- 计划：决定每一步"跑 / 跳过" ----------
    # 判定顺序很重要：先看 enabled（例如权重已存在），再看产物是否齐全。
    plan = []
    for s in chosen:
        if not s.get("enabled", lambda: True)():
            plan.append((s, "skip", "本地权重已就绪，无需下载"))
            continue
        if not args.force and not s.get("always"):
            done = [o for o in s["outputs"] if rel_exists(o)]
            if len(done) == len(s["outputs"]):
                plan.append((s, "skip", f"{len(done)} 个产物已存在（--force 可强制重跑）"))
                continue
            elif done:
                plan.append((s, "run", f"产物不齐 {len(done)}/{len(s['outputs'])}，重跑"))
                continue
        plan.append((s, "run", ""))

    # ---------- 打印执行计划 ----------
    hr()
    print("阶段一流水线")
    hr()
    print(f"解释器 : {sys.executable}")
    print(f"工作目录: {ROOT}\n")
    for s, act, why in plan:
        mark = "RUN " if act == "run" else "SKIP"
        print(f"  [{mark}] {s['title']}" + (f"   ({why})" if why else ""))

    # ---------- 前置检查 ----------
    # 只对真的要跑的步骤做检查：跳过的步骤缺前置也无所谓。
    for s, act, _ in plan:
        if act != "run":
            continue
        lack = missing(s["requires"])
        if lack:
            print()
            hr("!")
            print(f"前置文件缺失，无法执行「{s['title']}」：")
            for r in lack:
                print(f"    - {r}")
            # 给出"应该先跑哪一步"的提示：谁把缺失文件作为产物，就提示谁
            hints = []
            for other in STEPS:
                if other["key"] == s["key"]:
                    continue
                if any(r in other["outputs"] for r in lack):
                    hints.append(f"      先运行 --only {other['key']}（{other['script']}）")
            if hints:
                print("  建议：")
                for h in hints:
                    print(h)
            hr("!")
            return 2

    # ---------- 执行 ----------
    results = []          # (步骤标题, 返回码, 耗时)
    for s, act, _ in plan:
        if act != "run":
            continue

        cmd = [sys.executable, os.path.join(ROOT, "scripts", s["script"])] + list(s["args"])
        # 验收步骤支持 --full 透传
        if s["key"] == "verify" and args.verify_full:
            cmd.append("--full")

        print()
        hr()
        print(f">>> {s['title']}")
        print("    " + " ".join(
            [os.path.basename(cmd[0])] + [os.path.relpath(c, ROOT) if os.path.isabs(c)
                                          else c for c in cmd[1:]]))
        hr()

        if args.dry_run:
            results.append((s["title"], None, 0.0))
            continue

        t0 = time.time()
        # 不捕获输出：让子进程直接写终端，进度条与日志即时可见
        rc = subprocess.call(cmd, cwd=ROOT)
        el = time.time() - t0
        results.append((s["title"], rc, el))

        if rc != 0:
            print()
            hr("!")
            print(f"步骤失败（退出码 {rc}）：{s['title']}")
            print("后续步骤已中止 —— 依赖它的产物可能不可信，不要继续往下跑。")
            hr("!")
            break

        # 产物存在性检查：这一步"成功了"但没产出文件，说明脚本行为异常
        lack = missing(s["outputs"])
        if lack and s["key"] != "encoder":
            print()
            print(f"  ⚠ 退出码为 0，但以下产物不存在：{', '.join(lack)}")

    # ---------- 汇总 ----------
    print()
    hr()
    print("汇总")
    hr()
    for title, rc, el in results:
        if rc is None:
            print(f"  [DRY ] {title}")
        elif rc == 0:
            print(f"  [ OK ] {title}   ({human(el)})")
        else:
            print(f"  [FAIL] {title}   退出码 {rc}   ({human(el)})")

    # 验收结果单独回显：这是阶段一唯一"通过/不通过"的权威判据。
    # dry-run 时必须屏蔽 —— 否则会把上一次的旧结论当成这次的结果读出来，
    # 这正好是验收环节要防的那类错误（拿过期产物当依据）。
    acc = os.path.join(ROOT, PROC, "stage1_acceptance.json")
    if args.dry_run:
        print("\n  （--dry-run：未执行任何步骤，故不回显验收结论）")
    elif os.path.exists(acc):
        try:
            with open(acc, encoding="utf-8") as f:
                d = json.load(f)
            print(f"\n  验收结果：{d.get('passed')}/{d.get('total')} 项通过，"
                  f"{d.get('failed')} 项失败（{d.get('elapsed_sec')}s）")
            for c in d.get("checks", []):
                if not c.get("ok"):
                    print(f"    ✗ {c.get('check')}  {c.get('detail')}")
        except Exception as e:
            print(f"  （读取验收结果失败：{e}）")

    failed = any(rc not in (0, None) for _, rc, _ in results)
    if args.dry_run:
        print("\n（dry-run 结束：以上为将要执行的步骤，未实际运行）")
    elif not failed:
        print("\n阶段一流水线执行完毕，全部步骤成功。")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
