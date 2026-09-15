# -*- coding: utf-8 -*-
"""M2.8 编排脚本的计划构建测试 —— 只测 build_plan 的纯逻辑，不碰数据集/GPU。"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.run_experiments import (  # noqa: E402
    BLOCKED_GROUPS,
    IMPLEMENTED_GROUPS,
    build_plan,
    load_yaml,
    resolve_seeds,
)

CFG = load_yaml("configs/experiment.yaml")
SCALE_YAML = load_yaml("configs/scale.yaml")


def test_e1_plan_contains_three_seeds_per_neural_model():
    plan = build_plan(CFG, SCALE_YAML, ["E1_overall"],
                      existing_baselines=set())
    trains = [s for s in plan if s["kind"] == "train"]
    by_model = {}
    for s in trains:
        by_model.setdefault(s["model"], []).append(s["seed"])
    # sasrec / ours 各 3 种子（42/2024/2025，来自 main 档 spec）
    assert sorted(by_model["sasrec"]) == [42, 2024, 2025]
    assert sorted(by_model["ours"]) == [42, 2024, 2025]
    # gru4rec 也是 3 种子（走 run_baselines 而非 train.py）
    gru = [s for s in plan if s["kind"] == "baselines" and s["model"] == "gru4rec"]
    assert len(gru) == 3
    # 无已有报告时，一切标为"不复用"
    assert not any(s.get("reuse") for s in plan)


def test_e1_plan_marks_existing_baseline_reports_as_reuse():
    existing = {("gru4rec", "main", "test", 42), ("popularity", "main", "test", 42),
                ("itemcf", "main", "test", 42)}
    plan = build_plan(CFG, SCALE_YAML, ["E1_overall"],
                      existing_baselines=existing)
    reused = {(s["model"], s.get("desc", "")) for s in plan if s.get("reuse")}
    assert any(m == "gru4rec" for m, _ in reused)
    assert all(s.get("reuse") for s in plan
               if s["kind"] == "baselines" and s["model"] in ("popularity", "itemcf"))


def test_e2_plan_reuses_two_groups_and_trains_two():
    plan = build_plan(CFG, SCALE_YAML, ["E2_ablation"], existing_baselines=set())
    trains = [s for s in plan if s["kind"] == "train"]
    evals = [s for s in plan if s["kind"] == "eval"]
    # 只有 E2_2 / E2_3 需要新训练（E2_1/E2_4 复用 E1 权重）
    assert {s["model"] for s in trains} == {"E2_2_multi_interest", "E2_3_content_fusion"}
    reused = {s["model"] for s in evals if s.get("reuse")}
    assert reused == {"E2_1_pure_sasrec", "E2_4_full"}


def test_e3_is_blocked_with_explicit_reason():
    plan = build_plan(CFG, SCALE_YAML, ["E3_coldstart"])
    assert len(plan) == 1 and plan[0]["kind"] == "blocked"
    assert "gpu-queue" in plan[0]["desc"]
    assert "E3_coldstart" in BLOCKED_GROUPS


def test_scale_override_propagates_to_estimates():
    main = build_plan(CFG, SCALE_YAML, ["E1_overall"], existing_baselines=set())
    smoke = build_plan(CFG, SCALE_YAML, ["E1_overall"], existing_baselines=set(),
                       scale_override="smoke")
    main_hours = sum(s.get("hours") or 0 for s in main if s["kind"] == "train")
    smoke_hours = sum(s.get("hours") or 0 for s in smoke if s["kind"] == "train")
    assert smoke_hours < main_hours / 20   # smoke 档 3 epoch vs main 30 epoch × 2 倍步数


def test_resolve_seeds_prefers_override_then_config_then_scale():
    assert resolve_seeds({}, SCALE_YAML["scales"]["main"], "42,7") == [42, 7]
    assert resolve_seeds({"seeds": [1]}, SCALE_YAML["scales"]["main"], None) == [1]
    assert resolve_seeds({}, SCALE_YAML["scales"]["main"], None) == [42, 2024, 2025]
    assert resolve_seeds({}, SCALE_YAML["scales"]["dev"], None) == [42]


def test_implemented_groups_are_subset_of_yaml_keys():
    for g in IMPLEMENTED_GROUPS:
        assert g in CFG, f"{g} 在 experiment.yaml 里不存在（配置与编排脚本脱节）"
