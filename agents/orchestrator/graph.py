# -*- coding: utf-8 -*-
"""A0 的流程编排图（`docs/agents.md` §六 的"典型协作场景"落地）。

为什么是"静态图 + 参数化"而不是 LangGraph / 自由编排
--------------------------------------------------
`docs/agent-interaction-protocol.md` §二 明确论证过：**不用 AutoGen 式自由对话**，
理由是"流程不可预测、无法保证延迟、答辩讲不清"。所以这里把每个意图的
调用顺序**写死成表**，好处有三：

1. **延迟可预测**：每条流程的最大串行深度是常数，可以逐段核预算；
2. **可审计**：`PIPELINES` 就是"系统会调用哪些 Agent"的唯一答案；
3. **降级明确**：每个步骤标了 `optional`，链路上谁挂了会走哪条分支一目了然。

M1（同步）与 M2（异步）的边界
---------------------------
本模块**只做 M1**：A0 收到的请求要走完这些步骤才能返回。
行为触发的增量重排（协议 §6.3）走 Redis Stream，不在 `invoke()` 里 ——
否则一个用户点"收藏"会让 API 多等几百毫秒。

预算分配（协议 §5.2）
-------------------
A0 总预算 800ms。各步骤的 `timeout_ms` 之和会超过 800ms（因为 A2∥A3 并行、
且 A5 可跳），真正约束是**串行关键路径**：

    A1(150) → max(A2(300), A3(300)) → A4(100) = 550ms

留 250ms 给网关、序列化与抖动。`PIPELINES` 里的预算逐项可核对。

⚠️ 这组预算与早期文档写的 `A2(200) / A3(150)` **不一样**，是照实测改的：
按旧值打 12 次真实请求，A3 有 **9 次**超时降级（它的 p95 本身就有 141ms，
而旧预算是 150ms）。实测数据与安全系数的推导写在下面 `BUDGET_A2_MS` 附近。
**改这些数字前请先重新测一遍**，不要照文档里的旧值调。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence

__all__ = ["PIPELINES", "Pipeline", "PipelineResult", "Stage", "Step", "run_pipeline"]


@dataclass(frozen=True)
class Step:
    """一次 Agent 调用。"""

    agent: str
    action: str
    timeout_ms: int
    optional: bool = False
    # 跳过该步骤的条件（对当前已聚合的结果求值）
    skip_if: Optional[str] = None


@dataclass(frozen=True)
class Stage:
    """一个阶段：`steps` 之间**并行**（`asyncio.gather`），阶段之间串行。"""

    steps: tuple[Step, ...]
    optional: bool = False


@dataclass
class Pipeline:
    intent: str
    stages: tuple[Stage, ...]
    description: str = ""


# ---------------------------------------------------------------- 流程定义

# ⚠️ 单步预算是**实测出来的，不是照文档拍的**。
#
# 2026-09-14 在 dev 档真实服务上连打 12 次 `orchestrate.route`
# （`top_n=20`，A2 与 A3 在同一 `asyncio.gather` 里并发）得到的分布（ms）：
#
#     A1  p50 0    p95 0    max 0      （缓存命中；未命中时约 20~40）
#     A2  p50 145  p95 149  max 177     ← 序列召回（torch 前向）
#     A3  p50 133  p95 141  max 141     ← 内容召回（numpy 矩阵乘）
#     A4  p50 0    p95 1    max 1
#     A5  p50 0    p95 0    max 0       （无 LLM 时是纯模板拼装）
#
# 预算是把 p95 乘上 ~2 倍安全系数得来的。为什么必须留得比"看起来够用"宽：
# A2 与 A3 是**两个 CPU 密集任务并发**，在 GIL 上互相顶牛，快的那条会被
# 慢的拖慢。早期把 A3 设成 150ms（≈它的 p95 141ms）时，12 次里 **9 次**
# 超时降级 —— 单看每一步都"就差一点点"，整体却是个每 4 次坏 3 次的接口，
# 而且只在 `meta.degraded` 里留一行痕迹，页面看上去完全正常。
#
# 串行关键路径（含并行）随之为 A1(150) → max(A2,A3)(300) → A4(100) = 550ms，
# 仍在 A0 的 800ms 总预算内（`OrchestratorConfig.total_timeout_ms`）。
BUDGET_A2_MS = 300
BUDGET_A3_MS = 300

PIPELINES: dict[str, Pipeline] = {
    # 首页推荐：A1 → (A2 ∥ A3) → A4 → A5（协议 §6.1）
    "RECOMMEND_FEED": Pipeline(
        intent="RECOMMEND_FEED",
        description="主链路：画像 → 双路并行召回 → 融合排序 → 解释",
        stages=(
            Stage((Step("A1", "profile.get", 150),)),
            Stage((Step("A2", "recall.sequence", BUDGET_A2_MS),
                   Step("A3", "content.retrieve", BUDGET_A3_MS, optional=True))),
            Stage((Step("A4", "rank.fusion", 100),)),
            Stage((Step("A5", "explain.generate", 200, optional=True),),
                  optional=True),
        ),
    ),
    # 分题材推荐：与首页同构，A4 带 genre_id 过滤
    "RECOMMEND_BY_GENRE": Pipeline(
        intent="RECOMMEND_BY_GENRE",
        description="按题材：画像 → 双路召回 → 融合（带 genre 约束）→ 解释",
        stages=(
            Stage((Step("A1", "profile.get", 150),)),
            Stage((Step("A2", "recall.sequence", BUDGET_A2_MS),
                   Step("A3", "content.retrieve", BUDGET_A3_MS, optional=True))),
            Stage((Step("A4", "rank.fusion", 100),)),
            Stage((Step("A5", "explain.generate", 200, optional=True),),
                  optional=True),
        ),
    ),
    # 新番专区：跳过 A2（协议 §6.2）—— 单跑 A3，没有并发顶牛，但保持同一预算
    "RECOMMEND_NEW_ANIME": Pipeline(
        intent="RECOMMEND_NEW_ANIME",
        description="新番专区：画像 → 内容召回（跳过序列召回）→ 融合（冷启 5:5）→ 解释",
        stages=(
            Stage((Step("A1", "profile.get", 150),)),
            Stage((Step("A3", "content.retrieve", BUDGET_A3_MS),)),
            Stage((Step("A4", "rank.fusion", 100),)),
            Stage((Step("A5", "explain.generate", 200, optional=True),),
                  optional=True),
        ),
    ),
    # 单条解释：A1（取画像做信号）+ A5
    "EXPLAIN_RECOMMEND": Pipeline(
        intent="EXPLAIN_RECOMMEND",
        description="单条解释：画像 → 解释生成（A4 的信号由请求体带入）",
        stages=(
            Stage((Step("A1", "profile.get", 150, optional=True),)),
            Stage((Step("A5", "explain.generate", 3000),)),
        ),
    ),
    # 兴趣看板：A1 → A6
    "ANALYZE_INTEREST": Pipeline(
        intent="ANALYZE_INTEREST",
        description="兴趣漂移：画像 → 漂移分析（内部读画像）",
        stages=(
            Stage((Step("A1", "profile.get", 150, optional=True),)),
            Stage((Step("A6", "drift.analyze", 2000),)),
        ),
    ),
    # 对话推荐：直接交给 A7（内部再调 A2/A3/A4/A5）
    "CHAT_RECOMMEND": Pipeline(
        intent="CHAT_RECOMMEND",
        description="多轮对话：A7 自行编排工具调用（含召回与融合）",
        stages=(Stage((Step("A7", "chat.reply", 5000),)),),
    ),
    # 检索：不走 Agent 链（A7 的 search_anime 是同一份实现，但检索接口
    # 直接用 gateway 更快，故这里只做「无重活」的直通）
    "SEARCH_ANIME": Pipeline(
        intent="SEARCH_ANIME",
        description="动漫检索：无 Agent 依赖，由 API 层直查（此处仅占位）",
        stages=(),
    ),
    # 系统查询：A9
    "SYSTEM_QUERY": Pipeline(
        intent="SYSTEM_QUERY",
        description="系统状态：A9 的在线快照",
        stages=(Stage((Step("A9", "eval.online_snapshot", 3000),)),),
    ),
}


Serializer = Callable[[dict], dict]          # 预留：将来加"大对象走 Redis ref"


@dataclass
class PipelineResult:
    """流水线产物。`stages` 逐阶段记录，供 A0 组装 `agent_chain` 与归因。"""

    values: dict[str, Any] = field(default_factory=dict)      # 步骤键 → payload
    chain: list[str] = field(default_factory=list)            # 调用顺序
    degraded: list[dict] = field(default_factory=list)        # 降级记录
    errors: list[dict] = field(default_factory=list)

    def get(self, agent: str, action: str, default: Any = None) -> Any:
        return self.values.get(f"{agent}:{action}", default)


def _step_key(s: Step) -> str:
    return f"{s.agent}:{s.action}"


async def run_pipeline(
    pipeline: Pipeline,
    *,
    call: Callable[[Step, "PipelineResult"], Awaitable[Optional[dict]]],
    on_result: Optional[Callable[[Step, Optional[dict], Optional[Exception]], None]] = None,
) -> PipelineResult:
    """执行一条流水线。

    参数
    ----
    call      实际调用函数：`call(step, result) -> payload | None`。
              由 `OrchestratorAgent` 实现（它负责走 `client.call_agent`，
              并按 `result` 里已有的上游产出**动态构造**本步入参）。
              **`None` 表示该步骤失败且不可降级**。
    on_result 每步结束的回调（A0 用它累积 `agent_chain` 与降级记录）。

    ⚠️ 为什么 `call` 拿到的是 `result` 而不是"预先算好的入参"：
    A4 的入参必须由 A2/A3 的**实际产出**构造（候选集合），A5 的入参又要
    由 A4 的第一条产出构造。预先算入参等于把流程写第二遍，而且两处
    一旦不一致就会出现"A4 收到的是空候选"这种只在运行时暴露的错。
    把累积结果交给 `call`，编排逻辑只有一处。

    并行语义
    -------
    同一 `Stage` 内的步骤并发（`asyncio.gather`）。单个步骤失败：
    * `optional=True` → 记录降级，继续；
    * `optional=False` → 记录错误，**但流程继续**（协议 §5.1 的 L3/L4
      由 A0 的 `fallback` 决定是否整链兜底）。这里不中断是有意的：
      A4 拿到"只有行为路候选"仍然能给出结果，比直接 500 好得多。
    """

    async def _one(step: Step) -> tuple[str, Optional[dict], Optional[Exception]]:
        try:
            out = await call(step, result)
            return _step_key(step), out, None
        except Exception as exc:                # 单步异常不炸整条链
            return _step_key(step), None, exc

    result = PipelineResult()
    for stage in pipeline.stages:
        outs = await asyncio.gather(*(_one(s) for s in stage.steps))
        for step, (key, value, exc) in zip(stage.steps, outs):
            result.chain.append(step.agent)
            if exc is not None:
                result.errors.append({"step": key, "error": f"{type(exc).__name__}: {exc}",
                                      "optional": step.optional})
                if on_result:
                    on_result(step, None, exc)
                continue
            if value is None:
                result.degraded.append({"step": key, "reason": "returned_none",
                                        "optional": step.optional})
                if on_result:
                    on_result(step, None, None)
                continue
            result.values[key] = value
            if isinstance(value, dict) and value.get("used_fallback"):
                result.degraded.append({"step": key,
                                        "reason": value.get("degraded_reason") or "fallback"})
            if on_result:
                on_result(step, value, None)
    return result
