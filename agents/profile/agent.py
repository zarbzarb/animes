# -*- coding: utf-8 -*-
"""A1 · 用户画像智能体（`ProfileAgent`）。

职责（`docs/agents.md` §A1）
---------------------------
维护"用户是谁"的唯一事实来源：长期兴趣画像（`user_profile`）+ 短期会话偏好（Redis）。

三条边界，越界即功能重复
-----------------------
* ❌ 不产生候选（那是 A2/A3）
* ❌ 不做排序（A4）
* ❌ 不碰模型权重（本 Agent 只有网关 + LLM，一行 torch 都没有）

降级矩阵（协议 §5.2：A1 超时 150ms、失败用 1h 前缓存画像、无缓存则空画像）
------------------------------------------------------------------------
| 情况 | 行为 |
|---|---|
| 网关不可用 | 返回空画像 + `degraded`（**不抛**，A2 会退化成"仅序列"模式） |
| 画像不存在 | 就地重算（一次），算不出来则空画像 |
| LLM 不可用 | `summary_text` 用模板兜底，其余字段照常 |
| 缓存不可用 | 绕过缓存直查 DB（`cache.backend == "none"`） |
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agents.common.base import BaseAgent
from agents.common.data import GatewayNotBound, get_gateway
from agents.common.envelope import Envelope
from agents.common.errors import AgentError, invalid_param, make
from agents.common.llm import get_llm
from agents.common.ports import get_runtime
from agents.profile import adapter, prompts
from agents.profile.config import ProfileConfig
from agents.profile.schemas import ProfileInput, SummarizeInput
from agents.profile.strategy import compute_profile

logger = logging.getLogger(__name__)

CFG = ProfileConfig()


class ProfileAgent(BaseAgent):
    agent_id = "A1"
    name = "profile"
    description = "用户画像：题材分布 / 活跃度 / 短期偏好"
    online = True

    # ------------------------------------------------------------ 入口
    async def invoke(self, payload: dict, env: Envelope) -> dict:
        action = env.header.action
        if action == "profile.get":
            return await self._get(payload, env)
        if action == "profile.refresh":
            return await self._refresh(payload, env)
        if action == "profile.summarize":
            return await self._summarize(payload, env)
        raise invalid_param(f"未支持的 action：{action}")

    # ------------------------------------------------------------ 读
    async def _get(self, payload: dict, env: Envelope) -> dict:
        inp = ProfileInput.model_validate(payload)
        cached = await self._cache_get(inp.user_id)
        if cached is not None and not inp.force_refresh:
            self._last_cache_hit = True
            return cached

        stored = self._load_stored(inp.user_id)
        if stored is not None and not inp.force_refresh and self._usable(stored):
            await self._cache_set(inp.user_id, stored)
            return stored

        # 没有画像、或已存画像是**半成品** → 就地算一次。
        # `persist=True`：半成品行要被覆盖掉，否则永远修不好。
        return await self._compute_and_store(inp.user_id, env, persist=True)

    @staticmethod
    def _usable(row: dict) -> bool:
        """已存画像是否**真的可用**（决定"直接返回"还是"重算"）。

        判据：`top_genres` 非空，**或者**该用户本来就没有任何追番记录。
        为什么必须有这个判断（真实踩过）：
        `user_profile` 里可能存在 `top_genres = []` 而 `total_records > 0` 的
        **半成品行** —— 入库脚本预热画像时题材关联(`anime_genre`)还没建好，
        于是画像被算成"零题材"落了库。旧逻辑 `if stored is not None: return stored`
        把这个空画像当成权威结果**永久返回**：用户再也不会被重算，雷达图恒为
        全 0、兴趣胶囊强度恒为 0，而且**没有任何报错** —— 只有把接口返回值
        和 `watch_record` 表对照着看才会发现。

        "有记录却零题材"不可能是终态，所以按缓存未命中处理；
        而"零记录零题材"是合法的（就是还没看番的新用户），直接复用，
        否则每来一个请求都会重算 + 重写一次画像行。
        """
        if not row:
            return False
        if row.get("top_genres") or row.get("genres"):
            return True
        return int(row.get("total_records") or 0) <= 0

    async def _refresh(self, payload: dict, env: Envelope) -> dict:
        inp = ProfileInput.model_validate(payload)
        return await self._compute_and_store(inp.user_id, env, persist=True)

    # ------------------------------------------------------------ 摘要
    async def _summarize(self, payload: dict, env: Envelope) -> dict:
        inp = SummarizeInput.model_validate(payload)
        genres = [g.model_dump() for g in inp.top_genres]
        allowed = [g["genre"] for g in genres if g.get("genre")]
        text, tokens, degraded = await self._summarize_text(
            genres, inp.activity_label, inp.preferred_types, inp.style)
        return {"user_id": inp.user_id, "summary_text": text,
                "tokens_used": tokens, "degraded": degraded,
                "prompt_ver": prompts.PROMPT_VERSION}

    async def _summarize_text(self, genres: list[dict], activity_label: str,
                              types: list[str], style: str) -> tuple[str, int, bool]:
        llm = get_llm()
        fb = prompts.fallback_summary(genres, activity_label, types)
        if not llm.available:
            return fb, 0, True
        res = await llm.complete(
            prompts.build_messages(genres, activity_label, types, style),
            temperature=0.4, max_tokens=80,
            timeout_ms=CFG.summarize_timeout_ms)
        if not res.ok:
            return fb, 0, True
        allowed = [g.get("genre", "") for g in genres if g.get("genre")]
        ok, cleaned = prompts.validate_summary(res.text, allowed_genres=allowed)
        if not ok:
            return fb, res.tokens_used, True
        return cleaned, res.tokens_used, False

    # ------------------------------------------------------------ 计算与落库
    async def _compute_and_store(self, user_id: int, env: Envelope,
                                 *, persist: bool) -> dict:
        gw = get_gateway()
        if gw is None:
            raise GatewayNotBound("A1 读不到画像来源")

        facts = adapter.collect_facts(user_id, CFG)
        profile = compute_profile(facts, CFG)

        # 摘要：模板优先（快、稳），只有配置了 LLM 才尝试润色
        text, tokens, degraded = await self._summarize_text(
            profile["top_genres"], profile["activity_label"],
            profile["preferred_types"], "concise")
        profile["summary_text"] = text
        profile["user_tag"] = profile["activity_label"]
        profile["degraded"] = degraded
        profile["tokens_used"] = tokens

        if persist:
            try:
                adapter.save_profile(user_id, {
                    k: v for k, v in profile.items()
                    if k not in ("degraded", "tokens_used", "activity_label",
                                 "preferred_era", "is_cold_start", "recent_items")
                })
            except Exception as exc:            # 落库失败不影响本次返回
                logger.warning("A1 画像落库失败（已忽略）：%s", exc)
        await self._cache_set(user_id, profile)
        return profile

    # ------------------------------------------------------------ 缓存
    async def _cache_get(self, user_id: int) -> Optional[dict]:
        """读缓存。键规范与 `server/core/cache.py::profile_key` 一致：
        `profile:{uid}`。**Agent 侧自己拼串**而不是 import 那个函数 ——
        否则又是一条 `agents/ → server/` 的反向依赖。
        `tests/test_agents/test_cache_keys.py` 会断言两侧拼出来的键相同。

        （对比：A4 的 `rec:` 键**不是**单键而是 `rec:{uid}:{scene}[:{gid}]`，
        所以它由 `rec_scope_prefix` + 前缀删除来失效，而非精确键删除。）
        """
        cache = get_runtime().cache
        try:
            data = await cache.get_json(self._key(user_id))
        except Exception as exc:                # 缓存不可用 → 绕过
            logger.debug("A1 读缓存失败（已忽略）：%s", exc)
            return None
        return dict(data) if isinstance(data, dict) else None

    async def _cache_set(self, user_id: int, profile: dict) -> None:
        cache = get_runtime().cache
        try:
            await cache.set_json(self._key(user_id), profile, ttl=CFG.cache_ttl)
        except Exception as exc:                # 缓存失败不影响返回
            logger.debug("A1 画像写缓存失败（已忽略）：%s", exc)

    @staticmethod
    def _key(user_id: int) -> str:
        return f"profile:{int(user_id)}"

    def _load_stored(self, user_id: int) -> Optional[dict]:
        try:
            row = adapter.load_cached_profile(user_id)
        except GatewayNotBound:
            raise
        except Exception as exc:
            logger.warning("A1 读取已存画像失败：%s", exc)
            return None
        if not row:
            return None
        row.setdefault("genres", row.get("top_genres"))
        row["activity_label"] = {0: "low", 1: "medium", 2: "high"}.get(
            int(row.get("activity_level", 0)), "low")
        # ⚠️ `user_profile` 表没有 `recent_items` 列（短期信息不入库），
        # 而它是 A4 `explain_signals.top_attn_items` 的唯一来源 ——
        # 不在这里补回来，凡是命中已落库画像的请求都拿不到"最近在看"，
        # 最好的那句解释模板就永远走不到（见 `adapter.recent_items` 的说明）。
        if not row.get("recent_items"):
            try:
                row["recent_items"] = adapter.recent_items(user_id, CFG)
            except Exception as exc:
                logger.warning("A1 补 recent_items 失败（置空）：%s", exc)
                row["recent_items"] = []
        return row

    # ------------------------------------------------------------ 降级
    async def fallback(self, payload: dict, env: Envelope,
                       error: AgentError) -> Optional[dict]:
        """协议 §5.2：A1 失败 → 用 1h 前缓存画像；无缓存则空画像。

        ⚠️ 这里**刻意不给空 list 之外的东西** —— 空画像会让 A2 退回"仅序列"模式，
        这是设计好的降级路径，不是错误。
        """
        user_id = int((payload or {}).get("user_id") or env.context.user_id or 0)
        cached = await self._cache_get(user_id)
        if cached is not None:
            return cached
        try:
            stored = self._load_stored(user_id)
            if stored:
                return stored
        except Exception:                       # pragma: no cover
            pass
        logger.info("A1 降级为空画像（user_id=%s, reason=%s）", user_id, error.name)
        return {
            "user_id": user_id, "top_genres": [], "activity_level": 0,
            "activity_label": "low", "watch_intensity": 0.0, "dropped_rate": 0.0,
            "preferred_types": [], "total_records": 0, "summary_text": None,
            "recent_items": [], "is_cold_start": True, "empty": True,
        }


__all__ = ["ProfileAgent"]
