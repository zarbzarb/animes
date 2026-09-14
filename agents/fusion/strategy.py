# -*- coding: utf-8 -*-
"""A4 的融合与重排算法 —— **纯函数，无 IO、无 LLM**。

这是全链路唯一决定"最终顺序"的地方，因此口径必须能逐项复算。

为什么融合前必须**各自 min-max 归一**
------------------------------------
行为分是 `user_vec · item_vec` 的内积，量级随维度与训练状态漂移（实测
TopK 分数在 2~6 之间）；内容分是余弦，恒在 [-0.2, 0.4] 附近。直接按
`0.7*行为 + 0.3*内容` 加权，行为分会**完全主导**（数量级差 10 倍），
内容路等于没加。归一化之后两路都在 [0,1]，权重才真的是"7:3"。

为什么把"归一化"放在这里而不是各自 Agent 里
-----------------------------------------
min-max 的分母依赖**候选集合**。A2 只知道自己那 135 个候选，
不知道自己会被和谁合并；同样 A3 也不知道。只有 A4 见到完整集合，
才有一套统一的归一化基准。这也是 `docs/agents.md` 把"归一化"划给 A4 的原因。

MMR 的相似度为什么用**题材向量**
-------------------------------
标准 MMR 需要 item-item 相似度矩阵。本项目的物品相似度有现成的 ItemCF
（`models/baselines/itemcf.py`），但它是**离线全量**产物、且只覆盖训练池，
而 A3 的候选是新番（可能不在 ItemCF 覆盖内）。题材多热向量的余弦
是无依赖、覆盖全库、且语义上正是"看起来像不像"的代理。代价是粒度粗，
但 MMR 只负责**打断**同类聚集，不需要精确相似度。

`final_score` 到底是什么分（这条口径改过一次，别改回去）
-------------------------------------------------------
`final_score` 是**决定位次的那个分数**，不是"融合出来的相关性"。
两者的区别在 MMR 上是本质的：MMR 的目标是
`λ·rel - (1-λ)·max_sim(已选)`，被它放进第 t 位的物品，
**恰恰可能是一个 rel 更低但更不重复的物品**。

实测踩到的坑（2026-09-14，dev 档 feed 冒烟）：
`#2 Guilty Crown rel=0.742` 排在了 `#3 Fate/Zero Season 2 rel=0.969` 前面。
这不是排序错，是**正常的 MMR 行为** —— Fate/Zero S2 与已入选的
Fate/Zero 同一题材体系，`sim≈1` 吃满重复惩罚。但当时的实现把
`final_score` 直接导出成 `rel`（MMR 的**输入**），于是落库的
`final_score` 与 `rank_no` 对不上号。

为什么必须对齐（而不是放宽断言）
* `rank_no` 要落 `recommend_result`、要被 A9 用于"位置-CTR"分析：
  位置与分数不一致，离线指标就无法解释，实验结论不可信；
* 这是 `quota_repair` docstring 里早就写下的设计不变量。

所以 `select_with_quota` 在返回前会把 `final_score` **重新导出**成
MMR 目标值（见 `_export_final_scores`）。MMR 目标值有一条可证明的性质：
**贪心过程中严格单调不增**。证明（`v_t` = 第 t 步选中的目标值）：

    记 f_t(i) = λ·rel(i) - (1-λ)·max_sim(S_t, i)
    第 t+1 步的已选集 S_{t+1} ⊇ S_t ⇒ max_sim 只增 ⇒ f_{t+1}(i) ≤ f_t(i)
    剩余池 R_{t+1} ⊆ R_t ⇒ v_{t+1} = max_{R_{t+1}} f_{t+1}
                                  ≤ max_{R_{t+1}} f_t ≤ max_{R_t} f_t = v_t  ∎

`quota_repair` 只做"**保序替换 + 补位到尾部**"，不改变本次相对顺序
（输出是 MMR 全序的子序列＋位置更靠后的替补），所以单调性穿过它仍然成立。
`_export_final_scores` 用的是 min-max，是**单调不减**变换，
再叠一次也不会破坏单调性 —— 但**不能**改成"按 rel 重排"，
那就把 MMR 的多样性收益整条抹掉了。
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

__all__ = [
    "export_final_scores",
    "fuse_scores",
    "genre_signals",
    "minmax",
    "mmr_select",
    "quota_repair",
    "select_with_quota",
]


def genre_signals(item_genre_ids: Sequence[int],
                  profile_top_genres: Sequence[dict],
                  ) -> tuple[list[str], float]:
    """行为路候选的题材信号 → `(matched_genres, genre_overlap)`。

    为什么 A4 得自己算这个（这是修一个"一屏 20 条同一句解释"的坑）
    ----------------------------------------------------------
    `matched_genres` / `genre_overlap` 原本**只从 A3（内容召回）的候选带过来**
    （见 `agent.py::_assemble` 只读 `m.get(...)`）。而线上真实流量里，
    融合后的 Top20 几乎全由 A2（行为召回）贡献 —— 于是这两个字段**恒为空**，
    解释模板一路落到最后的兜底分支，每条推荐都是同一句
    「根据你近期的观看偏好，为你推荐这部作品」，看不出为什么是这一部。

    这不是模型能力问题，而是**信号没被算出来**：A4 手上已经有物品题材
    （`anime_facts` 取回的 `genres`）与用户题材偏好（画像 `top_genres`），
    两者的交集是纯集合运算 —— 不是向量运算，仍然守得住
    「ADRs-1：A4 不做任何向量计算」的边界。

    `genre_overlap` 定义为「**该物品覆盖了用户多少偏好权重**」：
    `sum(命中题材 strength) / sum(全部题材 strength)`，落在 [0,1]。
    用权重和而不是命中个数：命中一个 strength=1.0 的主兴趣，
    显然比命中三个 strength=0.05 的边缘兴趣更配得上"因为你在追这类"。
    """
    if not item_genre_ids or not profile_top_genres:
        return [], 0.0

    strength: dict[int, tuple[float, str]] = {}
    for g in profile_top_genres:
        gid = g.get("genre_id")
        if gid is None:
            continue
        strength[int(gid)] = (float(g.get("strength") or 0.0),
                              str(g.get("genre") or ""))
    total = sum(v[0] for v in strength.values())
    if total <= 0:
        return [], 0.0

    hits = {int(x) for x in item_genre_ids} & set(strength)
    if not hits:
        return [], 0.0
    # 按用户偏好强度降序排名字：说"因为你在看热血战斗"比
    # 先提一个 strength=0.02 的边缘题材更贴近用户自己的认知
    ordered = sorted(hits, key=lambda g: -strength[g][0])
    names = [strength[g][1] for g in ordered if strength[g][1]]
    overlap = sum(strength[g][0] for g in ordered) / total
    return names, round(min(1.0, overlap), 4)


def minmax(values: Sequence[float], eps: float = 1e-9) -> list[float]:
    """min-max 归一化到 [0,1]。

    ⚠️ 三个边界都实测碰到过：
    * 空序列 → 返回空（不能返回 [0]）；
    * 全部相等 → 返回**全 1.0**；返回全 0 会让该路在融合中彻底静默，
      而这恰恰是"该路只有唯一候选"的常见情形（A3 只召回 1 条）。
      全 1.0 的语义是"这条路内部没有区分度，交给另一路决定"，更合理。
    * 长度为 1 → 返回 [1.0]（同上）。
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo <= eps:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def fuse_scores(norm_behavior: float, norm_content: float, *,
                n_interactions: Optional[int], w_behavior: float, w_content: float,
                w_behavior_cold: float, w_content_cold: float,
                cold_threshold: int) -> float:
    """双路加权。候选交互数 < 阈值 → 用 5:5 权重（协议 §A4 第 ③ 步）。

    `n_interactions` 未知（无元数据）时**按冷启动处理**：
    保守地给内容路更高权重，总比让一条没有任何行为证据的候选
    被行为分（此时恒为 0）压到最底下要好。
    """
    cold = n_interactions is None or int(n_interactions) < int(cold_threshold)
    wb = w_behavior_cold if cold else w_behavior
    wc = w_content_cold if cold else w_content
    total = wb + wc
    if total <= 0:
        wb, wc, total = 0.5, 0.5, 1.0
    return round((wb / total) * norm_behavior + (wc / total) * norm_content, 6)


def _cosine_multi_hot(a: Iterable[int], b: Iterable[int]) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / ((len(sa) ** 0.5) * (len(sb) ** 0.5))


def mmr_select(items: list[dict], k: int, *, lam: float,
               genre_of: Callable[[int], Sequence[int]]) -> list[dict]:
    """MMR 贪心重排：`argmax λ·rel - (1-λ)·max_sim(已选)`。

    `items` 每项必须含 `anime_id` 与 `final_score`（已归一到 [0,1]）。
    ⚠️ `genre_of` 必须返回**题材 id 列表（int）**，不是题材名 ——
    `_cosine_multi_hot` 会对元素做 `int(x)`（类型不符会直接 ValueError，
    这是刻意的"响亮失败"：A4 的题材全部来自 `gateway.anime_genres()`，
    契约就是 `list[int]`，一旦有人塞进名字，说明上游换源了，
    静默按字符串比对会得到全 0 相似度、MMR 静默失效，比崩掉难查得多）。
    解释文案用的题材名是另一条路（`genre_signals` 接画像的 `strength`）。

    返回重排后的列表（长度 ≤ k）。**贪心而非全局最优**：全局最优需要
    枚举子集（NP-hard），而贪心的近似比有理论保证且 O(k·n）。

    选中项会被写入三个诊断字段（下游 `_export_final_scores` 依赖它们）：

    * `mmr_rel`   —— 归一化后的相关性（MMR 的**输入**，即旧版导出的 final_score）
    * `mmr_sim`   —— 与已选集合的最大题材相似度（重复度惩罚项）
    * `mmr_score` —— 目标值 `λ·rel - (1-λ)·sim`，**真正决定位次的分数**

    ⚠️ `mmr_score` 沿返回顺序**单调不增**（证明见模块 docstring），
    但 `mmr_rel` **不保证**单调 —— 这正是曾经把 final_score 导错的原因。
    调用方若要用 `final_score` 表示位次，必须走 `mmr_score`，不要用 `mmr_rel`。
    """
    if k <= 0 or not items:
        return []
    pool = list(items)
    rel = {int(it["anime_id"]): float(it.get("final_score", 0.0)) for it in pool}
    genres = {int(it["anime_id"]): list(genre_of(int(it["anime_id"]))) for it in pool}

    selected: list[dict] = []
    remaining = pool
    while remaining and len(selected) < int(k):
        best, best_val, best_sim = None, None, 0.0
        for it in remaining:
            iid = int(it["anime_id"])
            if selected:
                sim = max(_cosine_multi_hot(genres[iid], genres[int(s["anime_id"])])
                          for s in selected)
            else:
                sim = 0.0
            val = float(lam) * rel[iid] - (1.0 - float(lam)) * sim
            if best_val is None or val > best_val + 1e-12:
                best, best_val, best_sim = it, val, sim
        remaining = [it for it in remaining if it is not best]
        # 诊断字段写在**调用方传入的 dict 上**（与旧版写 mmr_rel 的行为一致）：
        # 这些 dict 由 A4 的 `_fuse` 构造，每轮调用都是新对象，不会跨请求串味。
        best["mmr_rel"] = round(rel[int(best["anime_id"])], 6)  # type: ignore[index]
        best["mmr_sim"] = round(float(best_sim), 6)             # type: ignore[index]
        best["mmr_score"] = round(float(best_val), 6)           # type: ignore[index]
        selected.append(best)  # type: ignore[arg-type]
    return selected


def quota_repair(ordered: list[dict], k: int, *, key: str = "interest_id",
                 per_group: int = 1) -> list[dict]:
    """在**已排好序**的列表上做配额修复：把"名额被吃光的组"补进 TopK。

    返回新的 TopK 列表（保持原有相对顺序）。

    为什么是"修复"而不是"预先插入"
    ----------------------------
    最初的实现是「先每组取 Top1 拼到队首，再 MMR 填剩余」。实测发现它破坏了
    排序语义：`interest_id` 升序拼接会让第 3 个位置出现一个分数明显更低的
    物品，而 `rank_no` 是要落库、要被 A9 用于"位置-CTR"分析的 ——
    **位置与分数不一致，离线指标就没法解释**。

    现在的做法：先在完整候选上跑 MMR 得到"多样化相关性序"，取前 K，
    再对"某组在 TopK 里条数 < per_group"的情况做**替换**（用该组里
    名次最高的物品，换掉一个来自"超额组"的末位物品）。顺序仍是 MMR 序，
    位置语义保持成立。

    ⚠️ 与 `_export_final_scores` 的配合（别改坏）
    替换是**保序**的：被保留的原 TopK 成员保持相对次序（它们是 MMR 序的
    子序列），替补来自 MMR 全序里**更靠后**的位置，因此重排后符合
    「`rank_no` 升序 ⟹ `mmr_score` 单调不增」。最后的
    `chosen.sort(key=pos_of)` 也是按 MMR 全序位置排，不是在按分数重排。
    """
    if per_group <= 0 or not ordered:
        return list(ordered[:int(k)])

    chosen = list(ordered[:int(k)])
    if len(chosen) < int(k):
        return chosen

    # 每个组允许的上限 = per_group；超额组（>per_group）才有"名额"可让出。
    # 若所有组都恰好 == per_group，则没有可让出的名额，跳过修复。
    def count_in(lst: list[dict], gid: object) -> int:
        return sum(1 for it in lst if it.get(key) == gid)

    chosen_ids = {id(it) for it in chosen}
    groups = sorted({it.get(key) for it in ordered},
                    key=lambda g: (g is None, str(g)))
    for gid in groups:
        have = count_in(chosen, gid)
        if have >= int(per_group):
            continue
        need = int(per_group) - have
        donors = [it for it in ordered
                  if it.get(key) == gid and id(it) not in chosen_ids]
        for cand in donors[:need]:
            # 从末位往前找"超额组"的成员让位
            replaced = False
            for pos in range(len(chosen) - 1, -1, -1):
                g = chosen[pos].get(key)
                if g == gid:
                    continue
                if count_in(chosen, g) > int(per_group):
                    chosen_ids.discard(id(chosen[pos]))
                    chosen[pos] = cand
                    chosen_ids.add(id(cand))
                    replaced = True
                    break
            if not replaced:
                break

    pos_of = {id(it): i for i, it in enumerate(ordered)}
    chosen.sort(key=lambda it: pos_of.get(id(it), 1 << 30))
    return chosen


def export_final_scores(chosen: list[dict], *, base: Sequence[float],
                        eps: float = 1e-9) -> None:
    """把 `final_score` 从"相关性"改写为"**决定位次的分数**"（就地修改）。

    为什么要有这一步
    ----------------
    MMR 会把"rel 略低但更不重复"的物品提到前面（这就是多样性的定义）。
    若把 MMR 的**输入** `rel` 当作 `final_score` 导出，落库后
    `final_score` 与 `rank_no` 顺序矛盾（实测 #2 rel=0.742 排在
    #3 rel=0.969 前面），A9 的"位置-CTR"分析就失去分母的物理意义。
    详见模块 docstring。

    `base` 传**归一化基准**，必须是 `mmr_select` 的**完整返回序列**
    （不是截断后的 TopK）：基准覆盖全池时，TopK 的分数才能反映出
    "它在整个候选池里有多好"，而不是被压缩成"20 个里排第几"。
    再用 min-max 把它压到 [0,1]，与 `docs/api-specification.md`
    的 `final_score ∈ [0,1]` 契约一致。

    单调性为什么不会被打乱
    ----------------------
    min-max 是单调不减变换；`chosen` 是 `base` 对应序列按
    `mmr_score` 单调不增顺序的**子序列**（配额修复只做保序替换 +
    补位到尾部）。故 `chosen` 上的 `final_score` 仍然单调不增。
    """
    if not chosen:
        return
    lo, hi = (min(base), max(base)) if base else (0.0, 0.0)
    span = hi - lo
    for it in chosen:
        v = float(it.get("mmr_score") or 0.0)
        it["final_score"] = round((v - lo) / span, 6) if span > eps else 1.0


def select_with_quota(items: list[dict], k: int, *, lam: float,
                      genre_of: Callable[[int], Sequence[int]],
                      key: str = "interest_id", per_group: int = 1) -> list[dict]:
    """A4 的实际选择入口：**MMR 全序 → 取前 K → 配额修复 → 校准 final_score**。

    MMR 在**完整候选**上跑（而不是先截断再跑）：先截断会让"被截掉的长尾
    里那个唯一的第 4 路兴趣"永远没机会进榜，配额修复也就无米可炊。

    最后一步 `export_final_scores` 是不可省的：它保证返回列表满足
    「`rank_no` 升序 ⟹ `final_score` 单调不增」这条落库不变量。
    调用方**不需要**、也**不应该**再自己改写 `final_score`。
    """
    if k <= 0 or not items:
        return []
    ordered = mmr_select(items, len(items), lam=lam, genre_of=genre_of)
    chosen = quota_repair(ordered, int(k), key=key, per_group=per_group)
    export_final_scores(
        chosen, base=[float(it.get("mmr_score") or 0.0) for it in ordered])
    return chosen
