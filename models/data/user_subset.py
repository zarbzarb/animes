"""确定性嵌套用户抽样 —— models/data/user_subset.py

职责
----
给定「用户 id 数组」与「目标比例」，返回「保留哪些用户」的布尔掩码。
档位制的实现基础，被 models/sasrec/dataset.py 与 scripts/train.py 调用。

本模块遵守模型层纪律：纯函数、无 IO、不读环境变量、不依赖全局随机状态。
（只用 hashlib + numpy 做确定性计算，因此结果在任何机器、任何进程都一致。）


为什么要"嵌套"而不是各档独立随机抽样
------------------------------------
configs/scale.yaml 规定 5% ⊂ 20% ⊂ 100%。若每档各抽各的，
5% 档调出来的超参照搬到 20% 档时面对的是另一批用户 —— 等于换了数据集，
超参迁移就失去依据，"小档调参大档直接用"这条省时策略直接作废。
嵌套抽样保证换档只是【放大验证】，不是【重做实验】。


为什么按 user_id 哈希，而不是按数组下标抽样
------------------------------------------
用户顺序来自 dataset.pkl 的 umap，正常情况固定；但一旦预处理重跑、
过滤阈值微调，顺序就可能变化。若按位置抽样，同一个 ratio 会选出
完全不同的用户集，历史实验结果立刻失去可比性。
按 uid 哈希则与数组顺序【完全无关】：

    nested_user_subset(uids, 0.05) 选中的用户
        == nested_user_subset(shuffle(uids), 0.05) 选中的用户（同一集合）

这也是本模块能被单元测试严格锁定的原因。


实现原理（三步）
----------------
1. 对每个 uid 计算 64 位稳定哈希键 key(uid, seed)；
2. 按 key 升序排序，得到一个【与 ratio 无关】的全序 order；
3. 取 order 的前 k = round(n * ratio) 个位置为 True。

因为第 2 步的全序不依赖 ratio，所以任何两个比例必然满足包含关系：
    ratio_a <= ratio_b  =>  集合(ratio_a) ⊆ 集合(ratio_b)
"""

from __future__ import annotations

import hashlib

import numpy as np

# 哈希命名空间：将来若更换抽样逻辑（例如改成按活跃度分层抽样），
# 必须同时升级这个版本号，以明确告知"新旧结果不可比"。
_HASH_NAMESPACE = "anirec.user_subset.v1"


def _uid_key(uid: int, seed: int) -> int:
    """把 (seed, uid) 映射为稳定的 64 位无符号整数。

    不使用 Python 内置 hash()：其字符串哈希受 PYTHONHASHSEED 影响，
    跨进程不稳定；blake2b 是密码学哈希，输出确定且分布均匀。
    """
    payload = f"{_HASH_NAMESPACE}:{int(seed)}:{int(uid)}".encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def user_hash_keys(user_ids, seed: int = 42) -> np.ndarray:
    """返回与 user_ids 等长的 uint64 哈希键数组（顺序与输入一一对应）。

    注意：这只是中间产物，真正的抽样依据是它的排序结果。
    """
    user_ids = np.asarray(user_ids)
    return np.fromiter(
        (_uid_key(u, seed) for u in user_ids),
        dtype=np.uint64,
        count=len(user_ids),
    )


def subset_order(user_ids, seed: int = 42) -> np.ndarray:
    """返回按哈希键升序排列的【位置索引】。

    关键性质：结果只依赖 (user_ids, seed)，与 ratio 无关，
    因此不同 ratio 生成的掩码天然嵌套。换档位时不要重算这个顺序的语义。
    """
    keys = user_hash_keys(user_ids, seed)
    # kind="stable"：64 位键撞车概率极低，但显式声明可保证
    # 在极端情况下（例如 uid 被重复传入）仍然完全可复现。
    return np.argsort(keys, kind="stable")


def _resolve_keep_count(n: int, ratio: float) -> int:
    """把比例换算成「保留多少个用户」，并做边界收敛。

    边界约定：
        ratio <= 0  -> 0        （显式要求空集）
        ratio >= 1  -> n        （全量）
        0 < ratio < 1 -> max(1, round(n * ratio))
            下限取 1 是为了避免小数据集（n 很小）因四舍五入取到 0，
            导致"抽了但一个都没留下"这种难排查的空集。
    """
    if n <= 0:
        return 0
    if ratio <= 0:
        return 0
    if ratio >= 1:
        return n
    return max(1, min(n, int(round(n * ratio))))


def nested_user_subset(user_ids, ratio: float, seed: int = 42) -> np.ndarray:
    """确定性嵌套抽样的主入口。

    参数
    ----
    user_ids : array-like[int]
        用户 id 数组（通常是数据集里的 umap 或过滤后的用户列表）。
        函数只关心「值」，不关心顺序。
    ratio : float
        保留比例，取值 (0, 1]。1.0 表示全量。
    seed : int
        抽样种子。不同 seed 得到不相关但同样嵌套的一系列子集。

    返回
    ----
    np.ndarray[bool]，与 user_ids 等长。
    True 表示该位置的用户被保留。可直接用于布尔索引：

        mask = nested_user_subset(uids, 0.05)
        train_uids = uids[mask]

    ⚠️ 抽样只作用于【用户】，物品池恒为全量 —— 这是档位制的硬约束
       （见 configs/scale.yaml 的 invariants.sample_users_not_items），
       不要在调用方顺手把物品也过滤了。

    性能：内部对 user_ids 逐个做哈希，实测 1,306,691 个用户约 1.6s。
    因此**应在训练启动时调用一次并复用返回的掩码**，
    不要放进 per-epoch / per-batch 的循环里。
    若需要多个比例，请用 subset_order() + mask_from_order()，哈希成本只付一次。
    """
    user_ids = np.asarray(user_ids)
    order = subset_order(user_ids, seed)
    return mask_from_order(len(user_ids), order, ratio)


def mask_from_order(n: int, order: np.ndarray, ratio: float) -> np.ndarray:
    """复用已经算好的全序生成掩码，避免重复支付哈希计算成本。

    同一批 user_ids 若需要多个比例（例如同时要训练掩码与评估掩码），
    应先用 subset_order() 算一次全序，再对本函数调用多次：

        order = subset_order(uids, seed=42)      # 约 1.6s（1.3M 用户），只算一次
        train_mask = mask_from_order(len(uids), order, 0.05)
        eval_mask  = mask_from_order(len(uids), order, 0.10)

    因为复用同一个全序，各比例之间仍然严格嵌套；
    同时评估掩码与训练掩码来自同一全序的前缀，两者天然不会互相污染
    （评估用户集合不依赖训练用户集合，只是碰巧共用了一把排序尺子）。
    """
    mask = np.zeros(n, dtype=bool)
    n_keep = _resolve_keep_count(n, ratio)
    if n_keep:
        mask[order[:n_keep]] = True
    return mask


def kept_user_ids(user_ids, ratio: float, seed: int = 42) -> np.ndarray:
    """便捷封装：直接返回被保留的用户 id（保序，与输入顺序一致）。

    保序的意义：下游若假设"用户顺序稳定"（例如按位置缓存 embedding），
    这里不会打乱它。
    """
    user_ids = np.asarray(user_ids)
    return user_ids[nested_user_subset(user_ids, ratio, seed)]


def is_nested(masks) -> bool:
    """校验一组掩码是否按规模严格嵌套（小集合必须是所有大集合的子集）。

    供单元测试与训练启动自检使用。传入的掩码应已按保留规模【升序】排列，
    例如 [mask_5pct, mask_20pct, mask_100pct]。

    返回 False 说明档位之间不再嵌套，"小档调参大档直接用"的前提已破坏，
    应当立刻停止实验并排查（通常是改了哈希命名空间或种子）。
    """
    masks = list(masks)
    if len(masks) <= 1:
        return True
    for smaller, larger in zip(masks[:-1], masks[1:]):
        if not np.all(larger[smaller]):
            return False
    return True


# =====================================================================
# 档位 -> 「哪些用户训练、哪些用户评估」
# =====================================================================
def rows_from_order(order: np.ndarray, ratio: float) -> np.ndarray:
    """从已有全序取前 `ratio` 比例的位置，返回**升序**行号。

    为什么返回升序而不是保持全序：下游（负采样、评估输入构造、滑窗数据集）
    都按行号顺序遍历，升序能让索引缓存友好、且让"评估子集"在日志里可读。
    注意升序**不会**破坏嵌套性 —— 嵌套是"哪些用户入选"的性质，
    与输出顺序无关。

    全序上的前缀而不是另抽一次，是本项目「嵌套抽样」这条不变式的唯一实现
    （见 `configs/scale.yaml` 的 invariants）。
    """
    n = int(len(order))
    n_keep = _resolve_keep_count(n, float(ratio))
    if n_keep <= 0:
        return np.zeros(0, dtype=np.int64)
    if n_keep >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.asarray(order[:n_keep], dtype=np.int64))


def resolve_scale_users(
    user_ids,
    user_ratio: float,
    eval_user_ratio: float,
    seed: int = 42,
) -> tuple:
    """档位制下「训练用户行号 / 评估用户行号」的**唯一实现**。

    返回 `(train_rows, eval_rows)`，两者都是**升序**行号数组。

    两条口径（与 `configs/scale.yaml` 的三条不变式一致，勿改）
    -------------------------------------------------------
    1. **只抽用户，不抽物品** —— 物品池恒为全量。
    2. **确定性嵌套** —— 各档位都基于同一个 `subset_order(uids, seed)`
       的前缀，故 5% ⊂ 10% ⊂ 100%。
    3. **评估用户从本档位用户中抽，比例相对本档位** ——
       `|eval_rows| = |train_rows| × eval_user_ratio`（**不是**相对全量用户池）。
       同档位内固定不变，因此跨 epoch 的指标可比、best model 不会被选错。

    为什么必须抽成公共函数：这段逻辑原先在 `scripts/train.py` 里，
    `scripts/diagnose_popularity_bias.py` 又抄了一份；M2.7 的基线脚本是
    第三个调用方。三份拷贝意味着"某天有人只改了其中一处"——
    而这类不一致**不会报错**，只会让两个脚本报出不可比的指标。
    """
    ids = np.asarray(user_ids)
    n = int(ids.size)
    order = subset_order(ids, seed)
    train_rows = rows_from_order(order, float(user_ratio))

    # 评估用户 = 本档位用户的子集。注意要的是"同一把哈希尺子的前缀"，
    # 所以从 order 里取，而不是从已升序的 train_rows 里取。
    n_eval_cap = max(1, int(round(train_rows.size * float(eval_user_ratio))))
    in_train = np.zeros(n, dtype=bool)
    in_train[train_rows] = True
    eval_rows = np.sort(
        np.asarray(order[in_train[order]][:n_eval_cap], dtype=np.int64))
    return train_rows, eval_rows


__all__ = [
    "user_hash_keys",
    "subset_order",
    "mask_from_order",
    "nested_user_subset",
    "kept_user_ids",
    "is_nested",
    "rows_from_order",
    "resolve_scale_users",
]
