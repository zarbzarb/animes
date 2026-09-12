# -*- coding: utf-8 -*-
"""滑动窗口训练集与评估输入构造（在线生成，不落盘）。

为什么在线生成
--------------
留一法切分下，每个用户长度为 `n` 的训练序列要展开成 `n` 个样本
`(prefix[0:t] → item[t])`，全量共 **109,081,471** 个（`Σ train_len`）。
按 `dtype=int64` 落盘光输入就是 1.09 亿 × 50 × 8B ≈ 43TB，完全不可行。
所以这里只保存**序列本身 + 累积偏移**，第 i 个样本按需现算——这也是
`docs/evaluation-plan.md` 2.3 明确要求的方式。

索引口径（三处数字必须自洽，改则实验不可比）
--------------------------------------------
| 量 | 值 | 出处 |
|---|---|---|
| 完整序列（train+val+test）上限 | **50** | `configs/data.yaml` 的 `sequence.max_len` |
| `val` / `test` 各占位置 | **1 / 1** | 留一法，各恰好 1 条 |
| 因此**输入序列**（prefix）上限 | **50 - 2 = 48** | 见下 |
| 窗口数 | `Σ train_len` = **109,081,471** | 基于**未截断**的 train 序列 |

关于最后两行的关系，最容易理解错：**窗口数是按完整的 train 序列算的，
被截断的是"每个样本能看多长的历史"**，不是"每个用户贡献几个样本"。
即第 `t` 个样本取「原序列中 `t` 之前的最近 48 条」作为输入，
而不是先把序列砍到 48 条再滑窗（那样窗口数会掉到 6,272 万，与阶段一
验收过的 109,081,471 对不上）。

`TARGET_SLOTS = 2` 的来由：完整序列截断到 50 条后，`test` 占最后 1 条、
`val` 占倒数第 2 条，所以训练侧与评估侧任何一次前向的输入都不会超过 48 条，
`val` 评估时输入 = `train`（≤48），`test` 评估时输入 = `train + val`（≤49）。
这样模型永远见不到超过 50 个位置的时间步，位置编码不会越界。

因果性（本文件最重要的一条不变量）
----------------------------------
第 `t` 个样本的输入**严格不含** `item[t]` 及其后的任何物品。
`tests/test_sasrec/test_dataset.py` 用「改写未来物品序列、断言该样本输入不变」
的方式锁定这条性质——它一旦被破坏，离线指标会显著虚高且极难发现。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "PAD_ITEM",
    "TARGET_SLOTS",
    "SlidingWindowDataset",
    "make_padded_id_array",
    "build_eval_inputs",
    "count_windows",
]

# 池内索引 0 保留给 PAD（`smap` 的 value 从 1 开始，见 models/data/negatives.py）
PAD_ITEM = 0

# val 与 test 各占用完整序列末尾的 1 个位置，故输入上限 = max_seq_len - 2
TARGET_SLOTS = 2


# =====================================================================
# 工具函数
# =====================================================================
def make_padded_id_array(
    sequences: Sequence[Sequence[int]],
    cap: int,
    pad_value: int = PAD_ITEM,
) -> np.ndarray:
    """把一批变长序列转成 `(N, cap)` 的左填充矩阵。

    三条口径：
    * **取尾部**：只保留每行最后 `cap` 个元素（保留最近的兴趣，这是推荐场景的
      通用做法，也与「序列过长时保留最近 50 条」的文档口径一致）；
    * **左填充**：空位补在左侧，使「最后一个位置」恒为序列的最新物品 ——
      这样模型取 `hidden[:, -1]` 或做因果注意力时，位置语义不随长度漂移；
    * **超长不报错**：静默截断到尾部 `cap` 个，长度统计由调用方另行负责。

    返回 `int64` 的 `(N, cap)` 数组（torch 的 `nn.Embedding` 要求 int64 索引）。
    """
    if cap <= 0:
        raise ValueError(f"cap 必须为正，收到 {cap}")
    n = len(sequences)
    out = np.full((n, cap), int(pad_value), dtype=np.int64)
    for i in range(n):
        s = sequences[i]
        k = len(s)
        if k == 0:
            continue
        take = cap if k > cap else k
        # s[-take:] 是尾部 take 个；放到矩阵右侧，左侧留 PAD
        out[i, cap - take:] = s[k - take:]
    return out


def _resolve_strip_mask(strip_items, n_items: Optional[int]):
    """把「要剥离的物品集合」转成按池内索引寻址的 `bytearray` 掩码。

    为什么不用 Python `set`：全量下一次构建要过 1.09 亿个元素，
    `set.__contains__` 比 `bytearray` 下标访问慢约 3 倍。
    `bytearray` 每个元素 1 字节，15,688 个物品只占 15KB。
    """
    if strip_items is None:
        return None
    arr = np.asarray(list(strip_items), dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return None
    upper = int(arr.max())
    if n_items is not None:
        upper = max(upper, int(n_items))
        if int(arr.min()) < 1 or int(arr.max()) > int(n_items):
            raise ValueError(
                f"strip_items 含越界索引：应在 [1, {n_items}]，"
                f"实际 [{arr.min()}, {arr.max()}]")
    mask = bytearray(upper + 1)
    for x in arr.tolist():
        mask[int(x)] = 1
    return mask


def _apply_strip(seq, mask):
    """按掩码过滤序列；若一个都没被过滤掉就**复用原对象**（零拷贝）。"""
    if mask is None:
        return seq
    kept = [x for x in seq if not mask[x]]
    return kept if len(kept) != len(seq) else seq


def count_windows(
    train_seqs,
    user_rows: Optional[Sequence[int]] = None,
    strip_items=None,
    n_items: Optional[int] = None,
) -> dict:
    """统计窗口数（= 训练样本数），用于与阶段一的 109,081,471 对账。

    返回 `{"n_users", "n_windows", "mean_windows", "max_windows"}`。
    这是本模块唯一的「规模口径」出口：任何脚本要报训练样本量都走这里，
    不要在别处再写一遍 `sum(len(s) for s in ...)`。
    """
    mask = _resolve_strip_mask(strip_items, n_items)
    rows = range(len(train_seqs)) if user_rows is None else user_rows
    lens = np.fromiter(
        (len(_apply_strip(train_seqs[int(r)], mask)) for r in rows),
        dtype=np.int64, count=len(rows),
    )
    total = int(lens.sum())
    return {
        "n_users": int(lens.size),
        "n_windows": total,
        "mean_windows": float(lens.mean()) if lens.size else 0.0,
        "max_windows": int(lens.max()) if lens.size else 0,
    }


# =====================================================================
# 训练集：滑动窗口
# =====================================================================
class SlidingWindowDataset(Dataset):
    """把每个用户的训练序列在线展开成 `n` 个 `(prefix → target)` 样本。

    参数
    ----
    train_seqs  按下标可取到用户训练序列的对象（list / dict 均可），
                下标是**用户行号**（`umap` 的 value，0-based）。
    max_seq_len 完整序列上限（来自 `configs/model.yaml`），默认 50。
                实际输入上限 = `max_seq_len - TARGET_SLOTS` = 48。
    pad_value   填充值，默认 0。
    user_rows   只用这些用户行（档位抽样）。None = 全部用户。
                传子集时窗口编号会重新排列，但**（行号, t）的对应关系不变**，
                因此换档位只是"少看一些用户"，不是"换一套样本"。
    strip_items 需要从序列中剥离的池内索引（E3 冷启动：剥离 year>=2021 的新番，
                等价于"这些番训练时还没上线"）。None = 不剥离。
    n_items     物品总数，用于校验 `strip_items` 是否越界（可选）。

    属性
    ----
    n_windows   窗口总数（= `Σ 过滤后 train_len`）
    input_cap   单次前向的输入长度上限（= `max_seq_len - TARGET_SLOTS`）

    用法
    ----
    >>> ds = SlidingWindowDataset(train_seqs, max_seq_len=50)
    >>> x, y = ds[0]          # x: LongTensor(48)，y: LongTensor(()) 标量
    >>> x.shape, int(y)       # 第一个窗口 prefix 为空，整行都是 PAD
    (torch.Size([48]), 1)
    """

    def __init__(
        self,
        train_seqs,
        max_seq_len: int = 50,
        pad_value: int = PAD_ITEM,
        user_rows: Optional[Sequence[int]] = None,
        strip_items=None,
        n_items: Optional[int] = None,
    ):
        if max_seq_len <= TARGET_SLOTS:
            raise ValueError(
                f"max_seq_len={max_seq_len} 扣除 {TARGET_SLOTS} 个 val/test 位置后"
                "没有剩余输入长度，请检查配置")

        self.train_seqs = train_seqs
        self.max_seq_len = int(max_seq_len)
        self.input_cap = int(max_seq_len) - TARGET_SLOTS
        self.pad_value = int(pad_value)
        self.strip_mask = _resolve_strip_mask(strip_items, n_items)
        self.n_stripped_items = (
            0 if self.strip_mask is None else int(sum(self.strip_mask)))

        # ---------- 用户行集合与序列引用 ----------
        if user_rows is None:
            self.user_rows = np.arange(len(train_seqs), dtype=np.int64)
        else:
            self.user_rows = np.asarray(user_rows, dtype=np.int64).reshape(-1)
            # 下标合法性必须分情况判：`seq_dataset.pkl` 里 train 是 **dict**
            # （key = 用户行号），此时「下标」就是 key，用 len() 卡上界会误报；
            # list / ndarray 才用 [0, len) 区间判断。
            if self.user_rows.size:
                if isinstance(train_seqs, Mapping):
                    bad = [int(r) for r in self.user_rows.tolist()
                           if int(r) not in train_seqs]
                    if bad:
                        raise ValueError(
                            f"user_rows 含不存在的用户行号：{bad[:5]} ...")
                else:
                    n_avail = len(train_seqs)
                    if self.user_rows.min() < 0 or self.user_rows.max() >= n_avail:
                        raise ValueError(
                            f"user_rows 越界：应落在 [0, {n_avail - 1}]，"
                            f"实际 [{self.user_rows.min()}, {self.user_rows.max()}]")

        # 逐用户解析出「最终参与训练」的序列引用。
        # strip 未命中时 _apply_strip 会原样返回，所以这一步几乎不额外占内存。
        self.seqs = [_apply_strip(train_seqs[int(r)], self.strip_mask)
                     for r in self.user_rows]

        # ---------- 累积偏移：窗口号 i -> (行, t) 的唯一索引结构 ----------
        # offsets[k] 是第 k 个用户第一个窗口的全局编号，offsets[N] = 总窗口数。
        # 用 int64：全量窗口数 1.09 亿，int32 上限 21.4 亿够用但没必要冒险。
        n = len(self.seqs)
        self.lens = np.fromiter((len(s) for s in self.seqs), dtype=np.int64, count=n)
        self.offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(self.lens, out=self.offsets[1:])
        self.n_windows = int(self.offsets[-1])

    # ------------------------------------------------------------------
    # 基础接口
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_windows

    def row_of(self, index: int) -> tuple:
        """把窗口号反解成 `(用户行号, t)`。`t` 是 target 在训练序列中的 0-based 位置。

        训练时用它把样本归因回用户（例如 A6 兴趣漂移 Agent 需要按用户聚合）。
        """
        i = int(index)
        if not 0 <= i < self.n_windows:
            raise IndexError(f"窗口号越界：{i} 不在 [0, {self.n_windows})")
        k = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return int(self.user_rows[k]), int(i - self.offsets[k])

    def _window(self, index: int):
        """核心：取第 index 个窗口的 `(输入序列, target)`。

        因果性由 `seq[start:t]` 这个**左闭右开**切片保证 —— 切片终点是 `t`，
        因此不含 `item[t]`，更不含其后任何物品。
        """
        i = int(index)
        if not 0 <= i < self.n_windows:
            raise IndexError(f"窗口号越界：{i} 不在 [0, {self.n_windows})")
        k = int(np.searchsorted(self.offsets, i, side="right") - 1)
        t = int(i - self.offsets[k])
        seq = self.seqs[k]

        target = int(seq[t])
        start = t - self.input_cap
        if start < 0:
            start = 0
        # 尾部对齐左填充：历史不足时左边补 PAD
        prefix = seq[start:t]
        x = np.full(self.input_cap, self.pad_value, dtype=np.int64)
        if prefix:
            x[self.input_cap - len(prefix):] = prefix
        return x, target

    def __getitem__(self, index: int):
        x, target = self._window(index)
        return torch.from_numpy(x), torch.tensor(target, dtype=torch.long)

    def __getitems__(self, indices):
        """批量取样本（PyTorch DataLoader 在 batch 内会优先走这个入口）。

        批量走一次向量化 `searchsorted`，比逐个 `__getitem__` 少 255 次二分，
        在 256 的 batch 下实测能省掉可观的 Python 开销。
        """
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        ks = np.searchsorted(self.offsets, idx, side="right") - 1
        ts = idx - self.offsets[ks]

        out_x = np.full((idx.size, self.input_cap), self.pad_value, dtype=np.int64)
        out_y = np.empty(idx.size, dtype=np.int64)
        for j in range(idx.size):
            seq = self.seqs[int(ks[j])]
            t = int(ts[j])
            start = t - self.input_cap
            if start < 0:
                start = 0
            prefix = seq[start:t]
            if prefix:
                out_x[j, self.input_cap - len(prefix):] = prefix
            out_y[j] = int(seq[t])
        return [(torch.from_numpy(out_x[j]), torch.tensor(int(out_y[j]), dtype=torch.long))
                for j in range(idx.size)]

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """规模与口径摘要，供日志与实验报告落盘（与阶段一验收口径对账用）。"""
        return {
            "n_users": int(self.lens.size),
            "n_windows": int(self.n_windows),
            "max_seq_len": int(self.max_seq_len),
            "input_cap": int(self.input_cap),
            "target_slots": int(TARGET_SLOTS),
            "pad_value": int(self.pad_value),
            "n_stripped_items": int(self.n_stripped_items),
            "mean_train_len": float(self.lens.mean()) if self.lens.size else 0.0,
            "max_train_len": int(self.lens.max()) if self.lens.size else 0,
            "truncated_users": int((self.lens > self.input_cap).sum()),
        }


# =====================================================================
# 评估输入构造
# =====================================================================
def build_eval_inputs(
    train_seqs,
    user_rows: Sequence[int],
    max_seq_len: int = 50,
    split: str = "test",
    val_items=None,
    test_items=None,
    pad_value: int = PAD_ITEM,
    strip_items=None,
    n_items: Optional[int] = None,
) -> np.ndarray:
    """构造评估时的输入序列，返回 `(N, input_cap)` 的 int64 矩阵。

    评估协议（`docs/evaluation-plan.md` 5.2）要求"用目标之前的全部历史预测目标"，
    因此两种切分的输入不同：

    | split | 输入 | 说明 |
    |---|---|---|
    | `"val"`  | `train` | 预测 val（倒数第 2 条），历史只有 train |
    | `"test"` | `train + val` | 预测 test（最后 1 条），历史含 val |

    冷启动（E3）传 `strip_items`，序列中的新番会被剥离 —— 语义是
    "站在新番上线那一刻，用户的历史里还没有这些番"。

    参数
    ----
    user_rows  要评估的用户行号（升序与否均可，输出顺序与输入顺序一致）。
    split      `"val"` / `"test"`，见上表。
    val_items  `(N,)` val 目标物品的池内索引；`split="test"` 时必填。
    test_items 保留参数，仅用于形状校验与文档完整性。
    """
    if split not in ("val", "test"):
        raise ValueError(f"split 只能是 'val' 或 'test'，收到 {split!r}")
    rows = np.asarray(user_rows, dtype=np.int64).reshape(-1)
    if split == "test" and val_items is None:
        raise ValueError("split='test' 时输入序列含 val，必须提供 val_items")

    mask = _resolve_strip_mask(strip_items, n_items)
    cap = int(max_seq_len) - TARGET_SLOTS
    n = rows.size

    out = np.full((n, cap), int(pad_value), dtype=np.int64)
    for j in range(n):
        r = int(rows[j])
        seq = _apply_strip(train_seqs[r], mask)
        if split == "test":
            vi = int(val_items[j])
            # val 目标也可能属于被剥离的新番（E3 中 val 与 test 同池），
            # 那种情况下"当时它还没上线"，不能进历史。
            if mask is None or not mask[vi]:
                seq = list(seq) + [vi]   # 仅此分支需要复制
        k = len(seq)
        if k:
            take = cap if k > cap else k
            out[j, cap - take:] = seq[k - take:]
    return out
