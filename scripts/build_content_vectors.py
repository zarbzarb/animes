# -*- coding: utf-8 -*-
"""阶段 1-B：构建动漫内容语义向量库（DistilBERT）。

用法
----
    # 默认：全部 20,237 条动漫，512 维，PCA 降维
    python scripts/build_content_vectors.py

    # 指定设备 / 批量 / 不降维（直接输出 768 维）
    python scripts/build_content_vectors.py --device cpu --batch-size 64 --reduce none

    # 只编码进入推荐池的 15,687 个物品
    python scripts/build_content_vectors.py --subset pool

产物（data/features/）
----------------------
    content_vec_512.npy        (15,687, 512)  float32 —— 与 smap 索引对齐（第 i-1 行 = 物品 idx i）
    content_vec_512_all.npy    (20,237, 512)  float32 —— 全量动漫，含未入池新番
    content_ids.npy            (20,237,)      int32   —— 全量矩阵每行对应的 anime_id
    content_pca.npz            PCA 投影矩阵（可复现，新番上线时用同一矩阵投影）
    content_meta.json          模型 / 池化 / 降维 / 维度的完整快照

关于文本（重要）
----------------
rq.md 原本要求「题材标签 + 剧情简介」，但 `dataset/animes.csv` **没有简介字段**
（列为 animeID/title/alternative_title/type/year/score/episodes/mal_url/sequel/
image_url/genres/genres_detailed）。经确认改用：

    标题 + 12 类中文题材 + genres_detailed 细分标签

`genres_detailed` 平均含数十个细分标签（如 "battle of wits"、"time skip"），
信息密度足以支撑语义向量；此偏差已同步记录到 docs/evaluation-plan.md。

关于降维（与文档的偏差）
------------------------
文档原写「[CLS] → 线性投影到 512 维」。未经训练的随机线性层会破坏语义结构，
故改为 **在语料上拟合 PCA 投影到 512 维**（确定性、可复现、对新增物品可复用）。
若要严格照搬文档，加 `--reduce random` 走固定种子高斯投影。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = "distilbert-base-multilingual-cased"
DETAILED_TAG_BUDGET = 24  # 细分标签最多取前 N 个，控制文本长度


def log(msg: str) -> None:
    """带时间戳打印进度（flush 保证编码长任务时能实时看到）。"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_proxy() -> None:
    """HuggingFace 直连不通（502 隧道失败），走本机代理。

    只在**在线加载**（本地权重目录缺失、回退到模型名）时才会调用；
    离线模式（本地目录存在）不需要代理，也就不会走到这里。
    设环境变量 ANIREC_NO_PROXY=1 可强制关闭。
    """
    if os.environ.get("ANIREC_NO_PROXY"):
        return
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.setdefault(k, "http://127.0.0.1:12450")


def _as_list(v) -> list:
    """把 parquet 里的列表列统一转成 Python list。

    关键坑：`anime_meta.parquet` 的 list 列读回来是 **np.ndarray**，
    此时 `x or []` 会抛 `ValueError: truth value of an array is ambiguous`。
    所以这里必须显式判断类型，不能图省事写 `v or []`。
    """
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v else []
    try:
        return list(v)
    except TypeError:
        return []


def build_texts(meta: pd.DataFrame, tags: pd.DataFrame, mode: str = "rich") -> list[str]:
    """把每条动漫拼成一段供 DistilBERT 编码的文本。

    rich 模式（默认）的拼接格式：
        标题 | 别名 | 题材：剧情、奇幻 | 标签：time skip, battle of wits, ...
    用 " | " 分隔不同字段、"、" 分隔题材、", " 分隔细分标签，让模型能分辨字段边界。

    title_only 模式只给标题，用于做消融（内容特征退化成"只有名字"）。
    """
    tag_map = dict(zip(tags["anime_id"], tags["genres_detailed"]))
    texts = []
    for _, r in meta.iterrows():
        # 标题作兜底：万一为空就写 "unknown"，避免产出全空文本（编码后是无意义向量）
        title = str(r.get("title") or "").strip() or "unknown"
        alt = str(r.get("alternative_title") or "").strip()
        cn = _as_list(r.get("genres_cn"))
        if mode == "title_only":
            texts.append(title)
            continue
        parts = [title]
        # alt.lower() != "nan"：原始数据里有字面量字符串 "nan"，要当缺失处理
        if alt and alt.lower() != "nan":
            parts.append(alt)
        if cn:
            parts.append("题材：" + "、".join(str(c) for c in cn))
        # 细分标签只取前 DETAILED_TAG_BUDGET(24) 个：控制文本长度，避免超出 max_len 被截断
        dt = _as_list(tag_map.get(r["anime_id"]))
        if dt:
            parts.append("标签：" + ", ".join(str(t) for t in dt[:DETAILED_TAG_BUDGET]))
        texts.append(" | ".join(parts))
    return texts



def resolve_model(model: str, model_dir: str | None, log_fn=print) -> str:
    """决定用「本地权重目录」还是「在线模型名」加载。

    优先本地：一旦走 huggingface_hub 联网，经本机代理时拉 tokenizer_config.json
    会返回空体导致 `JSONDecodeError`，是个已复现的兼容性问题。
    改成读本地目录 + 强制离线环境变量后，整个编码过程完全离线、可复现。

    本地目录由 scripts/download_content_encoder.py 准备。
    返回值为可直接喂给 AutoModel/AutoTokenizer 的路径或模型名。
    """
    if model_dir and os.path.isdir(model_dir):
        base = os.path.basename(model_dir.rstrip("/\\"))
        if base and base != model:
            # 目录名即模型名，视为可用
            pass
        # 双保险：hub 与 transformers 都置离线，杜绝任何联网尝试
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        log_fn(f"使用本地模型目录: {model_dir}（已置离线模式）")
        return model_dir
    log_fn(f"本地目录不存在({model_dir})，回退到在线模型名: {model}")
    return model


def encode(texts: list[str], model_name: str, batch_size: int, max_len: int,
           device: str) -> np.ndarray:
    """批量编码文本，返回 (n_texts, hidden_size) 的 float32 矩阵。

    池化方式：取 **[CLS]**（last_hidden_state 第 0 个 token）。
    DistilBERT 的 [CLS] 是句向量约定位置，比 mean-pooling 更贴近"整句语义"。
    输出维度 = 模型隐藏层维度（DistilBERT 多语言版为 768），降维在 reduce_dim() 里做。
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    log(f"加载 tokenizer / model: {model_name} (device={device})")
    tok = AutoTokenizer.from_pretrained(model_name)
    # .eval() 关闭 dropout，保证同一文本每次编码结果完全一致（可复现的前提）
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()

    embs = np.zeros((len(texts), mdl.config.hidden_size), dtype=np.float32)
    n_batch = (len(texts) + batch_size - 1) // batch_size
    t0 = time.time()
    with torch.no_grad():                        # 只推理不训练，省显存且更快
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            # padding=True 让同批长度对齐；truncation + max_length 防超长文本
            enc = tok(chunk, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            out = mdl(**enc)
            cls = out.last_hidden_state[:, 0, :]           # [CLS] 池化
            embs[i:i + len(chunk)] = cls.float().cpu().numpy()
            b = i // batch_size + 1
            # 每 20 批报一次进度 + 按当前速度线性外推剩余时间
            if b % 20 == 0 or b == n_batch:
                el = time.time() - t0
                log(f"  编码 {b}/{n_batch} 批  已用 {el:.0f}s  预计剩余 {el / b * (n_batch - b):.0f}s")
    log(f"编码完成，耗时 {time.time() - t0:.0f}s，形状 {embs.shape}")
    return embs


def reduce_dim(embs: np.ndarray, dim: int, method: str, seed: int, pca_path: str):
    """把 768 维 [CLS] 向量降到 dim(512) 维。返回 (降维后矩阵, 投影参数字典或 None)。

    三种模式：
        none   不降维，原样返回 768 维
        random 固定种子的高斯随机投影（严格照搬文档"线性投影"的写法，仅作对照）
        pca    在语料上拟合 PCA（默认）—— 确定性、可复现、新番可用同一矩阵投影

    为什么默认 PCA 而不是随机线性层：未经训练的随机投影会破坏语义结构，
    而降维本意是保留主要语义方向。PCA 的投影矩阵可以落盘（content_pca.npz），
    后续新番上线时用同一矩阵投影即可，不需要重新拟合整个语料。
    """
    if method == "none" or dim == embs.shape[1]:
        return embs, None
    if method == "random":
        rng = np.random.default_rng(seed)
        # 标准差取 1/sqrt(d_in)：让投影后各维方差与输入量级相当
        W = rng.normal(0, 1.0 / np.sqrt(embs.shape[1]), size=(embs.shape[1], dim)).astype(np.float32)
        proj = {"method": "random", "seed": seed, "W": W}
        return (embs @ W).astype(np.float32), proj
    # pca
    log(f"拟合 PCA: {embs.shape[1]} -> {dim}（n={embs.shape[0]}）")
    mu = embs.mean(axis=0, keepdims=True)   # 中心化（PCA 必须）
    X = embs - mu
    # 用 SVD 实现 PCA，避免依赖 sklearn 的大矩阵协方差
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[:dim].T.astype(np.float32)          # (d_in, dim)，取前 dim 个主成分方向
    proj = {"method": "pca", "mean": mu.astype(np.float32), "W": W}
    # 累计解释方差比 = 前 dim 个奇异值平方和 / 全部奇异值平方和（衡量降维损失了多少信息）
    evr = float((S[:dim] ** 2).sum() / (S ** 2).sum())
    log(f"  PCA 前 {dim} 个主成分累计解释方差比 = {evr:.4f}")
    proj["explained_variance_ratio"] = evr
    return ((X @ W)).astype(np.float32), proj


def main() -> int:
    """完整流程：读元数据 → 拼文本 → DistilBERT 编码 → PCA 降维 → 归一化 → 落盘。"""
    ap = argparse.ArgumentParser(description="构建内容语义向量库")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "data.yaml"))
    ap.add_argument("--model", default=DEFAULT_MODEL, help="模型名（本地目录缺失时的回退）")
    ap.add_argument("--model-dir", default=os.path.join(
        ROOT, "models", "content_encoder", "pretrained", DEFAULT_MODEL),
        help="本地权重目录（优先使用；由 download_content_encoder.py 准备）")
    ap.add_argument("--dim", type=int, default=512, help="降维后的维度")
    ap.add_argument("--reduce", choices=["pca", "random", "none"], default="pca",
                    help="降维方式：pca=语料拟合(默认) / random=随机投影(对照) / none=不降维")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128, help="tokenize 后的最大长度")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--subset", choices=["all", "pool"], default="all",
                    help="all=全部20237条, pool=仅smap中的15687个物品")
    ap.add_argument("--text-mode", choices=["rich", "title_only"], default="rich",
                    help="rich=标题+题材+细分标签；title_only=仅标题（消融用）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    proc = os.path.join(ROOT, cfg["output"]["processed_dir"].lstrip("./"))
    feat = os.path.join(ROOT, cfg["output"]["feature_dir"].lstrip("./"))
    os.makedirs(feat, exist_ok=True)

    # 元数据按 anime_id 升序排列 —— 这决定了后续「全量矩阵」的行序约定，
    # 一旦这里不排序，content_ids.npy 与矩阵行就会错位（验收脚本 E 段会抓到）。
    meta = pd.read_parquet(os.path.join(proc, "anime_meta.parquet"))
    tags = pd.read_parquet(os.path.join(proc, "anime_detailed_tags.parquet"))
    meta = meta.sort_values("anime_id").reset_index(drop=True)
    log(f"动漫元数据 {len(meta)} 条")

    # --subset pool：只编码推荐池内的 15687 个物品（用于快速迭代调试）
    if args.subset == "pool":
        import pickle
        with open(os.path.join(proc, cfg["output"]["sequence_file"]), "rb") as f:
            smap = pickle.load(f)["smap"]
        keep = set(int(k) for k in smap.keys())
        meta = meta[meta["anime_id"].isin(keep)].reset_index(drop=True)
        log(f"  限定推荐池物品: {len(meta)} 条")

    texts = build_texts(meta, tags, args.text_mode)
    log("文本示例（前 2 条）:")
    for t in texts[:2]:
        log("   " + t[:180])

    # ---- 设备选择 ----
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"自动选择设备: {device}")

    # ---- 模型加载：本地优先，本地缺失才联网（联网才需要代理）----
    model_ref = resolve_model(args.model, args.model_dir, log)
    if not os.path.isdir(args.model_dir or ""):
        ensure_proxy()  # 仅在线加载时才需要代理

    # ---- 编码：768 维 [CLS] 向量 ----
    embs = encode(texts, model_ref, args.batch_size, args.max_len, device)

    # ---- 降维 ----
    reduced, proj = reduce_dim(embs, args.dim, args.reduce, args.seed,
                               os.path.join(feat, "content_pca.npz"))
    # L2 归一化（FAISS 内积即余弦）
    # 先归一化再落盘，下游（召回/融合）就不用反复归一化了。
    # 范数为 0 的行置 1 避免除零（虽然正常不会出现）。
    norm = np.linalg.norm(reduced, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    vecs = (reduced / norm).astype(np.float32)

    # ---- 落盘 1：全量矩阵（含未入池的新番，冷启动实验要用）----
    out_all = os.path.join(feat, "content_vec_%d_all.npy" % args.dim)
    np.save(out_all, vecs)
    # content_ids 记录每行对应哪个 anime_id —— 与 vecs 的行严格一一对应
    np.save(os.path.join(feat, "content_ids.npy"), meta["anime_id"].to_numpy(np.int32))
    log(f"已写出 {out_all}  {vecs.shape}")

    # ---- 落盘 2：与 smap 对齐的池内矩阵 ----
    # 约定：第 (idx-1) 行 = smap 中 idx 对应的物品（smap 的 value 从 1 开始）。
    # 模型侧拿到的物品索引就是 smap 的 idx，直接用它查表即可，无需再转换。
    pool_mat = None
    try:
        import pickle
        with open(os.path.join(proc, cfg["output"]["sequence_file"]), "rb") as f:
            smap = pickle.load(f)["smap"]
        inv = {int(k): int(v) for k, v in smap.items()}          # anime_id -> idx
        # 为「全量矩阵的每一行」找出它在池内的 idx；不属池内的记 -1
        pos = np.array([inv.get(int(a), -1) for a in meta["anime_id"]], dtype=np.int64)
        if (pos >= 0).any():
            n_items = len(smap)
            pool_mat = np.zeros((n_items, vecs.shape[1]), dtype=np.float32)
            ok = pos >= 0
            pool_mat[pos[ok] - 1] = vecs[ok]      # smap 值从 1 开始
            np.save(os.path.join(feat, cfg["output"]["content_vec_file"]), pool_mat)
            log(f"已写出池内矩阵 {pool_mat.shape}（第 i-1 行 = 物品 idx i）")
    except FileNotFoundError:
        log("未找到 seq_dataset.pkl，跳过池内矩阵（先跑 preprocess.py）")

    # ---- 落盘 3：PCA 投影矩阵 ----
    # 新番上线时用同一组 (mean, W) 投影，保证新番向量与历史向量在同一空间。
    if proj is not None and proj["method"] == "pca":
        np.savez(os.path.join(feat, "content_pca.npz"), mean=proj["mean"], W=proj["W"])

    # ---- 落盘 4：构建快照（论文「方法」一节可直接引用）----
    meta_out = dict(
        model_name=args.model,
        model_source=model_ref,
        loaded_offline=os.path.isdir(args.model_dir or ""),
        hidden_size=int(embs.shape[1]),         # 768
        pooled="cls",                           # 池化方式
        output_dim=int(vecs.shape[1]),          # 512
        reduce=args.reduce,
        text_mode=args.text_mode,
        max_len=args.max_len,
        subset=args.subset,
        n_items=int(len(meta)),
        normalized=True,
        text_source="title + genres_cn(12类) + genres_detailed",
        note="animes.csv 无剧情简介字段，故未使用 synopsis；见 docs/evaluation-plan.md 2.4",
        explained_variance_ratio=proj.get("explained_variance_ratio") if proj else None,
    )
    with open(os.path.join(feat, "content_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    log("完成 -> " + feat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
