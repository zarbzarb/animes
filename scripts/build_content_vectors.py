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
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_proxy() -> None:
    """HuggingFace 直连不通（502 隧道失败），走本机代理。"""
    if os.environ.get("ANIREC_NO_PROXY"):
        return
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.setdefault(k, "http://127.0.0.1:12450")


def _as_list(v) -> list:
    """parquet 里的列表列读回来是 np.ndarray，直接 `x or []` 会抛 ValueError。"""
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v else []
    try:
        return list(v)
    except TypeError:
        return []


def build_texts(meta: pd.DataFrame, tags: pd.DataFrame, mode: str = "rich") -> list[str]:
    tag_map = dict(zip(tags["anime_id"], tags["genres_detailed"]))
    texts = []
    for _, r in meta.iterrows():
        title = str(r.get("title") or "").strip() or "unknown"
        alt = str(r.get("alternative_title") or "").strip()
        cn = _as_list(r.get("genres_cn"))
        if mode == "title_only":
            texts.append(title)
            continue
        parts = [title]
        if alt and alt.lower() != "nan":
            parts.append(alt)
        if cn:
            parts.append("题材：" + "、".join(str(c) for c in cn))
        dt = _as_list(tag_map.get(r["anime_id"]))
        if dt:
            parts.append("标签：" + ", ".join(str(t) for t in dt[:DETAILED_TAG_BUDGET]))
        texts.append(" | ".join(parts))
    return texts



def resolve_model(model: str, model_dir: str | None, log_fn=print) -> str:
    """优先用项目本地的权重目录，避免每次跑都联网（代理下 hub 拉取会失败）。

    本地目录由 scripts/download_content_encoder.py 准备。
    """
    if model_dir and os.path.isdir(model_dir):
        base = os.path.basename(model_dir.rstrip("/\\"))
        if base and base != model:
            # 目录名即模型名，视为可用
            pass
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        log_fn(f"使用本地模型目录: {model_dir}（已置离线模式）")
        return model_dir
    log_fn(f"本地目录不存在({model_dir})，回退到在线模型名: {model}")
    return model


def encode(texts: list[str], model_name: str, batch_size: int, max_len: int,
           device: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    log(f"加载 tokenizer / model: {model_name} (device={device})")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()

    embs = np.zeros((len(texts), mdl.config.hidden_size), dtype=np.float32)
    n_batch = (len(texts) + batch_size - 1) // batch_size
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tok(chunk, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            out = mdl(**enc)
            cls = out.last_hidden_state[:, 0, :]           # [CLS] 池化
            embs[i:i + len(chunk)] = cls.float().cpu().numpy()
            b = i // batch_size + 1
            if b % 20 == 0 or b == n_batch:
                el = time.time() - t0
                log(f"  编码 {b}/{n_batch} 批  已用 {el:.0f}s  预计剩余 {el / b * (n_batch - b):.0f}s")
    log(f"编码完成，耗时 {time.time() - t0:.0f}s，形状 {embs.shape}")
    return embs


def reduce_dim(embs: np.ndarray, dim: int, method: str, seed: int, pca_path: str):
    if method == "none" or dim == embs.shape[1]:
        return embs, None
    if method == "random":
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 1.0 / np.sqrt(embs.shape[1]), size=(embs.shape[1], dim)).astype(np.float32)
        proj = {"method": "random", "seed": seed, "W": W}
        return (embs @ W).astype(np.float32), proj
    # pca
    log(f"拟合 PCA: {embs.shape[1]} -> {dim}（n={embs.shape[0]}）")
    mu = embs.mean(axis=0, keepdims=True)
    X = embs - mu
    # 用 SVD 实现 PCA，避免依赖 sklearn 的大矩阵协方差
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[:dim].T.astype(np.float32)          # (d_in, dim)
    proj = {"method": "pca", "mean": mu.astype(np.float32), "W": W}
    evr = float((S[:dim] ** 2).sum() / (S ** 2).sum())
    log(f"  PCA 前 {dim} 个主成分累计解释方差比 = {evr:.4f}")
    proj["explained_variance_ratio"] = evr
    return ((X @ W)).astype(np.float32), proj


def main() -> int:
    ap = argparse.ArgumentParser(description="构建内容语义向量库")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "data.yaml"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--model-dir", default=os.path.join(
        ROOT, "models", "content_encoder", "pretrained", DEFAULT_MODEL),
        help="本地权重目录（优先使用；由 download_content_encoder.py 准备）")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--reduce", choices=["pca", "random", "none"], default="pca")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--subset", choices=["all", "pool"], default="all",
                    help="all=全部20237条, pool=仅smap中的15687个物品")
    ap.add_argument("--text-mode", choices=["rich", "title_only"], default="rich")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    proc = os.path.join(ROOT, cfg["output"]["processed_dir"].lstrip("./"))
    feat = os.path.join(ROOT, cfg["output"]["feature_dir"].lstrip("./"))
    os.makedirs(feat, exist_ok=True)

    meta = pd.read_parquet(os.path.join(proc, "anime_meta.parquet"))
    tags = pd.read_parquet(os.path.join(proc, "anime_detailed_tags.parquet"))
    meta = meta.sort_values("anime_id").reset_index(drop=True)
    log(f"动漫元数据 {len(meta)} 条")

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

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"自动选择设备: {device}")

    model_ref = resolve_model(args.model, args.model_dir, log)
    if not os.path.isdir(args.model_dir or ""):
        ensure_proxy()  # 仅在线加载时才需要代理

    embs = encode(texts, model_ref, args.batch_size, args.max_len, device)

    reduced, proj = reduce_dim(embs, args.dim, args.reduce, args.seed,
                               os.path.join(feat, "content_pca.npz"))
    # L2 归一化（FAISS 内积即余弦）
    norm = np.linalg.norm(reduced, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    vecs = (reduced / norm).astype(np.float32)

    out_all = os.path.join(feat, "content_vec_%d_all.npy" % args.dim)
    np.save(out_all, vecs)
    np.save(os.path.join(feat, "content_ids.npy"), meta["anime_id"].to_numpy(np.int32))
    log(f"已写出 {out_all}  {vecs.shape}")

    # 与 smap 对齐的池内矩阵
    pool_mat = None
    try:
        import pickle
        with open(os.path.join(proc, cfg["output"]["sequence_file"]), "rb") as f:
            smap = pickle.load(f)["smap"]
        inv = {int(k): int(v) for k, v in smap.items()}
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

    if proj is not None and proj["method"] == "pca":
        np.savez(os.path.join(feat, "content_pca.npz"), mean=proj["mean"], W=proj["W"])

    meta_out = dict(
        model_name=args.model,
        model_source=model_ref,
        loaded_offline=os.path.isdir(args.model_dir or ""),
        hidden_size=int(embs.shape[1]),
        pooled="cls",
        output_dim=int(vecs.shape[1]),
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
