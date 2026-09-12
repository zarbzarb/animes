# -*- coding: utf-8 -*-
"""下载 DistilBERT 权重到项目本地目录（离线可用）。

huggingface_hub 走本机代理时获取 tokenizer_config.json 会返回空体导致 JSON 解析失败，
故改为直接用 urllib + 代理逐文件抓取，落到 models/content_encoder/pretrained/ 下，
之后 transformers 用本地路径加载，完全离线、可复现。

用法：python scripts/download_content_encoder.py [--model distilbert-base-multilingual-cased]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROXY = "http://127.0.0.1:12450"

NEEDED = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "special_tokens_map.json",
    "model.safetensors",
]


def opener():
    h = {"http": PROXY, "https": PROXY}
    return urllib.request.build_opener(urllib.request.ProxyHandler(h))


def fetch(op, url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with op.open(req, timeout=timeout) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-multilingual-cased")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(ROOT, "models", "content_encoder", "pretrained", args.model)
    os.makedirs(out, exist_ok=True)
    op = opener()

    print(f"目标目录: {out}", flush=True)
    # 先用 API 确认文件清单（顺带验证代理可用）
    try:
        api = json.loads(fetch(op, f"https://huggingface.co/api/models/{args.model}"))
        siblings = [s["rfilename"] for s in api.get("siblings", [])]
        print(f"仓库文件数: {len(siblings)}", flush=True)
    except Exception as e:
        print(f"  API 查询失败({e})，回退到固定清单", flush=True)
        siblings = NEEDED

    todo = [f for f in NEEDED if f in siblings] or NEEDED
    for name in todo:
        dst = os.path.join(out, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  已存在，跳过: {name} ({os.path.getsize(dst)/1024**2:.1f} MB)", flush=True)
            continue
        url = f"https://huggingface.co/{args.model}/resolve/main/{name}"
        t0 = time.time()
        try:
            data = fetch(op, url, timeout=600)
            with open(dst, "wb") as f:
                f.write(data)
            print(f"  ✓ {name:28s} {len(data)/1024**2:8.2f} MB  {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"  ✗ {name:28s} 失败: {type(e).__name__}: {str(e)[:100]}", flush=True)
            return 1

    # 验证可加载
    import torch
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(out)
    mdl = AutoModel.from_pretrained(out)
    print(f"加载校验通过: hidden={cfg.hidden_size} layers={cfg.n_layers} "
          f"vocab={cfg.vocab_size} params={sum(p.numel() for p in mdl.parameters())/1e6:.1f}M", flush=True)
    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
