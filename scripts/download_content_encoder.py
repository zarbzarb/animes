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
    """构造一个走本机代理的 urllib opener。

    不用 huggingface_hub 的原因见模块 docstring：代理下它会拿到空响应体。
    这里直接手写 ProxyHandler，把 http/https 都指向本机代理。
    """
    h = {"http": PROXY, "https": PROXY}
    return urllib.request.build_opener(urllib.request.ProxyHandler(h))


def fetch(op, url: str, timeout: int = 60) -> bytes:
    """用给定 opener 抓一个 URL 的原始字节。伪装 UA 避免被网关拦截。"""
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with op.open(req, timeout=timeout) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser(description="离线拉取内容编码器权重")
    ap.add_argument("--model", default="distilbert-base-multilingual-cased")
    ap.add_argument("--out", default=None, help="默认 models/content_encoder/pretrained/<model>")
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
        # API 挂了不影响下载本身，退回 NEEDED 固定清单继续
        print(f"  API 查询失败({e})，回退到固定清单", flush=True)
        siblings = NEEDED

    # 只下载 NEEDED 里列出的那 6 个文件（其余如 onnx/tf 权重本项目用不到）
    todo = [f for f in NEEDED if f in siblings] or NEEDED
    for name in todo:
        dst = os.path.join(out, name)
        # 断点续传：已存在且非空就跳过，重跑不会重复下载 517MB 的大文件
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  已存在，跳过: {name} ({os.path.getsize(dst)/1024**2:.1f} MB)", flush=True)
            continue
        url = f"https://huggingface.co/{args.model}/resolve/main/{name}"
        t0 = time.time()
        try:
            data = fetch(op, url, timeout=600)   # 大文件给到 10 分钟超时
            with open(dst, "wb") as f:
                f.write(data)
            print(f"  ✓ {name:28s} {len(data)/1024**2:8.2f} MB  {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"  ✗ {name:28s} 失败: {type(e).__name__}: {str(e)[:100]}", flush=True)
            return 1

    # 验证可加载：真正把权重读进来一次，避免"文件下载完整但加载报错"的假成功
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
