# -*- coding: utf-8 -*-
"""
第一步: 建库(在你的电脑上运行一次)。

用法:
  python3 build_registry.py                          # 用自带语料 + distilgpt2
  python3 build_registry.py --corpus my_text.txt     # 用你自己的正常文本
  python3 build_registry.py --model distilgpt2 --layers 2 4 --k 2048

做什么: 让模型读一遍"正常文本",顺带把每层 FFN 的共激活单元组登记下来,
并记录正常状态的读数分布。产物是一个 registry_<model>.pt 文件,之后体检
和演示都靠它。全程 CPU,不训练、不改模型。
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from sentinel import AssemblySentinel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def make_batches(tok, corpus_path, n_batches, seq, bs, seed):
    text = open(corpus_path, encoding="utf-8").read()
    ids_all = tok(text, return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batches):
        starts = torch.randint(0, ids_all.shape[0] - seq - 1, (bs,), generator=g)
        out.append(torch.stack([ids_all[s:s + seq] for s in starts]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--corpus", default=os.path.join(HERE, "default_corpus.txt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--build-batches", type=int, default=120)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()

    print(f"加载模型 {args.model} …", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    sent = AssemblySentinel(model, args.layers, k_max=args.k, tau=args.tau)
    t0 = time.time()
    print(f"在正常文本上建库({args.build_batches} 批)…", flush=True)
    sent.build(make_batches(tok, args.corpus, args.build_batches, args.seq, args.bs, 111),
               log_every=30)
    print("校准正常读数分布…", flush=True)
    sent.calibrate(make_batches(tok, args.corpus, max(10, args.build_batches // 4),
                                 args.seq, args.bs, 222))

    out_path = os.path.join(HERE, f"registry_{args.model.replace('/', '_')}.pt")
    sent.save(out_path)
    for li in args.layers:
        r = sent.regs[li]
        print(f"  第{li}层: {r.n_groups} 个组", flush=True)
    print(f"完成({time.time()-t0:.0f}s)。登记库已保存: {out_path}")
    print("下一步: python3 check_health.py --text \"任意一段文本\"")


if __name__ == "__main__":
    main()
