# -*- coding: utf-8 -*-
"""
实验21: 哨兵两种修法叠加——混合领域建库 + 文档内相对判定。

实验17 结果: 混合领域建库把远领域误报 99%→1%(检出不掉); 文档内相对判定把同领域
误报 32%→10% 但对远领域无效。两者针对的是不同的失败模式, 叠加应当同时拿到两个好处。
本实验只跑这一个组合, 并与实验17的三个单项在同一批文档上并列。
"""

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from sentinel import AssemblySentinel  # noqa: E402
from demo_all import Fault  # noqa: E402
from eval_at_scale import make_docs  # noqa: E402
from run_expt17_multidomain import build, judge, CORPORA, LAYERS, N_DOCS  # noqa: E402

OUT = os.path.join(HERE, "expt21_combined.json")


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype=torch.float32)
    model.load_state_dict(torch.load(os.path.join(HERE, "distilgpt2_finetuned.pt"), weights_only=True))
    model.eval()
    fault = Fault(model, 4)
    sent = AssemblySentinel(model, LAYERS)

    docs = {k: make_docs(tok, os.path.join(HERE, v), N_DOCS, 8000 + i,
                          skip_frac=0.55 if k == "in" else 0.5)
            for i, (k, v) in enumerate(CORPORA.items())}

    print("建库(三领域混合)…", flush=True)
    build(model, sent, tok, list(CORPORA.values()), seed=9000)
    res = {}
    for pd in (False, True):
        name = "multi_perdoc" if pd else "multi_only"
        entry = {}
        for src, dl in docs.items():
            fault.off()
            ab = 0
            with torch.no_grad():
                for ch in dl:
                    _, v, _ = judge(sent, ch, perdoc=pd)
                    ab += int(v != "正常")
            entry[f"误报率_{src}"] = ab / len(dl)
        fault.set("drop", 0.3)
        det = hit4 = 0
        with torch.no_grad():
            for ch in docs["in"]:
                _, v, loc = judge(sent, ch, perdoc=pd)
                det += int(v != "正常")
                hit4 += int(v == "局部" and loc == 4)
        fault.off()
        entry["检出率_砍30%"] = det / len(docs["in"])
        entry["正确指向L4"] = hit4 / len(docs["in"])
        res[name] = entry
        print(f"{name}: 误报 in={entry['误报率_in']:.0%} near={entry['误报率_near']:.0%} "
              f"far={entry['误报率_far']:.0%} | 检出={entry['检出率_砍30%']:.0%} "
              f"指向L4={entry['正确指向L4']:.0%} ({time.time()-t0:.0f}s)", flush=True)

    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print(f"完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
