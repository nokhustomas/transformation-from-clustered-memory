# -*- coding: utf-8 -*-
"""
规模化评测: 不再看一篇文档,而是几百篇独立文档上逐篇判定,统计误报率/检出率/定位准确率。

设置:
- 模型: 继续预训练后的 distilgpt2; 档案: 边训边建。
- 文档: 每篇 2048 token(切16块),从语料的**留出区段**取,与建库/校准区不重叠。
- 三种文本来源:
    in     = 建库同领域(torch 文档英文)  —— 正常基线,量误报率
    near   = 近邻领域(numpy/sklearn/scipy 文档) —— 换个相近领域还准不准
    far    = 远领域(法律英文) —— 领域不符能否被判成"输入问题"而非"模型故障"
- 故障(全部打在第4层,另加一组打第2层做定位): 砍活跃单元10/30%、单元卡死10/30%、
  量化4/8级、高斯噪声0.5x。每种故障都在同一批文档上跑,与健康读数配对。
- 判定沿用工具里的规则: 每层"异常块比例">30% 即该层报警;
  只有一层报警=局部(疑似内部故障), 多层同时报警=全局(疑似输入领域不符)。

同时记录每种故障对模型输出的真实伤害(held-out loss 增量),用来回答
"漏掉的是不是无害的"这个关键问题。
"""

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from sentinel import AssemblySentinel  # noqa: E402
from demo_all import Fault  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = [2, 4]
DOC_TOKENS, CHUNK = 2048, 128
N_DOCS = 150
OUT = os.path.join(HERE, "eval_at_scale_results.json")


def make_docs(tok, path, n_docs, seed, skip_frac=0.55):
    """从语料靠后段落取 n_docs 篇互不重叠的文档(避开建库/校准用的前段)。"""
    text = open(path, encoding="utf-8").read()
    ids = tok(text[int(len(text) * skip_frac):], return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(seed)
    max_start = ids.shape[0] - DOC_TOKENS - 1
    starts = torch.randperm(max(1, max_start // DOC_TOKENS), generator=g)[:n_docs] * DOC_TOKENS
    return [ids[s:s + DOC_TOKENS].view(CHUNK and -1, CHUNK) for s in starts.tolist()
            if s + DOC_TOKENS <= ids.shape[0]]


def doc_verdict(sent, chunks, thr=0.30):
    """chunks: [16, 128] 一整篇文档作为一个批次前向。返回每层异常块比例与判定。"""
    s = sent.readings(chunks)
    out = {}
    flagged = []
    for li in LAYERS:
        ns = sent._stats_for(li, CHUNK)
        bad = sum(1 for r, v in zip(s[li]["rho_per_seq"], s[li]["vmis_per_seq"])
                  if r < ns["rho_p05"] or v > ns["vmis_p95"])
        frac = bad / len(s[li]["rho_per_seq"])
        out[li] = frac
        if frac > thr:
            flagged.append(li)
    if not flagged:
        verdict, loc = "正常", None
    elif len(flagged) == 1:
        verdict, loc = "局部", flagged[0]
    else:
        verdict, loc = "全局", max(out, key=out.get)
    return out, verdict, loc


def main():
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype=torch.float32)
    model.load_state_dict(torch.load(os.path.join(HERE, "distilgpt2_finetuned.pt"), weights_only=True))
    model.eval()
    f2, f4 = Fault(model, 2), Fault(model, 4)      # 故障钩子先注册
    sent = AssemblySentinel(model, LAYERS)
    sent.load(os.path.join(HERE, "registry_ready.pt"))

    docs = {
        "in": make_docs(tok, os.path.join(HERE, "default_corpus.txt"), N_DOCS, 4242),
        "near": make_docs(tok, os.path.join(HERE, "neardomain_corpus.txt"), N_DOCS, 4243, skip_frac=0.0),
    }
    far_path = os.path.join(HERE, "fardomain_corpus.txt")
    if os.path.exists(far_path):
        docs["far"] = make_docs(tok, far_path, N_DOCS, 4244, skip_frac=0.0)
    for k, v in docs.items():
        print(f"{k}: {len(v)} 篇文档 × {DOC_TOKENS} token", flush=True)

    FAULTS = [
        ("健康", None, None, None),
        ("L4砍10%活跃单元", f4, "drop", 0.10),
        ("L4砍30%活跃单元", f4, "drop", 0.30),
        ("L4卡死10%单元", f4, "stuck", 0.10),
        ("L4卡死30%单元", f4, "stuck", 0.30),
        ("L4量化4级", f4, "quant", 4),
        ("L4量化8级", f4, "quant", 8),
        ("L4高斯噪声0.5x", f4, "noise", 0.5),
        ("L2砍30%活跃单元", f2, "drop", 0.30),
    ]

    # 故障对模型输出的真实伤害
    print("\n先量化每种故障对模型输出的伤害…", flush=True)
    harm = {}
    loss_docs = docs["in"][:20]
    for name, fobj, kind, param in FAULTS:
        f2.off(); f4.off()
        if fobj is not None:
            fobj.set(kind, param)
        tot = []
        with torch.no_grad():
            for ch in loss_docs:
                tot.append(model(ch, labels=ch).loss)
        f2.off(); f4.off()
        harm[name] = float(torch.stack(tot).mean())
        print(f"  {name:18s} loss={harm[name]:.4f}  伤害={harm[name]-harm['健康']:+.4f}", flush=True)

    results = {"n_docs": {k: len(v) for k, v in docs.items()}, "harm": harm, "conditions": {}}

    for name, fobj, kind, param in FAULTS:
        t0 = time.time()
        f2.off(); f4.off()
        if fobj is not None:
            fobj.set(kind, param)
        entry = {}
        for src, dl in docs.items():
            if src != "in" and name != "健康":
                continue                      # 故障只在同领域文档上测
            counts = {"正常": 0, "局部": 0, "全局": 0}
            loc_right = 0
            fr = {li: [] for li in LAYERS}
            with torch.no_grad():
                for ch in dl:
                    fracs, verdict, loc = doc_verdict(sent, ch)
                    counts[verdict] += 1
                    for li in LAYERS:
                        fr[li].append(fracs[li])
                    if verdict == "局部" and fobj is not None:
                        want = 2 if fobj is f2 else 4
                        loc_right += int(loc == want)
            n = len(dl)
            entry[src] = {
                "n": n,
                "正常率": counts["正常"] / n, "局部率": counts["局部"] / n, "全局率": counts["全局"] / n,
                "定位正确率": (loc_right / max(1, counts["局部"])) if fobj is not None else None,
                "平均异常块比例": {li: sum(fr[li]) / n for li in LAYERS},
            }
            print(f"[{name}|{src}] n={n} 正常{entry[src]['正常率']:.0%} "
                  f"局部{entry[src]['局部率']:.0%} 全局{entry[src]['全局率']:.0%} "
                  f"块比例 L2={entry[src]['平均异常块比例'][2]:.0%} L4={entry[src]['平均异常块比例'][4]:.0%} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        results["conditions"][name] = entry
        f2.off(); f4.off()

    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n全部完成, 总耗时 {(time.time()-t_start)/60:.1f} 分钟。", flush=True)


if __name__ == "__main__":
    main()
