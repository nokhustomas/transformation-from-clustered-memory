# -*- coding: utf-8 -*-
"""
实验17: 修掉哨兵最严重的缺陷——领域不匹配被误报成"模型内部故障"。

已实测的问题: 健康模型 + 远领域文本, 150 篇里 91% 被判成"局部异常(疑似内部故障)"。
原因: 档案只在单一领域(torch 文档英文)上建, 正常范围随之只覆盖那个领域。

三个修法同场比较(全部用同一批评测文档与同一组故障):
  single   : 单领域档案(现状)                                —— 锚点
  multi    : 用三个领域(torch文档/近邻科学库文档/系统文档法律文本)混合建档案+校准
  perdoc   : 单领域档案, 但判定改为"文档内部相对基线"(用该文档自身各层读数的
             相对模式, 而不是与全局正常范围比绝对值)
指标(每个修法都算):
  误报率: 健康模型在 in/near/far 三种文本上被判异常的比例(越低越好)
  检出率: 第4层砍30%活跃单元时被判异常且指向第4层的比例(越高越好)
真正的成功标准是**两者兼得**: 远领域误报显著下降, 而故障检出不掉。
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
from registry import AssemblyRegistry  # noqa: E402
from demo_all import Fault  # noqa: E402
from eval_at_scale import make_docs, CHUNK  # noqa: E402

LAYERS = [2, 4]
N_DOCS = 100
OUT = os.path.join(HERE, "expt17_multidomain.json")
CORPORA = {"in": "default_corpus.txt", "near": "neardomain_corpus.txt", "far": "fardomain_corpus.txt"}


def build(model, sent, tok, paths, seed, n_batches_each=60):
    """在给定语料上流式建库 + 分档校准。"""
    for li in LAYERS:
        sent.regs[li] = AssemblyRegistry(3072, 2048, 0.2, 0.05, act_frac=0.5)
    for p in paths:
        text = open(os.path.join(HERE, p), encoding="utf-8").read()
        ids = tok(text[:int(len(text) * 0.5)], return_tensors="pt").input_ids[0]
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for _ in range(n_batches_each):
                s = torch.randint(0, ids.shape[0] - CHUNK - 1, (4,), generator=g)
                b = torch.stack([ids[x:x + CHUNK] for x in s])
                model(b)
                for li in LAYERS:
                    flat = sent.acts[li].reshape(-1, 3072)[::8].float()
                    sent.regs[li].register(sent.regs[li].extract_sets(flat), values=flat)
    sent.normal_stats = {}
    for L in (32, 64, 128, 256):
        batches = []
        for p in paths:
            text = open(os.path.join(HERE, p), encoding="utf-8").read()
            ids = tok(text[:int(len(text) * 0.5)], return_tensors="pt").input_ids[0]
            g = torch.Generator().manual_seed(seed + L)
            for _ in range(15):
                s = torch.randint(0, ids.shape[0] - L - 1, (4,), generator=g)
                batches.append(torch.stack([ids[x:x + L] for x in s]))
        with torch.no_grad():
            sent.calibrate(batches, length=L)


def judge(sent, chunks, thr=0.30, perdoc=False):
    s = sent.readings(chunks)
    fr = {}
    for li in LAYERS:
        ns = sent._stats_for(li, CHUNK)
        bad = sum(1 for r, v in zip(s[li]["rho_per_seq"], s[li]["vmis_per_seq"])
                  if r < ns["rho_p05"] or v > ns["vmis_p95"])
        fr[li] = bad / len(s[li]["rho_per_seq"])
    if perdoc:
        # 文档内相对: 只有当某层显著高于该文档其它层时才算局部异常
        vals = list(fr.values())
        mx = max(fr, key=fr.get)
        others = [fr[li] for li in LAYERS if li != mx]
        rel = fr[mx] - (sum(others) / len(others))
        flagged = [mx] if (rel > 0.35 and fr[mx] > thr) else []
    else:
        flagged = [li for li in LAYERS if fr[li] > thr]
    if not flagged:
        return fr, "正常", None
    if len(flagged) == 1:
        return fr, "局部", flagged[0]
    return fr, "全局", max(fr, key=fr.get)


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
    print({k: len(v) for k, v in docs.items()}, flush=True)

    res = {}
    for setup in ("single", "multi", "perdoc"):
        paths = [CORPORA["in"]] if setup != "multi" else list(CORPORA.values())
        if setup != "perdoc" or "built" not in res:
            print(f"\n=== {setup}: 建库 ({'单领域' if len(paths)==1 else '三领域混合'}) ===", flush=True)
            build(model, sent, tok, paths, seed=9000)
        entry = {"groups": {li: sent.regs[li].n_groups for li in LAYERS}}
        pd = (setup == "perdoc")
        # 误报率
        for src, dl in docs.items():
            fault.off()
            ab = 0
            with torch.no_grad():
                for ch in dl:
                    _, v, _ = judge(sent, ch, perdoc=pd)
                    ab += int(v != "正常")
            entry[f"误报率_{src}"] = ab / len(dl)
        # 检出率
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
        res[setup] = entry
        print(f"  误报 in={entry['误报率_in']:.0%} near={entry['误报率_near']:.0%} "
              f"far={entry['误报率_far']:.0%} | 检出={entry['检出率_砍30%']:.0%} "
              f"指向L4={entry['正确指向L4']:.0%} ({time.time()-t0:.0f}s)", flush=True)

    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
