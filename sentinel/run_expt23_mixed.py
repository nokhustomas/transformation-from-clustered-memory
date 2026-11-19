# -*- coding: utf-8 -*-
"""
实验23: 用**混合损伤 + 混合领域**一起拟合一个变换, 能否换来通用性?

实验22 结论: 单一条件拟合出的 A 是"某种损伤 × 某类数据"的专用算子——跨强度(68%)、
跨相近领域(77%)尚可, 跨损伤类型(21-33%)和跨远领域(-9%)塌掉甚至有害。
本实验检验最直接的补救: 把四种损伤 × 三个领域的 (damaged, healthy) 配对**混在一起**
拟合同一个 A, 再逐条件评测。

  若混合拟合后各条件都拿到可观恢复 => 通用修复算子可得, 只是需要多样化的拟合数据。
  若混合拟合把各条件都拉平到平庸 => 不同损伤需要的修正在线性空间里相互冲突,
     必须按损伤类型分别存(专用算子), 这本身是个干净的结构性结论。
对照: 实验22 里各条件的专用 A 的恢复率(同口径, 直接引用)。
"""

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from sentinel import AssemblySentinel, _find_act_module  # noqa: E402
from eval_at_scale import make_docs  # noqa: E402

LAYER, U = 4, 3072
FIT_DOCS_PER_CELL, EVAL_DOCS = 12, 40
RIDGE = 0.1
OUT = os.path.join(HERE, "expt23_mixed.json")
CORPORA = {"in": "default_corpus.txt", "near": "neardomain_corpus.txt", "far": "fardomain_corpus.txt"}
FAULTS = [("drop", 0.30), ("drop", 0.10), ("stuck", 0.30), ("quant", 4), ("noise", 0.5)]
FAULT_NAMES = {("drop", 0.30): "砍30%", ("drop", 0.10): "砍10%", ("stuck", 0.30): "卡死30%",
               ("quant", 4): "量化4级", ("noise", 0.5): "噪声0.5x"}


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype=torch.float32)
    model.load_state_dict(torch.load(os.path.join(HERE, "distilgpt2_finetuned.pt"), weights_only=True))
    model.eval()
    S = {"mode": "off", "capture": None, "healthy": None, "damaged": None,
         "A": None, "b": None, "fault": ("drop", 0.30)}

    def damage(flat):
        kind, p = S["fault"]
        g = torch.Generator().manual_seed(20260802)
        if kind == "drop":
            a0 = flat.clamp_min(0.0); mx = a0.max(-1, keepdim=True).values
            sets = (a0 > 0.5 * mx) & (mx > 0)
            return flat * (~((torch.rand(flat.shape, generator=g) < p) & sets)).float()
        if kind == "stuck":
            gm = torch.Generator().manual_seed(7)
            return flat * (~(torch.rand(flat.shape[-1], generator=gm) < p)).float()
        if kind == "quant":
            lo, hi = float(flat.min()), float(flat.max())
            step = (hi - lo) / max(1, int(p) - 1)
            return ((flat - lo) / step).round() * step + lo
        if kind == "noise":
            return flat + torch.randn(flat.shape, generator=g) * flat.std() * p
        raise ValueError(kind)

    def hook(_m, _i, out):
        shp = out.shape
        flat = out.reshape(-1, shp[-1])
        if S["capture"] == "healthy":
            S["healthy"] = flat.detach().clone(); return None
        if S["capture"] == "damaged":
            S["damaged"] = damage(flat).detach().clone(); return None
        if S["mode"] == "off":
            return None
        d = damage(flat)
        if S["mode"] == "damaged":
            new = d
        elif S["mode"] == "transform":
            new = d @ S["A"] + S["b"]
        elif S["mode"] == "patch":
            new = S["healthy"]
        else:
            raise ValueError(S["mode"])
        return new.reshape(shp)

    _find_act_module(model.transformer.h[LAYER]).register_forward_hook(hook)

    # ---- 混合拟合: 5种损伤 × 3个领域, 每格 FIT_DOCS_PER_CELL 篇 ----
    XtX = torch.zeros(U + 1, U + 1, dtype=torch.float64)
    XtY = torch.zeros(U + 1, U, dtype=torch.float64)
    n_rows = 0
    print("混合拟合数据采集(5损伤 × 3领域)…", flush=True)
    for src, path in CORPORA.items():
        docs = make_docs(tok, os.path.join(HERE, path), FIT_DOCS_PER_CELL, 7000,
                          skip_frac=0.55 if src == "in" else 0.5)
        for fault in FAULTS:
            S["fault"] = fault
            with torch.no_grad():
                for ch in docs:
                    S["mode"] = "off"
                    S["capture"] = "healthy"; model(ch)
                    S["capture"] = "damaged"; model(ch)
                    S["capture"] = None
                    X = S["damaged"].double(); Y = S["healthy"].double()
                    Xb = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], 1)
                    XtX += Xb.t() @ Xb; XtY += Xb.t() @ Y; n_rows += X.shape[0]
            print(f"  {src} × {FAULT_NAMES[fault]}: 累计 {n_rows} 行 ({time.time()-t0:.0f}s)", flush=True)
    W = torch.linalg.solve(XtX + RIDGE * torch.eye(U + 1, dtype=torch.float64), XtY)
    S["A"], S["b"] = W[:U].float(), W[U].float()
    print(f"混合 A 拟合完成 ({time.time()-t0:.0f}s)\n", flush=True)

    def evaluate(docs, fault):
        S["fault"] = fault
        out = {}
        for mode in ("off", "damaged", "transform", "patch"):
            S["mode"] = mode
            tot = []
            with torch.no_grad():
                for ch in docs:
                    if mode == "patch":
                        S["mode"] = "off"; S["capture"] = "healthy"; model(ch)
                        S["capture"] = None; S["mode"] = "patch"
                    tot.append(model(ch, labels=ch).loss)
            out[mode] = float(torch.stack(tot).mean())
        S["mode"] = "off"
        denom = out["damaged"] - out["off"]
        return {"harm": denom, "recovery": (out["damaged"] - out["transform"]) / denom if denom else float("nan")}

    res = {"n_fit_rows": n_rows, "cells": {}}
    print("=== 混合A 逐条件评测(同领域 in, 跨损伤) ===", flush=True)
    eval_in = make_docs(tok, os.path.join(HERE, CORPORA["in"]), EVAL_DOCS, 8888)
    for fault in FAULTS:
        r = evaluate(eval_in, fault)
        res["cells"][f"损伤_{FAULT_NAMES[fault]}"] = r
        print(f"  {FAULT_NAMES[fault]:10s} 伤害={r['harm']:+.4f} 混合A恢复={r['recovery']:.0%} "
              f"({time.time()-t0:.0f}s)", flush=True)

    print("\n=== 混合A 逐领域评测(损伤=砍30%) ===", flush=True)
    for src, path in CORPORA.items():
        docs = make_docs(tok, os.path.join(HERE, path), EVAL_DOCS, 9999,
                          skip_frac=0.55 if src == "in" else 0.5)
        r = evaluate(docs, ("drop", 0.30))
        res["cells"][f"领域_{src}"] = r
        print(f"  {src:6s} 伤害={r['harm']:+.4f} 混合A恢复={r['recovery']:.0%} "
              f"({time.time()-t0:.0f}s)", flush=True)

    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
