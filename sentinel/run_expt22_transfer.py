# -*- coding: utf-8 -*-
"""
实验22: 学到的变换是"通用修复算子"还是"针对某一种损伤的补丁"?

实验20 证明: 在"砍30%活跃单元"这种损伤上, 拟合出的线性变换 A·h+b 能在留出文档上
恢复 83%(oracle 87%)。但这个 A 是在**同一种损伤 + 同一个领域**上拟合的。真正决定它
有没有实用价值的是两个迁移问题:

  跨损伤类型: 用"砍30%"拟合的 A, 去修"量化4级""卡死30%""高斯噪声"造成的损伤。
  跨领域:     用同领域文本拟合的 A, 去修近邻领域/远领域文本上的同种损伤。

若两种迁移都塌掉 => A 只是记住了这一种损伤在这一个领域的统计, 实用价值有限。
若跨领域成立而跨损伤塌掉 => A 是"该损伤的逆", 需要按损伤类型各存一个。
若两种都成立 => A 接近"通用状态修复算子", 价值大得多。
对照: 每种损伤/领域各自的 oracle(健康真实状态)恢复率, 作为该情形的上界。
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

LAYER = 4
FIT_DOCS, EVAL_DOCS = 60, 40
RIDGE = 0.1
OUT = os.path.join(HERE, "expt22_transfer.json")
CORPORA = {"in": "default_corpus.txt", "near": "neardomain_corpus.txt", "far": "fardomain_corpus.txt"}


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype=torch.float32)
    model.load_state_dict(torch.load(os.path.join(HERE, "distilgpt2_finetuned.pt"), weights_only=True))
    model.eval()
    sent = AssemblySentinel(model, [LAYER]); sent.load(os.path.join(HERE, "registry_ready.pt")); sent.remove()
    U = 3072

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

    # ---- 在"砍30% + 同领域"上拟合 A ----
    fit_docs = make_docs(tok, os.path.join(HERE, CORPORA["in"]), FIT_DOCS, 7777)
    XtX = torch.zeros(U + 1, U + 1, dtype=torch.float64)
    XtY = torch.zeros(U + 1, U, dtype=torch.float64)
    S["fault"] = ("drop", 0.30)
    with torch.no_grad():
        for i, ch in enumerate(fit_docs):
            S["mode"] = "off"
            S["capture"] = "healthy"; model(ch)
            S["capture"] = "damaged"; model(ch)
            S["capture"] = None
            X = S["damaged"].double(); Y = S["healthy"].double()
            Xb = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], 1)
            XtX += Xb.t() @ Xb; XtY += Xb.t() @ Y
            if (i + 1) % 30 == 0:
                print(f"  拟合累积 {i+1}/{len(fit_docs)} ({time.time()-t0:.0f}s)", flush=True)
    W = torch.linalg.solve(XtX + RIDGE * torch.eye(U + 1, dtype=torch.float64), XtY)
    S["A"], S["b"] = W[:U].float(), W[U].float()
    print(f"A 拟合完成(损伤=砍30%, 领域=in) ({time.time()-t0:.0f}s)\n", flush=True)

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
        base, dmg = out["off"], out["damaged"]
        denom = dmg - base
        return {"healthy": base, "damaged": dmg, "harm": denom,
                "transform_recovery": (dmg - out["transform"]) / denom if denom else float("nan"),
                "patch_recovery": (dmg - out["patch"]) / denom if denom else float("nan")}

    res = {"fit": {"fault": "drop0.3", "domain": "in", "n_docs": FIT_DOCS}, "transfer": {}}

    print("=== 跨损伤类型迁移(同领域 in) ===", flush=True)
    eval_in = make_docs(tok, os.path.join(HERE, CORPORA["in"]), EVAL_DOCS, 8888)
    for name, fault in (("砍30%(拟合时的损伤)", ("drop", 0.30)), ("砍10%", ("drop", 0.10)),
                        ("卡死30%", ("stuck", 0.30)), ("量化4级", ("quant", 4)),
                        ("高斯噪声0.5x", ("noise", 0.5))):
        r = evaluate(eval_in, fault)
        res["transfer"][f"损伤_{name}"] = r
        print(f"  {name:16s} 伤害={r['harm']:+.4f}  变换恢复={r['transform_recovery']:.0%}  "
              f"oracle={r['patch_recovery']:.0%} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== 跨领域迁移(损伤固定为砍30%) ===", flush=True)
    for src, path in CORPORA.items():
        docs = make_docs(tok, os.path.join(HERE, path), EVAL_DOCS, 9999,
                          skip_frac=0.55 if src == "in" else 0.5)
        r = evaluate(docs, ("drop", 0.30))
        res["transfer"][f"领域_{src}"] = r
        print(f"  {src:6s} 伤害={r['harm']:+.4f}  变换恢复={r['transform_recovery']:.0%}  "
              f"oracle={r['patch_recovery']:.0%} ({time.time()-t0:.0f}s)", flush=True)

    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
