# -*- coding: utf-8 -*-
"""
实验16: "替换有效、叠加无效"在真实预训练模型(distilgpt2, 82M)上还成立吗?

小模型上的发现: 在 L1 用供体的完整状态**替换** -> 吃满 oracle; **叠加**增量 -> 0;
用档案平均态替换 -> 0。本实验把同一判别搬到真实模型, 换一个真实模型上做得了的
"缺陷-供体"配对:

  缺陷位置: 罕见词(按语料词频统计, 出现次数最低的那一档 token)所在位置。
  供体:     把该罕见词换成同上下文里的**高频近义替身**不可得(真实语言没有干净的同类),
            所以改用可严格定义的配对: **同一 token 在另一段上下文中的状态**不行(不同上下文),
            于是采用 corruption-donor 设计——
            对同一段文本, 把第4层激活按"损坏(砍30%活跃单元)"处理得到受损模型,
            然后在受损模型上分别用三种方式恢复:
              patch  : 用健康模型同位置的完整第4层状态替换      (完整真实状态)
              add    : 用档案里该组"该亮未亮"成员的典型值叠加   (旧方式)
              ema    : 用档案里该组的 EMA 平均态替换            (平均态)
            指标: 恢复了多少损坏造成的 loss(与实验3A同口径)。
这样在真实模型上就有了严格配对的"完整真实状态"参照, 三种形式同场竞技。
"""

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from sentinel import AssemblySentinel, _find_act_module  # noqa: E402
from eval_at_scale import make_docs  # noqa: E402

LAYER = 4
N_DOCS = 80
DROP = 0.30
OUT = os.path.join(HERE, "expt16_real_patching.json")


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype=torch.float32)
    model.load_state_dict(torch.load(os.path.join(HERE, "distilgpt2_finetuned.pt"), weights_only=True))
    model.eval()
    sent = AssemblySentinel(model, [LAYER])
    sent.load(os.path.join(HERE, "registry_ready.pt"))
    sent.remove()
    reg = sent.regs[LAYER]

    state = {"mode": "off", "capture": False, "healthy": None}

    def hook(_m, _i, out):
        if state["capture"]:
            state["healthy"] = out.detach().clone()
            return None
        if state["mode"] == "off":
            return None
        shp = out.shape
        flat = out.reshape(-1, shp[-1])
        g = torch.Generator().manual_seed(20260802)
        a0 = flat.clamp_min(0.0)
        mx = a0.max(-1, keepdim=True).values
        sets_pre = (a0 > 0.5 * mx) & (mx > 0)
        drop = (torch.rand(flat.shape, generator=g) < DROP) & sets_pre
        damaged = flat * (~drop).float()
        if state["mode"] == "damaged":
            return damaged.reshape(shp)
        sets = reg.extract_sets(damaged)
        a, rho = reg.assign(sets)
        hit = a >= 0
        k = a.clamp_min(0)
        if state["mode"] == "patch":
            out_new = torch.where(hit.unsqueeze(-1),
                                  state["healthy"].reshape(-1, shp[-1]), damaged)
        elif state["mode"] == "ema":
            out_new = torch.where(hit.unsqueeze(-1), reg.act_ema[k], damaged)
        elif state["mode"] == "add":
            miss = reg.membership[k] & (~sets) & hit.unsqueeze(-1)
            out_new = damaged + reg.act_ema[k] * miss.float() * rho.clamp_min(0).unsqueeze(-1)
        else:
            raise ValueError(state["mode"])
        return out_new.reshape(shp)

    _find_act_module(model.transformer.h[LAYER]).register_forward_hook(hook)
    docs = make_docs(tok, os.path.join(HERE, "default_corpus.txt"), N_DOCS, 6161)
    print(f"{len(docs)} 篇文档 × 2048 token, 损坏={DROP:.0%} 于第{LAYER}层", flush=True)

    modes = ["off", "damaged", "patch", "ema", "add"]
    losses = {m: [] for m in modes}
    with torch.no_grad():
        for i, ch in enumerate(docs):
            state["mode"] = "off"; state["capture"] = True
            model(ch)
            state["capture"] = False
            for m in modes:
                state["mode"] = m
                losses[m].append(model(ch, labels=ch).loss)
            state["mode"] = "off"
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(docs)} ({time.time()-t0:.0f}s)", flush=True)

    L = {m: torch.stack(losses[m]) for m in modes}
    base = float(L["off"].mean()); dmg = float(L["damaged"].mean())
    print(f"\n健康={base:.4f}  损坏={dmg:.4f}  (损坏代价 {dmg-base:+.4f})\n", flush=True)
    res = {"healthy": base, "damaged": dmg, "n_docs": len(docs), "arms": {}}
    for m in ("patch", "ema", "add"):
        v = float(L[m].mean())
        d = L[m] - L["damaged"]
        se = float(d.std() / d.shape[0] ** 0.5)
        rec = (dmg - v) / (dmg - base) if dmg != base else float("nan")
        res["arms"][m] = {"loss": v, "delta_vs_damaged": float(d.mean()), "se": se, "recovery": rec}
        print(f"  {m:6s} loss={v:.4f}  相对损坏 Δ={float(d.mean()):+.4f}±{se:.4f}  恢复={rec:.0%}", flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
