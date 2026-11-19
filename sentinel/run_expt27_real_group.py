# -*- coding: utf-8 -*-
"""
实验27: 把"只在同组内靠拢"搬到真实模型(distilgpt2, 82M)验证。

小模型上(实验25, n=11664)所有存状态-注入的做法里, 只有"限制在同组内、按相似度加权
混合、走几步"做出了真改善(-0.0094±0.0021, 拿到满分的13.3%, 4.5倍误差)。
这条线此前每次从小模型搬到真实模型都出过意外(平均态从"无效"变成"灾难性有害"),
所以必须单独验。

真实模型上没有天然的"生僻词-同类高频词"配对, 沿用实验16/20的口径: 第4层砍掉30%
活跃单元制造缺陷, 健康模型同位置的完整状态作为满分参照。恢复率同口径可比。

各组(同一批文档、配对):
  损坏不修            基线
  一次性换档案平均态     已知: -246%(灾难)
  一次性换最像的存档     未在真实模型上测过
  同组内靠拢(扫几档)    本实验主角
  换成健康真实状态       满分(实验16: 87%)
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

LAYER, DROP, U = 4, 0.30, 3072
N_STORE, COLLECT_DOCS, EVAL_DOCS = 1024, 40, 60
OUT = os.path.join(HERE, "expt27_real_group.json")


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype=torch.float32)
    model.load_state_dict(torch.load(os.path.join(HERE, "distilgpt2_finetuned.pt"), weights_only=True))
    model.eval()
    sent = AssemblySentinel(model, [LAYER]); sent.load(os.path.join(HERE, "registry_ready.pt")); sent.remove()
    reg = sent.regs[LAYER]

    S = {"mode": "off", "capture": None, "healthy": None, "beta": 10.0, "eta": 0.5, "T": 3}

    def damage(flat):
        g = torch.Generator().manual_seed(20260802)
        a0 = flat.clamp_min(0.0); mx = a0.max(-1, keepdim=True).values
        sets = (a0 > 0.5 * mx) & (mx > 0)
        return flat * (~((torch.rand(flat.shape, generator=g) < DROP) & sets)).float()

    def hook(_m, _i, out):
        shp = out.shape
        flat = out.reshape(-1, shp[-1])
        if S["capture"] == "healthy":
            S["healthy"] = flat.detach().clone(); return None
        if S["capture"] == "collect":
            S["pool"] = flat.detach().clone(); return None
        if S["mode"] == "off":
            return None
        d = damage(flat)
        if S["mode"] == "damaged":
            new = d
        elif S["mode"] == "patch":
            new = S["healthy"]
        elif S["mode"] == "ema":
            a, _ = reg.assign(reg.extract_sets(d))
            new = torch.where((a >= 0).unsqueeze(-1), reg.act_ema[a.clamp_min(0)], d)
        elif S["mode"] == "nearest":
            new = STORE[(F.normalize(d, dim=-1) @ STOREn.t()).argmax(-1)]
        elif S["mode"] == "group":
            a, _ = reg.assign(reg.extract_sets(d))
            gmask = (GRP.unsqueeze(0) == a.unsqueeze(1))
            has = gmask.any(-1)
            cur = d
            for _ in range(S["T"]):
                sim = (F.normalize(cur, dim=-1) @ STOREn.t()).masked_fill(~gmask, -1e9)
                w = torch.softmax(S["beta"] * sim, dim=-1)
                cur = torch.where(has.unsqueeze(-1), cur + S["eta"] * (w @ STORE - cur), cur)
            new = cur
        else:
            raise ValueError(S["mode"])
        return new.reshape(shp)

    _find_act_module(model.transformer.h[LAYER]).register_forward_hook(hook)

    print("收集真实状态存档…", flush=True)
    docs_c = make_docs(tok, os.path.join(HERE, "default_corpus.txt"), COLLECT_DOCS, 4141)
    pool = []
    with torch.no_grad():
        for ch in docs_c:
            S["mode"] = "off"; S["capture"] = "collect"; model(ch); S["capture"] = None
            pool.append(S["pool"][::64])
    pool = torch.cat(pool)
    g = torch.Generator().manual_seed(99)
    STORE = pool[torch.randperm(pool.shape[0], generator=g)[:N_STORE]].contiguous()
    STOREn = F.normalize(STORE, dim=-1)
    GRP, _ = reg.assign(reg.extract_sets(STORE))
    print(f"存档 {STORE.shape[0]} 条, 有组归属 {int((GRP>=0).sum())} 条 ({time.time()-t0:.0f}s)", flush=True)

    configs = [("健康", {"mode": "off"}), ("损坏不修", {"mode": "damaged"}),
               ("一次性换平均态", {"mode": "ema"}), ("一次性换最像的存档", {"mode": "nearest"})]
    for beta in (5.0, 10.0, 30.0):
        for T in (1, 3):
            configs.append((f"同组内靠拢 挑剔{beta:g} 次数{T}",
                            {"mode": "group", "beta": beta, "T": T, "eta": 0.5}))
    configs.append(("换成健康真实状态(满分)", {"mode": "patch"}))

    docs = make_docs(tok, os.path.join(HERE, "default_corpus.txt"), EVAL_DOCS, 8888)
    L = {}
    for name, cfg in configs:
        S["mode"] = cfg["mode"]
        for k in ("beta", "eta", "T"):
            if k in cfg:
                S[k] = cfg[k]
        tot = []
        with torch.no_grad():
            for ch in docs:
                if cfg["mode"] == "patch":
                    S["mode"] = "off"; S["capture"] = "healthy"; model(ch)
                    S["capture"] = None; S["mode"] = "patch"
                tot.append(model(ch, labels=ch).loss)
        L[name] = torch.stack(tot)
        S["mode"] = "off"
        print(f"  {name:26s} loss={float(L[name].mean()):.4f} ({time.time()-t0:.0f}s)", flush=True)

    base, dmg = float(L["健康"].mean()), float(L["损坏不修"].mean())
    print(f"\n健康={base:.4f} 损坏={dmg:.4f} (损坏代价 {dmg-base:+.4f})\n", flush=True)
    res = {"healthy": base, "damaged": dmg, "arms": {}}
    for name, _ in configs[2:]:
        v = float(L[name].mean())
        d = L[name] - L["损坏不修"]
        se = float(d.std() / d.shape[0] ** 0.5)
        rec = (dmg - v) / (dmg - base)
        res["arms"][name] = {"loss": v, "recovery": rec, "se": se}
        print(f"  {name:26s} 恢复={rec:6.0%}  (相对损坏 Δ={float(d.mean()):+.4f}±{se:.4f})", flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
