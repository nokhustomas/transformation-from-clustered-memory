# -*- coding: utf-8 -*-
"""
实验26: 把"只在同组内靠拢"这一档扫开, 找它的上限。

实验25(n=11664)发现: 所有做法里只有两个真改善, 最好的是"只在同组内靠拢"(-0.0094±0.0021,
拿到满分的13.3%, 4.5倍误差)。但那一档只测了一个配置(挑剔10、3次、步长0.5)。
同批数据里全局版的规律是"次数越少越好"(1次最好, 10次最差), 所以同组内很可能还没到最优。

本实验把同组内这一档的三个旋钮扫开: 挑剔度 × 次数 × 步长, 并保留同批对照。
另外加一档"同组内只走一步且步长可调"的精细扫描, 因为全局版的最优就出现在这个角落。

判断标准沿用: 改善超过误差两倍; 必须优于两个一次性对照; 不能伤到常用词。
"""

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))

from models import FullAttentionLM  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402
from concept_lm_data import make_concept_spec, generate_batch  # noqa: E402
from run_expt3_stress import EMIT_EXTREME  # noqa: E402
from run_expt2_completion import DIM, LAYERS, HEADS, SEQ_LEN, DATA_SEED, OUT_DIR  # noqa: E402

N_BATCHES, BS, MEMBERS = 25000, 32, 8
N_STORE, COLLECT_BATCHES = 512, 300
OUT = os.path.join(OUT_DIR, "expt26_group_sweep.json")


def main():
    t0 = time.time()
    spec = make_concept_spec(8, MEMBERS, emit_probs=EMIT_EXTREME)
    torch.manual_seed(0)
    model = FullAttentionLM(spec["vocab_size"], DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "expt3_model_seed0.pt")))
    model.eval()
    st = torch.load(os.path.join(OUT_DIR, "expt3_registries_seed0.pt"))["L1"]
    reg = AssemblyRegistry(**st["config"])
    for k in ("freq", "act_ema", "active", "count", "created_at"):
        setattr(reg, k, st[k])
    U = DIM * 4

    S = {"mode": "off", "capture": False, "buf": None, "mask": None, "donor": None,
         "beta": 10.0, "eta": 0.5, "T": 3}

    def hook(_m, _i, out):
        if S["capture"]:
            S["buf"] = out.detach().clone(); return None
        if S["mode"] == "off" or S["mask"] is None:
            return None
        m = S["mask"]
        h = out[m]
        if S["mode"] == "oracle":
            new = S["donor"][m]
        elif S["mode"] == "once_mean":
            new = MEAN.unsqueeze(0).expand_as(h)
        elif S["mode"] == "once_nearest":
            new = STORE[(F.normalize(h, dim=-1) @ STOREn.t()).argmax(-1)]
        elif S["mode"] == "group":
            a, _ = reg.assign(reg.extract_sets(h))
            gmask = (GRP.unsqueeze(0) == a.unsqueeze(1))
            has = gmask.any(-1)
            cur = h
            for _ in range(S["T"]):
                sim = (F.normalize(cur, dim=-1) @ STOREn.t()).masked_fill(~gmask, -1e9)
                w = torch.softmax(S["beta"] * sim, dim=-1)
                tgt = w @ STORE
                cur = torch.where(has.unsqueeze(-1), cur + S["eta"] * (tgt - cur), cur)
            new = cur
        else:
            raise ValueError(S["mode"])
        o = out.clone(); o[m] = new
        return o

    model.layers[1].ffn[1].register_forward_hook(hook)

    print("收集存档…", flush=True)
    pool = []
    gen = torch.Generator().manual_seed(DATA_SEED + 31337)
    with torch.no_grad():
        for _ in range(COLLECT_BATCHES):
            ids, _ = generate_batch(BS, SEQ_LEN, spec, generator=gen)
            S["capture"] = True; model(ids); S["capture"] = False
            flat = S["buf"].reshape(-1, U)
            pool.append(flat[::max(1, flat.shape[0] // 4)])
    pool = torch.cat(pool)
    g = torch.Generator().manual_seed(99)
    STORE = pool[torch.randperm(pool.shape[0], generator=g)[:N_STORE]].contiguous()
    STOREn = F.normalize(STORE, dim=-1)
    MEAN = STORE.mean(0)
    GRP, _ = reg.assign(reg.extract_sets(STORE))
    print(f"存档 {STORE.shape[0]} 条 ({time.time()-t0:.0f}s)", flush=True)

    configs = [("不动", {"mode": "off"}),
               ("一次性换平均值", {"mode": "once_mean"}),
               ("一次性换最像的一条", {"mode": "once_nearest"}),
               ("同组内 挑剔10 次数3 步长0.5(上次的)", {"mode": "group", "beta": 10., "T": 3, "eta": .5})]
    for beta in (2.0, 5.0, 10.0, 30.0):
        for T in (1, 2, 5):
            configs.append((f"同组内 挑剔{beta:g} 次数{T} 步长0.5",
                            {"mode": "group", "beta": beta, "T": T, "eta": 0.5}))
    for eta in (0.15, 0.3, 0.7, 1.0):
        configs.append((f"同组内 挑剔5 次数1 步长{eta}",
                        {"mode": "group", "beta": 5.0, "T": 1, "eta": eta}))
    configs.append(("换成正确状态(满分)", {"mode": "oracle"}))

    rare = {n: [] for n, _ in configs}
    fstat = {n: [0.0, 0] for n, _ in configs}
    gen = torch.Generator().manual_seed(DATA_SEED + 555)
    with torch.no_grad():
        for bi in range(N_BATCHES):
            ids, _ = generate_batch(BS, SEQ_LEN, spec, generator=gen)
            rk = ids % MEMBERS
            rm = rk >= 6
            if not bool(rm.any()):
                continue
            same = ids.clone(); same[rm] = (ids[rm] // MEMBERS) * MEMBERS
            S["capture"] = True; model(same); S["capture"] = False
            S["donor"] = S["buf"]; S["mask"] = rm
            bf = None
            for name, cfg in configs:
                S["mode"] = cfg["mode"]
                for k in ("beta", "eta", "T"):
                    if k in cfg:
                        S[k] = cfg[k]
                lg = model(ids)
                lp = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                                     ids[:, 1:].reshape(-1), reduction="none").view(BS, -1)
                rare[name].append(lp[rk[:, :-1] >= 6])
                fv = lp[rk[:, :-1] < 2]
                if name == "不动":
                    bf = fv
                else:
                    d = fv - bf
                    fstat[name][0] += float(d.sum()); fstat[name][1] += d.shape[0]
            S["mode"] = "off"; S["mask"] = None
            if (bi + 1) % 5000 == 0:
                print(f"  {bi+1}/{N_BATCHES} ({time.time()-t0:.0f}s)", flush=True)

    base = torch.cat(rare["不动"])
    ceil = float((torch.cat(rare["换成正确状态(满分)"]) - base).mean())
    print(f"\n生僻词位置 {base.shape[0]} 个; 满分 = {ceil:+.4f}\n", flush=True)
    res = {"n": int(base.shape[0]), "ceiling": ceil, "arms": {}}
    for name, _ in configs[1:]:
        d = torch.cat(rare[name]) - base
        se = float(d.std() / d.shape[0] ** 0.5)
        fm = fstat[name][0] / max(1, fstat[name][1])
        res["arms"][name] = {"delta": float(d.mean()), "se": se,
                             "pct_of_ceiling": float(d.mean()) / ceil, "freq_delta": fm}
        tag = "✓有效" if d.mean() < -2 * se else ("✗有害" if d.mean() > 2 * se else "≈无反应")
        print(f"  {name:34s} Δ={float(d.mean()):+.4f}±{se:.4f}  满分的{float(d.mean())/ceil*100:6.1f}%  "
              f"常用词Δ={fm:+.5f}  {tag}", flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
