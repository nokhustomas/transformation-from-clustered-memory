# -*- coding: utf-8 -*-
"""
实验25: 反复靠拢的大样本细网格版(约1小时)。

快跑(实验24, n=689)结论: 所有档位都变差, 且行为符合理论预期——挑剔度低+迭代多
滑向"全体平均"那个灾难, 挑剔度高收敛到"最像的一条", 两端都是已知失败, 中间无甜蜜点。

本实验的目的不是再确认一次"没效果", 而是把话说硬:
  1. 样本量约 17 倍(误差从 ±0.015 收到 ±0.004), 使"任何真实收益都小于某个上限"成为可陈述的结论;
  2. 网格加密(挑剔度 × 次数 × 步长), 排除甜蜜点藏在测点之间;
  3. 补一个更贴近原意的变体: **只在当前状态所属的那一组内部靠拢**(而不是在全部存档里靠拢)。

内存做法: 生僻词位置很少, 全部保留; 常用词位置极多, 只累加配对差值的和与平方和。
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
OUT = os.path.join(OUT_DIR, "expt25_settling_big.json")


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
         "beta": 10.0, "eta": 0.5, "T": 3, "scope": "global", "moves": []}

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
        elif S["mode"] == "settle":
            if S["scope"] == "group":
                a, _ = reg.assign(reg.extract_sets(h))          # 每个位置所属的组
                gmask = (GRP.unsqueeze(0) == a.unsqueeze(1))    # [P, N] 只允许同组存档
            cur = h
            for _ in range(S["T"]):
                sim = F.normalize(cur, dim=-1) @ STOREn.t()
                if S["scope"] == "group":
                    sim = sim.masked_fill(~gmask, -1e9)
                w = torch.softmax(S["beta"] * sim, dim=-1)
                tgt = w @ STORE
                nxt = cur + S["eta"] * (tgt - cur)
                S["moves"].append(float((nxt - cur).norm(dim=-1).mean()))
                cur = nxt
            new = cur
        else:
            raise ValueError(S["mode"])
        o = out.clone(); o[m] = new
        return o

    model.layers[1].ffn[1].register_forward_hook(hook)

    print("收集真实状态存档…", flush=True)
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
    GRP, _ = reg.assign(reg.extract_sets(STORE))                # 每条存档属于哪一组
    print(f"存档 {STORE.shape[0]} 条; 有组归属的 {int((GRP>=0).sum())} 条 ({time.time()-t0:.0f}s)", flush=True)

    configs = [("不动", {"mode": "off"}),
               ("一次性换平均值", {"mode": "once_mean"}),
               ("一次性换最像的一条", {"mode": "once_nearest"})]
    for beta in (5.0, 10.0, 20.0, 50.0):
        for T in (1, 3, 10):
            configs.append((f"靠拢 挑剔{beta:g} 次数{T}",
                            {"mode": "settle", "beta": beta, "T": T, "eta": 0.5, "scope": "global"}))
    for eta in (0.1, 0.25):
        configs.append((f"靠拢 挑剔10 次数3 步长{eta}",
                        {"mode": "settle", "beta": 10.0, "T": 3, "eta": eta, "scope": "global"}))
    configs.append(("靠拢 只在同组内 挑剔10 次数3",
                    {"mode": "settle", "beta": 10.0, "T": 3, "eta": 0.5, "scope": "group"}))
    configs.append(("换成正确状态(满分)", {"mode": "oracle"}))

    rare_acc = {n: [] for n, _ in configs}
    freq_stat = {n: [0.0, 0.0, 0] for n, _ in configs}      # sum, sumsq, count(配对差)
    moves = {}
    gen = torch.Generator().manual_seed(DATA_SEED + 555)
    with torch.no_grad():
        for bi in range(N_BATCHES):
            ids, _ = generate_batch(BS, SEQ_LEN, spec, generator=gen)
            rk = ids % MEMBERS
            rare = rk >= 6
            if not bool(rare.any()):
                continue
            same = ids.clone(); same[rare] = (ids[rare] // MEMBERS) * MEMBERS
            S["capture"] = True; model(same); S["capture"] = False
            S["donor"] = S["buf"]; S["mask"] = rare
            base_freq = None
            for name, cfg in configs:
                S["mode"] = cfg["mode"]
                for k in ("beta", "eta", "T", "scope"):
                    if k in cfg:
                        S[k] = cfg[k]
                S["moves"] = []
                lg = model(ids)
                if S["moves"]:
                    moves.setdefault(name, []).extend(S["moves"][:3])
                lp = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                                     ids[:, 1:].reshape(-1), reduction="none").view(BS, -1)
                rare_acc[name].append(lp[rk[:, :-1] >= 6])
                fv = lp[rk[:, :-1] < 2]
                if name == "不动":
                    base_freq = fv
                else:
                    d = fv - base_freq
                    s = freq_stat[name]
                    s[0] += float(d.sum()); s[1] += float((d * d).sum()); s[2] += d.shape[0]
            S["mode"] = "off"; S["mask"] = None
            if (bi + 1) % 5000 == 0:
                print(f"  {bi+1}/{N_BATCHES} ({time.time()-t0:.0f}s)", flush=True)

    base = torch.cat(rare_acc["不动"])
    ceil = float((torch.cat(rare_acc["换成正确状态(满分)"]) - base).mean())
    print(f"\n生僻词位置 {base.shape[0]} 个, 不动时 loss={float(base.mean()):.4f}", flush=True)
    print(f"满分(换成正确状态) = {ceil:+.4f}\n", flush=True)
    res = {"n": int(base.shape[0]), "base": float(base.mean()), "ceiling": ceil, "arms": {}}
    for name, _ in configs[1:]:
        d = torch.cat(rare_acc[name]) - base
        se = float(d.std() / d.shape[0] ** 0.5)
        s = freq_stat[name]
        fmean = s[0] / max(1, s[2])
        e = {"delta": float(d.mean()), "se": se, "pct_of_ceiling": float(d.mean()) / ceil,
             "freq_delta": fmean, "n_freq": s[2]}
        if name in moves:
            e["mean_move_first3"] = float(torch.tensor(moves[name]).mean())
        res["arms"][name] = e
        tag = "✓有效" if d.mean() < -2 * se else ("✗有害" if d.mean() > 2 * se else "≈无反应")
        print(f"  {name:26s} Δ={float(d.mean()):+.4f}±{se:.4f}  满分的{e['pct_of_ceiling']*100:6.1f}%  "
              f"常用词Δ={fmean:+.5f}  {tag}", flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
