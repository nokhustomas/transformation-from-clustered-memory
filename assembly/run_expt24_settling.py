# -*- coding: utf-8 -*-
"""
实验24: 叠加存储 + 反复靠拢(而不是一次性读出)。

用户指出: 我此前把"存下来的一坨"当成答案直接加进去/换进去, 全都失败;
但如果那一坨是拿来让当前状态**反复往里靠**、自己落定呢? 这个读法我从没做过。

存: 收集一批真实出现过的内部状态原样存着(不做平均——平均正是之前失败的地方)。
读: 从当前状态出发,
     1) 算它跟每条存档有多像
     2) 按像的程度加权, 混出一个"应该长什么样"的估计
     3) 朝这个估计挪一步
     4) 用挪过之后的新状态回到第1步, 反复几次
两个旋钮: 挑剔程度 beta(权重多集中), 靠拢步长 eta 与次数 T。
  beta 很小 -> 混出来接近全体平均(已知失败)
  beta 很大 -> 等于只取最像的一条(已知失败)
  中间地带是这次真正要看的。

对照(同批数据同次运行): 不动 / 一次性换平均值 / 一次性换最像的一条 / 换成正确状态(满分)。
另记录每步的移动幅度, 判断是否收敛(跑飞本身也是结果)。
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

N_BATCHES, BS, MEMBERS = 1500, 32, 8
N_STORE = 512            # 存多少条真实状态
COLLECT_BATCHES = 300
OUT = os.path.join(OUT_DIR, "expt24_settling.json")


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
         "beta": 10.0, "eta": 0.5, "T": 3, "move_log": []}

    def retrieve(h, P, Pn, beta):
        """按相似度加权混合存档: h [P_pos, U], P [N, U] -> [P_pos, U]"""
        q = F.normalize(h, dim=-1)
        sim = q @ Pn.t()                       # [P_pos, N]
        w = torch.softmax(beta * sim, dim=-1)
        return w @ P

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
            q = F.normalize(h, dim=-1)
            new = STORE[(q @ STOREn.t()).argmax(-1)]
        elif S["mode"] == "settle":
            cur = h
            for _ in range(S["T"]):
                tgt = retrieve(cur, STORE, STOREn, S["beta"])
                nxt = cur + S["eta"] * (tgt - cur)
                S["move_log"].append(float((nxt - cur).norm(dim=-1).mean()))
                cur = nxt
            new = cur
        else:
            raise ValueError(S["mode"])
        o = out.clone(); o[m] = new
        return o

    model.layers[1].ffn[1].register_forward_hook(hook)

    # ---- 收集真实状态存档(不做平均) ----
    print("收集真实状态存档…", flush=True)
    pool = []
    gen = torch.Generator().manual_seed(DATA_SEED + 31337)
    with torch.no_grad():
        for _ in range(COLLECT_BATCHES):
            ids, _ = generate_batch(BS, SEQ_LEN, spec, generator=gen)
            S["capture"] = True; model(ids); S["capture"] = False
            flat = S["buf"].reshape(-1, U)
            step = max(1, flat.shape[0] // 4)
            pool.append(flat[::step])
    pool = torch.cat(pool)
    g = torch.Generator().manual_seed(99)
    STORE = pool[torch.randperm(pool.shape[0], generator=g)[:N_STORE]].contiguous()
    STOREn = F.normalize(STORE, dim=-1)
    MEAN = STORE.mean(0)
    print(f"存档 {STORE.shape[0]} 条真实状态 ({time.time()-t0:.0f}s)", flush=True)

    # ---- 评测 ----
    configs = [("不动", {"mode": "off"}),
               ("一次性换平均值", {"mode": "once_mean"}),
               ("一次性换最像的一条", {"mode": "once_nearest"})]
    for beta in (2.0, 10.0, 50.0):
        for T in (3, 10):
            configs.append((f"反复靠拢 挑剔={beta:g} 次数={T}",
                            {"mode": "settle", "beta": beta, "T": T, "eta": 0.5}))
    configs.append(("反复靠拢 挑剔=10 次数=10 步长=1.0",
                    {"mode": "settle", "beta": 10.0, "T": 10, "eta": 1.0}))
    configs.append(("换成正确状态(满分)", {"mode": "oracle"}))

    acc = {name: [] for name, _ in configs}
    accf = {name: [] for name, _ in configs}
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
            for name, cfg in configs:
                S.update({"mode": cfg["mode"]})
                for k in ("beta", "eta", "T"):
                    if k in cfg:
                        S[k] = cfg[k]
                S["move_log"] = []
                lg = model(ids)
                if S["move_log"]:
                    moves.setdefault(name, []).extend(S["move_log"])
                lp = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                                     ids[:, 1:].reshape(-1), reduction="none").view(BS, -1)
                acc[name].append(lp[rk[:, :-1] >= 6])
                accf[name].append(lp[rk[:, :-1] < 2])
            S["mode"] = "off"; S["mask"] = None
            if (bi + 1) % 500 == 0:
                print(f"  {bi+1}/{N_BATCHES} ({time.time()-t0:.0f}s)", flush=True)

    base = torch.cat(acc["不动"]); basef = torch.cat(accf["不动"])
    ceil_d = float((torch.cat(acc["换成正确状态(满分)"]) - base).mean())
    print(f"\n生僻词位置 {base.shape[0]} 个, 不动时 loss={float(base.mean()):.4f}", flush=True)
    print(f"满分(换成正确状态)= {ceil_d:+.4f}\n", flush=True)
    res = {"n": int(base.shape[0]), "base": float(base.mean()), "ceiling_delta": ceil_d,
           "arms": {}}
    for name, _ in configs[1:]:
        d = torch.cat(acc[name]) - base
        df = torch.cat(accf[name]) - basef
        se = float(d.std() / d.shape[0] ** 0.5)
        frac = float(d.mean()) / ceil_d if ceil_d else float("nan")
        res["arms"][name] = {"delta": float(d.mean()), "se": se, "pct_of_ceiling": frac,
                             "freq_delta": float(df.mean())}
        if name in moves:
            mv = torch.tensor(moves[name])
            res["arms"][name]["mean_move"] = float(mv.mean())
        ok = "✓有效" if d.mean() < -2 * se else ("✗有害" if d.mean() > 2 * se else "≈无反应")
        print(f"  {name:32s} Δ={float(d.mean()):+.4f}±{se:.4f}  拿到满分的{frac*100:5.1f}%  "
              f"常用词Δ={float(df.mean()):+.5f}  {ok}", flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
