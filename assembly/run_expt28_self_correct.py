# -*- coding: utf-8 -*-
"""
实验28: 模型自己纠正自己——深层的判断 + 变换(而不是取状态)。

把两个已证结论组合起来:
  (a) 变换能搬, 状态搬不动(实验13/19/20);
  (b) 深层知道浅层错在哪: 生僻词的嵌入认不出自己属于哪类(约25%, 接近瞎猜),
      但同一个词到了第二层, 组匹配的命中率接近满分(实验7的定位性测量)。

此前只测过 (b) 的"状态版": 用深层判断去取一个存好的状态塞回浅层 -> 只拿到 11%。
本实验测"变换版": 用深层判断当**条件**, 学一个修正**变换**作用在嵌入上, 而不是取状态。
修正信号完全来自模型内部, 不需要外部教师。

各组(全部在留出上下文上评测, 与拟合批次不重叠):
  不动                      基线
  全局变换(不看深层)            一个 A,b 修所有生僻词嵌入
  按深层判断分组的变换            深层说它属于第 k 组, 就用第 k 组的 A_k,b_k
  按真实类别分组的变换(作弊)       用真实类别选变换, 上界参照: 说明"分组信息本身"值多少
  oracle 嵌入替换(满分)         已知上界
关键对比是第 3 行 vs 第 2 行: 深层的判断有没有带来额外收益。
第 3 行 vs 第 4 行: 深层判断比真实类别差多少(即深层判断有多可靠)。
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

FIT_BATCHES, EVAL_BATCHES, BS, MEMBERS = 8000, 3000, 32, 8
RIDGE_GRID = [1e-3, 1e-2, 1e-1, 1.0]
OUT = os.path.join(OUT_DIR, "expt28_self_correct.json")


def fit_ridge(X, Y, lam):
    N, U = X.shape
    Xb = torch.cat([X, torch.ones(N, 1)], 1)
    W = torch.linalg.solve(Xb.t() @ Xb + lam * torch.eye(U + 1), Xb.t() @ Y)
    return W[:U], W[U]


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

    E0 = model.tok_emb.weight.detach().clone()
    cls = torch.arange(spec["vocab_size"]) // MEMBERS
    rank = torch.arange(spec["vocab_size"]) % MEMBERS
    E_oracle = E0.clone()
    for t in range(spec["vocab_size"]):
        if rank[t] >= 6:
            E_oracle[t] = E0[int(((cls == cls[t]) & (rank == 0)).nonzero()[0])]

    S = {"capture": False, "buf": None, "sub": None}

    def l1_hook(_m, _i, out):
        if S["capture"]:
            S["buf"] = out.detach().clone()
        return None

    def emb_hook(_m, _i, out):
        return S["sub"] if S["sub"] is not None else None

    model.layers[1].ffn[1].register_forward_hook(l1_hook)
    model.tok_emb.register_forward_hook(emb_hook)

    def collect(n_batches, seed, tag):
        """收集 (生僻词的原嵌入, 应有的嵌入, 深层给出的组, 真实类别)。"""
        H, Y, G, C = [], [], [], []
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for bi in range(n_batches):
                ids, _ = generate_batch(BS, SEQ_LEN, spec, generator=gen)
                rk = ids % MEMBERS
                rare = rk >= 6
                if not bool(rare.any()):
                    continue
                S["sub"] = None; S["capture"] = True
                model(ids)                                  # 正常前向, 取第二层状态
                S["capture"] = False
                l1 = S["buf"]
                a, _ = reg.assign(reg.extract_sets(l1.reshape(-1, DIM * 4)))
                a = a.view(ids.shape)
                H.append(E0[ids[rare]])
                Y.append(E_oracle[ids[rare]])
                G.append(a[rare])
                C.append((ids[rare] // MEMBERS))
                if (bi + 1) % 3000 == 0:
                    print(f"  {tag} {bi+1}/{n_batches} ({time.time()-t0:.0f}s)", flush=True)
        return torch.cat(H), torch.cat(Y), torch.cat(G), torch.cat(C)

    print("收集拟合数据…", flush=True)
    Hf, Yf, Gf, Cf = collect(FIT_BATCHES, DATA_SEED + 4000, "拟合")
    print(f"拟合样本 {Hf.shape[0]} 条; 深层组的种类 {len(Gf.unique())}", flush=True)
    acc_deep = float((Gf >= 0).float().mean())
    print(f"深层能给出组归属的比例: {acc_deep:.1%}", flush=True)

    n = Hf.shape[0]; ntr = int(n * 0.8)
    best = None
    for lam in RIDGE_GRID:
        A, b = fit_ridge(Hf[:ntr], Yf[:ntr], lam)
        err = float(((Hf[ntr:] @ A + b) - Yf[ntr:]).pow(2).mean())
        if best is None or err < best[1]:
            best = (lam, err)
    lam = best[0]
    A_glob, b_glob = fit_ridge(Hf, Yf, lam)
    print(f"全局变换拟合完毕 (λ={lam})", flush=True)

    def fit_by_key(keys, n_keys):
        As = torch.eye(DIM).unsqueeze(0).repeat(n_keys, 1, 1)
        bs = torch.zeros(n_keys, DIM)
        used = 0
        for k in range(n_keys):
            sel = keys == k
            if int(sel.sum()) < 30:
                As[k], bs[k] = A_glob, b_glob      # 样本不足就退回全局
                continue
            As[k], bs[k] = fit_ridge(Hf[sel], Yf[sel], lam)
            used += 1
        return As, bs, used

    A_grp, b_grp, n_grp = fit_by_key(Gf.clamp_min(0), reg.K)
    A_cls, b_cls, n_cls = fit_by_key(Cf, 8)
    print(f"按深层组分的变换: {n_grp} 组有足够样本; 按真实类别分的: {n_cls} 类", flush=True)

    arms = ["不动", "全局变换", "按深层判断分组的变换", "按真实类别分组的变换(作弊)", "oracle嵌入替换(满分)"]
    acc = {a: [] for a in arms}
    gen = torch.Generator().manual_seed(DATA_SEED + 555)
    with torch.no_grad():
        for bi in range(EVAL_BATCHES):
            ids, _ = generate_batch(BS, SEQ_LEN, spec, generator=gen)
            rk = ids % MEMBERS
            rare = rk >= 6
            if not bool(rare.any()):
                continue
            S["sub"] = None; S["capture"] = True
            model(ids)
            S["capture"] = False
            a_deep, _ = reg.assign(reg.extract_sets(S["buf"].reshape(-1, DIM * 4)))
            a_deep = a_deep.view(ids.shape).clamp_min(0)
            emb0 = E0[ids]
            for name in arms:
                if name == "不动":
                    S["sub"] = None
                elif name == "oracle嵌入替换(满分)":
                    e = emb0.clone(); e[rare] = E_oracle[ids[rare]]; S["sub"] = e
                else:
                    e = emb0.clone()
                    h = emb0[rare]
                    if name == "全局变换":
                        e[rare] = h @ A_glob + b_glob
                    elif name == "按深层判断分组的变换":
                        k = a_deep[rare]
                        e[rare] = torch.einsum("pu,puv->pv", h, A_grp[k]) + b_grp[k]
                    else:
                        k = (ids[rare] // MEMBERS)
                        e[rare] = torch.einsum("pu,puv->pv", h, A_cls[k]) + b_cls[k]
                    S["sub"] = e
                lg = model(ids)
                lp = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                                     ids[:, 1:].reshape(-1), reduction="none").view(BS, -1)
                acc[name].append(lp[rk[:, :-1] >= 6])
            S["sub"] = None
    base = torch.cat(acc["不动"])
    ceil = float((torch.cat(acc["oracle嵌入替换(满分)"]) - base).mean())
    print(f"\n生僻词位置 {base.shape[0]} 个; 满分 = {ceil:+.4f}\n", flush=True)
    res = {"n": int(base.shape[0]), "ceiling": ceil, "deep_group_coverage": acc_deep, "arms": {}}
    for name in arms[1:]:
        d = torch.cat(acc[name]) - base
        se = float(d.std() / d.shape[0] ** 0.5)
        res["arms"][name] = {"delta": float(d.mean()), "se": se,
                             "pct_of_ceiling": float(d.mean()) / ceil}
        tag = "✓有效" if d.mean() < -2 * se else ("✗有害" if d.mean() > 2 * se else "≈无反应")
        print(f"  {name:26s} Δ={float(d.mean()):+.4f}±{se:.4f}  满分的{float(d.mean())/ceil*100:6.1f}%  {tag}", flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
