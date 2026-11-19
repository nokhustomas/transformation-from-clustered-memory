# -*- coding: utf-8 -*-
"""
实验19: 唯一还活着的方向——档案存"变换"而不是"状态"。

前情: 实验13/14/15/16 联合证明, 状态是上下文特异的, 任何"存下来再取出来用"的
状态(均值/真实样本)都无效甚至有害; 唯一有效的 oracle 供体之所以有效, 是因为它
用**当前上下文**现算。

那么: 能不能存一个**变换**, 让它在新上下文里把有缺陷的状态**算成**正确状态?
形式: donor ≈ A·h + b, 其中 h 是稀有词位置的当前(有缺陷)状态。
若 A 在**留出上下文**上仍有效 => 记忆的正确载体是"变换"而不是"状态",
用户最初构想以修改形式复活: 登记库存的应是"这一组该怎么被修正", 不是"这一组长什么样"。

各组(全部在留出批次上评测, 与拟合批次不重叠):
  identity        不动(基线)
  global_linear   全局一个 A,b (ridge 最小二乘拟合)
  per_group       每个组一套 A_g,b_g
  shift_only      只有偏置 b(即"加一个固定向量")——这正是此前失败的加常数形式, 作为对照
  oracle          用同上下文真实状态替换(上界)
若 global/per_group 显著优于 shift_only, 说明"乘性/方向性变换"携带了状态搬不动的信息。
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

FIT_BATCHES, EVAL_BATCHES, BS, MEMBERS = 6000, 1500, 32, 8
RIDGE = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
OUT = os.path.join(OUT_DIR, "expt19_transform.json")


def collect_pairs(model, reg, spec, n_batches, seed, state, tag=""):
    """收集 (h=稀有词位置当前状态, donor=同上下文同类高频词的状态, g=所属组)。"""
    H, D, G = [], [], []
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    with torch.no_grad():
        for bi in range(n_batches):
            ids, _ = generate_batch(BS, SEQ_LEN, spec, generator=gen)
            rk = ids % MEMBERS
            rare = rk >= 6
            if not bool(rare.any()):
                continue
            same_ids = ids.clone(); same_ids[rare] = (ids[rare] // MEMBERS) * MEMBERS
            state["capture"] = True
            model(ids); h_all = state["buf"]
            model(same_ids); d_all = state["buf"]
            state["capture"] = False
            m = rare
            h = h_all[m]; d = d_all[m]
            sets = reg.extract_sets(h)
            a, _ = reg.assign(sets)
            H.append(h); D.append(d); G.append(a)
            if (bi + 1) % 2000 == 0:
                print(f"  {tag}收集 {bi+1}/{n_batches} ({time.time()-t0:.0f}s)", flush=True)
    return torch.cat(H), torch.cat(D), torch.cat(G)


def fit_ridge(X, Y, lam):
    """最小二乘 Y ≈ X A + b (带 ridge)。X:[N,U] Y:[N,U] -> A:[U,U], b:[U]"""
    N, U = X.shape
    Xb = torch.cat([X, torch.ones(N, 1)], dim=1)          # [N, U+1]
    G = Xb.t() @ Xb + lam * torch.eye(U + 1)
    W = torch.linalg.solve(G, Xb.t() @ Y)                  # [U+1, U]
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

    U = DIM * 4
    state = {"capture": False, "buf": None, "mode": "off", "mask": None,
             "donor": None, "A": None, "b": None, "Ag": None, "bg": None}

    def hook(_m, _i, out):
        if state["capture"]:
            state["buf"] = out.detach().clone()
            return None
        if state["mode"] == "off" or state["mask"] is None:
            return None
        B, T, Uu = out.shape
        m = state["mask"]
        h = out[m]                                            # [P, U]
        if state["mode"] == "oracle":
            new_vals = state["donor"][m]
        elif state["mode"] == "global_linear":
            new_vals = h @ state["A"] + state["b"]
        elif state["mode"] == "shift_only":
            new_vals = h + state["b"]
        elif state["mode"] == "per_group":
            sets = reg.extract_sets(h)
            a, _ = reg.assign(sets)
            k = a.clamp_min(0)
            new_vals = torch.einsum("pu,puv->pv", h, state["Ag"][k]) + state["bg"][k]
        else:
            raise ValueError(state["mode"])
        new = out.clone(); new[m] = new_vals
        return new

    model.layers[1].ffn[1].register_forward_hook(hook)

    print("收集拟合数据…", flush=True)
    Hf, Df, Gf = collect_pairs(model, reg, spec, FIT_BATCHES, DATA_SEED + 9000, state, "拟合:")
    print(f"拟合样本 {Hf.shape[0]} 条, 维度 {U}", flush=True)

    # 内部划分, 选 ridge 系数
    n = Hf.shape[0]; ntr = int(n * 0.8)
    best = None
    for lam in RIDGE:
        A, b = fit_ridge(Hf[:ntr], Df[:ntr], lam)
        err = float(((Hf[ntr:] @ A + b) - Df[ntr:]).pow(2).mean())
        base_err = float((Hf[ntr:] - Df[ntr:]).pow(2).mean())
        print(f"  ridge λ={lam}: 留出重构误差={err:.4f} (不动={base_err:.4f})", flush=True)
        if best is None or err < best[1]:
            best = (lam, err)
    lam = best[0]
    A, b = fit_ridge(Hf, Df, lam)
    state["A"], state["b"] = A, b
    print(f"选定 λ={lam}", flush=True)

    # 每组
    Ag = torch.eye(U).unsqueeze(0).repeat(reg.K, 1, 1)
    bg = torch.zeros(reg.K, U)
    for k in Gf[Gf >= 0].unique().tolist():
        sel = Gf == k
        if int(sel.sum()) < 50:
            continue
        Ak, bk = fit_ridge(Hf[sel], Df[sel], lam)
        Ag[k] = Ak; bg[k] = bk
    state["Ag"], state["bg"] = Ag, bg
    print(f"每组变换: {int((Gf[Gf>=0].bincount(minlength=reg.K) >= 50).sum())} 个组有足够样本", flush=True)

    arms = ["off", "global_linear", "per_group", "shift_only", "oracle"]
    acc = {a: [] for a in arms}
    gen = torch.Generator().manual_seed(DATA_SEED + 555)      # 与拟合不重叠的评测流
    with torch.no_grad():
        for bi in range(EVAL_BATCHES):
            ids, _ = generate_batch(BS, SEQ_LEN, spec, generator=gen)
            rk = ids % MEMBERS
            rare = rk >= 6
            if not bool(rare.any()):
                continue
            same_ids = ids.clone(); same_ids[rare] = (ids[rare] // MEMBERS) * MEMBERS
            state["capture"] = True; model(same_ids); state["capture"] = False
            state["donor"] = state["buf"]; state["mask"] = rare
            for a in arms:
                state["mode"] = a
                lg = model(ids)
                lp = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                                     ids[:, 1:].reshape(-1), reduction="none").view(BS, -1)
                acc[a].append(lp[rk[:, :-1] >= 6])
            state["mode"] = "off"; state["mask"] = None

    base = torch.cat(acc["off"])
    print(f"\n评测: 稀有词 n={base.shape[0]}, 基线={float(base.mean()):.4f}", flush=True)
    res = {"n": int(base.shape[0]), "base": float(base.mean()), "ridge": lam,
           "n_fit": int(Hf.shape[0]), "arms": {}}
    for a in arms[1:]:
        d = torch.cat(acc[a]) - base
        se = float(d.std() / d.shape[0] ** 0.5)
        res["arms"][a] = {"delta": float(d.mean()), "se": se}
        tag = "✓改善" if d.mean() < -2 * se else ("✗恶化" if d.mean() > 2 * se else "≈无变化")
        print(f"  {a:16s} Δ={float(d.mean()):+.4f}±{se:.4f}  {tag}", flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
