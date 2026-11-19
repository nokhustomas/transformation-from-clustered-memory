# -*- coding: utf-8 -*-
"""
实验14: 把"叠加 vs 替换"从二选一变成连续谱, 并检验"什么样的状态才算合法"。

实验13 得到: 用供体的完整 L1 状态替换 -> 吃满 oracle(-0.052); 用增量叠加 -> 0;
用档案里的平均典型态替换 -> 0。本实验拆两个问题:

问题A(形式): 增量本身是否非法, 还是"增量指向的方向"才是关键?
  做法: h <- h + λ(donor - h), λ 从 0 扫到 1。λ=1 即替换, λ<1 是增量叠加。
  若效果随 λ 线性上升 => 增量是合法通道, 之前失败是因为加错了方向(加的是存储均值,
  不是指向真实状态的差向量)。若只有 λ≈1 才有效 => 只有完整替换合法。

问题B(内容): 什么状态算"合法"?
  patch_same_class      同类高频词、同上下文的真实状态   (实验13已知: 满额)
  patch_diff_class      异类词的真实状态                 (对照: 应该有害)
  patch_other_context   同类词但取自**另一条序列**的状态 (真实状态, 但上下文不符)
  patch_avg_two_real    两个真实状态的平均               (仍在流形附近?)
  patch_registry_ema    档案里的跨语境平均               (实验13已知: 无效)
  这条链能定位"合法状态"的边界: 是要真实? 要上下文一致? 还是只要不是均值就行?
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
OUT = os.path.join(OUT_DIR, "expt14_interp.json")


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

    state = {"mode": "off", "lam": 1.0, "capture": None, "donor": {}, "mask": None}

    def hook(_m, _i, out):
        c = state["capture"]
        if c is not None:
            state["donor"][c] = out.detach().clone()
            return None
        if state["mode"] == "off" or state["mask"] is None:
            return None
        B, T, U = out.shape
        m = state["mask"]
        tgt = None
        mo = state["mode"]
        if mo == "interp":
            tgt = state["donor"]["same"]
        elif mo == "same":
            tgt = state["donor"]["same"]
        elif mo == "diff_class":
            tgt = state["donor"]["diff"]
        elif mo == "other_context":
            tgt = state["donor"]["same"].roll(1, dims=0)      # 换一条序列的同位置
        elif mo == "avg_two_real":
            tgt = 0.5 * (state["donor"]["same"] + state["donor"]["same"].roll(1, dims=0))
        elif mo == "registry_ema":
            flat = out.reshape(-1, U).detach()
            sets = reg.extract_sets(flat)
            a, _ = reg.assign(sets)
            tgt = reg.act_ema[a.clamp_min(0)].view(B, T, U)
        else:
            raise ValueError(mo)
        lam = state["lam"] if mo == "interp" else 1.0
        new = out.clone()
        new[m] = out[m] + lam * (tgt[m] - out[m])
        return new

    model.layers[1].ffn[1].register_forward_hook(hook)

    LAMS = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    arms = [("off", None)] + [(f"interp_λ={l}", ("interp", l)) for l in LAMS] + \
           [("patch_diff_class", ("diff_class", 1.0)),
            ("patch_other_context", ("other_context", 1.0)),
            ("patch_avg_two_real", ("avg_two_real", 1.0)),
            ("patch_registry_ema", ("registry_ema", 1.0))]
    acc = {n: [] for n, _ in arms}

    gen = torch.Generator().manual_seed(DATA_SEED + 555)
    with torch.no_grad():
        for bi in range(N_BATCHES):
            ids, _ = generate_batch(BS, SEQ_LEN, spec, generator=gen)
            rk = ids % MEMBERS
            rare = rk >= 6
            if not bool(rare.any()):
                continue
            same_ids = ids.clone(); same_ids[rare] = (ids[rare] // MEMBERS) * MEMBERS
            diff_ids = ids.clone()
            diff_ids[rare] = (((ids[rare] // MEMBERS) + 4) % 8) * MEMBERS   # 异类的高频词
            state["mode"] = "off"; state["mask"] = None
            state["capture"] = "same"; model(same_ids)
            state["capture"] = "diff"; model(diff_ids)
            state["capture"] = None
            state["mask"] = rare
            for name, cfg in arms:
                if cfg is None:
                    state["mode"] = "off"
                else:
                    state["mode"], state["lam"] = cfg
                lg = model(ids)
                lp = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                                     ids[:, 1:].reshape(-1), reduction="none").view(BS, -1)
                acc[name].append(lp[rk[:, :-1] >= 6])
            state["mode"] = "off"; state["mask"] = None
            if (bi + 1) % 400 == 0:
                print(f"  {bi+1}/{N_BATCHES} ({time.time()-t0:.0f}s)", flush=True)

    base = torch.cat(acc["off"])
    print(f"\n稀有词 n={base.shape[0]}, 基线={float(base.mean()):.4f}", flush=True)
    res = {"n": int(base.shape[0]), "base": float(base.mean()), "arms": {}}
    for name, _ in arms[1:]:
        d = torch.cat(acc[name]) - base
        se = float(d.std() / d.shape[0] ** 0.5)
        res["arms"][name] = {"delta": float(d.mean()), "se": se}
        tag = "✓改善" if d.mean() < -2 * se else ("✗恶化" if d.mean() > 2 * se else "≈无变化")
        print(f"  {name:22s} Δ={float(d.mean()):+.4f}±{se:.4f}  {tag}", flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
