# -*- coding: utf-8 -*-
"""
实验15: 修复方案候选——档案存"真实样本状态"而不是"跨语境平均"。

诊断链: 替换有效 / 叠加无效 / 平均态替换无效 => 病根可能不是"机制不行",
而是**存错了东西**: EMA 平均出的典型向量不是流形上的合法状态。
本实验把登记库改成同时保存每组若干条**真实激活向量**(exemplar),
部署时用"当前状态最像的那条 exemplar"整条替换进去。

三个版本对比(全部只在稀有词位置动手, 与实验13/14同一模型、同一评测流):
  ema_patch       用组的 EMA 平均态替换                 (已知无效, 锚点)
  exemplar_patch  用组内最近的一条真实 exemplar 替换     (本实验主角)
  exemplar_random 用组内随机一条 exemplar 替换           (检验"最近"是否重要)
  oracle_patch    用同类高频词同上下文的真实状态替换      (上界, 实验13已确认满额)

exemplar 的采集: 在训练分布上跑一遍, 每组保留 E 条真实激活向量(蓄水池抽样,
确定性: 用固定种子的 Generator)。这一步不需要梯度、不需要标注,
与原机制一样是"训练时顺带"可做的。
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
N_EXEMPLAR = 16          # 每组保留多少条真实状态
COLLECT_BATCHES = 200
OUT = os.path.join(OUT_DIR, "expt15_exemplar.json")


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
    state = {"mode": "off", "capture": False, "buf": None, "mask": None, "donor": None}

    def hook(_m, _i, out):
        if state["capture"]:
            state["buf"] = out.detach().clone()
            return None
        if state["mode"] == "off" or state["mask"] is None:
            return None
        B, T, Uu = out.shape
        m = state["mask"]
        flat = out.reshape(-1, Uu).detach()
        if state["mode"] == "oracle_patch":
            tgt = state["donor"]
        else:
            sets = reg.extract_sets(flat)
            a, _ = reg.assign(sets)
            k = a.clamp_min(0)
            if state["mode"] == "ema_patch":
                tgt = reg.act_ema[k].view(B, T, Uu)
            else:
                ex = EX[k]                     # [P, E, U] 该位置所属组的 exemplar 池
                cnt = EXN[k]                   # [P] 该组实际存了几条
                if state["mode"] == "exemplar_patch":
                    q = F.normalize(flat, dim=-1).unsqueeze(1)          # [P,1,U]
                    sim = (q * F.normalize(ex, dim=-1)).sum(-1)          # [P,E]
                    ar = torch.arange(ex.shape[1]).unsqueeze(0)
                    sim = sim.masked_fill(ar >= cnt.unsqueeze(1), -2.0)
                    pick = sim.argmax(-1)
                else:                                                    # exemplar_random
                    g = torch.Generator().manual_seed(4242)
                    pick = (torch.rand(ex.shape[0], generator=g) * cnt.clamp(min=1).float()).long()
                tgt = ex[torch.arange(ex.shape[0]), pick].view(B, T, Uu)
        new = out.clone()
        new[m] = tgt[m]
        return new

    model.layers[1].ffn[1].register_forward_hook(hook)

    # ---- 采集 exemplar: 每组 N_EXEMPLAR 条真实状态(蓄水池抽样) ----
    print(f"采集 exemplar (每组 {N_EXEMPLAR} 条)…", flush=True)
    EX = torch.zeros(reg.K, N_EXEMPLAR, U)
    EXN = torch.zeros(reg.K, dtype=torch.long)
    seen = torch.zeros(reg.K, dtype=torch.long)
    gsel = torch.Generator().manual_seed(777)
    gen = torch.Generator().manual_seed(DATA_SEED + 31337)
    with torch.no_grad():
        for _ in range(COLLECT_BATCHES):
            ids, _ = generate_batch(BS, SEQ_LEN, spec, generator=gen)
            state["capture"] = True; model(ids); state["capture"] = False
            flat = state["buf"].reshape(-1, U)
            a, _ = reg.assign(reg.extract_sets(flat))
            for p in torch.nonzero(a >= 0).flatten().tolist():
                k = int(a[p]); seen[k] += 1
                if EXN[k] < N_EXEMPLAR:
                    EX[k, EXN[k]] = flat[p]; EXN[k] += 1
                else:
                    j = int(torch.randint(0, int(seen[k]), (1,), generator=gsel))
                    if j < N_EXEMPLAR:
                        EX[k, j] = flat[p]
    print(f"  有 exemplar 的组: {int((EXN>0).sum())}/{reg.n_groups}, "
          f"平均每组 {float(EXN[EXN>0].float().mean()):.1f} 条 ({time.time()-t0:.0f}s)", flush=True)

    arms = ["off", "ema_patch", "exemplar_patch", "exemplar_random", "oracle_patch"]
    acc = {a: [] for a in arms}
    gen = torch.Generator().manual_seed(DATA_SEED + 555)
    with torch.no_grad():
        for bi in range(N_BATCHES):
            ids, _ = generate_batch(BS, SEQ_LEN, spec, generator=gen)
            rk = ids % MEMBERS
            rare = rk >= 6
            if not bool(rare.any()):
                continue
            same_ids = ids.clone(); same_ids[rare] = (ids[rare] // MEMBERS) * MEMBERS
            state["capture"] = True; model(same_ids); state["capture"] = False
            state["donor"] = state["buf"]
            state["mask"] = rare
            for a in arms:
                state["mode"] = a
                lg = model(ids)
                lp = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                                     ids[:, 1:].reshape(-1), reduction="none").view(BS, -1)
                acc[a].append(lp[rk[:, :-1] >= 6])
            state["mode"] = "off"; state["mask"] = None
            if (bi + 1) % 400 == 0:
                print(f"  {bi+1}/{N_BATCHES} ({time.time()-t0:.0f}s)", flush=True)

    base = torch.cat(acc["off"])
    print(f"\n稀有词 n={base.shape[0]}, 基线={float(base.mean()):.4f}", flush=True)
    res = {"n": int(base.shape[0]), "base": float(base.mean()), "n_exemplar": N_EXEMPLAR, "arms": {}}
    for a in arms[1:]:
        d = torch.cat(acc[a]) - base
        se = float(d.std() / d.shape[0] ** 0.5)
        res["arms"][a] = {"delta": float(d.mean()), "se": se}
        tag = "✓改善" if d.mean() < -2 * se else ("✗恶化" if d.mean() > 2 * se else "≈无变化")
        print(f"  {a:18s} Δ={float(d.mean()):+.4f}±{se:.4f}  {tag}", flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
