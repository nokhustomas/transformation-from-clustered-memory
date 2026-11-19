# -*- coding: utf-8 -*-
"""
实验13: 替换(activation patching) vs 叠加(addition)——信息进入网络的合法通道是哪种形式。

此前所有补全实验做的都是**加法**: h <- h + 增量。标准的 activation patching 是**替换**:
h <- 供体的完整隐状态。若替换能在 L1 吃到 oracle 的一部分而加法不能, 病灶描述就从
"信息必须从源头(嵌入)进"细化为"信息必须以完整状态的形式存在, 不能以增量形式叠加"。

供体怎么取(关键): 把序列里的稀有词换成**同类最高频词**再跑一遍前向, 取该位置的完整
隐状态。因为任务的下一词分布只取决于**类**、与类内是哪个成员无关, 所以换成同类成员
不改变正确答案 —— 这正是嵌入层 oracle 能填平缺口(-0.079)的原因。稀有词占比约 2e-4,
整批替换对上下文的影响可忽略。

各组(全部配对, 同一批数据, 只在稀有词位置动手):
  off                    不动(基线)
  patch_ffn_L1           用供体的 L1 FFN 激活**替换**  <- 用户提议的判别实验
  patch_resid_L1         用供体的 L1 层输出(残差流)替换 <- 标准 patching 位置
  patch_typical_ffn_L1   用登记库里该组的典型激活替换(可部署版, 非 oracle)
  add_ffn_L1             旧方式: 加上"该亮未亮"成员的典型值(此前唯一测过的形式)
  oracle_embedding       已知上界: 稀有词嵌入换成同类高频词嵌入
"""

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))

from models import FullAttentionLM  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402
from concept_lm_data import make_concept_spec, generate_batch  # noqa: E402
from run_expt3_stress import EMIT_EXTREME  # noqa: E402
from run_expt2_completion import DIM, LAYERS, HEADS, SEQ_LEN, DATA_SEED, OUT_DIR  # noqa: E402

N_BATCHES = 1500
BS = 32
MEMBERS = 8
OUT = os.path.join(OUT_DIR, "expt13_patching.json")


def main():
    t0 = time.time()
    spec = make_concept_spec(8, MEMBERS, emit_probs=EMIT_EXTREME)
    torch.manual_seed(0)
    model = FullAttentionLM(spec["vocab_size"], DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "expt3_model_seed0.pt")))
    model.eval()

    st = torch.load(os.path.join(OUT_DIR, "expt3_registries_seed0.pt"))
    d = st["L1"]
    reg = AssemblyRegistry(**d["config"])
    for k in ("freq", "act_ema", "active", "count", "created_at"):
        setattr(reg, k, d[k])

    E0 = model.tok_emb.weight.detach().clone()
    cls = torch.arange(spec["vocab_size"]) // MEMBERS
    rank = torch.arange(spec["vocab_size"]) % MEMBERS
    # oracle 嵌入: 稀有词 <- 同类 rank0 词的嵌入
    E_oracle = E0.clone()
    for t in range(spec["vocab_size"]):
        if rank[t] >= 6:
            donor = int(((cls == cls[t]) & (rank == 0)).nonzero()[0])
            E_oracle[t] = E0[donor]

    state = {"mode": "off", "capture": None, "donor_ffn": None, "donor_resid": None, "mask": None}

    def ffn_hook(_m, _i, out):
        if state["capture"] == "ffn":
            state["donor_ffn"] = out.detach().clone()
            return None
        if state["mode"] not in ("patch_ffn", "patch_typical", "add_ffn") or state["mask"] is None:
            return None
        B, T, U = out.shape
        new = out.clone()
        m = state["mask"]
        if state["mode"] == "patch_ffn":
            new[m] = state["donor_ffn"][m]
        else:
            flat = out.reshape(-1, U).detach()
            sets = reg.extract_sets(flat)
            a, rho = reg.assign(sets)
            hit = (a >= 0)
            k = a.clamp_min(0)
            typ = reg.act_ema[k].view(B, T, U)
            if state["mode"] == "patch_typical":
                sel = m & hit.view(B, T)
                new[sel] = typ[sel]
            else:  # add_ffn
                miss = (reg.membership[k] & (~sets) & hit.unsqueeze(-1)).view(B, T, U)
                g = (rho.clamp_min(0.0)).view(B, T, 1)
                inc = typ * miss.float() * g
                new = new + inc * m.unsqueeze(-1).float()
        return new

    def resid_hook(_m, _i, out):
        if state["capture"] == "resid":
            state["donor_resid"] = out.detach().clone()
            return None
        if state["mode"] != "patch_resid" or state["mask"] is None:
            return None
        new = out.clone()
        new[state["mask"]] = state["donor_resid"][state["mask"]]
        return new

    model.layers[1].ffn[1].register_forward_hook(ffn_hook)
    model.layers[1].register_forward_hook(resid_hook)

    modes = ["off", "patch_ffn_L1", "patch_resid_L1", "patch_typical_ffn_L1",
             "add_ffn_L1", "oracle_embedding"]
    acc = {m: [] for m in modes}
    accf = {m: [] for m in modes}

    gen = torch.Generator().manual_seed(DATA_SEED + 555)
    with torch.no_grad():
        for bi in range(N_BATCHES):
            ids, _ = generate_batch(BS, SEQ_LEN, spec, generator=gen)
            rk = ids % MEMBERS
            rare_mask = rk >= 6                      # [B, T] 输入词是稀有词的位置
            if not bool(rare_mask.any()):
                continue
            # 供体序列: 稀有词 -> 同类 rank0
            donor_ids = ids.clone()
            donor_ids[rare_mask] = (ids[rare_mask] // MEMBERS) * MEMBERS
            # 抓供体的两处隐状态
            state["mode"] = "off"; state["mask"] = None
            state["capture"] = "ffn"; model(donor_ids)
            state["capture"] = "resid"; model(donor_ids)
            state["capture"] = None
            state["mask"] = rare_mask

            for mode in modes:
                if mode == "oracle_embedding":
                    model.tok_emb.weight.copy_(E_oracle)
                    state["mode"] = "off"
                else:
                    model.tok_emb.weight.copy_(E0)
                    state["mode"] = {"off": "off", "patch_ffn_L1": "patch_ffn",
                                     "patch_resid_L1": "patch_resid",
                                     "patch_typical_ffn_L1": "patch_typical",
                                     "add_ffn_L1": "add_ffn"}[mode]
                logits = model(ids)
                lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                     ids[:, 1:].reshape(-1), reduction="none").view(BS, -1)
                r = rk[:, :-1] >= 6
                f = rk[:, :-1] < 2
                acc[mode].append(lp[r]); accf[mode].append(lp[f])
            model.tok_emb.weight.copy_(E0)
            state["mode"] = "off"; state["mask"] = None
            if (bi + 1) % 300 == 0:
                print(f"  {bi+1}/{N_BATCHES} ({time.time()-t0:.0f}s)", flush=True)

    base = torch.cat(acc["off"]); basef = torch.cat(accf["off"])
    print(f"\n稀有词位置 n={base.shape[0]}, 基线 loss={float(base.mean()):.4f}; "
          f"高频词基线={float(basef.mean()):.4f}\n", flush=True)
    res = {"n_rare": int(base.shape[0]), "base_rare": float(base.mean()),
           "base_freq": float(basef.mean()), "arms": {}}
    for mode in modes[1:]:
        x = torch.cat(acc[mode]); xf = torch.cat(accf[mode])
        dr = x - base; df = xf - basef
        res["arms"][mode] = {"rare_delta": float(dr.mean()),
                             "rare_se": float(dr.std() / dr.shape[0] ** 0.5),
                             "freq_delta": float(df.mean()),
                             "freq_se": float(df.std() / df.shape[0] ** 0.5)}
        a = res["arms"][mode]
        sig = "✓有效" if a["rare_delta"] < -2 * a["rare_se"] else ("✗有害" if a["rare_delta"] > 2 * a["rare_se"] else "≈无效")
        print(f"{mode:24s} 稀有词 Δ={a['rare_delta']:+.4f}±{a['rare_se']:.4f} {sig}   "
              f"高频词 Δ={a['freq_delta']:+.5f}±{a['freq_se']:.5f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
