# -*- coding: utf-8 -*-
"""
实验七(正式): 跨层两遍推理补全——"哪一层坏了"与"哪一层知道答案"不是同一层。

背景(高n oracle已确立):
- 极端稀有词(11-77次出现)的 loss 缺口 +0.060±0.043,配对 oracle 证明其完全
  嵌入根源: 把稀有词嵌入换成类典型嵌入,配对Δ=-0.079±0.017,缺口填平。
- 但嵌入层自匹配不可行: 稀有词嵌入的最近类命中率只有 0.25(约等于瞎猜)。
- L1 的组匹配对稀有词 100% 命中(rho=0.565)——上下文+部分信号在深层足以识别类。

机制(两遍推理):
  pass1: 正常前向,读取每个位置在 L1 登记库中的组分配 g 与重合度 rho。
  查表:  组 -> 典型嵌入 emb_typ[g](流式构建: 训练分布上把每个位置的输入嵌入
         按其 L1 组分配做 EMA——高频词主导,得到类典型嵌入)。
  pass2: 嵌入替换 emb' = emb + beta * rho * (emb_typ[g] - emb)(吸引子式拉拢,
         不是只补缺失维——oracle 显示稀有词需要整向量修正),重新前向出 logits。
  安慰剂: 组->嵌入映射固定置换。
成本: 推理时两遍前向(约 2x 计算)。上界: oracle 的 -0.08(rare 配对)。
"""

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from models import FullAttentionLM  # noqa: E402
from concept_lm_data import make_concept_spec, generate_batch  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402
from run_expt2_completion import DIM, LAYERS, HEADS, SEQ_LEN, DATA_SEED, OUT_DIR  # noqa: E402
from run_expt3_stress import EMIT_EXTREME  # noqa: E402


def main():
    spec = make_concept_spec(8, 8, emit_probs=EMIT_EXTREME)
    torch.manual_seed(0)
    model = FullAttentionLM(spec["vocab_size"], DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "expt3_model_seed0.pt")))
    model.eval()

    st = torch.load(os.path.join(OUT_DIR, "expt3_registries_seed0.pt"))
    reg = AssemblyRegistry(**st["L1"]["config"])
    for key in ("freq", "act_ema", "active", "count", "created_at"):
        setattr(reg, key, st["L1"][key])

    acts = {}
    hcap = model.layers[1].ffn[1].register_forward_hook(lambda m, i, o: acts.__setitem__(1, o.detach()))

    # ---- 构建 组 -> 典型嵌入 表(流式 EMA, 高频词主导) ----
    print("构建 L1组->典型嵌入 表…", flush=True)
    emb_typ = torch.zeros(reg.K, DIM)
    emb_cnt = torch.zeros(reg.K)
    g = torch.Generator().manual_seed(DATA_SEED + 7777)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(120):
            ids, _ = generate_batch(16, SEQ_LEN, spec, generator=g)
            emb = model.tok_emb(ids).reshape(-1, DIM)
            model(ids)
            sets = reg.extract_sets(acts[1].reshape(-1, DIM * 4))
            a, _ = reg.assign(sets)
            for k in a[a >= 0].unique().tolist():
                sel = a == k
                n = int(sel.sum())
                w = n / (emb_cnt[k] + n)
                emb_typ[k] = (1 - w) * emb_typ[k] + w * emb[sel].mean(0)
                emb_cnt[k] += n
    used = (emb_cnt > 0).sum()
    print(f"  {int(used)} 个组有典型嵌入 ({time.time()-t0:.0f}s)", flush=True)

    # 安慰剂映射: 有典型嵌入的组循环移位
    perm = torch.arange(reg.K)
    act_idx = torch.nonzero(emb_cnt > 0).flatten()
    if len(act_idx) > 1:
        perm[act_idx] = act_idx.roll(1)

    # ---- 两遍推理配对评估 ----
    N_BATCHES = 1500
    betas = [0.5, 1.0]
    modes = [("twopass_b0.5", 0.5, False), ("twopass_b1.0", 1.0, False), ("placebo_b1.0", 1.0, True)]

    sub_emb = {"cur": None}

    def emb_hook(_m, _i, out):
        if sub_emb["cur"] is not None:
            return sub_emb["cur"]
        return None
    hemb = model.tok_emb.register_forward_hook(emb_hook)

    stream = torch.Generator().manual_seed(DATA_SEED + 555)
    res = {m[0]: {"rare": [], "freq": [], "all": []} for m in modes}
    res["off"] = {"rare": [], "freq": [], "all": []}
    hit_stats = {"n_pos": 0, "n_hit": 0, "rho_sum": 0.0, "rare_hit": 0, "rare_n": 0}

    t0 = time.time()
    with torch.no_grad():
        for bi in range(N_BATCHES):
            ids, _ = generate_batch(32, SEQ_LEN, spec, generator=stream)
            B, T = ids.shape
            rk = ids % 8

            # pass1 (也是 off 条件的评估前向)
            sub_emb["cur"] = None
            logits = model(ids)
            lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                 ids[:, 1:].reshape(-1), reduction="none").view(B, -1)
            rmask, fmask = (rk[:, :-1] >= 6), (rk[:, :-1] < 2)
            res["off"]["rare"].append(lp[rmask]); res["off"]["freq"].append(lp[fmask]); res["off"]["all"].append(lp.reshape(-1))

            sets = reg.extract_sets(acts[1].reshape(-1, DIM * 4))
            a, rho = reg.assign(sets)
            hit = (a >= 0) & (emb_cnt[a.clamp_min(0)] > 0)
            hit_stats["n_pos"] += a.shape[0]; hit_stats["n_hit"] += int(hit.sum())
            hit_stats["rho_sum"] += float(rho[hit].sum())
            rflat = rk.reshape(-1)
            hit_stats["rare_hit"] += int((hit & (rflat >= 6)).sum()); hit_stats["rare_n"] += int((rflat >= 6).sum())

            emb0 = model.tok_emb(ids).reshape(-1, DIM)
            for mname, beta, is_placebo in modes:
                k = a.clamp_min(0)
                if is_placebo:
                    k = perm[k]
                typ = emb_typ[k]
                gain = (beta * rho.clamp_min(0.0)).unsqueeze(-1) * hit.unsqueeze(-1).float()
                emb2 = emb0 + gain * (typ - emb0)
                sub_emb["cur"] = emb2.view(B, T, DIM)
                logits2 = model(ids)
                lp2 = F.cross_entropy(logits2[:, :-1].reshape(-1, logits2.shape[-1]),
                                      ids[:, 1:].reshape(-1), reduction="none").view(B, -1)
                res[mname]["rare"].append(lp2[rmask]); res[mname]["freq"].append(lp2[fmask]); res[mname]["all"].append(lp2.reshape(-1))
            sub_emb["cur"] = None
            if (bi + 1) % 300 == 0:
                print(f"  {bi+1}/{N_BATCHES} ({time.time()-t0:.0f}s)", flush=True)

    hemb.remove(); hcap.remove()

    out = {"hit_stats": {"hit_rate": hit_stats["n_hit"]/hit_stats["n_pos"],
                          "mean_rho": hit_stats["rho_sum"]/max(1,hit_stats["n_hit"]),
                          "rare_hit_rate": hit_stats["rare_hit"]/max(1,hit_stats["rare_n"])}}
    print(f"\n命中率={out['hit_stats']['hit_rate']:.3f} rho={out['hit_stats']['mean_rho']:.3f} "
          f"稀有位置命中率={out['hit_stats']['rare_hit_rate']:.3f}")
    for bucket in ("rare", "freq", "all"):
        ref = torch.cat(res["off"][bucket])
        line = f"{bucket:5s} n={ref.shape[0]:8d} off={float(ref.mean()):.4f}"
        out[bucket] = {"n": ref.shape[0], "loss_off": float(ref.mean())}
        for mname, _, _ in modes:
            d = torch.cat(res[mname][bucket]) - ref
            out[bucket][f"delta_{mname}"] = float(d.mean())
            out[bucket][f"se_{mname}"] = float(d.std() / d.shape[0] ** 0.5)
            line += f" | {mname} Δ={float(d.mean()):+.4f}(±{float(d.std()/d.shape[0]**0.5):.4f})"
        print(line, flush=True)

    with open(os.path.join(OUT_DIR, "expt7_twopass_seed0.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n完成。参照: oracle 上界 rare 配对Δ=-0.0786±0.0166", flush=True)


if __name__ == "__main__":
    main()
