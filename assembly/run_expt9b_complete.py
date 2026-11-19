# -*- coding: utf-8 -*-
"""
实验九 v2 主测试: 在 k=16(基线 39.6%)上,补全能不能把没见过的组合拼对。

假设: 训练时"做N遍"的状态在不同动作词之间共亮的 unit 会进组(动作词相关的
unit 因为换来换去进不了组)——所以组=修饰词语义的稳定核。没见过的组合上,
状态里"修饰词该有的样子"若有残缺,补全把它填上,重复次数就该对。

协议(配对):
- 三种模式: 不补全 / 补全 / 注错组的对照,都在同一份 144 个没见过的组合上测。
- 指标: 整题全对率 + 按修饰词分 + 翻转计数(补全后 错->对 有几个, 对->错 有几个)。
- 无害检查: 见过的 112 个组合,补全开着还应该是满分附近。
- 诊断: 组和修饰词到底对不对得上(训练数据上每个组对应哪个修饰词、纯度多少),
  没见过组合上命中率/重合度/每次补几个 unit。
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from models import FullAttentionLM  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402
from run_expt3_stress import StressHook  # noqa: E402
from run_expt9b_scaled import (  # noqa: E402
    N_P, VOCAB, DIM, LAYERS, HEADS, SEQ_LEN, SEP, REPS,
    make_seq, train_combos, heldout_combos, answer_mask, combo_correct, OUT_DIR,
)

K = 16


def load():
    torch.manual_seed(0)
    model = FullAttentionLM(VOCAB, DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"expt9b_model_k{K}_seed0.pt")))
    model.eval()
    st = torch.load(os.path.join(OUT_DIR, f"expt9b_registries_k{K}_seed0.pt"))
    regs = {}
    for li in (0, 1):
        r = AssemblyRegistry(**st[f"L{li}"]["config"])
        for key in ("freq", "act_ema", "active", "count", "created_at"):
            setattr(r, key, st[f"L{li}"][key])
        regs[li] = r
    return model, regs


def diag_group_mod_alignment(model, reg, li, combos):
    """训练组合的答案区位置: 每个组对应哪个修饰词,纯度如何。"""
    acts = {}
    h = model.layers[li].ffn[1].register_forward_hook(lambda m, i, o: acts.__setitem__(0, o.detach()))
    ids = torch.tensor([make_seq(*c) for c in combos], dtype=torch.long)
    with torch.no_grad():
        model(ids)
    h.remove()
    B, T = ids.shape
    flat = acts[0].reshape(-1, acts[0].shape[-1])
    sets = reg.extract_sets(flat)
    a, rho = reg.assign(sets)
    # 答案区 mask(展平)
    am = torch.zeros(B, T, dtype=torch.bool)
    m1 = answer_mask(ids)
    am[:, :-1] |= m1  # 位置 t(预测t+1)属于答案区
    am_flat = am.reshape(-1)
    mods = torch.tensor([c[0] for c in combos]).unsqueeze(1).expand(B, T).reshape(-1)

    sel = am_flat & (a >= 0)
    cnt = torch.zeros(reg.K, 4)
    idx = a[sel] * 4 + mods[sel]
    cnt.view(-1).scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
    out = {}
    for k in range(reg.K):
        n = int(cnt[k].sum())
        if n == 0:
            continue
        top = int(cnt[k].argmax())
        out[k] = {"n": n, "top_mod": top, "purity": float(cnt[k][top] / n)}
    cov = float((a[am_flat] >= 0).float().mean())
    return out, cov


def main():
    model, regs = load()
    tr, ho = train_combos(K), heldout_combos(K)

    print("== 诊断: 答案区状态的组 <-> 修饰词对齐(训练组合上) ==", flush=True)
    for li in (0, 1):
        align, cov = diag_group_mod_alignment(model, regs[li], li, tr)
        pure = [v for v in align.values() if v["n"] >= 50]
        mean_pur = sum(v["purity"] for v in pure) / max(1, len(pure))
        print(f"  L{li}: 覆盖率={cov:.3f} 大组数={len(pure)} 平均修饰词纯度={mean_pur:.3f}", flush=True)
        for k, v in sorted(align.items(), key=lambda kv: -kv[1]["n"])[:8]:
            print(f"    组{k:3d} n={v['n']:5d} 主修饰词=做{REPS[v['top_mod']]}遍 纯度={v['purity']:.2f}", flush=True)

    print("\n== 配对测试: 144 个没见过的组合 ==", flush=True)
    results = {}
    base_ok = combo_correct(model, ho)
    seen_base = sum(combo_correct(model, tr)) / len(tr)
    print(f"  不补全: 没见过 {sum(base_ok)/len(ho):.3f} | 见过 {seen_base:.3f}", flush=True)
    results["off"] = {"heldout": sum(base_ok) / len(ho), "seen": seen_base}

    for site_label, layer_list in (("L1", [1]), ("L0+L1", [0, 1])):
        for mode in ("on", "placebo"):
            hooks = [StressHook(model, li, regs[li], alpha=1.0, mode=mode, drop_frac=0.0)
                     for li in layer_list]
            ho_ok = combo_correct(model, ho)
            seen_acc = sum(combo_correct(model, tr)) / len(tr)
            for h in hooks:
                h.remove()
            flips_up = sum(1 for a, b in zip(base_ok, ho_ok) if (not a) and b)
            flips_dn = sum(1 for a, b in zip(base_ok, ho_ok) if a and (not b))
            per_mod = {}
            for m in (1, 2, 3):
                sub = [i for i, c in enumerate(ho) if c[0] == m]
                per_mod[f"做{REPS[m]}遍"] = sum(ho_ok[i] for i in sub) / len(sub)
            label = f"{site_label}-{mode}"
            results[label] = {"heldout": sum(ho_ok) / len(ho), "seen": seen_acc,
                              "flips_up": flips_up, "flips_down": flips_dn, "per_mod": per_mod}
            print(f"  {label:12s}: 没见过 {results[label]['heldout']:.3f} "
                  f"(错->对 {flips_up}, 对->错 {flips_dn}) | 见过 {seen_acc:.3f} | "
                  + " ".join(f"{kk}={vv:.2f}" for kk, vv in per_mod.items()), flush=True)

    with open(os.path.join(OUT_DIR, "expt9b_completion_k16.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n完成。", flush=True)


if __name__ == "__main__":
    main()
