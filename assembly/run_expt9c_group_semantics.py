# -*- coding: utf-8 -*-
"""
实验9c: 组到底按什么组织的 —— 把论文里"组编码执行进度、不编码修饰词语义"这个
结论补成落盘结果。

背景(必须说清楚): 实验九的这个诊断当初是打印在控制台上的, 没有存结果文件, 而它
是论文解释"为什么组合泛化失败"的核心依据。本脚本重跑, 并把两种解释同时量化, 让
读者自己看哪个说得通:
  解释A: 组 = 修饰词语义("做2遍/做3遍/做4遍"分别有自己的稳定核)
  解释B: 组 = 执行进度(还剩几步要输出), 跨修饰词共用
对每个组算两个纯度: 对修饰词的纯度、对"剩余步数"的纯度。哪个高, 组就按哪个组织。
还给出各自的"瞎猜基线"(该组内最大类别的边际占比), 免得把边际不平衡看成纯度。

只看答案区的位置(即真正要靠修饰词决定重复几次的那些位置), 用训练见过的组合,
两层都测, 同时报同一份数据上的两套登记库(通用登记库 + 只在答案区登记的定向库)。
"""

import json
import os
import sys
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from models import FullAttentionLM  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402
from run_expt9b_scaled import (  # noqa: E402
    VOCAB, DIM, LAYERS, HEADS, SEQ_LEN, REPS, SEP, EOS,
    make_seq, train_combos, answer_mask, OUT_DIR,
)

K = 16
OUT = os.path.join(OUT_DIR, "expt9c_group_semantics.json")


def load_regs(path):
    st = torch.load(path)
    regs = {}
    for li in (0, 1):
        r = AssemblyRegistry(**st[f"L{li}"]["config"])
        for key in ("freq", "act_ema", "active", "count", "created_at"):
            setattr(r, key, st[f"L{li}"][key])
        regs[li] = r
    return regs


def purity(counter):
    n = sum(counter.values())
    return (max(counter.values()) / n, max(counter, key=counter.get), n) if n else (None, None, 0)


def analyze(model, reg, li, combos):
    acts = {}
    h = model.layers[li].ffn[1].register_forward_hook(
        lambda m, i, o: acts.__setitem__(0, o.detach()))
    ids = torch.tensor([make_seq(*c) for c in combos], dtype=torch.long)
    with torch.no_grad():
        model(ids)
    h.remove()
    B, T = ids.shape
    flat = acts[0].reshape(-1, acts[0].shape[-1])
    a, _ = reg.assign(reg.extract_sets(flat))
    a = a.view(B, T)

    am = torch.zeros(B, T, dtype=torch.bool)
    am[:, :-1] |= answer_mask(ids)

    # 每个答案区位置的两个标签: 修饰词下标, 以及"还剩几步"(到 EOS 的距离)
    mods, remain, groups = [], [], []
    for b in range(B):
        eos = int((ids[b] == EOS).nonzero()[0])
        for t in range(T):
            if not bool(am[b, t]) or int(a[b, t]) < 0:
                continue
            mods.append(combos[b][0]); remain.append(eos - t); groups.append(int(a[b, t]))
    cov = float((a[am] >= 0).float().mean())

    by_mod, by_rem = {}, {}
    for g, m, r in zip(groups, mods, remain):
        by_mod.setdefault(g, Counter())[m] += 1
        by_rem.setdefault(g, Counter())[r] += 1

    # 全局边际(瞎猜基线)
    marg_mod = Counter(mods); marg_rem = Counter(remain)
    n_all = len(groups)
    base_mod = max(marg_mod.values()) / n_all if n_all else None
    base_rem = max(marg_rem.values()) / n_all if n_all else None

    rows = []
    for g in sorted(by_mod, key=lambda k: -sum(by_mod[k].values())):
        pm, tm, n = purity(by_mod[g])
        pr, tr, _ = purity(by_rem[g])
        rows.append({"group": g, "n": n,
                     "purity_modifier": pm, "top_modifier_reps": REPS[tm],
                     "purity_remaining_steps": pr, "top_remaining_steps": tr})
    big = [r for r in rows if r["n"] >= 50]
    return {
        "coverage_answer_region": cov, "n_positions": n_all,
        "chance_purity_modifier": base_mod, "chance_purity_remaining_steps": base_rem,
        "n_groups_used": len(rows), "n_groups_big": len(big),
        "mean_purity_modifier_big": (sum(r["purity_modifier"] for r in big) / len(big)) if big else None,
        "mean_purity_remaining_big": (sum(r["purity_remaining_steps"] for r in big) / len(big)) if big else None,
        "groups": rows[:12],
    }


def main():
    torch.manual_seed(0)
    model = FullAttentionLM(VOCAB, DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"expt9b_model_k{K}_seed0.pt")))
    model.eval()
    combos = train_combos(K)

    res = {"k": K, "n_train_combos": len(combos), "archives": {}}
    for tag, path in (("general", f"expt9b_registries_k{K}_seed0.pt"),
                      ("answer_region_targeted", f"expt9b_ansregs_k{K}.pt")):
        p = os.path.join(OUT_DIR, path)
        if not os.path.exists(p):
            print(f"跳过 {tag}(找不到 {path})", flush=True); continue
        regs = load_regs(p)
        res["archives"][tag] = {}
        for li in (0, 1):
            d = analyze(model, regs[li], li, combos)
            res["archives"][tag][f"L{li}"] = d
            print(f"[{tag}] L{li}: 覆盖率={d['coverage_answer_region']:.3f} "
                  f"大组数={d['n_groups_big']}", flush=True)
            print(f"    平均纯度 —— 对修饰词 {d['mean_purity_modifier_big']} "
                  f"(瞎猜 {d['chance_purity_modifier']:.3f}) | "
                  f"对剩余步数 {d['mean_purity_remaining_big']} "
                  f"(瞎猜 {d['chance_purity_remaining_steps']:.3f})", flush=True)
            for r in d["groups"][:5]:
                print(f"      组{r['group']:3d} n={r['n']:5d}  修饰词纯度={r['purity_modifier']:.2f}"
                      f"(做{r['top_modifier_reps']}遍)  剩余步数纯度={r['purity_remaining_steps']:.2f}"
                      f"(剩{r['top_remaining_steps']}步)", flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n写入 {OUT}", flush=True)


if __name__ == "__main__":
    main()
