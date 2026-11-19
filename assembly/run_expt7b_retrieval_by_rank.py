# -*- coding: utf-8 -*-
"""
实验7b: 按词频档位测"取到的组对不对" —— 把论文的核心交叉点数字补成落盘结果。

背景(必须说清楚): 论文 §6 的核心论证是"需要补全的词和能可靠检索的词是反相关的":
出现约77次的词组匹配准确率接近满分, 出现约11次的词掉到瞎猜水平, 而"自信地换错组"
的代价远大于换对的收益, 所以有效窗口近乎为空。这个准确率当初是临时测的(n=70),
没有留下结果文件。本脚本重跑并落盘, 且把样本量提高两个数量级。

"对不对"的定义(不看任何外部标签之外的东西):
  先用每个类里最高频的成员词(rank 0-1)在同一模型上跑, 统计每个类最常被分到哪个组,
  把这个众数组定为该类的"正确组"(覆盖率与纯度一并记录, 作为这个定义是否站得住的证据);
  再对每个 rank 的位置, 算它被分到的组是否等于它所属类的正确组。
同时记录"分到了任何组"的比例(此前实验三报的是这个量, 两者不是一回事, 分开报)。
"""

import json
import os
import sys
import time
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))

from models import FullAttentionLM  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402
from concept_lm_data import make_concept_spec, generate_batch  # noqa: E402
from run_expt3_stress import EMIT_EXTREME  # noqa: E402
from run_expt2_completion import DIM, LAYERS, HEADS, SEQ_LEN, DATA_SEED, OUT_DIR  # noqa: E402

N_BATCHES, BS, MEMBERS, N_CLASSES = 4000, 32, 8, 8
OUT = os.path.join(OUT_DIR, "expt7b_retrieval_by_rank.json")


def main():
    t0 = time.time()
    spec = make_concept_spec(N_CLASSES, MEMBERS, emit_probs=EMIT_EXTREME)
    torch.manual_seed(0)
    model = FullAttentionLM(spec["vocab_size"], DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "expt3_model_seed0.pt")))
    model.eval()
    st = torch.load(os.path.join(OUT_DIR, "expt3_registries_seed0.pt"))["L1"]
    reg = AssemblyRegistry(**st["config"])
    for k in ("freq", "act_ema", "active", "count", "created_at"):
        setattr(reg, k, st[k])
    U = DIM * 4

    buf = {}

    def hook(_m, _i, out):
        buf["a"] = out.detach()
        return None

    model.layers[1].ffn[1].register_forward_hook(hook)

    # rank -> Counter(group) ; 逐 rank 累计组分布
    per_rank = {r: [Counter() for _ in range(N_CLASSES)] for r in range(MEMBERS)}
    per_rank_any = {r: [0, 0] for r in range(MEMBERS)}          # [有组归属, 总数]

    gen = torch.Generator().manual_seed(DATA_SEED + 777)
    with torch.no_grad():
        for bi in range(N_BATCHES):
            ids, _ = generate_batch(BS, SEQ_LEN, spec, generator=gen)
            model(ids)
            a, _ = reg.assign(reg.extract_sets(buf["a"].reshape(-1, U)))
            a = a.view(ids.shape)
            rk = (ids % MEMBERS).reshape(-1)
            cl = (ids // MEMBERS).reshape(-1)
            af = a.reshape(-1)
            for r in range(MEMBERS):
                sel = rk == r
                if not bool(sel.any()):
                    continue
                per_rank_any[r][0] += int((af[sel] >= 0).sum())
                per_rank_any[r][1] += int(sel.sum())
                hit = sel & (af >= 0)
                for c, g in zip(cl[hit].tolist(), af[hit].tolist()):
                    per_rank[r][c][g] += 1
            if (bi + 1) % 1000 == 0:
                print(f"  {bi+1}/{N_BATCHES} ({time.time()-t0:.0f}s)", flush=True)

    # 用 rank 0-1(最高频成员)定义每个类的"正确组"
    canon, canon_purity = {}, {}
    for c in range(N_CLASSES):
        agg = Counter()
        for r in (0, 1):
            agg.update(per_rank[r][c])
        if not agg:
            canon[c] = -1; canon_purity[c] = 0.0
            continue
        g, n = agg.most_common(1)[0]
        canon[c] = int(g)
        canon_purity[c] = n / sum(agg.values())
    print("\n每个类的正确组(由高频成员定义)及其纯度:")
    for c in range(N_CLASSES):
        print(f"  类{c}: 组{canon[c]:3d}  纯度 {canon_purity[c]:.3f}", flush=True)
    distinct = len(set(canon.values()))
    print(f"  → {distinct}/{N_CLASSES} 个类映射到互不相同的组\n", flush=True)

    # 词频档位: 每个 rank 在语料里的期望出现次数
    emit = EMIT_EXTREME
    rows = []
    for r in range(MEMBERS):
        tot = correct = 0
        for c in range(N_CLASSES):
            tot += sum(per_rank[r][c].values())
            correct += per_rank[r][c].get(canon[c], 0)
        anyv, ntot = per_rank_any[r]
        rows.append({
            "rank": r,
            "emit_prob_within_class": float(emit[r]) if r < len(emit) else None,
            "n_positions": ntot,
            "frac_assigned_to_any_group": anyv / max(1, ntot),
            "n_assigned": tot,
            "correct_group_acc_among_assigned": (correct / tot) if tot else None,
            "correct_group_acc_over_all_positions": (correct / ntot) if ntot else None,
        })
        acc = rows[-1]["correct_group_acc_among_assigned"]
        print(f"  rank {r}: 位置 {ntot:7d}  有组归属 {anyv/max(1,ntot):.3f}  "
              f"分对组 {acc if acc is None else f'{acc:.3f}'}", flush=True)

    res = {"n_batches": N_BATCHES, "batch_size": BS, "seq_len": SEQ_LEN,
           "chance_level": 1.0 / N_CLASSES,
           "canonical_group_per_class": canon,
           "canonical_group_purity": canon_purity,
           "n_distinct_canonical_groups": distinct,
           "rows": rows}
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n写入 {OUT}；用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
