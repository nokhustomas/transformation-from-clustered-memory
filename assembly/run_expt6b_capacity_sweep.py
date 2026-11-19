# -*- coding: utf-8 -*-
"""
实验6b: 真实文本的容量扫描 —— 把论文里引用过、但此前只以图里的硬编码数字存在的
那条"容量律"补成真正跑出来、落盘的结果。

背景(必须说清楚): 论文 §4 声称"K=512 时覆盖率到 1.0, 而 K=32 只有 0.08", 并给出
自发生成的组数。这个扫描当初是临时跑的, 数字被手抄进了 gen_figures.py 的字面量,
没有留下结果文件。这与本项目"图里的每个数都直接读结果文件"的规矩相冲突。本脚本
重跑该扫描并落盘, 让论文那条曲线有据可查。

做法: 载入实验六训练好的字符级模型(冻结), 在同一批探针文本上取第二层 FFN 激活,
对每个 (K_max, τ, act_frac) 组合新建一个空登记器、流式登记一遍, 记录:
  组数 / 覆盖率 / 与三种天然标注的对齐程度(字符身份、5类粗分类、是否词首)
覆盖率 = 有多少位置的激活集合能匹配到某个已登记组(Jaccard ≥ τ)。
"""

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))

from registry import AssemblyRegistry  # noqa: E402
from models import FullAttentionLM  # noqa: E402
from run_expt6_realtext import (  # noqa: E402
    ActCapture, char_category, nmi, load_corpus,
    DIM, LAYERS, HEADS, SEQ_LEN, PROBE_BATCH, FREQ_LR, OUT_DIR,
)

K_GRID = [32, 128, 512, 1024]
TAU_GRID = [0.2, 0.3]
ACT_GRID = [0.5, 0.75]
OUT = os.path.join(OUT_DIR, "expt6b_capacity_sweep.json")


def main():
    t0 = time.time()
    data, stoi, itos = load_corpus()
    vocab_size = len(itos)

    torch.manual_seed(0)
    model = FullAttentionLM(vocab_size, DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "expt6_model_seed0.pt")))
    model.eval()

    gen = torch.Generator().manual_seed(1234)
    starts = torch.randint(0, len(data) - SEQ_LEN - 1, (PROBE_BATCH,), generator=gen)
    probe = torch.stack([data[s:s + SEQ_LEN] for s in starts])

    cap = ActCapture(model)
    with torch.no_grad():
        model(probe)
    for h in cap.handles:
        h.remove()

    flat_ids = probe.reshape(-1)
    cats = torch.tensor([char_category(itos[int(i)]) for i in flat_ids])
    ps = probe.clone(); ps[:, 1:] = probe[:, :-1]; ps[:, 0] = 0
    prev_space = torch.tensor([1 if itos[int(i)] in " \t\n\r" else 0
                               for i in ps.reshape(-1)])

    acts = cap.acts[1].reshape(-1, cap.acts[1].shape[-1])       # 第二层(论文用的那层)
    U = acts.shape[-1]
    print(f"探针位置 {acts.shape[0]} 个, 单元数 {U}, 词表 {vocab_size}", flush=True)

    rows = []
    for act_frac in ACT_GRID:
        probe_reg = AssemblyRegistry(U, 8, 0.3, FREQ_LR, act_frac=act_frac)
        sets = probe_reg.extract_sets(acts)
        set_size = float(sets.float().sum(-1).mean())
        for tau in TAU_GRID:
            for K in K_GRID:
                reg = AssemblyRegistry(U, K, tau, FREQ_LR, act_frac=act_frac)
                reg.register(sets)
                a, _ = reg.assign(sets)
                ok = a >= 0
                cov = float(ok.float().mean())
                row = {
                    "act_frac": act_frac, "tau": tau, "K_max": K,
                    "mean_set_size": set_size,
                    "n_groups": int(reg.n_groups),
                    "saturated": bool(reg.n_groups >= K),
                    "merge_events": int(reg.merge_events),
                    "coverage": cov,
                    "nmi_char": nmi(a[ok], flat_ids[ok], K, vocab_size) if ok.any() else None,
                    "nmi_cat5": nmi(a[ok], cats[ok], K, 5) if ok.any() else None,
                    "nmi_wordstart": nmi(a[ok], prev_space[ok], K, 2) if ok.any() else None,
                }
                rows.append(row)
                print(f"  act_frac={act_frac} τ={tau} K={K:5d} → 组数 {row['n_groups']:4d}"
                      f"{' (满容量)' if row['saturated'] else '        '}"
                      f"  覆盖率 {cov:.3f}  对齐(字符/5类/词首) "
                      f"{row['nmi_char']:.3f}/{row['nmi_cat5']:.3f}/{row['nmi_wordstart']:.3f}",
                      flush=True)

    res = {"layer": 1, "n_positions": int(acts.shape[0]), "n_units": U,
           "vocab_size": vocab_size, "probe_batch": PROBE_BATCH, "seq_len": SEQ_LEN,
           "rows": rows}
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n写入 {OUT}；用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
