# -*- coding: utf-8 -*-
"""
实验八: 把实验三A的"损坏修复"因果结果延伸到真实语言。

设置: 冻结实验六的字符级模型(真实英文, loss 1.33),在 L1 上用实验六敏感性扫描
选出的容量配置(K=512, tau=0.2, act_frac=0.5——该配置下快照覆盖率 100%)流式
重建带典型激活值的登记库,然后跑与实验三A完全相同的配对协议:
  clean_off / corrupt_off / corrupt_on / corrupt_placebo, drop ∈ {0.3, 0.5}
损坏掩码按批号重播种,跨模式一致。

问题: "部分线索 -> 整组补全"的修复效应,在没有人工类结构、组数以百计的
真实语言表征上还成立吗?
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
from registry import AssemblyRegistry  # noqa: E402
from run_expt3_stress import StressHook  # noqa: E402
from run_expt6_realtext import (  # noqa: E402
    load_corpus, make_batch, DIM, LAYERS, HEADS, SEQ_LEN, DATA_SEED,
)

K_BIG, TAU_BIG, AF, FLR = 512, 0.2, 0.5, 0.05
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    ids_corpus, stoi, itos = load_corpus()
    V = max(itos.keys()) + 1
    torch.manual_seed(0)
    model = FullAttentionLM(V, DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "expt6_model_seed0.pt")))
    model.eval()

    # 流式重建 L1 大容量登记库(带典型值)
    print(f"重建 L1 登记库 K={K_BIG} tau={TAU_BIG}…", flush=True)
    reg = AssemblyRegistry(DIM * 4, K_BIG, TAU_BIG, FLR, act_frac=AF)
    acts = {}
    h = model.layers[1].ffn[1].register_forward_hook(lambda m, i, o: acts.__setitem__(1, o.detach()))
    g = torch.Generator().manual_seed(DATA_SEED + 8888)
    t0 = time.time()
    with torch.no_grad():
        for b in range(150):
            ids = make_batch(ids_corpus, 16, SEQ_LEN, g)
            model(ids)
            flat = acts[1].reshape(-1, DIM * 4)[::8]
            reg.register(reg.extract_sets(flat), values=flat)
    h.remove()
    print(f"  组数={reg.n_groups} 合并={reg.merge_events} 登记={reg.total_registered} ({time.time()-t0:.0f}s)", flush=True)

    # 覆盖率抽查
    probe_g = torch.Generator().manual_seed(DATA_SEED + 555)
    probe = make_batch(ids_corpus, 32, SEQ_LEN, probe_g)
    h = model.layers[1].ffn[1].register_forward_hook(lambda m, i, o: acts.__setitem__(1, o.detach()))
    with torch.no_grad():
        model(probe)
    h.remove()
    sets = reg.extract_sets(acts[1].reshape(-1, DIM * 4))
    a, _ = reg.assign(sets)
    print(f"  覆盖率(探针)={float((a>=0).float().mean()):.3f}", flush=True)

    results = {}
    n_batches = 100
    eval_g = torch.Generator().manual_seed(DATA_SEED + 555)
    all_ids = [make_batch(ids_corpus, 16, SEQ_LEN, eval_g) for _ in range(n_batches)]

    for drop in (0.3, 0.5):
        acc = {}
        stats = {}
        for mode_name, mode, df in (("clean_off", "off", 0.0), ("corrupt_off", "off", drop),
                                     ("corrupt_on", "on", drop), ("corrupt_placebo", "placebo", drop)):
            hook = StressHook(model, 1, reg, alpha=1.0, mode=mode, drop_frac=df)
            losses = []
            with torch.no_grad():
                for bi, ids in enumerate(all_ids):
                    hook.batch_idx = bi
                    logits = model(ids)
                    lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                         ids[:, 1:].reshape(-1), reduction="none")
                    losses.append(lp)
            acc[mode_name] = torch.cat(losses)
            stats[mode_name] = hook.stats()
            hook.remove()

        ref = acc["clean_off"]
        row = {"loss_clean": float(ref.mean()), "hook_stats": stats}
        for mname in ("corrupt_off", "corrupt_on", "corrupt_placebo"):
            d = acc[mname] - ref
            row[f"delta_{mname}"] = float(d.mean())
            row[f"se_{mname}"] = float(d.std() / d.shape[0] ** 0.5)
        recov = (row["delta_corrupt_off"] - row["delta_corrupt_on"]) / max(1e-9, row["delta_corrupt_off"])
        row["recovery_frac"] = recov
        results[str(drop)] = row
        hs = stats["corrupt_on"]
        print(f"drop={drop}: 损坏Δ=+{row['delta_corrupt_off']:.4f} | 补全Δ=+{row['delta_corrupt_on']:.4f} "
              f"(恢复 {recov*100:.0f}%) | 安慰剂Δ=+{row['delta_corrupt_placebo']:.4f} | "
              f"命中率={hs['hit_rate']:.3f} rho={hs['mean_rho']:.3f} 注入/命中={hs['mean_injected_units_per_hit']:.1f}", flush=True)

    with open(os.path.join(OUT_DIR, "expt8_realrepair_seed0.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("完成。", flush=True)


if __name__ == "__main__":
    main()
