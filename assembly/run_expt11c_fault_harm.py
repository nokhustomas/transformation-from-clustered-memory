# -*- coding: utf-8 -*-
"""
实验11c: 每种故障到底伤模型多少 —— 把"检测得到 vs 伤得重"这条对照补成落盘结果。

背景(必须说清楚): 论文 §8 说"8级量化确实伤模型(+0.09)却只被部分检出", 这是一个
关键论点(它说明检出率不跟伤害走), 但当初只临时测过、没有存结果文件。本脚本把
实验十一里全部故障档位的伤害(loss 变化)一次测全并落盘, 和已有的 AUC 表配套。

同一批文本、同一模型、逐档注入故障, 报注入后 loss 与健康 loss 的差。
"""

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))

from models import FullAttentionLM  # noqa: E402
from run_expt6_realtext import load_corpus, DIM, LAYERS, HEADS, SEQ_LEN, OUT_DIR  # noqa: E402
from run_expt11_harden import FaultInjector  # noqa: E402

N_BATCH, BS = 40, 32
LAYER = 1
OUT = os.path.join(OUT_DIR, "expt11c_fault_harm.json")

FAULTS = [
    ("drop 10%", "drop", 0.10), ("drop 30%", "drop", 0.30), ("drop 50%", "drop", 0.50),
    ("noise 0.5x", "noise", 0.5), ("noise 1.0x", "noise", 1.0),
    ("stuck 10%", "stuck", 0.10), ("stuck 30%", "stuck", 0.30),
    ("quant 4 levels", "quant", 4), ("quant 8 levels", "quant", 8),
]


def main():
    t0 = time.time()
    data, stoi, itos = load_corpus()
    vocab_size = len(itos)
    torch.manual_seed(0)
    model = FullAttentionLM(vocab_size, DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "expt6_model_seed0.pt")))
    model.eval()

    gen = torch.Generator().manual_seed(20260803)
    batches = []
    for _ in range(N_BATCH):
        st = torch.randint(0, len(data) - SEQ_LEN - 1, (BS,), generator=gen)
        batches.append(torch.stack([data[s:s + SEQ_LEN] for s in st]))

    inj = FaultInjector(model.layers[LAYER].ffn[1])

    def run():
        tot = []
        with torch.no_grad():
            for bi, ids in enumerate(batches):
                inj.batch_idx = bi
                lg = model(ids)
                tot.append(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                                           ids[:, 1:].reshape(-1)))
        return torch.stack(tot)

    inj.kind = "none"
    healthy = run()
    hb = float(healthy.mean())
    print(f"健康 loss = {hb:.4f}  ({N_BATCH}×{BS} 条 × {SEQ_LEN} 字符)\n", flush=True)

    res = {"layer": LAYER, "healthy_loss": hb, "n_batches": N_BATCH,
           "batch_size": BS, "seq_len": SEQ_LEN, "faults": {}}
    for name, kind, param in FAULTS:
        inj.kind = kind
        inj.param = param
        if kind == "stuck":
            inj.set_stuck(param, DIM * 4)
        v = run()
        d = v - healthy                              # 配对差(同一批文本)
        se = float(d.std() / d.shape[0] ** 0.5)
        res["faults"][name] = {"loss": float(v.mean()), "harm": float(d.mean()), "se": se}
        print(f"  {name:16s} loss={float(v.mean()):.4f}  伤害 {float(d.mean()):+.4f} ± {se:.4f}",
              flush=True)
    inj.kind = "none"
    inj.remove()
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n写入 {OUT}；用时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
