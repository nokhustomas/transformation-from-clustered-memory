# -*- coding: utf-8 -*-
"""快速联通性 + 速度测试:每种条件跑几十步,确认能训练、并估算单步耗时以规划正式实验规模。"""
import time
import torch
import torch.nn.functional as F

from mqar_data import make_vocab, generate_batch
from models import build_model, count_params

torch.manual_seed(0)

VOCAB = make_vocab(key_vocab=32, value_vocab=32, filler_vocab=32)
DIM, LAYERS, HEADS = 64, 1, 4
CHUNK, SEQ_LEN, NUM_PAIRS = 32, 256, 8
BUDGET = 16

data_gen = torch.Generator().manual_seed(123)


def eval_batch(model, condition, batch_size=32):
    ids, pos, tgt = generate_batch(batch_size, SEQ_LEN, CHUNK, NUM_PAIRS, VOCAB, generator=data_gen)
    logits = model(ids)
    B = ids.shape[0]
    b_idx = torch.arange(B).unsqueeze(1).expand(B, NUM_PAIRS)
    sel_logits = logits[b_idx, pos]
    loss = F.cross_entropy(sel_logits.reshape(-1, sel_logits.shape[-1]), tgt.reshape(-1))
    acc = (sel_logits.argmax(-1) == tgt).float().mean()
    return loss, acc


for condition in ["nomemory", "fullattn", "topkraw", "cma"]:
    torch.manual_seed(0)
    kwargs = dict(budget=BUDGET) if condition in ("topkraw", "cma") else {}
    model = build_model(condition, VOCAB["vocab_size"], DIM, LAYERS, HEADS, CHUNK, SEQ_LEN, **kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    n_params = count_params(model)

    t0 = time.time()
    n_steps = 20
    for step in range(n_steps):
        opt.zero_grad()
        loss, acc = eval_batch(model, condition, batch_size=32)
        loss.backward()
        opt.step()
    dt = time.time() - t0
    print(f"[{condition:9s}] params={n_params:7d}  {n_steps}steps={dt:6.2f}s  "
          f"({dt/n_steps*1000:.1f} ms/step)  last_loss={loss.item():.3f}  last_acc={acc.item():.3f}")
