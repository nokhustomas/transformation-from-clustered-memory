# -*- coding: utf-8 -*-
"""最终配置下的精确测速,用于敲定正式实验的 batch/steps/seeds/budget 组合。"""
import time
import torch
import torch.nn.functional as F

from mqar_data import make_vocab, generate_batch
from models import build_model

VOCAB = make_vocab(key_vocab=16, value_vocab=16, filler_vocab=32)
DIM, LAYERS, HEADS = 64, 2, 4
CHUNK, SEQ_LEN, NUM_PAIRS = 32, 192, 4
BATCH = 16

data_gen = torch.Generator().manual_seed(123)


def eval_batch(model, batch_size=BATCH):
    ids, pos, tgt = generate_batch(batch_size, SEQ_LEN, CHUNK, NUM_PAIRS, VOCAB, generator=data_gen)
    logits = model(ids)
    B = ids.shape[0]
    b_idx = torch.arange(B).unsqueeze(1).expand(B, NUM_PAIRS)
    sel_logits = logits[b_idx, pos]
    loss = F.cross_entropy(sel_logits.reshape(-1, sel_logits.shape[-1]), tgt.reshape(-1))
    return loss


configs = [
    ("nomemory", {}),
    ("fullattn", {}),
    ("topkraw", dict(budget=4)),
    ("topkraw", dict(budget=8)),
    ("cma", dict(budget=4)),
    ("cma", dict(budget=8)),
]

for condition, kwargs in configs:
    torch.manual_seed(0)
    model = build_model(condition, VOCAB["vocab_size"], DIM, LAYERS, HEADS, CHUNK, SEQ_LEN, **kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    n_steps = 20
    t0 = time.time()
    for step in range(n_steps):
        opt.zero_grad()
        loss = eval_batch(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    dt = time.time() - t0
    print(f"[{condition:9s} {str(kwargs):16s}] {dt/n_steps*1000:6.1f} ms/step")
