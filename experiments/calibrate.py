# -*- coding: utf-8 -*-
"""校准训练步数:每种条件跑一个 seed、较多步数,每隔一段打印准确率曲线，用来决定正式实验的步数。"""
import time
import torch
import torch.nn.functional as F

from mqar_data import make_vocab, generate_batch
from models import build_model

VOCAB = make_vocab(key_vocab=32, value_vocab=32, filler_vocab=32)
DIM, LAYERS, HEADS = 64, 1, 4
CHUNK, SEQ_LEN, NUM_PAIRS = 32, 256, 8
N_STEPS = 900
EVAL_EVERY = 100
BATCH = 32

train_gen = torch.Generator().manual_seed(777)
eval_gen = torch.Generator().manual_seed(999)


def get_batch(g, batch_size=BATCH):
    ids, pos, tgt = generate_batch(batch_size, SEQ_LEN, CHUNK, NUM_PAIRS, VOCAB, generator=g)
    return ids, pos, tgt


def forward_loss_acc(model, ids, pos, tgt):
    logits = model(ids)
    B = ids.shape[0]
    b_idx = torch.arange(B).unsqueeze(1).expand(B, NUM_PAIRS)
    sel_logits = logits[b_idx, pos]
    loss = F.cross_entropy(sel_logits.reshape(-1, sel_logits.shape[-1]), tgt.reshape(-1))
    acc = (sel_logits.argmax(-1) == tgt).float().mean()
    return loss, acc


for condition, budget in [("nomemory", None), ("fullattn", None), ("topkraw", 16), ("cma", 16)]:
    torch.manual_seed(0)
    kwargs = dict(budget=budget) if budget is not None else {}
    model = build_model(condition, VOCAB["vocab_size"], DIM, LAYERS, HEADS, CHUNK, SEQ_LEN, **kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    t0 = time.time()
    curve = []
    for step in range(1, N_STEPS + 1):
        ids, pos, tgt = get_batch(train_gen)
        opt.zero_grad()
        loss, acc = forward_loss_acc(model, ids, pos, tgt)
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                eids, epos, etgt = get_batch(eval_gen, batch_size=128)
                eloss, eacc = forward_loss_acc(model, eids, epos, etgt)
            model.train()
            curve.append((step, eloss.item(), eacc.item()))
    dt = time.time() - t0
    print(f"\n== {condition} (budget={budget}) 用时 {dt:.1f}s ==")
    for step, l, a in curve:
        print(f"  step {step:4d}  eval_loss={l:.3f}  eval_acc={a:.3f}")
