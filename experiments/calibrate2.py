# -*- coding: utf-8 -*-
"""RoPE + 2层后的可学性校准:先只跑 fullattn(最快)确认任务本身能被解出来。"""
import time
import sys
import math
import torch
import torch.nn.functional as F

from mqar_data import make_vocab, generate_batch
from models import build_model

VOCAB = make_vocab(key_vocab=16, value_vocab=16, filler_vocab=32)
DIM, LAYERS, HEADS = 64, 2, 4
CHUNK, SEQ_LEN, NUM_PAIRS = 32, 192, 4
BATCH = 32
LR = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
EVAL_EVERY = 100

train_gen = torch.Generator().manual_seed(777)
eval_gen = torch.Generator().manual_seed(999)


def get_batch(g, batch_size=BATCH):
    return generate_batch(batch_size, SEQ_LEN, CHUNK, NUM_PAIRS, VOCAB, generator=g)


def forward_loss_acc(model, ids, pos, tgt):
    logits = model(ids)
    B = ids.shape[0]
    b_idx = torch.arange(B).unsqueeze(1).expand(B, NUM_PAIRS)
    sel_logits = logits[b_idx, pos]
    loss = F.cross_entropy(sel_logits.reshape(-1, sel_logits.shape[-1]), tgt.reshape(-1))
    acc = (sel_logits.argmax(-1) == tgt).float().mean()
    return loss, acc


torch.manual_seed(0)
model = build_model("fullattn", VOCAB["vocab_size"], DIM, LAYERS, HEADS, CHUNK, SEQ_LEN)
opt = torch.optim.Adam(model.parameters(), lr=LR)
WARMUP = 100
sched = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lambda s: (s + 1) / WARMUP if s < WARMUP else 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / max(1, N_STEPS - WARMUP))),
)
print(f"lr={LR} n_steps={N_STEPS} layers={LAYERS} (warmup+cosine)")

t0 = time.time()
for step in range(1, N_STEPS + 1):
    ids, pos, tgt = get_batch(train_gen)
    opt.zero_grad()
    loss, acc = forward_loss_acc(model, ids, pos, tgt)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()

    if step % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            eids, epos, etgt = get_batch(eval_gen, batch_size=128)
            eloss, eacc = forward_loss_acc(model, eids, epos, etgt)
        model.train()
        dt = time.time() - t0
        print(f"  step {step:4d}  train_loss={loss.item():.3f}  eval_loss={eloss.item():.3f}  eval_acc={eacc.item():.3f}  ({dt:.0f}s elapsed)")
