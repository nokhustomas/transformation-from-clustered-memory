# -*- coding: utf-8 -*-
"""
诊断脚本(不产生正式结果,只用于解释现象): 检查 cma / topkraw 在训练过程中
是否把几乎全部注意力权重都放在了 null_kv(空槽)上,即"学会了忽略检索到的
记忆,只依赖 null 兜底槽"这一退化解 —— 这是解释"budget=4/8/16 三种预算、
topkraw/cma 两种记忆组织方式,在同一 seed 下逐 step 输出几乎逐位对齐"这一
反常现象的首要假设。

复用 run_experiment.py 完全相同的超参数和数据生成方式,只是额外读取
model.last_diags 里的 null_attn_weight_mean 并打印均值。
"""
import math
import torch
import torch.nn.functional as F

from mqar_data import make_vocab, generate_batch
from models import build_model

VOCAB = make_vocab(key_vocab=16, value_vocab=16, filler_vocab=32)
DIM, LAYERS, HEADS = 64, 2, 4
CHUNK, SEQ_LEN, NUM_PAIRS = 32, 192, 4
TRAIN_BATCH, EVAL_BATCH = 16, 256
LR, WARMUP, N_STEPS = 3e-3, 100, 700
GRAD_CLIP = 1.0
DATA_SEED_BASE = 100000


def get_lr_lambda(n_steps, warmup):
    def fn(s):
        if s < warmup:
            return (s + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, n_steps - warmup)))
    return fn


def forward_loss_acc(model, ids, pos, tgt):
    logits = model(ids)
    B = ids.shape[0]
    b_idx = torch.arange(B).unsqueeze(1).expand(B, NUM_PAIRS)
    sel_logits = logits[b_idx, pos]
    loss = F.cross_entropy(sel_logits.reshape(-1, sel_logits.shape[-1]), tgt.reshape(-1))
    acc = (sel_logits.argmax(-1) == tgt).float().mean()
    return loss, acc


def mean_null_weight(model):
    if not model.last_diags:
        return float("nan")
    vals = [d["null_attn_weight_mean"] for d in model.last_diags]
    return sum(vals) / len(vals)


def run(condition, budget, seed=0):
    torch.manual_seed(seed)
    model = build_model(condition, VOCAB["vocab_size"], DIM, LAYERS, HEADS, CHUNK, SEQ_LEN, budget=budget)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, get_lr_lambda(N_STEPS, WARMUP))
    train_gen = torch.Generator().manual_seed(DATA_SEED_BASE + seed)
    eval_gen = torch.Generator().manual_seed(DATA_SEED_BASE + 555 + seed)

    print(f"\n=== {condition} budget={budget} ===")
    for step in range(1, N_STEPS + 1):
        ids, pos, tgt = generate_batch(TRAIN_BATCH, SEQ_LEN, CHUNK, NUM_PAIRS, VOCAB, generator=train_gen)
        opt.zero_grad()
        loss, acc = forward_loss_acc(model, ids, pos, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()

        if step % 100 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                eids, epos, etgt = generate_batch(EVAL_BATCH, SEQ_LEN, CHUNK, NUM_PAIRS, VOCAB, generator=eval_gen)
                eloss, eacc = forward_loss_acc(model, eids, epos, etgt)
                null_w = mean_null_weight(model)
            model.train()
            print(f"step {step:4d}/{N_STEPS}  eval_acc={eacc.item():.4f}  "
                  f"null_attn_weight_mean={null_w:.4f}", flush=True)


if __name__ == "__main__":
    import sys
    if "--pertoken" in sys.argv:
        # 后续实验(逐 token 判别式检索, top_k_sim < 容量)下重新做一次同样的诊断,
        # 确认"没有坍缩到 null 槽"这一结论在修正后的检索机制下依然成立。
        run("topkraw_pertoken", budget=4)
        run("cma_pertoken", budget=4)
    else:
        run("topkraw", budget=4)
        run("cma", budget=4)
