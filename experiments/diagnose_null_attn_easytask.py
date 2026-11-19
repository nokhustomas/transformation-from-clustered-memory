# -*- coding: utf-8 -*-
"""
诊断脚本(不产生正式结果): 复用 diagnose_null_attn.py 的诊断逻辑,但换成
run_experiment_easytask.py 的真实超参数/词表/seed=0/budget=4 配置,直接检验
一个在分析 easytask 结果时发现、此前从未检查过的现象——

topkraw_easytask(budget=4)、topkraw_easytask(budget=8)、cma_easytask(budget=8)
三个 seed=0 的 run,eval_acc 曲线在全部 13 个 checkpoint 上逐位对齐(不是最终值
巧合相同,是整条曲线数值相同),但与 nomemory_easytask 从 step 600 起分叉。

代码复查已经确认: ClusterMemoryBank 和 RawTopKMemoryBank 都不含任何
nn.Parameter(记忆内容是 reset() 里创建的普通 tensor,不参与随机初始化),
budget/capacity 也不改变模型的参数量或结构。因此在同一个 seed 下,"rawtopk"
和 "cluster" 两种 block_type、以及不同的 top_k_sim,得到的可学习参数初始值是
逐位相同的——如果训练过程中记忆交叉注意力分支对 loss 的梯度贡献也恒为零
(即 null_attn_weight 饱和到 ~1.0,检索到什么内容不再影响输出),那么这三个
条件在整个训练轨迹上逐位相同就有了一个具体、可检验的解释,而不是巧合。

这里只用 diagnose_null_attn.py 已经验证过的诊断字段(null_attn_weight_mean),
换上 easytask 的超参数重新跑一次,不引入任何新逻辑。
"""
import math
import torch
import torch.nn.functional as F

from mqar_data import make_vocab, generate_batch
from models import build_model

# 与 run_experiment_easytask.py 完全一致的超参数
VOCAB = make_vocab(key_vocab=8, value_vocab=8, filler_vocab=32)
DIM, LAYERS, HEADS = 64, 2, 4
CHUNK, SEQ_LEN, NUM_PAIRS = 32, 192, 2
INSERT_REGION_FRAC = 0.4
TRAIN_BATCH, EVAL_BATCH = 16, 256
LR, WARMUP, N_STEPS = 3e-3, 100, 1300
GRAD_CLIP = 1.0
DATA_SEED_BASE = 200000


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


def gen_batch(bs, generator):
    return generate_batch(bs, SEQ_LEN, CHUNK, NUM_PAIRS, VOCAB,
                           insert_region_frac=INSERT_REGION_FRAC, generator=generator)


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

    print(f"\n=== {condition} budget={budget} (easytask 超参数) ===", flush=True)
    for step in range(1, N_STEPS + 1):
        ids, pos, tgt = gen_batch(TRAIN_BATCH, train_gen)
        opt.zero_grad()
        loss, acc = forward_loss_acc(model, ids, pos, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()

        if step % 100 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                eids, epos, etgt = gen_batch(EVAL_BATCH, eval_gen)
                eloss, eacc = forward_loss_acc(model, eids, epos, etgt)
                null_w = mean_null_weight(model)
            model.train()
            print(f"step {step:4d}/{N_STEPS}  eval_acc={eacc.item():.4f}  "
                  f"null_attn_weight_mean={null_w:.4f}", flush=True)


if __name__ == "__main__":
    run("topkraw_pertoken", budget=4)
    run("cma_pertoken", budget=4)
