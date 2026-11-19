# -*- coding: utf-8 -*-
"""
后续实验: 检索方式从"整 chunk 池化 query"换成"逐 token query",同时把 top_k_sim
降到容量的一半(见 models.py build_model 里 "_pertoken" 两个 condition 的注释),
使检索"筛选"这一步真正依赖 query 内容而不是恒等操作。

背景: 已交付论文第 5.2 节把 TopKRaw/CMA 在 B=4 时都停在随机水平的现象,归因于
"pooled_query 导致检索无法区分 chunk 内的不同 query"。但检查代码后发现,原始实验
里 top_k_sim 被设成等于各自的全部容量,检索"筛选"本来就是恒等操作(不管 query 是
什么,返回的都是全部记忆)——这意味着那个归因站不住脚。这个脚本运行的是"让检索
筛选真正起作用"之后的版本,用来诚实回答:这样修改之后,CMA 能否在 B=4 时反超
TopKRaw?

与 run_experiment.py 共用同一份 results/raw_results.jsonl(新 condition 名字
"topkraw_pertoken" / "cma_pertoken" 不会与旧记录冲突),同样支持断点续跑。
nomemory / fullattn 两个基线不受这次改动影响,直接复用 run_experiment.py 已经
跑出的结果作参照,不重新训练。

超参数(学习率、步数、batch、模型规模)与 run_experiment.py 完全一致,只有
condition 名字和背后的 top_k_sim / 检索方式不同,确保这仍然是一次公平对照。
"""
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from mqar_data import make_vocab, generate_batch
from models import build_model, count_params

VOCAB = make_vocab(key_vocab=16, value_vocab=16, filler_vocab=32)
DIM, LAYERS, HEADS = 64, 2, 4
CHUNK, SEQ_LEN, NUM_PAIRS = 32, 192, 4
TRAIN_BATCH, EVAL_BATCH = 16, 256
LR, WARMUP, N_STEPS = 3e-3, 100, 1300
EVAL_EVERY = 100
GRAD_CLIP = 1.0

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "raw_results.jsonl")
DATA_SEED_BASE = 100000  # 与 run_experiment.py 完全相同,保证同一 seed 看到相同的数据流


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


def run_one(condition, seed, budget, log_prefix=""):
    torch.manual_seed(seed)
    model = build_model(condition, VOCAB["vocab_size"], DIM, LAYERS, HEADS, CHUNK, SEQ_LEN, budget=budget)
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, get_lr_lambda(N_STEPS, WARMUP))

    train_gen = torch.Generator().manual_seed(DATA_SEED_BASE + seed)
    eval_gen = torch.Generator().manual_seed(DATA_SEED_BASE + 555 + seed)

    curve = []
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        ids, pos, tgt = generate_batch(TRAIN_BATCH, SEQ_LEN, CHUNK, NUM_PAIRS, VOCAB, generator=train_gen)
        opt.zero_grad()
        loss, acc = forward_loss_acc(model, ids, pos, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()

        if step % EVAL_EVERY == 0 or step == N_STEPS:
            model.eval()
            with torch.no_grad():
                eids, epos, etgt = generate_batch(EVAL_BATCH, SEQ_LEN, CHUNK, NUM_PAIRS, VOCAB, generator=eval_gen)
                eloss, eacc = forward_loss_acc(model, eids, epos, etgt)
            model.train()
            curve.append({"step": step, "eval_loss": eloss.item(), "eval_acc": eacc.item()})
            print(f"{log_prefix} step {step:4d}/{N_STEPS}  eval_acc={eacc.item():.3f}  "
                  f"eval_loss={eloss.item():.3f}  ({time.time()-t0:.0f}s)", flush=True)

    dt = time.time() - t0
    result = {
        "condition": condition, "seed": seed, "budget": budget, "n_params": n_params,
        "n_steps": N_STEPS, "wall_clock_sec": dt, "final_eval_acc": curve[-1]["eval_acc"],
        "final_eval_loss": curve[-1]["eval_loss"], "curve": curve,
    }
    return result


def append_result(result):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def already_done(condition, seed, budget):
    if not os.path.exists(RESULTS_PATH):
        return False
    with open(RESULTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] == condition and r["seed"] == seed and r["budget"] == budget:
                return True
    return False


def main():
    plan = []
    # 主实验: budget=4, 3 个种子, 2 个 "_pertoken" 条件(与旧实验的 3-seed 主实验对齐)
    for seed in [0, 1, 2]:
        plan.append(("topkraw_pertoken", seed, 4))
        plan.append(("cma_pertoken", seed, 4))
    # 补充预算扫描: budget in {8, 16}, 单种子(seed=0),与旧实验的补充扫描对齐
    for budget in [8, 16]:
        plan.append(("topkraw_pertoken", 0, budget))
        plan.append(("cma_pertoken", 0, budget))

    print(f"总计划 {len(plan)} 个 run, N_STEPS={N_STEPS}")
    for i, (condition, seed, budget) in enumerate(plan):
        if already_done(condition, seed, budget):
            print(f"[{i+1}/{len(plan)}] 跳过(已完成): {condition} seed={seed} budget={budget}")
            continue
        prefix = f"[{i+1}/{len(plan)} {condition} seed={seed} budget={budget}]"
        print(f"\n=== 开始 {prefix} ===", flush=True)
        result = run_one(condition, seed, budget, log_prefix=prefix)
        append_result(result)
        print(f"=== 完成 {prefix}  final_acc={result['final_eval_acc']:.3f} "
              f"用时{result['wall_clock_sec']:.0f}s ===", flush=True)

    print("\n全部实验完成。")


if __name__ == "__main__":
    main()
