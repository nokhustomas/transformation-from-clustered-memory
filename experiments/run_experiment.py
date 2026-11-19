# -*- coding: utf-8 -*-
"""
正式实验:在 MQAR 合成长程记忆任务上训练并评测 4 种模型条件,如实记录结果到
results/raw_results.jsonl(每完成一个 run 就追加写入一行,方便中途查看进度/断点)。

主实验(primary):budget=4(紧张预算,4个键值对却只有4个记忆槽位,必须"压缩"才够用)，
                  3 个随机种子 × {nomemory, fullattn, topkraw, cma}。
补充实验(sweep):  budget in {8, 16},1 个随机种子 × {topkraw, cma}，
                  用于展示"预算-准确率"趋势,明确标注为单种子、示意性结果。

所有超参数(学习率、步数、batch、模型规模)在校准阶段(calibrate.py / calibrate2.py /
speed_test*.py)确定,记录在下面的常量里,正式实验不再调参、不"挑种子"。
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
DATA_SEED_BASE = 100000  # 数据生成用的种子基数,与模型初始化种子分开


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


def run_one(condition, seed, budget=None, log_prefix=""):
    torch.manual_seed(seed)
    kwargs = dict(budget=budget) if budget is not None else {}
    model = build_model(condition, VOCAB["vocab_size"], DIM, LAYERS, HEADS, CHUNK, SEQ_LEN, **kwargs)
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, get_lr_lambda(N_STEPS, WARMUP))

    # 数据种子与模型种子分离,同一 seed 下不同 condition/budget 看到完全相同的训练/评测数据序列
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
    # 主实验: budget=4, 3 个种子, 4 种条件
    for seed in [0, 1, 2]:
        plan.append(("nomemory", seed, None))
        plan.append(("fullattn", seed, None))
        plan.append(("topkraw", seed, 4))
        plan.append(("cma", seed, 4))
    # 补充预算扫描: budget in {8, 16}, 单种子(seed=0), 仅 topkraw / cma
    for budget in [8, 16]:
        plan.append(("topkraw", 0, budget))
        plan.append(("cma", 0, budget))

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
