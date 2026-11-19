# -*- coding: utf-8 -*-
"""
探索性诊断,不是正式实验轮次:把训练步数从 1300 拉长到 6000(约 4.6 倍),只在
主任务(原始、较难的那个: key_vocab=16, value_vocab=16, num_pairs=4)、budget=4、
单一 seed=0 上跑,只跑我们真正关心的两个条件(topkraw_pertoken, cma_pertoken)
外加 fullattn 作参照(判断这个任务在更长训练下是否还有空间可涨,而不是已经
到顶)。

目的:论文里唯一没测试过的"剩余假设"是——记忆交叉注意力的内容匹配电路,
可能只是在 1300 步内没学会,不是任何方法/预算/任务难度的问题。这是一个便宜的
单种子探测,只看"曲线有没有翘起来的苗头",不是要产出可发表的严谨结果(那需要
多种子、显著性检验,是一次更大的投入,论文里已经写清楚了为什么没做)。

关键实现点:学习率的 cosine decay 必须按 6000 步重新计算,不能沿用 1300 步那版
schedule 再往后延伸——否则后面 4700 步的学习率会一直卡在接近 0,训练效果和"真的
多训练"完全不是一回事,那样测出来毫无意义。

单独写入 results/longtrain_probe.jsonl,不写入 raw_results.jsonl,不混入论文
正式报告的 40 组数据——这是探索性检查,如果真的看到苗头,再决定要不要正式纳入。
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
LR, WARMUP, N_STEPS = 3e-3, 100, 6000
EVAL_EVERY = 100
GRAD_CLIP = 1.0
DATA_SEED_BASE = 100000  # 与 run_experiment.py / run_experiment_pertoken.py 相同,数据流可比

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "longtrain_probe.jsonl")

PLAN = [
    ("fullattn", 0, None),
    ("topkraw_pertoken", 0, 4),
    ("cma_pertoken", 0, 4),
    # 复现性检查(round 2): cma_pertoken 在 seed=0 的后段(约 step 4300 起)出现了
    # chance(0.0625) -> ~0.086 的小幅抬升,topkraw_pertoken 全程没有。单一种子无法
    #区分"真实但弱的信号"和"这一次初始化的偶然轨迹",所以加 4 个新种子(1-4),
    # 步数、budget、其余超参数完全不变,只看这个后段抬升会不会在多数种子上重复出现。
    ("topkraw_pertoken", 1, 4),
    ("cma_pertoken", 1, 4),
    ("topkraw_pertoken", 2, 4),
    ("cma_pertoken", 2, 4),
    ("topkraw_pertoken", 3, 4),
    ("cma_pertoken", 3, 4),
    ("topkraw_pertoken", 4, 4),
    ("cma_pertoken", 4, 4),
]


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
    return generate_batch(bs, SEQ_LEN, CHUNK, NUM_PAIRS, VOCAB, generator=generator)


def run_one(condition, seed, budget, log_prefix=""):
    torch.manual_seed(seed)
    kwargs = dict(budget=budget) if budget is not None else {}
    model = build_model(condition, VOCAB["vocab_size"], DIM, LAYERS, HEADS, CHUNK, SEQ_LEN, **kwargs)
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, get_lr_lambda(N_STEPS, WARMUP))  # 按 6000 步重新算,不是延用 1300 步那版

    train_gen = torch.Generator().manual_seed(DATA_SEED_BASE + seed)
    eval_gen = torch.Generator().manual_seed(DATA_SEED_BASE + 555 + seed)

    curve = []
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        ids, pos, tgt = gen_batch(TRAIN_BATCH, train_gen)
        opt.zero_grad()
        loss, acc = forward_loss_acc(model, ids, pos, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()

        if step % EVAL_EVERY == 0 or step == N_STEPS:
            model.eval()
            with torch.no_grad():
                eids, epos, etgt = gen_batch(EVAL_BATCH, eval_gen)
                eloss, eacc = forward_loss_acc(model, eids, epos, etgt)
            model.train()
            curve.append({"step": step, "eval_loss": eloss.item(), "eval_acc": eacc.item()})
            print(f"{log_prefix} step {step:5d}/{N_STEPS}  eval_acc={eacc.item():.4f}  "
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
    print(f"探测性长训练: N_STEPS={N_STEPS}(原始 1300 的 {N_STEPS/1300:.1f} 倍), "
          f"chance={1/VOCAB['value_vocab']:.4f}, 计划 {len(PLAN)} 组", flush=True)
    for i, (condition, seed, budget) in enumerate(PLAN):
        if already_done(condition, seed, budget):
            print(f"[{i+1}/{len(PLAN)}] 跳过(已完成): {condition} seed={seed} budget={budget}")
            continue
        prefix = f"[{i+1}/{len(PLAN)} {condition} seed={seed} budget={budget}]"
        print(f"\n=== 开始 {prefix} ===", flush=True)
        result = run_one(condition, seed, budget, log_prefix=prefix)
        append_result(result)
        print(f"=== 完成 {prefix}  final_acc={result['final_eval_acc']:.4f}  "
              f"用时{result['wall_clock_sec']:.0f}s ===", flush=True)
    print("\n全部探测性长训练完成。")


if __name__ == "__main__":
    main()
