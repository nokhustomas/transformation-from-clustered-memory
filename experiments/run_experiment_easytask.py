# -*- coding: utf-8 -*-
"""
第二次后续实验: 把 MQAR 任务本身的难度降到最低,只用已经修正过的逐 token 判别式
检索机制(topkraw_pertoken / cma_pertoken),直接检验当前的领先假设——
"交叉注意力的内容匹配电路在 1300 步内没学会,与记忆组织方式/检索粒度无关"。

与原任务的差异,只有两处,都复用 mqar_data.py 里已经测试过、开放的参数,没有新写
数据生成逻辑:
  - num_pairs:  4 -> 2   (需要消歧的候选变少)
  - key/value 词表: 16 -> 16 -> 8       (更小、更容易区分的词表)
  - insert_region_frac: 0.6 -> 0.4  (写入区和查询区之间的间隔从 ~75 拉大到 ~130)
故意不引入"同一个 key 多次写入、查询要最新值"这类时序/覆盖语义——那类任务存在
一条比"按内容匹配"简单得多的退化路径("优先选最近写入的"),即使模型在那类任务上
分离出来,也不能确认 content-matching 电路本身可学,可能只是学会了时间启发式。
这里坚持纯内容匹配语义,只调"难度"和"间隔"两个旋钮,确保测的还是同一个电路。

预算设计(仍用于 exemplars_per_cluster=2 的 CMA): budget=4 对应 K_max=2 个簇,
k_sim=1,强制模型必须二选一,不再是"预算=容量"的恒等操作(见已交付论文第 5.2 节
的更正);budget=8 作为单种子的宽裕预算参照。

同样写入 results/raw_results.jsonl,condition 名字加 "_easytask" 后缀,与之前
两批记录不冲突,同样支持断点续跑。
"""
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from mqar_data import make_vocab, generate_batch
from models import build_model, count_params

VOCAB = make_vocab(key_vocab=8, value_vocab=8, filler_vocab=32)
DIM, LAYERS, HEADS = 64, 2, 4
CHUNK, SEQ_LEN, NUM_PAIRS = 32, 192, 2
INSERT_REGION_FRAC = 0.4
TRAIN_BATCH, EVAL_BATCH = 16, 256
LR, WARMUP, N_STEPS = 3e-3, 100, 1300
EVAL_EVERY = 100
GRAD_CLIP = 1.0

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "raw_results.jsonl")
DATA_SEED_BASE = 200000  # 换一批数据种子(与前两批实验完全独立,避免任何巧合关联)

CONDITION_MAP = {
    "nomemory_easytask": "nomemory",
    "fullattn_easytask": "fullattn",
    "topkraw_easytask": "topkraw_pertoken",
    "cma_easytask": "cma_pertoken",
}


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


def run_one(condition_label, seed, budget, log_prefix=""):
    real_condition = CONDITION_MAP[condition_label]
    torch.manual_seed(seed)
    kwargs = dict(budget=budget) if budget is not None else {}
    model = build_model(real_condition, VOCAB["vocab_size"], DIM, LAYERS, HEADS, CHUNK, SEQ_LEN, **kwargs)
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, get_lr_lambda(N_STEPS, WARMUP))

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
            print(f"{log_prefix} step {step:4d}/{N_STEPS}  eval_acc={eacc.item():.3f}  "
                  f"eval_loss={eloss.item():.3f}  ({time.time()-t0:.0f}s)", flush=True)

    dt = time.time() - t0
    result = {
        "condition": condition_label, "seed": seed, "budget": budget, "n_params": n_params,
        "n_steps": N_STEPS, "wall_clock_sec": dt, "final_eval_acc": curve[-1]["eval_acc"],
        "final_eval_loss": curve[-1]["eval_loss"], "curve": curve,
    }
    return result


def append_result(result):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def already_done(condition_label, seed, budget):
    if not os.path.exists(RESULTS_PATH):
        return False
    with open(RESULTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] == condition_label and r["seed"] == seed and r["budget"] == budget:
                return True
    return False


def main():
    plan = []
    for seed in [0, 1, 2]:
        plan.append(("nomemory_easytask", seed, None))
        plan.append(("fullattn_easytask", seed, None))
        plan.append(("topkraw_easytask", seed, 4))
        plan.append(("cma_easytask", seed, 4))
    for budget in [8]:
        plan.append(("topkraw_easytask", 0, budget))
        plan.append(("cma_easytask", 0, budget))

    print(f"总计划 {len(plan)} 个 run, N_STEPS={N_STEPS}, num_pairs={NUM_PAIRS}, "
          f"vocab_size={VOCAB['vocab_size']}, chance={1/VOCAB['value_vocab']:.4f}")
    for i, (condition_label, seed, budget) in enumerate(plan):
        if already_done(condition_label, seed, budget):
            print(f"[{i+1}/{len(plan)}] 跳过(已完成): {condition_label} seed={seed} budget={budget}")
            continue
        prefix = f"[{i+1}/{len(plan)} {condition_label} seed={seed} budget={budget}]"
        print(f"\n=== 开始 {prefix} ===", flush=True)
        result = run_one(condition_label, seed, budget, log_prefix=prefix)
        append_result(result)
        print(f"=== 完成 {prefix}  final_acc={result['final_eval_acc']:.3f} "
              f"用时{result['wall_clock_sec']:.0f}s ===", flush=True)

    print("\n全部实验完成。")


if __name__ == "__main__":
    main()
