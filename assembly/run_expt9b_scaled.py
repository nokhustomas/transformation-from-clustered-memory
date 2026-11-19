# -*- coding: utf-8 -*-
"""
实验九 v2: 按用户三点方法论修正重设计的组合泛化任务。

用户的三点(2026-08-02):
1. v1 里 1000 步 loss 归零 => 背诵免费,模型没有压力总结规律。要让背诵变贵:
   组合数大增 + 模型容量压小。
2. v1 测试集只有 8-12 项 => 刻度 8-12%,分辨率不够;加种子没用,要加测试项数。
3. 结果解读表先定后看数。
巧合: 本任务确定性(每组合唯一正确答案),加测试项 == 加组合总数 == 让背诵变贵,
一刀两用。

v2 设计:
- 64 个动作词 x 4 个修饰词(做1/2/3/4遍),组合总数 256。
- 覆盖度 k: 每个非plain修饰词只见过 k 个动作词(错开轮转),plain 见全部。
  训练组合 = 64 + 3k;没见过的组合 = 3*(64-k)   [k=8:168项, k=16:144项, k=32:96项]
- 模型缩小: dim 32(v1 是 64), 2层4头。
- 评估: teacher-forcing 下答案区(SEP后到EOS)全对才算该组合正确。
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from models import FullAttentionLM, count_params  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402

N_P = 64
PAD, SEP, EOS = 0, 1, 2
MOD0 = 3                       # plain/twice/thrice/four = 3/4/5/6
N_MODS = 4
P0 = MOD0 + N_MODS             # 7..70
O0 = P0 + N_P                  # 71..134
VOCAB = O0 + N_P               # 135
REPS = {0: 1, 1: 2, 2: 3, 3: 4}  # mod 下标 -> 重复次数
SEQ_LEN = 10                   # MOD P SEP + 最多4答案 + EOS = 8, 余量到10

DIM, LAYERS, HEADS = 32, 2, 4
LR, WARMUP = 3e-3, 100
GRAD_CLIP = 1.0
K_MAX, J_NEW, ACT_FRAC, FREQ_LR = 32, 0.3, 0.5, 0.05
DATA_SEED = 910000
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def make_seq(mod_idx, p_idx):
    rep = REPS[mod_idx]
    seq = [MOD0 + mod_idx, P0 + p_idx, SEP] + [O0 + p_idx] * rep + [EOS]
    return seq + [PAD] * (SEQ_LEN - len(seq))


def train_combos(k):
    combos = [(0, i) for i in range(N_P)]                     # plain x 全部
    for m in (1, 2, 3):                                        # 每个非plain修饰词见 k 个,错开
        start = (m - 1) * k % N_P
        combos += [(m, (start + j) % N_P) for j in range(k)]
    return combos


def heldout_combos(k):
    seen = set(train_combos(k))
    return [(m, i) for m in (1, 2, 3) for i in range(N_P) if (m, i) not in seen]


def gen_batch(combos, bs, generator):
    idx = torch.randint(0, len(combos), (bs,), generator=generator)
    return torch.tensor([make_seq(*combos[i]) for i in idx.tolist()], dtype=torch.long)


def answer_mask(ids):
    B, T = ids.shape
    mask = torch.zeros(B, T - 1, dtype=torch.bool)
    for b in range(B):
        sep = (ids[b] == SEP).nonzero()[0].item()
        eos = (ids[b] == EOS).nonzero()[0].item()
        mask[b, sep:eos] = True
    return mask


def combo_correct(model, combos, bs=128):
    ok = []
    with torch.no_grad():
        for s in range(0, len(combos), bs):
            chunk = combos[s:s + bs]
            ids = torch.tensor([make_seq(*c) for c in chunk], dtype=torch.long)
            pred = model(ids)[:, :-1].argmax(-1)
            tgt = ids[:, 1:]
            m = answer_mask(ids)
            for b in range(len(chunk)):
                ok.append(bool((pred[b][m[b]] == tgt[b][m[b]]).all()))
    return ok


def train_model(k, steps, seed, with_registry=True, log_prefix=""):
    torch.manual_seed(seed)
    model = FullAttentionLM(VOCAB, DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    regs = [AssemblyRegistry(DIM * 4, K_MAX, J_NEW, FREQ_LR, act_frac=ACT_FRAC) for _ in range(LAYERS)]
    acts = {}
    handles = [model.layers[li].ffn[1].register_forward_hook(
        lambda m, i, o, li=li: acts.__setitem__(li, o.detach())) for li in range(LAYERS)]
    combos = train_combos(k)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def lr_fn(s):
        if s < WARMUP:
            return (s + 1) / WARMUP
        return 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / max(1, steps - WARMUP)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    gen = torch.Generator().manual_seed(DATA_SEED + seed)

    t0 = time.time()
    for step in range(1, steps + 1):
        ids = gen_batch(combos, 32, gen)
        logits = model(ids)
        m = answer_mask(ids)
        lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                             ids[:, 1:].reshape(-1), reduction="none").view(ids.shape[0], -1)
        loss = lp[m].mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        if with_registry:
            for li in range(LAYERS):
                flat = acts[li].reshape(-1, acts[li].shape[-1])[::4]
                regs[li].register(regs[li].extract_sets(flat), values=flat)
        if step % 1000 == 0:
            print(f"{log_prefix} step {step}/{steps} loss={float(loss):.4f} ({time.time()-t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()
    return model, regs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landscape", action="store_true", help="一档覆盖度一个种子, 看基线全景")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"v2: {N_P}动作 x {N_MODS}修饰 = {N_P*N_MODS}组合; 模型 dim={DIM} "
          f"参数量={count_params(FullAttentionLM(VOCAB, DIM, LAYERS, HEADS, max_len=SEQ_LEN))}", flush=True)

    if args.landscape:
        out = {}
        for k in args.ks:
            tr, ho = train_combos(k), heldout_combos(k)
            print(f"\n== k={k}: 训练组合 {len(tr)} 个, 没见过 {len(ho)} 个 ==", flush=True)
            model, regs = train_model(k, args.steps, args.seed, with_registry=True,
                                       log_prefix=f"[k={k}]")
            model.eval()
            seen_acc = sum(combo_correct(model, tr)) / len(tr)
            ho_ok = combo_correct(model, ho)
            ho_acc = sum(ho_ok) / len(ho)
            print(f"k={k}: 见过 {seen_acc:.3f} | 没见过 {ho_acc:.3f} ({len(ho)}项, 刻度{1/len(ho)*100:.1f}%)", flush=True)
            out[k] = {"seen": seen_acc, "heldout": ho_acc, "n_heldout": len(ho),
                      "heldout_per_mod": {}}
            for m in (1, 2, 3):
                sub = [i for i, c in enumerate(ho) if c[0] == m]
                out[k]["heldout_per_mod"][m] = sum(ho_ok[i] for i in sub) / len(sub)
            print(f"  按修饰词分: " + " ".join(f"做{REPS[m]}遍={out[k]['heldout_per_mod'][m]:.2f}" for m in (1,2,3)), flush=True)
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"expt9b_model_k{k}_seed{args.seed}.pt"))
            torch.save({f"L{li}": regs[li].state_dict_plain() for li in range(LAYERS)},
                       os.path.join(OUT_DIR, f"expt9b_registries_k{k}_seed{args.seed}.pt"))
        with open(os.path.join(OUT_DIR, f"expt9b_landscape_seed{args.seed}.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nlandscape 完成。", flush=True)


if __name__ == "__main__":
    main()
