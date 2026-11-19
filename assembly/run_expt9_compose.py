# -*- coding: utf-8 -*-
"""
实验九: 组合泛化——机制第一次有机会"创造新能力"而不是"修复损伤"。

任务(命令语言):
  序列格式: [MOD, P, SEP, 答案...]
  - P ∈ {p0..p7} 8 个动作词,各自对应一个输出 token o_i
  - MOD ∈ {plain, twice, thrice}: 答案 = o_i 重复 1/2/3 次,最后 EOS
  训练覆盖(制造组合缺口):
  - plain  x 全部 8 个 P
  - twice  x 只有 p0..p_{k-1}
  - thrice x 只有 p_{8-k}..p7 (与 twice 的覆盖错开)
  测试: 没见过的 (MOD, P) 组合。模型要答对,必须把"MOD 的含义"和"P 的身份"
  分开学并拼起来——这正是"组=稳定共同点"可能帮上忙的形态:
  "twice 组"的成员 unit 是跨不同 P 稳定共亮的(修饰语义),换着 P 变的 unit
  不会进组(EMA 频率过不了 0.5)——补全因此只补"twice 该有的样子",不碰 P 身份。

先跑 pilot(--pilot): 扫覆盖度 k∈{2,3,4},看基线模型在 held-out 组合上的
准确率落在哪——需要一个"挣扎但不是零"的设置,补全才有可测的空间。
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

from models import FullAttentionLM  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402

# ---- 词表 ----
N_P = 8
PAD, SEP, EOS = 0, 1, 2
MOD0 = 3                      # plain, twice, thrice = 3,4,5
P0 = 6                        # p0..p7 = 6..13
O0 = P0 + N_P                 # o0..o7 = 14..21
VOCAB = O0 + N_P              # 22
MODS = {"plain": (MOD0, 1), "twice": (MOD0 + 1, 2), "thrice": (MOD0 + 2, 3)}
SEQ_LEN = 8                   # MOD P SEP a a a EOS (最长 3+3+1=7) + pad

DIM, LAYERS, HEADS = 64, 2, 4
LR, WARMUP = 3e-3, 100
GRAD_CLIP = 1.0
K_MAX, J_NEW, ACT_FRAC, FREQ_LR = 16, 0.3, 0.5, 0.05
DATA_SEED = 900000
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def make_seq(mod_name, p_idx):
    mod_tok, rep = MODS[mod_name]
    seq = [mod_tok, P0 + p_idx, SEP] + [O0 + p_idx] * rep + [EOS]
    seq = seq + [PAD] * (SEQ_LEN - len(seq))
    return seq


def train_combos(k):
    combos = [("plain", i) for i in range(N_P)]
    combos += [("twice", i) for i in range(k)]
    combos += [("thrice", i) for i in range(N_P - k, N_P)]
    return combos


def heldout_combos(k):
    seen = set(train_combos(k))
    return [(m, i) for m in ("twice", "thrice") for i in range(N_P) if (m, i) not in seen]


def gen_batch(combos, bs, generator):
    idx = torch.randint(0, len(combos), (bs,), generator=generator)
    seqs = [make_seq(*combos[i]) for i in idx.tolist()]
    return torch.tensor(seqs, dtype=torch.long)


def answer_positions(ids):
    """答案区 = SEP 之后到 EOS(含)。返回 mask [B, T-1](对 t 预测 t+1 的位置)。"""
    B, T = ids.shape
    mask = torch.zeros(B, T - 1, dtype=torch.bool)
    for b in range(B):
        sep = (ids[b] == SEP).nonzero()[0].item()
        eos = (ids[b] == EOS).nonzero()[0].item()
        mask[b, sep:eos] = True  # 位置 sep 预测第一个答案 token,位置 eos-1 预测 EOS
    return mask


def seq_exact_match(model, combos, hooks=None):
    """对每个 combo: teacher-forcing 下答案区全部预测正确才算对。"""
    ids = torch.tensor([make_seq(*c) for c in combos], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)
    pred = logits[:, :-1].argmax(-1)
    tgt = ids[:, 1:]
    mask = answer_positions(ids)
    ok = []
    for b in range(len(combos)):
        ok.append(bool((pred[b][mask[b]] == tgt[b][mask[b]]).all()))
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

    for step in range(1, steps + 1):
        ids = gen_batch(combos, 32, gen)
        logits = model(ids)
        mask = answer_positions(ids)
        lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                             ids[:, 1:].reshape(-1), reduction="none").view(ids.shape[0], -1)
        loss = lp[mask].mean()
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
            print(f"{log_prefix} step {step}/{steps} loss={float(loss):.4f}", flush=True)

    for h in handles:
        h.remove()
    return model, regs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    if args.pilot:
        print("== pilot: 扫覆盖度 k(每个修饰词见过几个动作词), 基线在没见过组合上的表现 ==", flush=True)
        out = {}
        for k in (2, 3, 4):
            accs_seen, accs_ho = [], []
            for seed in args.seeds:
                model, _ = train_model(k, args.steps, seed, with_registry=False,
                                        log_prefix=f"[k={k} seed={seed}]")
                model.eval()
                seen_ok = seq_exact_match(model, train_combos(k))
                ho_ok = seq_exact_match(model, heldout_combos(k))
                accs_seen.append(sum(seen_ok) / len(seen_ok))
                accs_ho.append(sum(ho_ok) / len(ho_ok))
                print(f"  k={k} seed={seed}: 见过的组合 {accs_seen[-1]:.2f} | "
                      f"没见过的 {accs_ho[-1]:.2f} ({len(heldout_combos(k))}个)", flush=True)
            out[k] = {"seen": accs_seen, "heldout": accs_ho}
            print(f"k={k}: 没见过组合平均 {sum(accs_ho)/len(accs_ho):.2f}", flush=True)
        with open(os.path.join(OUT_DIR, "expt9_pilot.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("pilot 完成。", flush=True)
        return


if __name__ == "__main__":
    main()
