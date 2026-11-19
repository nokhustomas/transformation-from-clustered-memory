# -*- coding: utf-8 -*-
"""
实验六: 真实语言(字符级)上的组涌现复测——没有人工埋的概念类时,组还涌现吗,
对齐的是什么结构?

语料: 本地抽取的 torch docstring 英文(4MB,领域偏技术文档,如实标注)。
模型: 与前面同骨架的字符级 LM(dim 64, 2层, seq 128)。
登记: 与实验一同机制,K_MAX 放宽到 32(真实文本的状态结构未知,给出余量;
超参选择如实记录,未做敏感性扫描)。

评估里没有"真类别",改用三种天然标注,各测 NMI,看组到底编码什么:
  - char_id:    当前字符身份(~90类)
  - category5:  小写/大写/数字/空白/标点 5 大类(实验一"概念类"的天然对应物)
  - word_start: 前一字符是否空白(词边界状态,2类)
原始集合 Jaccard 同样做三分: 同字符 / 同大类不同字符 / 跨大类——与实验一的
"同token/同类异token/跨类"完全平行,训练前后对比是涌现判据。

loss 参照: 语料的 unigram 熵与 bigram 条件熵(模型至少要显著低于 bigram
才说明学到了超出二元统计的结构)。
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from models import FullAttentionLM, count_params  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402

DIM, LAYERS, HEADS = 64, 2, 4
SEQ_LEN, TRAIN_BATCH, PROBE_BATCH = 128, 16, 64
LR, WARMUP = 3e-3, 100
GRAD_CLIP = 1.0
REG_STRIDE = 8
K_MAX, J_NEW, ACT_FRAC, FREQ_LR = 32, 0.3, 0.5, 0.05
DATA_SEED = 500000

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "data", "docstring_corpus.txt")
OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------- 语料与标注

def load_corpus(max_vocab=96):
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()
    counts = Counter(text)
    keep = [c for c, _ in counts.most_common(max_vocab - 1)]
    stoi = {c: i + 1 for i, c in enumerate(sorted(keep))}  # 0 = <unk>
    ids = torch.tensor([stoi.get(c, 0) for c in text], dtype=torch.long)
    itos = {i: c for c, i in stoi.items()}
    itos[0] = "�"
    return ids, stoi, itos


def char_category(ch):
    if ch.islower():
        return 0
    if ch.isupper():
        return 1
    if ch.isdigit():
        return 2
    if ch in " \t\n\r":
        return 3
    return 4


CAT_NAMES = ["小写", "大写", "数字", "空白", "标点"]


def make_batch(ids_corpus, bs, seq_len, generator):
    starts = torch.randint(0, ids_corpus.shape[0] - seq_len - 1, (bs,), generator=generator)
    return torch.stack([ids_corpus[s:s + seq_len] for s in starts])


def entropy_refs(ids_corpus, vocab_size, sample=2_000_000):
    x = ids_corpus[:sample]
    uni = torch.bincount(x, minlength=vocab_size).float()
    p = uni / uni.sum()
    h_uni = float(-(p[p > 0] * p[p > 0].log()).sum())
    pair = x[:-1] * vocab_size + x[1:]
    big = torch.bincount(pair, minlength=vocab_size * vocab_size).float().view(vocab_size, vocab_size)
    rowsum = big.sum(-1, keepdim=True)
    cond = big / rowsum.clamp_min(1)
    h_rows = -(cond * cond.clamp_min(1e-12).log()).sum(-1)
    w = (rowsum.squeeze(-1) / rowsum.sum())
    h_big = float((w * h_rows).sum())
    return h_uni, h_big


# ---------------------------------------------------------------- 指标(复用实验一逻辑)

def nmi(a, b, na, nb):
    n = a.shape[0]
    if n == 0:
        return float("nan")
    joint = torch.bincount(a * nb + b, minlength=na * nb).float().view(na, nb)
    p = joint / n
    pa, pb = p.sum(1), p.sum(0)
    outer = pa.unsqueeze(1) * pb.unsqueeze(0)
    nz = p > 0
    mi = (p[nz] * (p[nz] / outer[nz].clamp_min(1e-12)).log()).sum()
    ha = -(pa[pa > 0] * pa[pa > 0].log()).sum()
    hb = -(pb[pb > 0] * pb[pb > 0].log()).sum()
    denom = float(ha + hb)
    return float(2 * mi / denom) if denom > 0 else 0.0


def raw_jaccard_three_way(sets, char_ids, cats, max_pairs_side=4096):
    S = sets[:max_pairs_side].float()
    ci = char_ids[:max_pairs_side]
    ct = cats[:max_pairs_side]
    sz = S.sum(-1)
    inter = S @ S.t()
    union = sz.unsqueeze(0) + sz.unsqueeze(1) - inter
    J = inter / union.clamp_min(1e-9)
    same_char = ci.unsqueeze(0) == ci.unsqueeze(1)
    same_cat = ct.unsqueeze(0) == ct.unsqueeze(1)
    off = ~torch.eye(S.shape[0], dtype=torch.bool)
    ne = sets[:max_pairs_side].any(-1)
    valid = off & ne.unsqueeze(0) & ne.unsqueeze(1)
    m1 = valid & same_char
    m2 = valid & same_cat & ~same_char
    m3 = valid & ~same_cat
    f = lambda m: float(J[m].mean()) if bool(m.any()) else float("nan")
    return f(m1), f(m2), f(m3)


class ActCapture:
    def __init__(self, model):
        self.acts = {}
        self.handles = []
        for li, layer in enumerate(model.layers):
            self.handles.append(layer.ffn[1].register_forward_hook(self._mk(li)))
        self.n_layers = len(model.layers)

    def _mk(self, li):
        def hook(_m, _i, out):
            self.acts[li] = out.detach()
        return hook


def probe_metrics(model, cap, registries, probe_ids, itos, vocab_size):
    model.eval()
    with torch.no_grad():
        logits = model(probe_ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), probe_ids[:, 1:].reshape(-1))
    model.train()

    flat_ids = probe_ids.reshape(-1)
    cats = torch.tensor([char_category(itos[int(i)]) for i in flat_ids])
    prev_space = torch.zeros_like(flat_ids, dtype=torch.long)
    ps = probe_ids.clone()
    ps[:, 1:] = probe_ids[:, :-1]
    ps[:, 0] = 0
    prev_space = torch.tensor([1 if itos[int(i)] in " \t\n\r" else 0 for i in ps.reshape(-1)])

    out = {"probe_loss": float(loss)}
    for li in range(cap.n_layers):
        acts = cap.acts[li].reshape(-1, cap.acts[li].shape[-1])
        reg = registries[li]
        sets = reg.extract_sets(acts)
        j1, j2, j3 = raw_jaccard_three_way(sets, flat_ids, cats)

        snap = AssemblyRegistry(reg.n_units, K_MAX, J_NEW, FREQ_LR, act_frac=ACT_FRAC)
        snap.register(sets)
        a_s, _ = snap.assign(sets)
        ok = a_s >= 0
        d_snap = {
            "n_groups": snap.n_groups, "coverage": float(ok.float().mean()),
            "nmi_char": nmi(a_s[ok], flat_ids[ok], K_MAX, vocab_size),
            "nmi_cat5": nmi(a_s[ok], cats[ok], K_MAX, 5),
            "nmi_wordstart": nmi(a_s[ok], prev_space[ok], K_MAX, 2),
        }
        a_o, _ = reg.assign(sets)
        ok2 = a_o >= 0
        d_on = {
            "n_groups": reg.n_groups, "coverage": float(ok2.float().mean()),
            "nmi_char": nmi(a_o[ok2], flat_ids[ok2], K_MAX, vocab_size),
            "nmi_cat5": nmi(a_o[ok2], cats[ok2], K_MAX, 5),
            "nmi_wordstart": nmi(a_o[ok2], prev_space[ok2], K_MAX, 2),
            "merge_events": reg.merge_events,
        }
        out[f"L{li}"] = {
            "set_size_mean": float(sets.float().sum(-1).mean()),
            "raw_jaccard": {"same_char": j1, "same_cat_diff_char": j2, "cross_cat": j3},
            "snapshot": d_snap, "online": d_on,
        }
    return out


def group_char_profile(reg, sets, flat_ids, itos, topn=8):
    """每个在线组: 位置数最多的前几个字符(定性解读用)。"""
    a, _ = reg.assign(sets)
    prof = {}
    for k in a[a >= 0].unique().tolist():
        ids_k = flat_ids[a == k]
        cnt = Counter(itos[int(i)] for i in ids_k)
        prof[int(k)] = {"n": int((a == k).sum()), "top_chars": cnt.most_common(topn)}
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ids_corpus, stoi, itos = load_corpus()
    V = max(itos.keys()) + 1
    h_uni, h_big = entropy_refs(ids_corpus, V)
    print(f"语料 {ids_corpus.shape[0]/1e6:.2f}M 字符, 词表 {V}, "
          f"unigram熵={h_uni:.4f} bigram条件熵={h_big:.4f}", flush=True)

    torch.manual_seed(args.seed)
    model = FullAttentionLM(V, DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    print(f"参数量 {count_params(model)}", flush=True)
    cap = ActCapture(model)
    regs = [AssemblyRegistry(DIM * 4, K_MAX, J_NEW, FREQ_LR, act_frac=ACT_FRAC) for _ in range(LAYERS)]

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def lr_fn(s):
        if s < WARMUP:
            return (s + 1) / WARMUP
        return 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / max(1, args.steps - WARMUP)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    gen = torch.Generator().manual_seed(DATA_SEED + args.seed)
    probe_gen = torch.Generator().manual_seed(DATA_SEED + 555 + args.seed)
    probe_ids = make_batch(ids_corpus, PROBE_BATCH, SEQ_LEN, probe_gen)

    results_path = os.path.join(OUT_DIR, f"expt6_realtext_seed{args.seed}.jsonl")
    f_out = open(results_path, "w")
    checkpoints = sorted(set([0, 250, 500, 1000, 2000, 3000, 4500, args.steps]))

    def ck(step):
        m = probe_metrics(model, cap, regs, probe_ids, itos, V)
        m["step"] = step
        f_out.write(json.dumps(m, ensure_ascii=False) + "\n")
        f_out.flush()
        l1 = m["L1"]
        print(f"[step {step:5d}] loss={m['probe_loss']:.4f} | "
              f"L1 raw 同字符/同类异字符/跨类={l1['raw_jaccard']['same_char']:.3f}/"
              f"{l1['raw_jaccard']['same_cat_diff_char']:.3f}/{l1['raw_jaccard']['cross_cat']:.3f} | "
              f"在线 组数={l1['online']['n_groups']} NMI字符/类5/词首="
              f"{l1['online']['nmi_char']:.3f}/{l1['online']['nmi_cat5']:.3f}/{l1['online']['nmi_wordstart']:.3f} "
              f"(cov={l1['online']['coverage']:.2f})", flush=True)

    t0 = time.time()
    ck(0)
    for step in range(1, args.steps + 1):
        ids = make_batch(ids_corpus, TRAIN_BATCH, SEQ_LEN, gen)
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        for li in range(LAYERS):
            acts = cap.acts[li].reshape(-1, cap.acts[li].shape[-1])[::REG_STRIDE]
            regs[li].register(regs[li].extract_sets(acts), values=acts)
        if step in checkpoints:
            ck(step)

    print(f"训练完成 {time.time()-t0:.0f}s。生成组的字符画像…", flush=True)
    model.eval()
    with torch.no_grad():
        model(probe_ids)
    flat_ids = probe_ids.reshape(-1)
    profiles = {}
    for li in range(LAYERS):
        acts = cap.acts[li].reshape(-1, cap.acts[li].shape[-1])
        sets = regs[li].extract_sets(acts)
        profiles[f"L{li}"] = group_char_profile(regs[li], sets, flat_ids, itos)
    f_out.write(json.dumps({"group_profiles": profiles}, ensure_ascii=False) + "\n")
    f_out.close()

    torch.save({f"L{li}": regs[li].state_dict_plain() for li in range(LAYERS)},
               os.path.join(OUT_DIR, f"expt6_registries_seed{args.seed}.pt"))
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"expt6_model_seed{args.seed}.pt"))
    print(f"完成。结果: {results_path}", flush=True)


if __name__ == "__main__":
    main()
