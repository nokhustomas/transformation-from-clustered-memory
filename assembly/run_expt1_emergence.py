# -*- coding: utf-8 -*-
"""
实验一: 共激活 unit 组是否在训练中涌现,并对齐数据里埋好的概念类。

只测"组的真实性"(用户三选一的第一优先),不做补全干预。登记按用户选择全程在线:
训练的每个 step 都把当前训练批的 FFN 激活集合流式登记进 AssemblyRegistry
(与部署时同一条代码路径);训练结束后再在纯推理流上继续登记,检验组数是否失控。

三层证据,从"不依赖登记器"到"依赖登记器",防止把登记器超参数的伪影当成结论:

  A. 原始集合相似度(不经过登记器): 探针批上直接算激活集合两两 Jaccard,分三组
     统计——同 token、同类不同 token、跨类。涌现的判据是训练后"同类不同 token"
     相对"跨类"拉开差距(训练前的随机网络应该拉不开,是天然对照)。
  B. 快照登记(每个 checkpoint 用只见当前探针批的一次性登记器): 排除在线登记器
     "历史累积混入早期脏组"的因素,干净地看当前表征结构切出来的组。
  C. 在线登记(真机制): 训练全程流式累积的那个登记器,评估其分配与类/token 的
     NMI、同类合并率 vs 跨类合并率、组数、合并事件数。

防坑(已在给用户的说明里写明): 如果组只是"每 token 一个检测器",按类算纯度会
虚高——所以核心判据是"同类不同 token 进同一组"的率,以及 NMI(组,类) 与
NMI(组,token) 的对比,而不是单看纯度。

超参数(k_max=16, jaccard_new=0.3, act_frac=0.5, 登记步长 stride=8)是第一版
选择,敏感性未扫描——如实记录,后续如需下结论需要做敏感性检查。
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
from concept_lm_data import make_concept_spec, generate_batch, optimal_loss  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402

DIM, LAYERS, HEADS = 64, 2, 4
SEQ_LEN, TRAIN_BATCH, PROBE_BATCH = 64, 16, 64
LR, WARMUP = 3e-3, 100
GRAD_CLIP = 1.0
REG_STRIDE = 8            # 每 step 从 B*T 个位置里按固定步长抽 1/8 登记(控制成本,保持确定性)
K_MAX, J_NEW, ACT_FRAC, FREQ_LR = 16, 0.3, 0.5, 0.05
DATA_SEED = 300000

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------- 激活捕获

class ActCapture:
    """挂在每层 FFN 的 GELU 输出上,捕获 [B, T, ffn_dim]。"""

    def __init__(self, model):
        self.acts = {}
        self.handles = []
        for li, layer in enumerate(model.layers):
            gelu = layer.ffn[1]
            self.handles.append(gelu.register_forward_hook(self._make_hook(li)))
        self.n_layers = len(model.layers)

    def _make_hook(self, li):
        def hook(_mod, _inp, out):
            self.acts[li] = out.detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------- 指标

def nmi(a: torch.Tensor, b: torch.Tensor, na: int, nb: int) -> float:
    """归一化互信息(算术平均归一)。a,b: 等长 long。"""
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


def pair_same_group_rates(assign, token_ids, classes, n_tokens):
    """同类不同token vs 跨类 的"进同一组"率。只统计 assign>=0 的位置。"""
    ok = assign >= 0
    a, v = assign[ok], token_ids[ok]
    if a.shape[0] == 0:
        return float("nan"), float("nan"), 0.0
    K = int(a.max()) + 1
    A = torch.zeros(K, n_tokens)
    idx = a * n_tokens + v
    A.view(-1).scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    n_v = A.sum(0)                       # [V] 每 token 的位置数
    M = A.t() @ A                        # [V, V] token 对之间的同组配对数
    tot = n_v.unsqueeze(1) * n_v.unsqueeze(0)  # [V, V] 全部配对数

    tok_cls = torch.arange(n_tokens) // (n_tokens // classes_n_global)
    same_cls = tok_cls.unsqueeze(1) == tok_cls.unsqueeze(0)
    diff_tok = ~torch.eye(n_tokens, dtype=torch.bool)

    m1 = same_cls & diff_tok
    m2 = ~same_cls
    r_same = float(M[m1].sum() / tot[m1].sum().clamp_min(1.0))
    r_cross = float(M[m2].sum() / tot[m2].sum().clamp_min(1.0))
    coverage = float(ok.float().mean())
    return r_same, r_cross, coverage


def raw_jaccard_stats(sets, token_ids, classes):
    """不经过登记器的原始集合两两 Jaccard,分同token/同类不同token/跨类三组求均值。"""
    S = sets.float()
    sz = S.sum(-1)
    inter = S @ S.t()
    union = sz.unsqueeze(0) + sz.unsqueeze(1) - inter
    J = inter / union.clamp_min(1e-9)

    same_tok = token_ids.unsqueeze(0) == token_ids.unsqueeze(1)
    same_cls = classes.unsqueeze(0) == classes.unsqueeze(1)
    off_diag = ~torch.eye(sets.shape[0], dtype=torch.bool)
    nonempty = sets.any(-1)
    valid = off_diag & nonempty.unsqueeze(0) & nonempty.unsqueeze(1)

    m_tok = valid & same_tok
    m_cls = valid & same_cls & ~same_tok
    m_cross = valid & ~same_cls
    return (
        float(J[m_tok].mean()) if bool(m_tok.any()) else float("nan"),
        float(J[m_cls].mean()) if bool(m_cls.any()) else float("nan"),
        float(J[m_cross].mean()) if bool(m_cross.any()) else float("nan"),
    )


def probe_metrics(model, cap, registries, spec, probe_ids, probe_classes):
    """在冻结探针批上做一轮只读评估,返回逐层指标 dict。"""
    model.eval()
    with torch.no_grad():
        logits = model(probe_ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), probe_ids[:, 1:].reshape(-1)
        )
    model.train()

    V = spec["vocab_size"]
    tok_flat = probe_ids.reshape(-1)
    cls_flat = probe_classes.reshape(-1)
    out = {"probe_loss": float(loss)}

    for li in range(cap.n_layers):
        acts = cap.acts[li].reshape(-1, cap.acts[li].shape[-1])  # [N, U]
        reg = registries[li]
        sets = reg.extract_sets(acts)

        # A. 原始集合相似度(登记器无关)
        j_tok, j_cls, j_cross = raw_jaccard_stats(sets, tok_flat, cls_flat)

        # B. 快照登记器: 只见当前探针批
        snap = AssemblyRegistry(reg.n_units, K_MAX, J_NEW, FREQ_LR, act_frac=ACT_FRAC)
        snap.register(sets)
        a_snap, _ = snap.assign(sets)
        ok = a_snap >= 0
        nmi_cls_s = nmi(a_snap[ok], cls_flat[ok], K_MAX, spec["n_classes"])
        nmi_tok_s = nmi(a_snap[ok], tok_flat[ok], K_MAX, V)
        rs_s, rc_s, cov_s = pair_same_group_rates(a_snap, tok_flat, cls_flat, V)

        # C. 在线登记器(训练全程累积的真机制)
        a_on, _ = reg.assign(sets)
        ok2 = a_on >= 0
        nmi_cls_o = nmi(a_on[ok2], cls_flat[ok2], K_MAX, spec["n_classes"])
        nmi_tok_o = nmi(a_on[ok2], tok_flat[ok2], K_MAX, V)
        rs_o, rc_o, cov_o = pair_same_group_rates(a_on, tok_flat, cls_flat, V)

        out[f"L{li}"] = {
            "set_size_mean": float(sets.float().sum(-1).mean()),
            "raw_jaccard": {"same_token": j_tok, "same_class_diff_token": j_cls, "cross_class": j_cross},
            "snapshot": {"n_groups": snap.n_groups, "nmi_class": nmi_cls_s, "nmi_token": nmi_tok_s,
                          "rate_same_class": rs_s, "rate_cross_class": rc_s, "coverage": cov_s},
            "online": {"n_groups": reg.n_groups, "nmi_class": nmi_cls_o, "nmi_token": nmi_tok_o,
                        "rate_same_class": rs_o, "rate_cross_class": rc_o, "coverage": cov_o,
                        "merge_events": reg.merge_events, "empty_member_events": reg.empty_member_events,
                        "total_registered": reg.total_registered},
        }
    return out


# ---------------------------------------------------------------- 主流程

classes_n_global = None  # pair_same_group_rates 里用(由 main 设定,避免层层传参)


def main():
    global classes_n_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_steps = 60 if args.smoke else args.steps
    checkpoints = sorted(set(
        [0, 30, 60] if args.smoke else [0, 100, 250, 500, 1000, 1500, 2000, 2500, n_steps]
    ))

    spec = make_concept_spec(n_classes=8, members=8)
    classes_n_global = spec["n_classes"]

    torch.manual_seed(args.seed)
    model = FullAttentionLM(spec["vocab_size"], DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    print(f"模型参数量: {count_params(model)}, FFN unit/层: {DIM * 4}, "
          f"最优loss参照: {optimal_loss(spec):.4f}, 均匀猜测: {math.log(spec['vocab_size']):.4f}", flush=True)

    cap = ActCapture(model)
    registries = [AssemblyRegistry(DIM * 4, K_MAX, J_NEW, FREQ_LR, act_frac=ACT_FRAC)
                  for _ in range(LAYERS)]

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    warm = WARMUP if not args.smoke else 10

    def lr_fn(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n_steps - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    seed_off = args.seed * 10
    train_gen = torch.Generator().manual_seed(DATA_SEED + seed_off)
    probe_gen = torch.Generator().manual_seed(DATA_SEED + 555 + seed_off)
    stream_gen = torch.Generator().manual_seed(DATA_SEED + 777 + seed_off)
    probe_ids, probe_classes = generate_batch(PROBE_BATCH, SEQ_LEN, spec, generator=probe_gen)

    suffix = f"_seed{args.seed}" if args.seed != 0 else ""
    results_path = os.path.join(
        OUT_DIR, f"expt1_emergence{suffix}.jsonl" if not args.smoke else "expt1_smoke.jsonl")
    f_out = open(results_path, "w")

    def checkpoint_eval(step):
        m = probe_metrics(model, cap, registries, spec, probe_ids, probe_classes)
        m["step"] = step
        f_out.write(json.dumps(m, ensure_ascii=False) + "\n")
        f_out.flush()
        l0, l1 = m["L0"], m["L1"]
        print(f"[step {step:5d}] loss={m['probe_loss']:.4f} | "
              f"L0 raw同类/跨类={l0['raw_jaccard']['same_class_diff_token']:.3f}/{l0['raw_jaccard']['cross_class']:.3f} "
              f"L1 raw同类/跨类={l1['raw_jaccard']['same_class_diff_token']:.3f}/{l1['raw_jaccard']['cross_class']:.3f} | "
              f"L1快照 NMI类/NMI词={l1['snapshot']['nmi_class']:.3f}/{l1['snapshot']['nmi_token']:.3f}(cov={l1['snapshot']['coverage']:.2f}) "
              f"L1在线 组数={l1['online']['n_groups']} NMI类={l1['online']['nmi_class']:.3f} "
              f"同类率/跨类率={l1['online']['rate_same_class']:.3f}/{l1['online']['rate_cross_class']:.3f}"
              f"(cov={l1['online']['coverage']:.2f})", flush=True)

    t0 = time.time()
    # step 0 基线: 未训练网络。先跑一次探针前向填充 cap.acts
    checkpoint_eval(0)

    for step in range(1, n_steps + 1):
        ids, _classes = generate_batch(TRAIN_BATCH, SEQ_LEN, spec, generator=train_gen)
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()

        # 在线登记: 用刚才训练前向捕获的激活,固定步长抽样控制成本
        for li in range(LAYERS):
            acts = cap.acts[li].reshape(-1, cap.acts[li].shape[-1])[::REG_STRIDE]
            registries[li].register(registries[li].extract_sets(acts))

        if step in checkpoints:
            checkpoint_eval(step)

    print(f"\n训练+在线登记完成, 用时 {time.time()-t0:.0f}s。进入纯推理流登记阶段…", flush=True)

    # 训练后: 纯推理流上继续登记(不再训练),检验组数是否失控增长
    stream_log = []
    model.eval()
    n_stream = 10 if args.smoke else 100
    with torch.no_grad():
        for i in range(1, n_stream + 1):
            ids, _ = generate_batch(TRAIN_BATCH, SEQ_LEN, spec, generator=stream_gen)
            model(ids)
            for li in range(LAYERS):
                acts = cap.acts[li].reshape(-1, cap.acts[li].shape[-1])[::REG_STRIDE]
                registries[li].register(registries[li].extract_sets(acts))
            if i % max(1, n_stream // 10) == 0:
                snap = {"stream_batch": i,
                        "n_groups": [registries[li].n_groups for li in range(LAYERS)],
                        "merge_events": [registries[li].merge_events for li in range(LAYERS)]}
                stream_log.append(snap)
                print(f"  推理流 {i}/{n_stream}: 组数={snap['n_groups']} 合并次数={snap['merge_events']}", flush=True)

    f_out.write(json.dumps({"inference_stream": stream_log}, ensure_ascii=False) + "\n")
    f_out.close()

    torch.save({f"L{li}": registries[li].state_dict_plain() for li in range(LAYERS)},
               os.path.join(OUT_DIR, f"expt1_registries{suffix}.pt" if not args.smoke else "expt1_registries_smoke.pt"))
    torch.save(model.state_dict(),
               os.path.join(OUT_DIR, f"expt1_model{suffix}.pt" if not args.smoke else "expt1_model_smoke.pt"))
    print(f"完成。结果: {results_path}", flush=True)


if __name__ == "__main__":
    main()
