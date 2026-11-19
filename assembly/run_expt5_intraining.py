# -*- coding: utf-8 -*-
"""
实验五: 训练中开启补全——完整闭环,最接近用户原始构想的形态
("预训练时就开始标记出组合unit…以后激活 b 的时候…引入 a 的权重")。

设计:
- 登记与注入都发生在训练前向里,同一条在线代码路径。
- 时间表: 前 WARM_REG 步只登记不注入(组先成型);之后 alpha 从 0 线性爬升到
  ALPHA_MAX(RAMP 步),避免一开始就注入还没成型的组把训练带崩。
- 注入项对 autograd 是常数(集合/匹配经过布尔化,无梯度路径;act_ema 无梯度)
  ——语义就是"把组的典型激活作为外部内容引入",梯度只经下游权重传播,
  模型可以学会利用这份注入,但不会去"优化"注入本身。
- 登记用注入前的激活(记录该层自然产生的共激活,不记自己注入后的结果,
  避免"注入强化自己组"的正反馈;这是一个设计自由度,先取保守选项,如实标注)。

条件(同数据种子,配对):
  base     : 全程不注入(登记照常运行) == 实验二的训练方式
  complete : 按时间表注入正确组
  placebo  : 同时间表注错组(固定置换)——如果它也"有效",收益就是平凡效应

数据: EMIT_EXTREME(rank6≈77次, rank7≈11次)——实验三B里冻结模型救不回来的
稀有词,看"从头活在补全机制里"的模型会不会不一样。

评估(训练结束后,每个模型两种方式各评一遍):
  - 带补全评估(与训练条件一致的部署形态)
  - 关补全评估(测模型是否已把功能"外包"给补全——依赖性检查)
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
from concept_lm_data import make_concept_spec, generate_batch, optimal_loss  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402
from run_expt2_completion import (  # noqa: E402
    DIM, LAYERS, HEADS, SEQ_LEN, TRAIN_BATCH, LR, WARMUP, N_STEPS, GRAD_CLIP,
    K_MAX, J_NEW, ACT_FRAC, FREQ_LR, DATA_SEED, OUT_DIR,
)
from run_expt3_stress import EMIT_EXTREME  # noqa: E402

WARM_REG = 600     # 只登记不注入
RAMP = 600         # alpha 线性爬升步数
ALPHA_MAX = 1.0


class TrainSiteHook:
    """训练中在线站点: 每次前向 (1)按注入前激活登记(stride 抽样), (2)按 alpha 注入。
    mode: off/on/placebo。alpha 由外部按步更新。"""

    def __init__(self, module, registry, mode="off", register_stride=8, use_abs=False, name=""):
        self.reg = registry
        self.mode = mode
        self.alpha = 0.0
        self.register_stride = register_stride
        self.use_abs = use_abs
        self.name = name
        self.registration_enabled = True
        self.handle = module.register_forward_hook(self._hook)

    def _perm(self):
        act_idx = torch.nonzero(self.reg.active).flatten()
        perm = torch.arange(self.reg.K)
        if len(act_idx) > 1:
            perm[act_idx] = act_idx.roll(1)
        return perm

    def _hook(self, _m, _i, out):
        B, T, U = out.shape
        with torch.no_grad():
            flat_d = out.detach().reshape(-1, U)
            basis = flat_d.abs() if self.use_abs else flat_d
            sets = self.reg.extract_sets(basis)
            if self.registration_enabled:
                sub = slice(None, None, self.register_stride)
                self.reg.register(sets[sub], values=flat_d[sub])
            if self.mode == "off" or self.alpha <= 0:
                return None
            assign, best_j = self.reg.assign(sets)
            hit = assign >= 0
            if not bool(hit.any()):
                return None
            k = assign.clamp_min(0)
            if self.mode == "placebo":
                k = self._perm()[k]
            mem = self.reg.membership[k]
            typ = self.reg.act_ema[k]
            missing = mem & (~sets) & hit.unsqueeze(-1)
            gain = (self.alpha * best_j.clamp_min(0.0)).unsqueeze(-1)
            inj = (typ * missing.float() * gain)  # 对 autograd 是常数
        return (out.reshape(-1, U) + inj).view(B, T, U)

    def remove(self):
        self.handle.remove()


def alpha_at(step):
    if step <= WARM_REG:
        return 0.0
    return min(1.0, (step - WARM_REG) / RAMP) * ALPHA_MAX


def train_one(spec, mode, seed, log_every=500):
    torch.manual_seed(seed)
    model = FullAttentionLM(spec["vocab_size"], DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    regs = [AssemblyRegistry(DIM * 4, K_MAX, J_NEW, FREQ_LR, act_frac=ACT_FRAC) for _ in range(LAYERS)]
    hooks = [TrainSiteHook(model.layers[li].ffn[1], regs[li], mode=mode, name=f"L{li}")
             for li in range(LAYERS)]

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def lr_fn(s):
        if s < WARMUP:
            return (s + 1) / WARMUP
        return 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / max(1, N_STEPS - WARMUP)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    gen = torch.Generator().manual_seed(DATA_SEED + seed * 10)

    t0 = time.time()
    diverged = False
    for step in range(1, N_STEPS + 1):
        a = alpha_at(step)
        for h in hooks:
            h.alpha = a
        ids, _ = generate_batch(TRAIN_BATCH, SEQ_LEN, spec, generator=gen)
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        lv = float(loss.detach())
        if not math.isfinite(lv) or lv > 10.0:
            print(f"  [{mode} seed{seed}] 训练发散于 step {step} (loss={lv:.3f})", flush=True)
            diverged = True
            break
        if step % log_every == 0 or step == N_STEPS:
            print(f"  [{mode} seed{seed} step {step}/{N_STEPS}] loss={lv:.4f} alpha={a:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model, regs, hooks, diverged


def eval_buckets(model, spec, hooks, alpha, n_batches=300, eval_bs=32):
    """hooks 的 alpha 设为给定值(0=关补全),登记关闭,配对分桶评估。"""
    members = spec["members"]
    buckets = {"frequent(rank0-1)": (0, 1), "mid(rank2-5)": (2, 5), "rare(rank6-7)": (6, 7),
               "all": (0, members - 1)}
    for h in hooks:
        h.alpha = alpha
        h.registration_enabled = False
    gen = torch.Generator().manual_seed(DATA_SEED + 555)
    model.eval()
    out = {}
    accs = {b: [] for b in buckets}
    with torch.no_grad():
        for _ in range(n_batches):
            ids, _ = generate_batch(eval_bs, SEQ_LEN, spec, generator=gen)
            logits = model(ids)
            lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                 ids[:, 1:].reshape(-1), reduction="none").view(ids.shape[0], -1)
            rank = ids[:, :-1] % members
            for bname, (lo, hi) in buckets.items():
                m = (rank >= lo) & (rank <= hi)
                accs[bname].append(lp[m])
    model.train()
    for h in hooks:
        h.registration_enabled = True
    for bname in buckets:
        x = torch.cat(accs[bname])
        out[bname] = {"n": x.shape[0], "loss": float(x.mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()

    spec = make_concept_spec(n_classes=8, members=8, emit_probs=EMIT_EXTREME)
    print(f"实验五: 训练中补全。emit={EMIT_EXTREME}, 最优loss={optimal_loss(spec):.4f}, "
          f"时间表: {WARM_REG}步纯登记 -> {RAMP}步alpha爬升 -> alpha={ALPHA_MAX}", flush=True)

    results = {"config": {"warm_reg": WARM_REG, "ramp": RAMP, "alpha_max": ALPHA_MAX}, "runs": []}
    for seed in args.seeds:
        for mode in ("off", "on", "placebo"):
            label = {"off": "base", "on": "complete", "placebo": "placebo"}[mode]
            print(f"\n== 训练 {label} seed={seed} ==", flush=True)
            model, regs, hooks, diverged = train_one(spec, mode, seed)
            run = {"mode": label, "seed": seed, "diverged": diverged}
            if not diverged:
                ev_with = eval_buckets(model, spec, hooks, alpha=(ALPHA_MAX if mode != "off" else 0.0))
                ev_without = eval_buckets(model, spec, hooks, alpha=0.0)
                run["eval_with_completion"] = ev_with
                run["eval_without_completion"] = ev_without
                r_w, r_wo = ev_with["rare(rank6-7)"], ev_without["rare(rank6-7)"]
                a_w = ev_with["all"]
                print(f"  [{label} seed{seed}] all={a_w['loss']:.4f} | "
                      f"rare 带补全={r_w['loss']:.4f} 关补全={r_wo['loss']:.4f} (n={r_w['n']})", flush=True)
                torch.save(model.state_dict(), os.path.join(OUT_DIR, f"expt5_{label}_seed{seed}.pt"))
            for h in hooks:
                h.remove()
            results["runs"].append(run)

    path = os.path.join(OUT_DIR, "expt5_intraining.json")
    with open(path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n完成。结果: {path}", flush=True)


if __name__ == "__main__":
    main()
