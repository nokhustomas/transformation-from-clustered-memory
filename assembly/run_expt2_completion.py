# -*- coding: utf-8 -*-
"""
实验二: 重合度门控的激活补全,能不能帮到"学得差"的稀有词。

机制(用户构想的"激活层面"版,权重版留待后续): 冻结训练好的模型,推理时在指定
FFN 层上,每个位置的激活集合与登记库匹配(Jaccard);命中组 k(重合度 rho >= 门槛)
时,把组内"该亮而未亮"的 unit(membership(k) 且不在当前集合里)按典型激活值
act_ema[k] * g(rho) 注入,g(rho) = alpha * rho——重合度越高,越确信命中,注入越强。
这实现了"输入 b 与 a 组重合 40% -> 引入 a 组其余部分"的模式补全语义。

实验对象: 数据的类内词频带"超稀有尾部"(每类 8 个同义词,最后两个在训练里只出现
几十到几百次)。稀有词嵌入学得差 -> 其激活集合残缺 -> 如果残缺集合仍能命中正确的
类组,补全就把"这个类该有的样子"填进去。预测(如果机制有效): 稀有词位置的下一
token 预测 loss,开补全 < 关补全;高频词位置基本不变(它们的集合本来就完整,
没什么可注入)。

三个必须一起跑的对照,防止把平凡效应当成结论:
  - OFF:      不补全(基线)。
  - ON:       正常补全。
  - PLACEBO:  组身份被固定置换(命中组 k 后注入的是另一组 perm[k] 的缺失 pattern,
              注入强度/规模与 ON 同量级)。如果 PLACEBO 也"有效",那收益就不是来自
              "补的内容对",而是来自"随便多加点激活"这种平凡效应——这是上一条
              研究线里用户提出的"随机分组对照"思想在新范式里的直接沿用。

评估配对进行: 同一批评估数据分别在三种模式下前向,逐位置 loss 相减,按当前输入
token 的稀有度分桶(frequent=rank0-1, mid=rank2-5, rare=rank6-7)报告。
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
SEQ_LEN, TRAIN_BATCH = 64, 16
LR, WARMUP, N_STEPS = 3e-3, 100, 3000
GRAD_CLIP = 1.0
REG_STRIDE = 8
K_MAX, J_NEW, ACT_FRAC, FREQ_LR = 16, 0.3, 0.5, 0.05
DATA_SEED = 400000

# 超稀有尾部: rank6 约 1e-3、rank7 约 4e-4(3000 步 x 1024 token/步 x 1/8 类
# ≈ 384k 次/类 -> rank6 约出现 380 次, rank7 约 150 次;高频 rank0 约 13 万次)
EMIT_ULTRA = [0.36, 0.24, 0.15, 0.10, 0.08, 0.0686, 0.001, 0.0004]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)


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

    def remove(self):
        for h in self.handles:
            h.remove()


class CompletionHook:
    """挂在某一层 FFN 的 GELU 输出上,做重合度门控的激活补全。

    mode: "off" | "on" | "placebo"(固定置换组身份注入错误内容,强度同量级)。
    """

    def __init__(self, model, layer_idx, registry, alpha=1.0, mode="off"):
        self.reg = registry
        self.alpha = alpha
        self.mode = mode
        self.stats_reset()
        self.handle = model.layers[layer_idx].ffn[1].register_forward_hook(self._hook)
        # placebo 的固定置换: 活跃组循环移位(确定性,无 RNG)
        act_idx = torch.nonzero(registry.active).flatten()
        perm = torch.arange(registry.K)
        if len(act_idx) > 1:
            perm[act_idx] = act_idx.roll(1)
        self.perm = perm

    def stats_reset(self):
        self.n_positions = 0
        self.n_hit = 0
        self.injected_units_sum = 0.0
        self.rho_sum = 0.0

    def _hook(self, _m, _i, out):
        if self.mode == "off":
            return None
        B, T, U = out.shape
        flat = out.reshape(-1, U)
        sets = self.reg.extract_sets(flat)
        assign, best_j = self.reg.assign(sets)
        hit = assign >= 0
        self.n_positions += flat.shape[0]
        self.n_hit += int(hit.sum())
        if not bool(hit.any()):
            return None

        k = assign.clamp_min(0)
        if self.mode == "placebo":
            k = self.perm[k]
        mem = self.reg.membership[k]                    # [P, U] 组的成员集合
        typ = self.reg.act_ema[k]                       # [P, U] 组的典型激活
        missing = mem & (~sets) & hit.unsqueeze(-1)     # 该亮而未亮
        gain = (self.alpha * best_j.clamp_min(0.0)).unsqueeze(-1)  # g(rho) = alpha * rho
        inj = typ * missing.float() * gain
        self.injected_units_sum += float(missing.float().sum())
        self.rho_sum += float(best_j[hit].sum())
        return (flat + inj).view(B, T, U)

    def stats(self):
        return {
            "hit_rate": self.n_hit / max(1, self.n_positions),
            "mean_injected_units_per_hit": self.injected_units_sum / max(1, self.n_hit),
            "mean_rho": self.rho_sum / max(1, self.n_hit),
        }

    def remove(self):
        self.handle.remove()


def train_model_with_registry(spec, seed=0, log_every=500):
    torch.manual_seed(seed)
    model = FullAttentionLM(spec["vocab_size"], DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    cap = ActCapture(model)
    regs = [AssemblyRegistry(DIM * 4, K_MAX, J_NEW, FREQ_LR, act_frac=ACT_FRAC) for _ in range(LAYERS)]
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def lr_fn(s):
        if s < WARMUP:
            return (s + 1) / WARMUP
        return 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / max(1, N_STEPS - WARMUP)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    gen = torch.Generator().manual_seed(DATA_SEED + seed * 10)

    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        ids, _ = generate_batch(TRAIN_BATCH, SEQ_LEN, spec, generator=gen)
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        for li in range(LAYERS):
            acts = cap.acts[li].reshape(-1, cap.acts[li].shape[-1])[::REG_STRIDE]
            regs[li].register(regs[li].extract_sets(acts), values=acts)  # 带典型激活值登记
        if step % log_every == 0 or step == N_STEPS:
            print(f"  [train step {step}/{N_STEPS}] loss={float(loss):.4f} "
                  f"L1组数={regs[1].n_groups} ({time.time()-t0:.0f}s)", flush=True)
    cap.remove()
    return model, regs


def paired_eval(model, spec, regs, layer_idx, alpha, n_batches=200, eval_bs=32, seed_off=555):
    """同一批数据在 off/on/placebo 三种模式下前向,逐位置 loss 按稀有度分桶配对比较。"""
    members = spec["members"]
    buckets = {"frequent(rank0-1)": (0, 1), "mid(rank2-5)": (2, 5), "rare(rank6-7)": (6, 7)}
    acc = {m: {b: [] for b in buckets} for m in ("off", "on", "placebo")}
    hooks = {}

    model.eval()
    gen = torch.Generator().manual_seed(DATA_SEED + seed_off)
    all_ids = [generate_batch(eval_bs, SEQ_LEN, spec, generator=gen)[0] for _ in range(n_batches)]

    for mode in ("off", "on", "placebo"):
        hook = CompletionHook(model, layer_idx, regs[layer_idx], alpha=alpha, mode=mode)
        with torch.no_grad():
            for ids in all_ids:
                logits = model(ids)
                lp = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    ids[:, 1:].reshape(-1), reduction="none",
                ).view(ids.shape[0], -1)                     # [B, T-1] 位置 t 预测 t+1
                rank = (ids[:, :-1] % members)               # 当前输入 token 的类内 rank
                for bname, (lo, hi) in buckets.items():
                    m = (rank >= lo) & (rank <= hi)
                    acc[mode][bname].append(lp[m])
        hooks[mode] = hook.stats()
        hook.remove()

    out = {"hook_stats": hooks}
    for bname in buckets:
        off = torch.cat(acc["off"][bname])
        on = torch.cat(acc["on"][bname])
        pl = torch.cat(acc["placebo"][bname])
        d_on = on - off      # 配对差,负=补全有帮助
        d_pl = pl - off
        n = off.shape[0]
        out[bname] = {
            "n_positions": n,
            "loss_off": float(off.mean()), "loss_on": float(on.mean()), "loss_placebo": float(pl.mean()),
            "delta_on": float(d_on.mean()), "delta_on_se": float(d_on.std() / max(1, n) ** 0.5),
            "delta_placebo": float(d_pl.mean()), "delta_placebo_se": float(d_pl.std() / max(1, n) ** 0.5),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--layer", type=int, default=1)
    args = ap.parse_args()

    spec = make_concept_spec(n_classes=8, members=8, emit_probs=EMIT_ULTRA)
    print(f"实验二: 超稀有尾部词频 {EMIT_ULTRA}", flush=True)
    print(f"最优loss参照 {optimal_loss(spec):.4f}; 训练 {N_STEPS} 步后冻结,补全层=L{args.layer}", flush=True)

    model, regs = train_model_with_registry(spec, seed=args.seed)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"expt2_model_seed{args.seed}.pt"))
    torch.save({f"L{li}": regs[li].state_dict_plain() for li in range(LAYERS)},
               os.path.join(OUT_DIR, f"expt2_registries_seed{args.seed}.pt"))

    results = {"seed": args.seed, "emit": EMIT_ULTRA, "layer": args.layer, "alphas": {}}
    for alpha in args.alphas:
        print(f"\n== 配对评估 alpha={alpha} ==", flush=True)
        r = paired_eval(model, spec, regs, args.layer, alpha)
        results["alphas"][str(alpha)] = r
        for b in ("frequent(rank0-1)", "mid(rank2-5)", "rare(rank6-7)"):
            d = r[b]
            print(f"  {b:18s} n={d['n_positions']:6d} off={d['loss_off']:.4f} | "
                  f"ON Δ={d['delta_on']:+.4f}(±{d['delta_on_se']:.4f}) | "
                  f"PLACEBO Δ={d['delta_placebo']:+.4f}(±{d['delta_placebo_se']:.4f})", flush=True)
        hs = r["hook_stats"]
        print(f"  命中率 on={hs['on']['hit_rate']:.3f} 每命中注入unit数 on={hs['on']['mean_injected_units_per_hit']:.1f} "
              f"placebo={hs['placebo']['mean_injected_units_per_hit']:.1f} 平均rho={hs['on']['mean_rho']:.3f}", flush=True)

    path = os.path.join(OUT_DIR, f"expt2_completion_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n完成。结果: {path}", flush=True)


if __name__ == "__main__":
    main()
