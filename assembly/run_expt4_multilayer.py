# -*- coding: utf-8 -*-
"""
实验四: 补全站点从 L1 单层扩到 嵌入层+L0+L1 三处,针对实验三B的发现
("自然欠训练的病灶分布在整条通路,只补 L1 够不着")再测极端稀有词。

站点与实现说明:
- L0/L1: FFN GELU 输出,与实验二/三完全相同的机制,登记库直接用实验三训练时
  在线积累的那两个(expt3_registries_seed0.pt)。
- 嵌入层(实验性延伸,单独标注): 嵌入向量有正负,"超过该位置最大激活 50%"的
  集合提取规则改在 |值| 上做;注入时用带符号的典型值(act_ema 登记的是原始
  带符号值)。嵌入层登记库不是训练时建的(当时没挂),这里用冻结模型在训练
  分布上流式补建(同一条 register() 在线代码路径,只是时间点在训练后)——
  如实标注这一差异。
- L0 的已知弱点(实验一): 集合被 token 身份主导、组搅动多,但已分配组仍类纯。
  补全在 L0 的命中率/收益单独报告,不与 L1 混。

评估: 与实验三B同一冻结模型(EMIT_EXTREME, rank7≈11次出现),配对评估,
站点组合 off / L1 / L0+L1 / emb+L0+L1 / 安慰剂(emb+L0+L1 全站点注错内容),
评估批数加大到 600(稀有桶样本量 ~260)。
"""

import json
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
from concept_lm_data import make_concept_spec, generate_batch  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402
from run_expt2_completion import (  # noqa: E402
    DIM, LAYERS, HEADS, SEQ_LEN, K_MAX, J_NEW, ACT_FRAC, FREQ_LR, DATA_SEED, OUT_DIR,
)
from run_expt3_stress import EMIT_EXTREME  # noqa: E402


class SiteHook:
    """一个补全站点(挂在某个 module 的输出上)。mode: off/on/placebo。
    use_abs=True 时集合按 |激活| 提取(嵌入层用),注入仍用带符号 act_ema。"""

    def __init__(self, module, registry, alpha=1.0, mode="off", use_abs=False, name=""):
        self.reg = registry
        self.alpha = alpha
        self.mode = mode
        self.use_abs = use_abs
        self.name = name
        self.stats_reset()
        self.handle = module.register_forward_hook(self._hook)
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
        basis = flat.abs() if self.use_abs else flat
        sets = self.reg.extract_sets(basis)
        assign, best_j = self.reg.assign(sets)
        hit = assign >= 0
        self.n_positions += flat.shape[0]
        self.n_hit += int(hit.sum())
        if not bool(hit.any()):
            return None
        k = assign.clamp_min(0)
        if self.mode == "placebo":
            k = self.perm[k]
        mem = self.reg.membership[k]
        typ = self.reg.act_ema[k]
        missing = mem & (~sets) & hit.unsqueeze(-1)
        gain = (self.alpha * best_j.clamp_min(0.0)).unsqueeze(-1)
        flat = flat + typ * missing.float() * gain
        self.injected_units_sum += float(missing.float().sum())
        self.rho_sum += float(best_j[hit].sum())
        return flat.view(B, T, U)

    def stats(self):
        return {"site": self.name,
                "hit_rate": self.n_hit / max(1, self.n_positions),
                "mean_injected_units_per_hit": self.injected_units_sum / max(1, self.n_hit),
                "mean_rho": self.rho_sum / max(1, self.n_hit)}

    def remove(self):
        self.handle.remove()


def build_emb_registry(model, spec, n_batches=100, bs=16):
    """冻结模型上流式补建嵌入层登记库(带符号典型值,|值|集合)。"""
    reg = AssemblyRegistry(DIM, K_MAX, J_NEW, FREQ_LR, act_frac=ACT_FRAC)
    gen = torch.Generator().manual_seed(DATA_SEED + 4444)
    with torch.no_grad():
        for _ in range(n_batches):
            ids, _ = generate_batch(bs, SEQ_LEN, spec, generator=gen)
            emb = model.tok_emb(ids).reshape(-1, DIM)
            sub = emb[::4]  # 步长抽样控制成本
            reg.register(reg.extract_sets(sub.abs()), values=sub)
    return reg


def main():
    spec = make_concept_spec(n_classes=8, members=8, emit_probs=EMIT_EXTREME)
    torch.manual_seed(0)
    model = FullAttentionLM(spec["vocab_size"], DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "expt3_model_seed0.pt")))
    model.eval()

    st = torch.load(os.path.join(OUT_DIR, "expt3_registries_seed0.pt"))
    regs = {}
    for li in (0, 1):
        r = AssemblyRegistry(**st[f"L{li}"]["config"])
        for key in ("freq", "act_ema", "active", "count", "created_at"):
            setattr(r, key, st[f"L{li}"][key])
        regs[li] = r

    print("流式补建嵌入层登记库…", flush=True)
    t0 = time.time()
    emb_reg = build_emb_registry(model, spec)
    print(f"  嵌入层登记库: {emb_reg.n_groups} 组, 合并 {emb_reg.merge_events} 次 ({time.time()-t0:.0f}s)", flush=True)

    # 站点组合: (名字, [(module, registry, use_abs, site_name)], mode)
    SITES_L1 = [(model.layers[1].ffn[1], regs[1], False, "L1")]
    SITES_L0L1 = [(model.layers[0].ffn[1], regs[0], False, "L0")] + SITES_L1
    SITES_ALL = [(model.tok_emb, emb_reg, True, "emb")] + SITES_L0L1

    configs = [
        ("off", [], "off"),
        ("L1", SITES_L1, "on"),
        ("L0+L1", SITES_L0L1, "on"),
        ("emb+L0+L1", SITES_ALL, "on"),
        ("placebo(emb+L0+L1)", SITES_ALL, "placebo"),
    ]

    members = spec["members"]
    buckets = {"frequent(rank0-1)": (0, 1), "mid(rank2-5)": (2, 5),
               "rare(rank6-7)": (6, 7), "all": (0, members - 1)}
    n_batches, eval_bs = 600, 32
    gen = torch.Generator().manual_seed(DATA_SEED + 555)
    all_ids = [generate_batch(eval_bs, SEQ_LEN, spec, generator=gen)[0] for _ in range(n_batches)]

    acc = {}
    site_stats = {}
    for cname, sites, mode in configs:
        hooks = [SiteHook(m, r, alpha=1.0, mode=mode, use_abs=ab, name=nm) for m, r, ab, nm in sites]
        acc[cname] = {b: [] for b in buckets}
        with torch.no_grad():
            for ids in all_ids:
                logits = model(ids)
                lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                     ids[:, 1:].reshape(-1), reduction="none").view(ids.shape[0], -1)
                rank = ids[:, :-1] % members
                for bname, (lo, hi) in buckets.items():
                    m = (rank >= lo) & (rank <= hi)
                    acc[cname][bname].append(lp[m])
        site_stats[cname] = [h.stats() for h in hooks]
        for h in hooks:
            h.remove()
        print(f"  [{cname}] 评估完成", flush=True)

    results = {"site_stats": site_stats, "buckets": {}}
    print("\n== 配对结果(相对 off) ==", flush=True)
    for bname in buckets:
        ref = torch.cat(acc["off"][bname])
        row = {"n_positions": ref.shape[0], "loss_off": float(ref.mean())}
        line = f"  {bname:18s} n={ref.shape[0]:7d} off={float(ref.mean()):.4f}"
        for cname, _, _ in configs[1:]:
            x = torch.cat(acc[cname][bname])
            d = x - ref
            row[f"delta[{cname}]"] = float(d.mean())
            row[f"se[{cname}]"] = float(d.std() / max(1, d.shape[0]) ** 0.5)
            line += f" | {cname} Δ={float(d.mean()):+.4f}(±{float(d.std()/max(1,d.shape[0])**0.5):.4f})"
        results["buckets"][bname] = row
        print(line, flush=True)

    print("\n站点统计(emb+L0+L1 配置):", flush=True)
    for s in site_stats["emb+L0+L1"]:
        print(f"  {s['site']:4s} 命中率={s['hit_rate']:.3f} 注入/命中={s['mean_injected_units_per_hit']:.1f} rho={s['mean_rho']:.3f}", flush=True)

    path = os.path.join(OUT_DIR, "expt4_multilayer_seed0.json")
    with open(path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n完成。结果: {path}", flush=True)


if __name__ == "__main__":
    main()
