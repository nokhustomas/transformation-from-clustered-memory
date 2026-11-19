# -*- coding: utf-8 -*-
"""
实验三: 给补全机制制造"真的有东西可补"的场景。

实验二的结果(如实): 补全机制匹配近乎完美(命中率 1.0, rho=0.90)但无事可做——
每次命中平均只有 0.7 个缺失 unit,因为这个任务里连稀有词(150-380 次出现)的
激活集合都已经学得完整了。安慰剂对照证明注入通路有因果效力(注错内容 loss 明显
变差),所以问题不是机制没接上,是这个 regime 里没有缺损可修。

两个压力场景:

A. 损坏修复(部分线索 -> 整组补全的直接演示): 冻结实验二的模型,评估时在补全层
   把当前激活集合里随机 drop_frac 比例的活跃 unit 强行置零(模拟残缺线索),
   然后看补全能不能把损失恢复回来。四条配对: 干净基线 / 损坏+不补 / 损坏+补全 /
   损坏+安慰剂。损坏掩码在三种补全模式间逐批完全相同(按批号重播种的独立
   Generator),保证配对干净。预测(如果构想成立): 损坏拉高 loss,补全恢复其中
   相当一部分,安慰剂进一步恶化。

B. 真欠训练词: 把类内最稀有两个词压到训练中只出现 ~10-80 次(rank6≈2e-4,
   rank7≈3e-5),重训模型,重复实验二的配对评估。这时稀有词的嵌入/激活集合
   应该真的残缺——看补全有没有救,以及残缺集合还能不能命中正确的组
   (如果 rho 掉到门槛以下连命中都不发生,那也是结论: 检索的失效点找到了)。
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
from concept_lm_data import make_concept_spec, generate_batch, optimal_loss  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402
from run_expt2_completion import (  # noqa: E402
    train_model_with_registry, EMIT_ULTRA,
    DIM, LAYERS, HEADS, SEQ_LEN, K_MAX, J_NEW, ACT_FRAC, FREQ_LR, DATA_SEED, OUT_DIR,
)

EMIT_EXTREME = [0.37, 0.25, 0.15, 0.10, 0.08, 0.04957, 2e-4, 3e-5]  # rank6≈77次, rank7≈11次
CORRUPT_SEED = 987654


class StressHook:
    """损坏(可选) + 补全(off/on/placebo)。损坏掩码由 batch_idx 重播种,跨模式一致。"""

    def __init__(self, model, layer_idx, registry, alpha=1.0, mode="off", drop_frac=0.0):
        self.reg = registry
        self.alpha = alpha
        self.mode = mode
        self.drop_frac = drop_frac
        self.batch_idx = 0
        self.stats_reset()
        self.handle = model.layers[layer_idx].ffn[1].register_forward_hook(self._hook)
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
        B, T, U = out.shape
        flat = out.reshape(-1, U)

        if self.drop_frac > 0:
            g = torch.Generator().manual_seed(CORRUPT_SEED + self.batch_idx)  # 跨模式一致
            sets_pre = self.reg.extract_sets(flat)
            drop = (torch.rand(flat.shape, generator=g) < self.drop_frac) & sets_pre
            flat = flat * (~drop).float()

        if self.mode != "off":
            sets = self.reg.extract_sets(flat)
            assign, best_j = self.reg.assign(sets)
            hit = assign >= 0
            self.n_positions += flat.shape[0]
            self.n_hit += int(hit.sum())
            if bool(hit.any()):
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
        return {"hit_rate": self.n_hit / max(1, self.n_positions),
                "mean_injected_units_per_hit": self.injected_units_sum / max(1, self.n_hit),
                "mean_rho": self.rho_sum / max(1, self.n_hit)}

    def remove(self):
        self.handle.remove()


def paired_eval_stress(model, spec, registry, layer_idx, alpha, drop_frac,
                        modes, n_batches=100, eval_bs=32, seed_off=555):
    members = spec["members"]
    buckets = {"frequent(rank0-1)": (0, 1), "mid(rank2-5)": (2, 5), "rare(rank6-7)": (6, 7),
               "all": (0, members - 1)}
    gen = torch.Generator().manual_seed(DATA_SEED + seed_off)
    all_ids = [generate_batch(eval_bs, SEQ_LEN, spec, generator=gen)[0] for _ in range(n_batches)]

    acc = {m: {b: [] for b in buckets} for m in modes}
    hook_stats = {}
    model.eval()
    for mode_name, mode, df in modes_cfg(modes, drop_frac):
        hook = StressHook(model, layer_idx, registry, alpha=alpha, mode=mode, drop_frac=df)
        with torch.no_grad():
            for bi, ids in enumerate(all_ids):
                hook.batch_idx = bi
                logits = model(ids)
                lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                     ids[:, 1:].reshape(-1), reduction="none").view(ids.shape[0], -1)
                rank = ids[:, :-1] % members
                for bname, (lo, hi) in buckets.items():
                    m = (rank >= lo) & (rank <= hi)
                    acc[mode_name][bname].append(lp[m])
        hook_stats[mode_name] = hook.stats()
        hook.remove()

    out = {"hook_stats": hook_stats, "buckets": {}}
    base = "clean_off" if "clean_off" in acc else modes[0]
    for bname in buckets:
        ref = torch.cat(acc[base][bname])
        row = {"n_positions": ref.shape[0], f"loss_{base}": float(ref.mean())}
        for m in modes:
            if m == base:
                continue
            x = torch.cat(acc[m][bname])
            d = x - ref
            row[f"loss_{m}"] = float(x.mean())
            row[f"delta_{m}"] = float(d.mean())
            row[f"delta_{m}_se"] = float(d.std() / max(1, d.shape[0]) ** 0.5)
        out["buckets"][bname] = row
    return out


def modes_cfg(modes, drop_frac):
    cfg = []
    for m in modes:
        if m == "clean_off":
            cfg.append((m, "off", 0.0))
        else:
            cfg.append((m, m.replace("corrupt_", ""), drop_frac))
    return cfg


def main():
    results = {}

    # ---------- A. 损坏修复(用实验二已训练的模型/登记库) ----------
    print("== A. 损坏修复测试(实验二冻结模型) ==", flush=True)
    spec2 = make_concept_spec(n_classes=8, members=8, emit_probs=EMIT_ULTRA)
    torch.manual_seed(0)
    model = FullAttentionLM(spec2["vocab_size"], DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "expt2_model_seed0.pt")))
    st = torch.load(os.path.join(OUT_DIR, "expt2_registries_seed0.pt"))
    reg = AssemblyRegistry(**st["L1"]["config"])
    for key in ("freq", "act_ema", "active", "count", "created_at"):
        setattr(reg, key, st["L1"][key])

    results["A_corruption"] = {}
    for drop in (0.3, 0.5, 0.7):
        r = paired_eval_stress(model, spec2, reg, layer_idx=1, alpha=1.0, drop_frac=drop,
                                modes=["clean_off", "corrupt_off", "corrupt_on", "corrupt_placebo"])
        results["A_corruption"][str(drop)] = r
        a = r["buckets"]["all"]
        hs = r["hook_stats"]
        recov = (a["delta_corrupt_off"] - a["delta_corrupt_on"]) / max(1e-9, a["delta_corrupt_off"])
        print(f"  drop={drop}: 损坏Δ=+{a['delta_corrupt_off']:.4f} | 补全Δ=+{a['delta_corrupt_on']:.4f} "
              f"(恢复 {recov*100:.0f}%) | 安慰剂Δ=+{a['delta_corrupt_placebo']:.4f} | "
              f"rho={hs['corrupt_on']['mean_rho']:.3f} 注入/命中={hs['corrupt_on']['mean_injected_units_per_hit']:.1f} "
              f"命中率={hs['corrupt_on']['hit_rate']:.3f}", flush=True)

    # ---------- B. 真欠训练词(重训) ----------
    print(f"\n== B. 极端稀有尾部重训: emit={EMIT_EXTREME} ==", flush=True)
    spec3 = make_concept_spec(n_classes=8, members=8, emit_probs=EMIT_EXTREME)
    print(f"  最优loss参照 {optimal_loss(spec3):.4f}", flush=True)
    model3, regs3 = train_model_with_registry(spec3, seed=0)
    torch.save(model3.state_dict(), os.path.join(OUT_DIR, "expt3_model_seed0.pt"))
    torch.save({f"L{li}": regs3[li].state_dict_plain() for li in range(LAYERS)},
               os.path.join(OUT_DIR, "expt3_registries_seed0.pt"))

    r = paired_eval_stress(model3, spec3, regs3[1], layer_idx=1, alpha=1.0, drop_frac=0.0,
                            modes=["clean_off", "on", "placebo"], n_batches=300)
    results["B_extreme_rare"] = r
    for b in ("frequent(rank0-1)", "mid(rank2-5)", "rare(rank6-7)"):
        d = r["buckets"][b]
        print(f"  {b:18s} n={d['n_positions']:6d} off={d['loss_clean_off']:.4f} | "
              f"ON Δ={d['delta_on']:+.4f}(±{d['delta_on_se']:.4f}) | "
              f"PLACEBO Δ={d['delta_placebo']:+.4f}(±{d['delta_placebo_se']:.4f})", flush=True)
    hs = r["hook_stats"]
    print(f"  命中率 on={hs['on']['hit_rate']:.3f} 注入/命中={hs['on']['mean_injected_units_per_hit']:.1f} "
          f"rho={hs['on']['mean_rho']:.3f}", flush=True)

    # 诊断: 极端稀有词的激活集合残缺程度与命中情况(按 rank 分桶)
    print("\n  [诊断] 按 rank 看集合残缺度/命中率:", flush=True)
    gen = torch.Generator().manual_seed(DATA_SEED + 999)
    ids, _ = generate_batch(256, SEQ_LEN, spec3, generator=gen)
    acts = {}
    h = model3.layers[1].ffn[1].register_forward_hook(lambda m, i, o: acts.__setitem__("x", o.detach()))
    model3.eval()
    with torch.no_grad():
        model3(ids)
    h.remove()
    flat = acts["x"].reshape(-1, DIM * 4)
    sets = regs3[1].extract_sets(flat)
    assign, best_j = regs3[1].assign(sets)
    ranks = (ids % 8).reshape(-1)
    diag = {}
    for lo, hi, name in ((0, 1, "frequent"), (2, 5, "mid"), (6, 6, "rank6(~77次)"), (7, 7, "rank7(~11次)")):
        m = (ranks >= lo) & (ranks <= hi)
        hit = (assign >= 0) & m
        mem = regs3[1].membership[assign.clamp_min(0)]
        missing = (mem & ~sets).float().sum(-1)
        diag[name] = {"n": int(m.sum()), "hit_rate": float((assign[m] >= 0).float().mean()),
                      "mean_rho": float(best_j[m].mean()),
                      "mean_missing_when_hit": float(missing[hit].mean()) if bool(hit.any()) else float("nan")}
        print(f"    {name:14s} n={diag[name]['n']:5d} 命中率={diag[name]['hit_rate']:.3f} "
              f"rho={diag[name]['mean_rho']:.3f} 命中时缺失unit={diag[name]['mean_missing_when_hit']:.1f}", flush=True)
    results["B_diag_by_rank"] = diag

    path = os.path.join(OUT_DIR, "expt3_stress_seed0.json")
    with open(path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n完成。结果: {path}", flush=True)


if __name__ == "__main__":
    main()
