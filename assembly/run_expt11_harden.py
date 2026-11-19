# -*- coding: utf-8 -*-
"""
实验十一: 把"内部故障哨兵"结果加固到论文强度。

三个加固:
1. 故障类型从"随机砍高激活"扩到四种(都是实际会发生的损伤形态):
   - drop:   随机把 30% 高激活 unit 置零(实验十的原始形态,连续性参照)
   - noise:  高斯噪声,幅度 = 激活标准差的 0.5x / 1.0x
   - stuck:  随机 10%/30% 的 unit 永久卡死为 0(硬件坏道式,同一批 unit 每个输入都坏)
   - quant:  激活四舍五入到 4/8 个等级(粗暴量化压缩的失真)
2. 第二个种子: 正常参照流 + 登记库构建流都换种子重跑一遍,看 AUC 稳不稳。
3. 定位演示: 故障只打在 L0 或只打在 L1,两层各自的登记库各自打分。
   预期的不对称性: L1 故障 -> L0 检测器安静(上游没坏)、L1 检测器报警;
   L0 故障 -> 两层都报警(坏信号会往下游传)。"最早报警的层 = 病灶层"。
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from models import FullAttentionLM  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402
from run_expt6_realtext import load_corpus, make_batch, DIM, LAYERS, HEADS, SEQ_LEN, DATA_SEED  # noqa: E402
from run_expt10_detector import auc, K_BIG, TAU_BIG, AF, FLR, N_SEQ  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FAULT_SEED = 555000


class FaultInjector:
    """挂在某层 GELU 输出上。kind: none/drop/noise/stuck/quant。掩码按批号重播种。"""

    def __init__(self, module):
        self.kind = "none"
        self.param = 0.0
        self.batch_idx = 0
        self.stuck_mask = None  # stuck 模式: 固定坏 unit 集合
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        if self.kind == "none":
            return None
        B, T, U = out.shape
        flat = out.reshape(-1, U)
        g = torch.Generator().manual_seed(FAULT_SEED + self.batch_idx)
        if self.kind == "drop":
            a = flat.clamp_min(0.0)
            mx = a.max(-1, keepdim=True).values
            sets = (a > 0.5 * mx) & (mx > 0)
            m = (torch.rand(flat.shape, generator=g) < self.param) & sets
            flat = flat * (~m).float()
        elif self.kind == "noise":
            std = flat.std()
            flat = flat + torch.randn(flat.shape, generator=g) * std * self.param
        elif self.kind == "stuck":
            flat = flat * (~self.stuck_mask).float()  # 固定 unit 置零
        elif self.kind == "quant":
            lo, hi = float(flat.min()), float(flat.max())
            levels = self.param
            step = (hi - lo) / max(1, levels - 1)
            flat = ((flat - lo) / step).round() * step + lo
        return flat.view(B, T, U)

    def set_stuck(self, frac, n_units):
        g = torch.Generator().manual_seed(FAULT_SEED + 42)
        self.stuck_mask = torch.rand(n_units, generator=g) < frac

    def remove(self):
        self.handle.remove()


def build_registry(model, ids_corpus, layer, seed):
    reg = AssemblyRegistry(DIM * 4, K_BIG, TAU_BIG, FLR, act_frac=AF)
    acts = {}
    h = model.layers[layer].ffn[1].register_forward_hook(lambda m, i, o: acts.__setitem__(0, o.detach()))
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _ in range(150):
            ids = make_batch(ids_corpus, 16, SEQ_LEN, g)
            model(ids)
            flat = acts[0].reshape(-1, DIM * 4)[::8]
            reg.register(reg.extract_sets(flat), values=flat)
    h.remove()
    return reg


def main():
    ids_corpus, stoi, itos = load_corpus()
    V = max(itos.keys()) + 1
    torch.manual_seed(0)
    model = FullAttentionLM(V, DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "expt6_model_seed0.pt")))
    model.eval()

    all_results = {}
    for seed_tag, reg_seed, eval_seed in (("seedA", DATA_SEED + 8888, DATA_SEED + 424242),
                                           ("seedB", DATA_SEED + 5151, DATA_SEED + 616161)):
        print(f"\n===== {seed_tag} =====", flush=True)
        # 故障注入器(注册在捕获钩子之前 => 捕获到的是故障后的激活)
        injectors = {li: FaultInjector(model.layers[li].ffn[1]) for li in range(LAYERS)}
        injectors[1].set_stuck(0.1, DIM * 4)
        caps = {}
        handles = [model.layers[li].ffn[1].register_forward_hook(
            lambda m, i, o, li=li: caps.__setitem__(li, o.detach())) for li in range(LAYERS)]

        regs = {li: build_registry_with_caps(model, ids_corpus, li, reg_seed + li, caps) for li in range(LAYERS)}
        print(f"  登记库: L0={regs[0].n_groups}组 L1={regs[1].n_groups}组", flush=True)

        def scores(batches, fault_layer=None, kind="none", param=0.0, stuck_frac=None):
            out = {0: [], 1: []}
            inj = injectors.get(fault_layer)
            if inj is not None:
                inj.kind = kind
                inj.param = param
                if kind == "stuck":
                    inj.set_stuck(stuck_frac, DIM * 4)
            for bi, ids in enumerate(batches):
                if inj is not None:
                    inj.batch_idx = bi + 1
                with torch.no_grad():
                    model(ids)
                for li in range(LAYERS):
                    flat = caps[li].reshape(-1, DIM * 4)
                    _, rho = regs[li].assign(regs[li].extract_sets(flat))
                    out[li] += (-rho.clamp_min(0.0).view(ids.shape[0], -1).mean(1)).tolist()
            if inj is not None:
                inj.kind = "none"
            return out

        eg = torch.Generator().manual_seed(eval_seed)
        batches = [make_batch(ids_corpus, 16, SEQ_LEN, eg) for _ in range(N_SEQ // 16)]
        normal = scores(batches)

        res = {}
        cases = [
            ("drop30", dict(kind="drop", param=0.3)),
            ("noise0.5x", dict(kind="noise", param=0.5)),
            ("noise1.0x", dict(kind="noise", param=1.0)),
            ("stuck10%", dict(kind="stuck", param=0, stuck_frac=0.10)),
            ("stuck30%", dict(kind="stuck", param=0, stuck_frac=0.30)),
            ("quant4级", dict(kind="quant", param=4)),
            ("quant8级", dict(kind="quant", param=8)),
        ]
        # 主表: 故障打在 L1, 用 L1 检测器(与实验十同构)
        for name, kw in cases:
            s = scores(batches, fault_layer=1, **kw)
            a, se = auc(s[1], normal[1])
            res[name] = {"auc_L1det": a, "se": se}
            print(f"  {name:10s} (打L1, L1检测): AUC={a:.3f}±{se:.3f}", flush=True)

        # 定位演示: drop30 分别打 L0 / L1, 两层检测器各自打分
        loc = {}
        for fl in (0, 1):
            s = scores(batches, fault_layer=fl, kind="drop", param=0.3)
            for dl in (0, 1):
                a, se = auc(s[dl], normal[dl])
                loc[f"故障在L{fl}_检测器L{dl}"] = {"auc": a, "se": se}
                print(f"  定位: 故障在L{fl}, L{dl}检测器: AUC={a:.3f}±{se:.3f}", flush=True)
        res["localization"] = loc
        all_results[seed_tag] = res

        for h in handles:
            h.remove()
        for inj in injectors.values():
            inj.remove()

    with open(os.path.join(OUT_DIR, "expt11_harden.json"), "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=1)
    print("\n完成。", flush=True)


def build_registry_with_caps(model, ids_corpus, layer, seed, caps):
    """用已存在的捕获钩子建库(避免重复注册钩子)。"""
    reg = AssemblyRegistry(DIM * 4, K_BIG, TAU_BIG, FLR, act_frac=AF)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _ in range(150):
            ids = make_batch(ids_corpus, 16, SEQ_LEN, g)
            model(ids)
            flat = caps[layer].reshape(-1, DIM * 4)[::8]
            reg.register(reg.extract_sets(flat), values=flat)
    return reg


if __name__ == "__main__":
    main()
