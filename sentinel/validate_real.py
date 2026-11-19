# -*- coding: utf-8 -*-
"""
真实预训练模型(distilgpt2, 82M)上的哨兵验证——论文声明的规模化开放问题的实测。

流程:
1. 正常英文语料(本地 docstring 4MB)切成序列,一部分建库、一部分校准、一部分评估。
2. 覆盖率检查: K=2048 在 3072-unit 的真实 FFN 层上够不够(论文的容量律预言
   真实模型需要的组数会更多——先测,不猜)。
3. 故障注入(第4层): 砍30%活跃单元 / 等幅高斯噪声 / 30%单元卡死 / 4级量化,
   两通道读数各算 AUC(240 异常 vs 240 正常)。
4. 定位: 故障打在第2层 vs 第4层,两层检测器各自打分。
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from sentinel import AssemblySentinel  # noqa: E402

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "assembly", "data", "docstring_corpus.txt")
SEQ, BS = 128, 4
LAYERS = [2, 4]
N_BUILD, N_CAL, N_EVAL = 120, 30, 60   # 批数: 建库480序列/校准120/评估240
FAULT_SEED = 424242
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_results.json")


def batches_from_corpus(tok, n_batches, seed):
    text = open(CORPUS, encoding="utf-8").read()
    ids_all = tok(text, return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batches):
        starts = torch.randint(0, ids_all.shape[0] - SEQ - 1, (BS,), generator=g)
        out.append(torch.stack([ids_all[s:s + SEQ] for s in starts]))
    return out


class Fault:
    def __init__(self, model, layer):
        from sentinel import _find_act_module
        self.kind, self.param, self.batch_idx = "none", 0.0, 0
        self.stuck_mask = None
        self.h = _find_act_module(model.transformer.h[layer]).register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        if self.kind == "none":
            return None
        shp = out.shape
        flat = out.reshape(-1, shp[-1])
        g = torch.Generator().manual_seed(FAULT_SEED + self.batch_idx)
        if self.kind == "drop":
            a = flat.clamp_min(0.0)
            mx = a.max(-1, keepdim=True).values
            sets = (a > 0.5 * mx) & (mx > 0)
            m = (torch.rand(flat.shape, generator=g) < self.param) & sets
            flat = flat * (~m).float()
        elif self.kind == "noise":
            flat = flat + torch.randn(flat.shape, generator=g) * flat.std() * self.param
        elif self.kind == "stuck":
            if self.stuck_mask is None:
                gm = torch.Generator().manual_seed(FAULT_SEED + 7)
                self.stuck_mask = torch.rand(shp[-1], generator=gm) < self.param
            flat = flat * (~self.stuck_mask).float()
        elif self.kind == "quant":
            lo, hi = float(flat.min()), float(flat.max())
            step = (hi - lo) / max(1, int(self.param) - 1)
            flat = ((flat - lo) / step).round() * step + lo
        return flat.reshape(shp)

    def off(self):
        self.kind = "none"
        self.stuck_mask = None


def auc(pos, neg):
    p, n = torch.tensor(pos), torch.tensor(neg)
    cmp = (p.unsqueeze(1) > n.unsqueeze(0)).float() + 0.5 * (p.unsqueeze(1) == n.unsqueeze(0)).float()
    a = float(cmp.mean())
    n1, n2 = len(p), len(n)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = ((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2)) ** 0.5
    return a, se


def collect(sent, batches, faults=None, which_fault=("none", 0.0, None)):
    per = {li: {"rho": [], "vmis": []} for li in sent.layers}
    kind, param, stuck = which_fault
    if faults is not None:
        faults.kind, faults.param = kind, param
        if kind == "stuck":
            faults.param = stuck
            faults.stuck_mask = None
    for bi, ids in enumerate(batches):
        if faults is not None:
            faults.batch_idx = bi + 1
        s = sent.readings(ids)
        for li in sent.layers:
            per[li]["rho"] += [-x for x in s[li]["rho_per_seq"]]     # 越高越可疑
            per[li]["vmis"] += s[li]["vmis_per_seq"]
    if faults is not None:
        faults.off()
    return per


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype=torch.float32)
    model.eval()
    print(f"模型就绪 ({time.time()-t0:.0f}s)", flush=True)

    # 关键顺序: 故障钩子必须先于哨兵的捕获钩子注册——PyTorch 按注册顺序链式执行
    # forward hook,后注册的哨兵才能读到"故障之后"的激活。反过来接,哨兵读到的
    # 是故障前的干净激活,AUC 恒等于 0.500(第一次跑就是这么错的,如实留档)。
    fault2 = Fault(model, 2)
    fault4 = Fault(model, 4)

    sent = AssemblySentinel(model, LAYERS, k_max=2048, tau=0.2)
    build = batches_from_corpus(tok, N_BUILD, 111)
    cal = batches_from_corpus(tok, N_CAL, 222)
    ev = batches_from_corpus(tok, N_EVAL, 333)

    t0 = time.time()
    sent.build(build, log_every=30)
    print(f"建库完成 ({time.time()-t0:.0f}s)", flush=True)
    for li in LAYERS:
        r = sent.regs[li]
        print(f"  L{li}: 组数={r.n_groups}/{r.K} 合并={r.merge_events} 登记={r.total_registered}", flush=True)

    sent.calibrate(cal)
    # 覆盖率
    s = sent.readings(ev[0])
    for li in LAYERS:
        print(f"  L{li} 覆盖率(评估样本)={s[li]['coverage']:.3f}", flush=True)

    results = {"model": "distilgpt2", "layers": LAYERS, "k_max": 2048}
    normal = collect(sent, ev)

    cases = [("drop30", ("drop", 0.3, None)), ("noise1.0x", ("noise", 1.0, None)),
             ("stuck30%", ("stuck", 0.0, 0.3)), ("quant4", ("quant", 4, None))]
    print("\n== 故障打在第4层, 第4层检测器 ==", flush=True)
    for name, wf in cases:
        f = collect(sent, ev, fault4, wf)
        a_r, se_r = auc(f[4]["rho"], normal[4]["rho"])
        a_v, se_v = auc(f[4]["vmis"], normal[4]["vmis"])
        results[name] = {"rho": [a_r, se_r], "vmis": [a_v, se_v]}
        print(f"  {name:10s}: 开关读数 AUC={a_r:.3f}±{se_r:.3f} | 亮度读数 AUC={a_v:.3f}±{se_v:.3f}", flush=True)

    print("\n== 定位: drop30 分别打第2/第4层 ==", flush=True)
    loc = {}
    for fl, fobj in ((2, fault2), (4, fault4)):
        f = collect(sent, ev, fobj, ("drop", 0.3, None))
        for dl in LAYERS:
            a_r, _ = auc(f[dl]["rho"], normal[dl]["rho"])
            a_v, _ = auc(f[dl]["vmis"], normal[dl]["vmis"])
            loc[f"fault_L{fl}_det_L{dl}"] = {"rho": a_r, "vmis": a_v}
            print(f"  故障在L{fl}, L{dl}检测器: 开关={a_r:.3f} 亮度={a_v:.3f}", flush=True)
    results["localization"] = loc

    sent.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "distilgpt2_registry.pt"))
    with open(OUT, "w") as fo:
        json.dump(results, fo, ensure_ascii=False, indent=1)
    print(f"\n完成。登记库已存盘。", flush=True)


if __name__ == "__main__":
    main()
