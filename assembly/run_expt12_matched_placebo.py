# -*- coding: utf-8 -*-
"""
实验12: 等量安慰剂对照——把"内容错了"和"扰动量大了10倍"分开。

背景(用户指出): 此前所有用 missing = M\\S 定义的实验里,治疗组与安慰剂组的注入量
根本不等量(实测 L1: 治疗每位置缺失成员 1.73 个、范数 2.41; 安慰剂 17.50 个、范数 9.38)。
所以"治疗≈0、安慰剂有害"这个对比里混着一个纯量级效应,不能归因于内容。

本实验的三个安慰剂,逐级把量级因素扣掉:
  placebo_raw     : 原来的做法(错位组的 missing),量级不受控 —— 复现锚点
  placebo_matched : 错位组的 missing 里**随机抽到与治疗组相同的单元数**,
                    再把整体范数缩放到与治疗组逐位置相等 => 量级完全匹配,只有内容错
  noise_matched   : 不用任何组的内容,注入随机方向,单元数与范数同样匹配
                    => 连"错位组的内容结构"都拿掉,只剩"扰动本身"
治疗组(正确组 missing)作为参照。若治疗 ≈ 等量安慰剂 ≈ 等量噪声,则说明内容压根不重要;
若治疗明显好于两者,则内容有价值,只是效应小。

任务/模型: 与实验3B相同(极端稀有尾部合成语言, 已训练好的 expt3_model_seed0.pt),
指标: 稀有词位置的预测损失,配对比较,n≈700。
"""

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))

from models import FullAttentionLM  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402
from concept_lm_data import make_concept_spec, generate_batch  # noqa: E402
from run_expt3_stress import EMIT_EXTREME  # noqa: E402
from run_expt2_completion import DIM, LAYERS, HEADS, SEQ_LEN, DATA_SEED, OUT_DIR  # noqa: E402

N_BATCHES = 1200
OUT = os.path.join(OUT_DIR, "expt12_matched_placebo.json")


class Injector:
    def __init__(self, model, layer, reg):
        self.reg = reg
        self.mode = "off"
        self.alpha = 1.0
        self.gen = torch.Generator().manual_seed(1234)
        act_idx = torch.nonzero(reg.active).flatten()
        perm = torch.arange(reg.K)
        if len(act_idx) > 1:
            perm[act_idx] = act_idx.roll(1)
        self.perm = perm
        self.stats = {"n_units": 0.0, "norm": 0.0, "n": 0}
        self.h = model.layers[layer].ffn[1].register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        if self.mode == "off":
            return None
        shp = out.shape
        flat = out.reshape(-1, shp[-1])
        sets = self.reg.extract_sets(flat)
        a, rho = self.reg.assign(sets)
        hit = a >= 0
        if not bool(hit.any()):
            return None
        gain = (self.alpha * rho.clamp_min(0.0)).unsqueeze(-1)

        # --- 治疗组的注入(所有条件都要先算它, 用来定"等量"的标准) ---
        k_t = a.clamp_min(0)
        miss_t = self.reg.membership[k_t] & (~sets) & hit.unsqueeze(-1)
        inj_t = self.reg.act_ema[k_t] * miss_t.float() * gain
        n_t = miss_t.float().sum(-1)                     # [P] 治疗组每位置注入几个单元
        norm_t = inj_t.norm(dim=-1)                      # [P] 治疗组每位置注入范数

        if self.mode == "treat":
            inj = inj_t
        else:
            k_p = self.perm[a.clamp_min(0)]
            if self.mode == "placebo_raw":
                miss_p = self.reg.membership[k_p] & (~sets) & hit.unsqueeze(-1)
                inj = self.reg.act_ema[k_p] * miss_p.float() * gain
            elif self.mode == "placebo_matched":
                miss_p = self.reg.membership[k_p] & (~sets) & hit.unsqueeze(-1)
                # 从错位组的候选单元里随机抽 n_t 个(逐位置)
                r = torch.rand(miss_p.shape, generator=self.gen) * miss_p.float()
                thresh = torch.zeros_like(n_t)
                srt, _ = r.sort(dim=-1, descending=True)
                idx = (n_t.long().clamp(min=0, max=r.shape[-1] - 1))
                thresh = srt.gather(1, idx.unsqueeze(1)).squeeze(1)
                keep = (r > thresh.unsqueeze(-1)) & miss_p
                inj = self.reg.act_ema[k_p] * keep.float() * gain
            elif self.mode == "noise_matched":
                cand = (~sets) & hit.unsqueeze(-1)       # 只往"当前没亮"的单元注入,与治疗同域
                r = torch.rand(cand.shape, generator=self.gen) * cand.float()
                srt, _ = r.sort(dim=-1, descending=True)
                idx = (n_t.long().clamp(min=0, max=r.shape[-1] - 1))
                thresh = srt.gather(1, idx.unsqueeze(1)).squeeze(1)
                keep = (r > thresh.unsqueeze(-1)) & cand
                inj = torch.randn(flat.shape, generator=self.gen).abs() * keep.float()
            else:
                raise ValueError(self.mode)
            # 逐位置把范数缩放到与治疗组相同
            cur = inj.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            inj = inj / cur * norm_t.unsqueeze(-1)

        self.stats["n_units"] += float((inj != 0).float().sum(-1)[hit].sum())
        self.stats["norm"] += float(inj.norm(dim=-1)[hit].sum())
        self.stats["n"] += int(hit.sum())
        return (flat + inj).reshape(shp)

    def reset_stats(self):
        self.stats = {"n_units": 0.0, "norm": 0.0, "n": 0}

    def summary(self):
        n = max(1, self.stats["n"])
        return {"每位置注入单元数": self.stats["n_units"] / n, "每位置注入范数": self.stats["norm"] / n}


def main():
    t0 = time.time()
    spec = make_concept_spec(8, 8, emit_probs=EMIT_EXTREME)
    torch.manual_seed(0)
    model = FullAttentionLM(spec["vocab_size"], DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "expt3_model_seed0.pt")))
    model.eval()

    st = torch.load(os.path.join(OUT_DIR, "expt3_registries_seed0.pt"))
    d = st["L1"]
    reg = AssemblyRegistry(**d["config"])
    for k in ("freq", "act_ema", "active", "count", "created_at"):
        setattr(reg, k, d[k])

    inj = Injector(model, 1, reg)
    modes = ["off", "treat", "placebo_raw", "placebo_matched", "noise_matched"]
    names = {"off": "不注入(基线)", "treat": "治疗(正确组)", "placebo_raw": "旧安慰剂(量级失控)",
             "placebo_matched": "等量安慰剂(内容错)", "noise_matched": "等量随机噪声"}

    res = {}
    per_seq = {}
    for mode in modes:
        inj.mode = mode
        inj.reset_stats()
        gen = torch.Generator().manual_seed(DATA_SEED + 555)
        rare, freq = [], []
        with torch.no_grad():
            for _ in range(N_BATCHES):
                ids, _ = generate_batch(32, SEQ_LEN, spec, generator=gen)
                logits = model(ids)
                lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                     ids[:, 1:].reshape(-1), reduction="none").view(32, -1)
                rk = ids[:, :-1] % 8
                rare.append(lp[rk >= 6]); freq.append(lp[rk < 2])
        per_seq[mode] = (torch.cat(rare), torch.cat(freq))
        res[mode] = {"注入量": inj.summary()}
        print(f"{names[mode]:22s} 注入 {inj.summary()['每位置注入单元数']:.2f} 个单元 / "
              f"范数 {inj.summary()['每位置注入范数']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    inj.mode = "off"

    base_r, base_f = per_seq["off"]
    print(f"\n基线: 稀有词 loss={float(base_r.mean()):.4f} (n={base_r.shape[0]}), "
          f"高频词 loss={float(base_f.mean()):.4f}\n", flush=True)
    for mode in modes[1:]:
        r, f = per_seq[mode]
        d_r = r - base_r
        d_f = f - base_f
        res[mode].update({"rare_delta": float(d_r.mean()), "rare_se": float(d_r.std() / d_r.shape[0] ** 0.5),
                          "freq_delta": float(d_f.mean()), "freq_se": float(d_f.std() / d_f.shape[0] ** 0.5)})
        print(f"{names[mode]:22s} 稀有词 Δ={res[mode]['rare_delta']:+.5f}±{res[mode]['rare_se']:.5f}   "
              f"高频词 Δ={res[mode]['freq_delta']:+.5f}±{res[mode]['freq_se']:.5f}", flush=True)

    res["n_rare"] = int(base_r.shape[0])
    res["base_rare"] = float(base_r.mean())
    with open(OUT, "w") as fo:
        json.dump(res, fo, ensure_ascii=False, indent=1)
    print(f"\n完成 {(time.time()-t0)/60:.1f} 分钟 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
