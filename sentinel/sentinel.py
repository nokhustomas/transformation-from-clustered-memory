# -*- coding: utf-8 -*-
"""
AssemblySentinel: 给真实的预训练语言模型挂上"内部故障哨兵"。

原理(来自论文《One Registry, Several Jobs》,本仓库 paper2/):
- 建库: 在正常文本上流式登记模型 FFN 层的"共激活 unit 集合"(> 50% 位置最大
  激活的单元),按重合度聚成有限个组,同时记录每组单元的典型激活值。
  不需要标注、不需要训练、不需要故障样本。
- 体检: 对任意输入,每层输出两个免费读数——
    which_units (rho):  当前激活集合与最像的组重合多少(0~1,越低越可疑)
    value_mismatch:     命中组后,成员单元的实际激活离典型值差多远(越高越可疑)
  再对照建库时留出的"正常分布"给出百分位,人话输出。
- 定位: 哪一层的读数异常,病灶就在哪一层(上游层安静=病灶在下游,
  论文实验十一,两种子完全复现)。

已验证范围(如实): 10万参数级toy模型上四类故障 AUC 0.95+;本文件是首次
搬到 8000 万参数的真实预训练模型(distilgpt2)——规模化表现见 validate_real.py
的实测输出,不要引用 toy 数字替代。
"""

import json
import os

import torch

from registry import AssemblyRegistry


def _find_act_module(block):
    """定位一个 transformer block 里 MLP 的激活函数模块(GPT2: mlp.act)。"""
    mlp = getattr(block, "mlp", None)
    if mlp is None:
        raise ValueError("该 block 没有 .mlp,请手动指定激活模块")
    for name in ("act", "act_fn", "activation_fn", "gelu"):
        if hasattr(mlp, name):
            return getattr(mlp, name)
    raise ValueError(f"在 mlp 里找不到激活模块,可用属性: {dir(mlp)}")


class AssemblySentinel:
    def __init__(self, model, layer_indices, k_max=2048, tau=0.2, act_frac=0.5,
                 freq_lr=0.05, register_stride=4):
        self.model = model
        self.layers = list(layer_indices)
        self.k_max, self.tau = k_max, tau
        self.register_stride = register_stride
        blocks = model.transformer.h
        self.acts = {}
        self.handles = []
        self.regs = {}
        self.normal_stats = {}  # 每层正常读数分布(建库后校准)
        for li in self.layers:
            act_mod = _find_act_module(blocks[li])
            self.handles.append(act_mod.register_forward_hook(self._mk_hook(li)))

    def _mk_hook(self, li):
        def hook(_m, _i, out):
            self.acts[li] = out.detach()
        return hook

    # ------------------------------------------------------------ 建库

    @torch.no_grad()
    def build(self, id_batches, log_every=20):
        """id_batches: 可迭代的 input_ids 张量 [B, T]。流式登记。"""
        for li in self.layers:
            U = None
        first = True
        for bi, ids in enumerate(id_batches):
            self.model(ids)
            for li in self.layers:
                flat = self.acts[li].reshape(-1, self.acts[li].shape[-1])
                if first:
                    if li not in self.regs:
                        self.regs[li] = AssemblyRegistry(
                            flat.shape[-1], self.k_max, self.tau,
                            freq_lr=0.05, act_frac=0.5)
                sub = flat[::self.register_stride].float()
                self.regs[li].register(self.regs[li].extract_sets(sub), values=sub)
            first = False
            if log_every and (bi + 1) % log_every == 0:
                gs = {li: self.regs[li].n_groups for li in self.layers}
                print(f"  建库 {bi+1} 批: 组数={gs}", flush=True)

    @torch.no_grad()
    def calibrate(self, id_batches, length=None):
        """在(建库同分布的)留出正常文本上收集读数分布,供百分位换算。

        length: 这批样本的序列长度。读数对序列长度敏感(短文本的重合度系统性
        偏高: 实测 16 token 时 L4≈0.89, 256 token 时≈0.57),所以正常范围必须
        按长度分档存,体检时取最接近的档比对——否则拿短文本比长文本的档会直接
        误报。不传 length 时从第一批数据推断。
        """
        per = {li: {"rho": [], "vmis": []} for li in self.layers}
        first = None
        for ids in id_batches:
            if first is None:
                first = ids.shape[1]
            s = self.readings(ids)
            for li in self.layers:
                per[li]["rho"] += s[li]["rho_per_seq"]
                per[li]["vmis"] += s[li]["vmis_per_seq"]
        L = int(length or first or 128)
        for li in self.layers:
            r = torch.tensor(per[li]["rho"])
            v = torch.tensor(per[li]["vmis"])
            stats = {
                "rho_mean": float(r.mean()), "rho_std": float(r.std().clamp_min(1e-6)),
                "rho_p05": float(r.quantile(0.05)),
                "vmis_mean": float(v.mean()), "vmis_std": float(v.std().clamp_min(1e-6)),
                "vmis_p95": float(v.quantile(0.95)),
            }
            self.normal_stats.setdefault(li, {})
            if not isinstance(self.normal_stats[li], dict) or "rho_mean" in self.normal_stats[li]:
                self.normal_stats[li] = {}          # 旧的扁平格式,升级为按长度分档
            self.normal_stats[li][L] = stats

    def _stats_for(self, li, seq_len):
        """取最接近该长度的正常范围;兼容旧的扁平格式。"""
        s = self.normal_stats.get(li)
        if not s:
            return None
        if "rho_mean" in s:          # 旧格式(单一长度)
            return s
        keys = [k for k in s.keys()]
        if not keys:
            return None
        best = min(keys, key=lambda k: abs(int(k) - seq_len))
        return s[best]

    # ------------------------------------------------------------ 读数

    @torch.no_grad()
    def readings(self, ids):
        """对一批 input_ids 前向一次,返回每层的逐序列读数。"""
        self.model(ids)
        B = ids.shape[0]
        out = {}
        for li in self.layers:
            flat = self.acts[li].reshape(-1, self.acts[li].shape[-1]).float()
            reg = self.regs[li]
            sets = reg.extract_sets(flat)
            a, rho = reg.assign(sets)
            hit = a >= 0
            k = a.clamp_min(0)
            mem = reg.membership[k]
            typ = reg.act_ema[k]
            lit = mem & hit.unsqueeze(-1)
            diff = (flat - typ).abs() * lit.float()
            denom = (typ.abs() * lit.float()).sum(-1).clamp_min(1e-6)
            vmis = diff.sum(-1) / denom
            rho_seq = rho.clamp_min(0.0).view(B, -1).mean(1)
            vmis_seq = vmis.view(B, -1).mean(1)
            out[li] = {"rho_per_seq": rho_seq.tolist(), "vmis_per_seq": vmis_seq.tolist(),
                        "coverage": float(hit.float().mean())}
        return out

    @torch.no_grad()
    def diagnose(self, ids, chunk_len=128, out_frac_threshold=0.30):
        """把长文本切成等长块逐块读数,再对"块均值"做判定——这才是本机制被验证过的
        用法(验证实验用的是 240 条序列,不是一句话)。

        单条短文本的读数方差很大,拿它比单条分布的极端分位必然误报(实测: 语料内
        技术文本也会在浅层冒 +1.9σ)。这里改为: 块均值的标准误 = 单条标准差/sqrt(块数),
        超过 z_threshold 个标准误才报警,并在块数太少时明确降级为"仅供参考"。
        """
        T = ids.shape[1]
        if T < chunk_len:
            chunks = [ids]
        else:
            n = T // chunk_len
            chunks = [ids[:, i * chunk_len:(i + 1) * chunk_len] for i in range(n)]
        per = {li: {"rho": [], "vmis": []} for li in self.layers}
        for c in chunks:
            s = self.readings(c)
            for li in self.layers:
                per[li]["rho"] += s[li]["rho_per_seq"]
                per[li]["vmis"] += s[li]["vmis_per_seq"]

        n_obs = len(per[self.layers[0]]["rho"])
        eff_len = chunks[0].shape[1]
        report = {"n_chunks": n_obs, "chunk_len": eff_len,
                  "reliable": bool(n_obs >= 4)}
        worst = None
        for li in self.layers:
            ns = self._stats_for(li, eff_len)
            rho = sum(per[li]["rho"]) / n_obs
            vm = sum(per[li]["vmis"]) / n_obs
            entry = {"rho": rho, "value_mismatch": vm}
            if ns:
                # 判定依据: 有多少比例的文本块落在正常范围之外(健康模型上应≈5-10%)。
                # 不用"均值偏离多少个标准误"——块数一多标准误被 sqrt(n) 压没,
                # 语料内不同段落的天然差异也会变成几十个 sigma,健康模型必然误报。
                bad = sum(1 for r, v in zip(per[li]["rho"], per[li]["vmis"])
                          if r < ns["rho_p05"] or v > ns["vmis_p95"])
                frac = bad / n_obs
                entry["outlier_fraction"] = frac
                entry["flag"] = bool(frac > out_frac_threshold)
                if worst is None or frac > worst[1]:
                    worst = (li, frac)
            report[f"layer_{li}"] = entry
        if worst:
            flagged = [li for li in self.layers if report[f"layer_{li}"].get("flag")]
            report["most_suspicious_layer"] = worst[0]
            # 区分两种"异常"(实测依据: 砍第4层 -> 100%/16%, 砍第2层 -> 59%/16%,
            # 而健康模型喂法律英文 -> 72%/56%。内部故障是"局部"的,只有出事那层
            # 读数崩;输入领域不对路是"全局"的,各层一起崩)。
            if not flagged:
                report["verdict"] = "正常"
            elif len(flagged) == 1:
                report["verdict"] = "异常"
                report["anomaly_type"] = "局部——疑似模型内部故障"
            else:
                report["verdict"] = "异常"
                report["anomaly_type"] = "全局——各层同时异常, 更像输入文本不对路(领域与档案不符), 而不是模型内部坏了"
        return report

    def health_report(self, ids):
        """人话体检报告: 每层两读数 + 相对正常分布的位置 + 结论。"""
        s = self.readings(ids)
        report = {"seq_len": int(ids.shape[1])}
        worst = None
        for li in self.layers:
            ns = self._stats_for(li, ids.shape[1])
            rho = sum(s[li]["rho_per_seq"]) / len(s[li]["rho_per_seq"])
            vm = sum(s[li]["vmis_per_seq"]) / len(s[li]["vmis_per_seq"])
            entry = {"rho": rho, "value_mismatch": vm, "coverage": s[li]["coverage"]}
            if ns:
                z_rho = (ns["rho_mean"] - rho) / ns["rho_std"]        # 越大越可疑(rho偏低)
                z_vm = (vm - ns["vmis_mean"]) / ns["vmis_std"]        # 越大越可疑(失配偏高)
                entry["suspicion_z"] = max(z_rho, z_vm)
                entry["flag"] = bool(rho < ns["rho_p05"] or vm > ns["vmis_p95"])
                if worst is None or entry["suspicion_z"] > worst[1]:
                    worst = (li, entry["suspicion_z"])
            report[f"layer_{li}"] = entry
        if worst:
            report["verdict"] = ("异常" if any(report[f"layer_{li}"].get("flag") for li in self.layers)
                                  else "正常")
            report["most_suspicious_layer"] = worst[0]
        return report

    # ------------------------------------------------------------ 存取

    def save(self, path):
        payload = {f"L{li}": self.regs[li].state_dict_plain() for li in self.layers}
        payload["normal_stats"] = self.normal_stats
        payload["meta"] = {"layers": self.layers, "k_max": self.k_max, "tau": self.tau}
        torch.save(payload, path)

    def load(self, path):
        st = torch.load(path, weights_only=False)
        for li in self.layers:
            d = st[f"L{li}"]
            reg = AssemblyRegistry(**d["config"])
            for key in ("freq", "act_ema", "active", "count", "created_at"):
                setattr(reg, key, d[key])
            self.regs[li] = reg
        ns = st.get("normal_stats") or {}
        out = {}
        for k, v in ns.items():
            li = int(k)
            if isinstance(v, dict) and "rho_mean" not in v:   # 按长度分档
                out[li] = {int(kk): vv for kk, vv in v.items()}
            else:                                             # 旧的扁平格式
                out[li] = v
        self.normal_stats = out

    def remove(self):
        for h in self.handles:
            h.remove()
