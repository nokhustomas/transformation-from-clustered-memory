# -*- coding: utf-8 -*-
"""
权重调节版(用户最初描述的形态: "引入 a 的权重"),对照此前只测过的"贴常数"版。

两种写入方式的本质区别:
  贴常数(此前唯一测过的, 已证失败):  h <- h + alpha*rho*典型激活值
      注入的是记忆里存的固定向量,与当前输入无关。叠加之下这是一团混合物。
  权重调节(本次, 从未测过):          h <- h * (1 + alpha*rho)   仅作用于命中组的成员单元
      等价于按重合度临时放大这组神经元在 W_in/W_out 上的通路增益。
      模型对该输入的计算被调节了,但算出来的仍是这个输入自己的东西。

三个对照(全部配对, 同一批文本):
  off       不动
  gain      正确组的成员单元被调节
  placebo   命中后改用另一个组(固定置换)的成员单元 —— 强度同量级、内容错位
另外扫 alpha, 因为增益调节的强度是新的自由度(贴常数版的 alpha 已知在 0.5/1.0 均无效)。

指标: 模型在留出文本上的预测损失(越低越好)。若 gain 显著低于 off 而 placebo 更差,
说明"按内容相似度放大对应通路"这件事本身有正向作用。
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from sentinel import AssemblySentinel, _find_act_module  # noqa: E402
from eval_at_scale import make_docs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = [2, 4]
N_DOCS = 60
OUT = os.path.join(HERE, "eval_gain_results.json")


class Writer:
    """挂在某层激活上,按 mode 改变该层的计算。"""

    def __init__(self, model, layer, reg):
        self.reg = reg
        self.mode = "off"          # off | gain | gain_placebo | add | add_placebo
        self.alpha = 0.0
        act_idx = torch.nonzero(reg.active).flatten()
        perm = torch.arange(reg.K)
        if len(act_idx) > 1:
            perm[act_idx] = act_idx.roll(1)
        self.perm = perm
        self.h = _find_act_module(model.transformer.h[layer]).register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        if self.mode == "off" or self.alpha == 0:
            return None
        shp = out.shape
        flat = out.reshape(-1, shp[-1]).float()
        sets = self.reg.extract_sets(flat)
        a, rho = self.reg.assign(sets)
        hit = a >= 0
        if not bool(hit.any()):
            return None
        k = a.clamp_min(0)
        if self.mode.endswith("placebo"):
            k = self.perm[k]
        mem = self.reg.membership[k] & hit.unsqueeze(-1)
        g = (self.alpha * rho.clamp_min(0.0)).unsqueeze(-1)
        if self.mode.startswith("gain"):
            # 权重调节: 放大命中组成员单元的通路增益(内容仍由当前输入算出)
            out_new = flat * (1.0 + g * mem.float())
        else:
            # 贴常数(对照): 把记忆里的典型值加到"该亮而未亮"的成员单元上
            missing = mem & (~sets)
            out_new = flat + self.reg.act_ema[k] * missing.float() * g
        return out_new.reshape(shp).to(out.dtype)

    def set(self, mode, alpha):
        self.mode, self.alpha = mode, alpha

    def off(self):
        self.set("off", 0.0)


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype=torch.float32)
    model.load_state_dict(torch.load(os.path.join(HERE, "distilgpt2_finetuned.pt"), weights_only=True))
    model.eval()
    sent = AssemblySentinel(model, LAYERS)
    sent.load(os.path.join(HERE, "registry_ready.pt"))
    sent.remove()          # 只借它的登记库,不需要捕获钩子

    writers = {li: Writer(model, li, sent.regs[li]) for li in LAYERS}
    docs = make_docs(tok, os.path.join(HERE, "default_corpus.txt"), N_DOCS, 5150)
    print(f"评测: {len(docs)} 篇 × 2048 token", flush=True)

    def loss_now():
        tot = []
        with torch.no_grad():
            for ch in docs:
                tot.append(model(ch, labels=ch).loss)
        t = torch.stack(tot)
        return float(t.mean()), float(t.std() / len(t) ** 0.5)

    for w in writers.values():
        w.off()
    base, base_se = loss_now()
    print(f"不动(基线): loss={base:.4f}±{base_se:.4f}\n", flush=True)

    results = {"base": base, "base_se": base_se, "runs": {}}
    plan = []
    for alpha in (0.25, 0.5, 1.0, 2.0):
        plan.append((f"权重调节 L4 α={alpha}", {4: ("gain", alpha)}))
    plan.append(("权重调节 L4 α=1.0 [注错组]", {4: ("gain_placebo", 1.0)}))
    for alpha in (0.5, 1.0):
        plan.append((f"权重调节 L2+L4 α={alpha}", {2: ("gain", alpha), 4: ("gain", alpha)}))
    plan.append(("贴常数 L4 α=1.0 [旧方式复核]", {4: ("add", 1.0)}))

    for name, cfg in plan:
        for w in writers.values():
            w.off()
        for li, (mode, alpha) in cfg.items():
            writers[li].set(mode, alpha)
        l, se = loss_now()
        d = l - base
        results["runs"][name] = {"loss": l, "se": se, "delta": d}
        flag = "↓好转" if d < -2 * se else ("↑变差" if d > 2 * se else "≈无变化")
        print(f"{name:30s} loss={l:.4f}  Δ={d:+.4f}  {flag}  ({time.time()-t0:.0f}s)", flush=True)

    for w in writers.values():
        w.off()
    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n完成, {(time.time()-t0)/60:.1f} 分钟。", flush=True)


if __name__ == "__main__":
    main()
