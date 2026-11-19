# -*- coding: utf-8 -*-
"""一次跑完六种情形,输出一张对照表。用的是继续预训练后的模型 + 边训边建的档案。"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from sentinel import AssemblySentinel, _find_act_module  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = [2, 4]


class Fault:
    def __init__(self, model, layer):
        self.kind, self.param = "none", 0.0
        self.stuck = None
        self.h = _find_act_module(model.transformer.h[layer]).register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        if self.kind == "none":
            return None
        shp = out.shape
        flat = out.reshape(-1, shp[-1])
        g = torch.Generator().manual_seed(12345)
        if self.kind == "drop":
            a = flat.clamp_min(0.0)
            mx = a.max(-1, keepdim=True).values
            sets = (a > 0.5 * mx) & (mx > 0)
            flat = flat * (~((torch.rand(flat.shape, generator=g) < self.param) & sets)).float()
        elif self.kind == "stuck":
            if self.stuck is None:
                gm = torch.Generator().manual_seed(7)
                self.stuck = torch.rand(shp[-1], generator=gm) < self.param
            flat = flat * (~self.stuck).float()
        elif self.kind == "quant":
            lo, hi = float(flat.min()), float(flat.max())
            step = (hi - lo) / max(1, int(self.param) - 1)
            flat = ((flat - lo) / step).round() * step + lo
        elif self.kind == "noise":
            flat = flat + torch.randn(flat.shape, generator=g) * flat.std() * self.param
        else:
            raise ValueError(f"未实现的故障类型: {self.kind}")   # 别再静默地什么都不做
        return flat.reshape(shp)

    def set(self, kind, param):
        self.kind, self.param, self.stuck = kind, param, None

    def off(self):
        self.set("none", 0.0)


def main():
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype=torch.float32)
    model.load_state_dict(torch.load(os.path.join(HERE, "distilgpt2_finetuned.pt"), weights_only=True))
    model.eval()

    f2, f4 = Fault(model, 2), Fault(model, 4)   # 故障钩子先注册
    sent = AssemblySentinel(model, LAYERS)
    sent.load(os.path.join(HERE, "registry_ready.pt"))

    corpus = open(os.path.join(HERE, "default_corpus.txt"), encoding="utf-8").read()
    ids_domain = tok(corpus[40000:56000], return_tensors="pt", truncation=True,
                     max_length=4096).input_ids
    legal = open("/tmp/legal.txt", encoding="utf-8").read() if os.path.exists("/tmp/legal.txt") else corpus
    ids_other = tok(legal, return_tensors="pt", truncation=True, max_length=4096).input_ids

    rows = []

    def run(name, ids, setup):
        f2.off(); f4.off()
        setup()
        with torch.no_grad():
            r = sent.diagnose(ids)
        f2.off(); f4.off()
        rows.append((name, r))

    run("① 模型健康 + 语料内文本", ids_domain, lambda: None)
    run("② 第4层 砍掉30%活跃单元", ids_domain, lambda: f4.set("drop", 0.3))
    run("③ 第2层 砍掉30%活跃单元", ids_domain, lambda: f2.set("drop", 0.3))
    run("④ 第4层 10%单元永久卡死", ids_domain, lambda: f4.set("stuck", 0.10))
    run("⑤ 第4层 压成4级(量化失真)", ids_domain, lambda: f4.set("quant", 4))
    run("⑥ 模型健康 + 法律英文", ids_other, lambda: None)

    print(f"\n{'情形':<26}{'第2层':>8}{'第4层':>8}   结论")
    print("-" * 78)
    for name, r in rows:
        a = r["layer_2"]["outlier_fraction"]
        b = r["layer_4"]["outlier_fraction"]
        v = r["verdict"]
        t = r.get("anomaly_type", "")
        tag = "正常" if v == "正常" else ("异常→第%d层" % r["most_suspicious_layer"])
        extra = ""
        if "局部" in t:
            extra = "  内部故障"
        elif "全局" in t:
            extra = "  输入领域不符(非故障)"
        print(f"{name:<26}{a:>7.0%}{b:>8.0%}   {tag}{extra}")
    print("-" * 78)
    print("（数字=文本块里读数落在正常范围外的比例；健康基线约16%，超过30%报警）")


if __name__ == "__main__":
    main()
