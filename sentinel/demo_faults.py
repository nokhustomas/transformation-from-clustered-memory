# -*- coding: utf-8 -*-
"""
第三步(可选): 眼见为实的故障演示。

用法: python3 demo_faults.py

做什么: 在你电脑上现场重演验证实验的一个缩小版——
  1. 正常文本过一遍模型,体检读数应该"正常";
  2. 人为损伤模型第4层(随机砍掉30%的活跃单元,模拟硬件/压缩损伤),
     同样的文本再过一遍,体检读数应该报"异常"且指向第4层;
  3. 打印前后对比。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from sentinel import AssemblySentinel, _find_act_module  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "distilgpt2"
FAULT_LAYER = 4

SAMPLE = ("Returns the tensor resulting from the elementwise addition of the two "
          "input tensors, broadcasting over their shapes when necessary. If the "
          "dtypes differ, the result follows the standard type promotion rules.")


class DropFault:
    def __init__(self, model, layer, frac=0.3):
        self.frac, self.on = frac, False
        self.h = _find_act_module(model.transformer.h[layer]).register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        if not self.on:
            return None
        shp = out.shape
        flat = out.reshape(-1, shp[-1])
        g = torch.Generator().manual_seed(12345)
        a = flat.clamp_min(0.0)
        mx = a.max(-1, keepdim=True).values
        sets = (a > 0.5 * mx) & (mx > 0)
        m = (torch.rand(flat.shape, generator=g) < self.frac) & sets
        return (flat * (~m).float()).reshape(shp)


def show(tag, rep):
    print(f"\n—— {tag} ——")
    for k, v in rep.items():
        if k.startswith("layer_"):
            print(f"  {k}: 重合度={v['rho']:.3f} 亮度失配={v['value_mismatch']:.3f} "
                  f"可疑度z={v.get('suspicion_z', float('nan')):.1f} "
                  f"{'⚠️ 异常' if v.get('flag') else '正常'}")
    print(f"  结论: {rep.get('verdict', '?')}"
          + (f" (最可疑: 第{rep['most_suspicious_layer']}层)" if rep.get("verdict") == "异常" else ""))


def main():
    reg_path = os.path.join(HERE, f"registry_{MODEL}.pt")
    if not os.path.exists(reg_path):
        print("请先运行 python3 build_registry.py")
        return
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    # 顺序要紧: 故障钩子先注册,哨兵后注册,哨兵才能读到"故障之后"的激活
    fault = DropFault(model, FAULT_LAYER, frac=0.3)
    sent = AssemblySentinel(model, [2, 4])
    sent.load(reg_path)
    ids = tok(SAMPLE, return_tensors="pt").input_ids
    with torch.no_grad():
        fault.on = False
        show("健康模型", sent.health_report(ids))
        fault.on = True
        show(f"第{FAULT_LAYER}层被砍掉30%活跃单元后", sent.health_report(ids))


if __name__ == "__main__":
    main()
