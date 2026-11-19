# -*- coding: utf-8 -*-
"""
第二步: 体检(想什么时候跑就什么时候跑)。

用法:
  python3 check_health.py --text "Any piece of English text to run through."
  python3 check_health.py --file some_document.txt
  python3 check_health.py --text "..." --model distilgpt2

做什么: 让模型读这段文本,输出每一层的两个健康读数和结论:
  重合度(rho):   这一层的内部状态和"正常时的样子"重合多少(偏低=可疑)
  亮度失配:      状态里该亮的单元亮得典型不典型(偏高=可疑)
读数会对照建库时记下的正常范围换算成"可疑度"(z 分数),超出正常范围
(重合度低于正常5%分位 或 失配高于正常95%分位)就标记异常,并指出最可疑的层。

注意(如实): 这套读数检测的是"模型内部状态异常"(硬件损伤/压缩失真/权重被改),
对"输入文本本身很怪"也会有反应但那不是它的强项——判断输入怪不怪,
用模型自己的困惑度就够了。
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from sentinel import AssemblySentinel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--text", default=None)
    ap.add_argument("--file", default=None)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--registry", default=None)
    args = ap.parse_args()

    text = args.text or (open(args.file, encoding="utf-8").read() if args.file else None)
    if not text:
        print("请用 --text 或 --file 提供文本"); return

    reg_path = args.registry or os.path.join(HERE, f"registry_{args.model.replace('/', '_')}.pt")
    if not os.path.exists(reg_path):
        print(f"找不到登记库 {reg_path}, 请先运行 build_registry.py"); return

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    sent = AssemblySentinel(model, args.layers)
    sent.load(reg_path)

    ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids
    with torch.no_grad():
        report = sent.health_report(ids)

    print(json.dumps(report, ensure_ascii=False, indent=1))
    if "verdict" in report:
        print(f"\n结论: {report['verdict']}", end="")
        if report["verdict"] == "异常":
            print(f"; 最可疑的层: 第 {report['most_suspicious_layer']} 层")
        else:
            print()


if __name__ == "__main__":
    main()
