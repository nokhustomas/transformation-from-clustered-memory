# -*- coding: utf-8 -*-
"""
最省事的试用入口: 用"继续预训练后的模型 + 边训边建的档案"给任意文本做体检,
并可现场制造一个故障看它报警。

首次运行会自动做一次校准(记录正常读数范围)并缓存,之后很快。

用法:
  python3 try_it.py --text "Some English text."          # 体检一段文本
  python3 try_it.py --file 你的文件.txt                   # 体检一个文件
  python3 try_it.py --text "..." --break-layer 4         # 同一文本, 人为砍伤第4层再体检(对比看)
  python3 try_it.py --base                               # 用原始未继续训练的模型+事后档案
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from sentinel import AssemblySentinel, _find_act_module  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = [2, 4]


class DropFault:
    """人为损伤: 把某层 30% 的活跃单元清零(模拟硬件故障/过度剪枝)。"""

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


def corpus_batches(tok, n, seed, seq=128, bs=4):
    text = open(os.path.join(HERE, "default_corpus.txt"), encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        st = torch.randint(0, ids.shape[0] - seq - 1, (bs,), generator=g)
        out.append(torch.stack([ids[s:s + seq] for s in st]))
    return out


def show(tag, rep):
    print(f"\n—— {tag} ——")
    for li in LAYERS:
        v = rep[f"layer_{li}"]
        mark = "⚠️  异常" if v.get("flag") else "正常"
        print(f"  第{li}层: 重合度={v['rho']:.3f}  亮度失配={v['value_mismatch']:.3f}  "
              f"异常块比例={v.get('outlier_fraction', 0):.0%}  {mark}")
    verdict = rep.get("verdict", "?")
    line = f"  → 结论: {verdict}"
    if verdict == "异常":
        line += f"（最可疑: 第{rep['most_suspicious_layer']}层）\n     类型: {rep.get('anomaly_type', '?')}"
    if not rep.get("reliable"):
        line += f"   [注意: 只有 {rep['n_chunks']} 个文本块, 样本太少, 结论仅供参考]"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None)
    ap.add_argument("--file", default=None)
    ap.add_argument("--break-layer", type=int, default=None,
                    help="人为砍伤这一层(2 或 4),对比健康/损伤两种读数")
    ap.add_argument("--base", action="store_true", help="用原始模型+事后档案,而不是继续训练版")
    args = ap.parse_args()

    default_text = open(os.path.join(HERE, "default_corpus.txt"), encoding="utf-8").read()[40000:56000]
    text = args.text or (open(args.file, encoding="utf-8").read()[:16000] if args.file else default_text)

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype=torch.float32)
    if args.base:
        reg_src, cache = os.path.join(HERE, "registry_distilgpt2.pt"), None
        tag = "原始 distilgpt2 + 事后档案"
    else:
        model.load_state_dict(torch.load(os.path.join(HERE, "distilgpt2_finetuned.pt"),
                                          weights_only=True))
        reg_src = os.path.join(HERE, "registry_during_training.pt")
        cache = os.path.join(HERE, "registry_ready.pt")
        tag = "继续预训练后的模型 + 边训边建的档案"
    model.eval()
    print(f"用的是: {tag}")

    fault = DropFault(model, args.break_layer) if args.break_layer else None
    sent = AssemblySentinel(model, LAYERS)

    if cache and os.path.exists(cache):
        sent.load(cache)
    else:
        sent.load(reg_src)
        print("首次运行: 按文本长度分档校准正常范围(约4分钟,只做一次)…", flush=True)
        sent.normal_stats = {}
        with torch.no_grad():
            for L in (32, 64, 128, 256):
                sent.calibrate(corpus_batches(tok, 40, 222 + L, seq=L), length=L)
        if cache:
            sent.save(cache)

    ids = tok(text, return_tensors="pt", truncation=True, max_length=4096).input_ids
    print(f"\n输入文本({ids.shape[1]} 个 token, 切成 {max(1, ids.shape[1]//128)} 块): "
          f"{text[:80].strip()}…")

    with torch.no_grad():
        if fault:
            fault.on = False
            show("模型健康时", sent.diagnose(ids))
            fault.on = True
            show(f"人为砍掉第{args.break_layer}层 30% 活跃单元后", sent.diagnose(ids))
        else:
            show("体检结果", sent.diagnose(ids))


if __name__ == "__main__":
    main()
