# -*- coding: utf-8 -*-
"""
盲区实验(叠加/多义性的直接后果):
同样杀掉 N 个神经元,一次专挑"是某个组的成员"的,一次专挑"从来不是任何组成员"的。
如果两者对模型输出的伤害相当、但检出率天差地别 => 这套读数存在结构性盲区,
而盲区的存在正是"特征叠加在神经元上、有意义的单位是方向不是神经元"的直接后果。

对照组:
  member    : 只杀成员神经元(在读数视野内)
  nonmember : 只杀非成员神经元(在读数视野外)
  random    : 随机杀(实际部署里故障的样子, 作为参照)
三组杀掉的神经元**数量完全相同**,唯一差别是挑选依据。

输出: 每组的 (模型伤害 loss增量, 逐篇判定的检出率, 平均异常块比例)。
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from sentinel import AssemblySentinel, _find_act_module  # noqa: E402
from eval_at_scale import make_docs, doc_verdict, DOC_TOKENS, CHUNK  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = [2, 4]
TARGET_LAYER = 4
N_DOCS = 120
OUT = os.path.join(HERE, "eval_blindspot_results.json")


class KillUnits:
    """把指定的一组神经元恒置零(卡死),挂在目标层。"""

    def __init__(self, model, layer):
        self.mask = None
        self.h = _find_act_module(model.transformer.h[layer]).register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        if self.mask is None:
            return None
        return out * (~self.mask).float()

    def set(self, mask):
        self.mask = mask

    def off(self):
        self.mask = None


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype=torch.float32)
    model.load_state_dict(torch.load(os.path.join(HERE, "distilgpt2_finetuned.pt"), weights_only=True))
    model.eval()
    killer = KillUnits(model, TARGET_LAYER)        # 故障钩子先注册
    sent = AssemblySentinel(model, LAYERS)
    sent.load(os.path.join(HERE, "registry_ready.pt"))

    mem = sent.regs[TARGET_LAYER].membership       # [K, U]
    is_member = mem.any(0)                          # [U] 该神经元是否属于任何组
    U = is_member.shape[0]
    n_mem, n_non = int(is_member.sum()), int((~is_member).sum())
    print(f"第{TARGET_LAYER}层: 共 {U} 个神经元, 组成员 {n_mem} ({n_mem/U:.0%}), "
          f"从不是成员 {n_non} ({n_non/U:.0%})", flush=True)

    N_KILL = int(0.10 * U)                          # 三组杀掉同样多
    if N_KILL > min(n_mem, n_non):
        N_KILL = min(n_mem, n_non)
    print(f"每组杀掉 {N_KILL} 个神经元(占全层 {N_KILL/U:.0%})", flush=True)

    g = torch.Generator().manual_seed(20260802)
    mem_idx = torch.nonzero(is_member).flatten()
    non_idx = torch.nonzero(~is_member).flatten()
    all_idx = torch.arange(U)

    def mask_from(idx_pool):
        pick = idx_pool[torch.randperm(len(idx_pool), generator=g)[:N_KILL]]
        m = torch.zeros(U, dtype=torch.bool)
        m[pick] = True
        return m

    conditions = {
        "健康": None,
        "杀成员神经元": mask_from(mem_idx),
        "杀非成员神经元": mask_from(non_idx),
        "随机杀": mask_from(all_idx),
    }

    docs = make_docs(tok, os.path.join(HERE, "default_corpus.txt"), N_DOCS, 4242)
    print(f"评测文档: {len(docs)} 篇 × {DOC_TOKENS} token\n", flush=True)

    results = {"n_units": U, "n_member": n_mem, "n_nonmember": n_non, "n_killed": N_KILL,
               "n_docs": len(docs), "conditions": {}}
    base_loss = None

    for name, mask in conditions.items():
        killer.off()
        if mask is not None:
            killer.set(mask)
        # 模型伤害
        tot = []
        with torch.no_grad():
            for ch in docs[:30]:
                tot.append(model(ch, labels=ch).loss)
        loss = float(torch.stack(tot).mean())
        if base_loss is None:
            base_loss = loss
        # 逐篇判定
        counts = {"正常": 0, "局部": 0, "全局": 0}
        fr = {li: [] for li in LAYERS}
        hit_target = 0
        with torch.no_grad():
            for ch in docs:
                fracs, verdict, loc = doc_verdict(sent, ch)
                counts[verdict] += 1
                for li in LAYERS:
                    fr[li].append(fracs[li])
                if verdict == "局部" and loc == TARGET_LAYER:
                    hit_target += 1
        killer.off()
        n = len(docs)
        entry = {"loss": loss, "harm": loss - base_loss,
                 "正常率": counts["正常"] / n, "局部率": counts["局部"] / n,
                 "全局率": counts["全局"] / n, "命中目标层率": hit_target / n,
                 "平均异常块比例": {li: sum(fr[li]) / n for li in LAYERS}}
        results["conditions"][name] = entry
        print(f"{name:14s} 伤害={entry['harm']:+.4f}  检出(报异常){1-entry['正常率']:.0%}  "
              f"指向第{TARGET_LAYER}层 {entry['命中目标层率']:.0%}  "
              f"块比例 L2={entry['平均异常块比例'][2]:.0%} L4={entry['平均异常块比例'][4]:.0%} "
              f"({time.time()-t0:.0f}s)", flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n完成, 用时 {(time.time()-t0)/60:.1f} 分钟。", flush=True)


if __name__ == "__main__":
    main()
