# -*- coding: utf-8 -*-
"""
直接检查注入张量: 治疗组(正确组)与安慰剂组(错位组)各自到底注入了什么。

要验的假设: missing = 组成员集合 M \ 当前激活集合 S。若匹配到的组的核心成员本来
就被当前输入全部点亮(M ⊆ S), 则 missing 恒空 -> 注入张量精确为 0 -> 治疗组是
数学上的恒等变换; 而安慰剂换成另一个组 M', M' ⊄ S, 注入非零。
若成立, 则"治疗组 exact no-op、安慰剂不是"就不是巧合, 而是这个 missing 定义的必然,
实验5(以及同用该定义的 2/4/9)的对照本身不等量, 不能算测过。

输出: 逐位置的 |missing| 计数、注入张量的范数、以及 M ⊆ S 的比例, 治疗/安慰剂并列。
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))

from models import FullAttentionLM  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402
from concept_lm_data import make_concept_spec, generate_batch  # noqa: E402
from run_expt3_stress import EMIT_EXTREME  # noqa: E402
from run_expt2_completion import DIM, LAYERS, HEADS, SEQ_LEN, TRAIN_BATCH, K_MAX, J_NEW, ACT_FRAC, FREQ_LR, DATA_SEED  # noqa: E402

WARM_STEPS = 600      # 与实验5相同: 前600步只登记不注入


def main():
    spec = make_concept_spec(8, 8, emit_probs=EMIT_EXTREME)
    torch.manual_seed(0)
    model = FullAttentionLM(spec["vocab_size"], DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    regs = [AssemblyRegistry(DIM * 4, K_MAX, J_NEW, FREQ_LR, act_frac=ACT_FRAC) for _ in range(LAYERS)]
    acts = {}
    for li in range(LAYERS):
        model.layers[li].ffn[1].register_forward_hook(
            lambda m, i, o, li=li: acts.__setitem__(li, o.detach()))

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(DATA_SEED)
    import torch.nn.functional as F
    print(f"复现实验5的前 {WARM_STEPS} 步(只登记不注入)…", flush=True)
    for step in range(1, WARM_STEPS + 1):
        ids = generate_batch(TRAIN_BATCH, SEQ_LEN, spec, generator=gen)[0]
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        for li in range(LAYERS):
            flat = acts[li].reshape(-1, acts[li].shape[-1])[::8]
            regs[li].register(regs[li].extract_sets(flat), values=flat)

    print(f"登记完成: L0={regs[0].n_groups}组 L1={regs[1].n_groups}组\n", flush=True)

    # 现在按实验5注入分支的逻辑,逐位置算注入张量
    ids = generate_batch(TRAIN_BATCH, SEQ_LEN, spec, generator=gen)[0]
    with torch.no_grad():
        model(ids)

    for li in range(LAYERS):
        reg = regs[li]
        flat = acts[li].reshape(-1, acts[li].shape[-1])
        sets = reg.extract_sets(flat)
        a, rho = reg.assign(sets)
        hit = a >= 0
        act_idx = torch.nonzero(reg.active).flatten()
        perm = torch.arange(reg.K)
        if len(act_idx) > 1:
            perm[act_idx] = act_idx.roll(1)

        print(f"--- 第{li}层 (命中率 {float(hit.float().mean()):.3f}, 平均重合度 {float(rho[hit].mean()):.3f}) ---")
        for tag, k in (("治疗(正确组)", a.clamp_min(0)), ("安慰剂(错位组)", perm[a.clamp_min(0)])):
            mem = reg.membership[k]
            typ = reg.act_ema[k]
            missing = mem & (~sets) & hit.unsqueeze(-1)
            gain = (1.0 * rho.clamp_min(0.0)).unsqueeze(-1)
            inj = typ * missing.float() * gain
            n_missing = missing.float().sum(-1)
            subset = ((mem & hit.unsqueeze(-1)).float().sum(-1) ==
                      ((mem & sets) & hit.unsqueeze(-1)).float().sum(-1))
            print(f"  {tag}: 每位置缺失成员数 均值={float(n_missing[hit].mean()):.2f} "
                  f"最大={float(n_missing[hit].max()):.0f} | "
                  f"注入张量范数 均值={float(inj.norm(dim=-1)[hit].mean()):.5f} "
                  f"全零位置占比={float((inj.norm(dim=-1)[hit] == 0).float().mean()):.1%} | "
                  f"M⊆S 的比例={float(subset[hit].float().mean()):.1%}")
        print()


if __name__ == "__main__":
    main()
