# -*- coding: utf-8 -*-
"""
用户指正后的补课实验: 不再"事后建档",而是**边训练边登记**。

背景: 之前在真实模型上的做法是拿训练完的 distilgpt2 事后跑一遍文本建档——
(a) 不是原始构想的形态(构想=预训练时就开始标记); (b) 与论文声称的差异点
("训练中免费建成,无需事后profiling")自相矛盾。本实验补上这块证据。

做法: 对 distilgpt2 做**继续预训练**(真实梯度更新,AdamW,在本地语料上,
权重真的变、域内损失真的降),登记器挂在第2/4层,在训练前向里流式登记。
训练结束后,三个档案在**同一个训练后的模型**上各自跑故障报警对比:

  A. 边训边建的档案(本实验主角——构想的忠实形态)
  B. 训练完后事后重建的档案(之前的做法,作为对照锚点)
  C. 训练前旧模型的档案直接硬用(过期测试: 权重变了,旧档案还能用吗)

如果 A ≈ B: "训练中免费建成、无需事后跑"的主张成立,论文的声称被兑现。
如果 A < B: 如实修正——训练中的档案会过期,需要训练尾声做一次刷新。
C 预期变差的程度,反过来量化"档案必须跟着训练走"这件事有多要紧。
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from sentinel import AssemblySentinel  # noqa: E402
from validate_real import Fault, auc, batches_from_corpus, collect  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = [2, 4]
SEQ, BS = 128, 2
N_STEPS = 400
LR = 5e-5
REG_EVERY = 1          # 每步都登记(流式)
N_EVAL = 60            # 评估批数(240 序列)
OUT = os.path.join(HERE, "finetune_registry_results.json")


def main():
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype=torch.float32)

    # ---------- 阶段1: 继续预训练,边训边登记 ----------
    sent_during = AssemblySentinel(model, LAYERS, k_max=2048, tau=0.2)  # 捕获钩子(训练期无故障钩子,顺序无碍)
    from validate_real import batches_from_corpus as bfc  # 签名 (tok, n_batches, seed), 批形状 4x128
    train_batches = bfc(tok, N_STEPS, 999)
    heldout = bfc(tok, 12, 555)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    def domain_loss():
        model.eval()
        tot = []
        with torch.no_grad():
            for ids in heldout:
                tot.append(model(ids, labels=ids).loss)
        model.train()
        return float(torch.stack(tot).mean())

    l0 = domain_loss()
    print(f"继续预训练前, 域内损失={l0:.4f}", flush=True)

    # 显式初始化登记器(AssemblySentinel 的懒初始化只在 build() 里走,这里绕过了它)
    from registry import AssemblyRegistry
    for li in LAYERS:
        sent_during.regs[li] = AssemblyRegistry(3072, 2048, 0.2, 0.05, act_frac=0.5)

    model.train()
    t0 = time.time()
    for step, ids in enumerate(train_batches, 1):
        out = model(ids, labels=ids)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        # 边训边登记: 用刚才训练前向捕获到的激活(真实训练路径,不是额外前向)
        if step % REG_EVERY == 0:
            for li in LAYERS:
                flat = sent_during.acts[li].reshape(-1, sent_during.acts[li].shape[-1])[::4].float()
                sent_during.regs[li].register(sent_during.regs[li].extract_sets(flat), values=flat)
        if step % 50 == 0:
            gs = {li: sent_during.regs[li].n_groups for li in LAYERS}
            print(f"  step {step}/{N_STEPS} loss={float(out.loss):.4f} 组数={gs} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    l1 = domain_loss()
    model.eval()
    print(f"继续预训练后, 域内损失={l1:.4f} (降幅 {l0-l1:+.4f} => 训练是真的)", flush=True)
    torch.save(model.state_dict(), os.path.join(HERE, "distilgpt2_finetuned.pt"))

    # ---------- 阶段2: 三个档案,同一个训练后模型,统一故障对比 ----------
    # 关键顺序教训: 故障钩子必须先于读数用的捕获钩子注册
    sent_during.remove()  # 摘掉训练期捕获钩子,重新按正确顺序布置
    fault2 = Fault(model, 2)
    fault4 = Fault(model, 4)
    sent = AssemblySentinel(model, LAYERS, k_max=2048, tau=0.2)

    # 档案A: 边训边建(把训练期登记器移植进来)
    regA = {li: sent_during.regs[li] for li in LAYERS}
    # 档案B: 训练后事后重建
    print("\n建档案B(训练后事后重建)…", flush=True)
    regB = {}
    from registry import AssemblyRegistry
    for li in LAYERS:
        regB[li] = AssemblyRegistry(3072, 2048, 0.2, 0.05, act_frac=0.5)
    t0 = time.time()
    with torch.no_grad():
        for ids in bfc(tok, 120, 111):
            sent.model(ids)
            for li in LAYERS:
                flat = sent.acts[li].reshape(-1, 3072)[::4].float()
                regB[li].register(regB[li].extract_sets(flat), values=flat)
    print(f"  档案B完成 ({time.time()-t0:.0f}s)", flush=True)
    # 档案C: 训练前旧模型的档案(过期测试)
    old = torch.load(os.path.join(HERE, "registry_distilgpt2.pt"), weights_only=False)
    regC = {}
    for li in LAYERS:
        d = old[f"L{li}"]
        r = AssemblyRegistry(**d["config"])
        for key in ("freq", "act_ema", "active", "count", "created_at"):
            setattr(r, key, d[key])
        regC[li] = r

    ev = bfc(tok, N_EVAL, 333)
    results = {"domain_loss_before": l0, "domain_loss_after": l1,
               "groups_during": {li: regA[li].n_groups for li in LAYERS}}

    def battery(tag, regs, cases, do_loc=False):
        sent.regs = regs
        normal = collect(sent, ev)
        # 覆盖率
        s = sent.readings(ev[0])
        cov = {li: s[li]["coverage"] for li in LAYERS}
        print(f"\n== 档案{tag}: 覆盖率={ {li: round(cov[li],3) for li in LAYERS} } ==", flush=True)
        r = {"coverage": cov}
        for name, wf in cases:
            f = collect(sent, ev, fault4, wf)
            a_r, se_r = auc(f[4]["rho"], normal[4]["rho"])
            a_v, se_v = auc(f[4]["vmis"], normal[4]["vmis"])
            r[name] = {"rho": [a_r, se_r], "vmis": [a_v, se_v]}
            print(f"  {name:10s}: 开关 AUC={a_r:.3f}±{se_r:.3f} | 亮度 AUC={a_v:.3f}±{se_v:.3f}", flush=True)
        if do_loc:
            loc = {}
            for fl, fobj in ((2, fault2), (4, fault4)):
                f = collect(sent, ev, fobj, ("drop", 0.3, None))
                for dl in LAYERS:
                    a_r, _ = auc(f[dl]["rho"], normal[dl]["rho"])
                    loc[f"fault_L{fl}_det_L{dl}"] = a_r
                    print(f"  定位: 故障L{fl}, L{dl}检测器: 开关={a_r:.3f}", flush=True)
            r["localization"] = loc
        return r

    cases_full = [("drop30", ("drop", 0.3, None)), ("stuck30%", ("stuck", 0.0, 0.3)),
                  ("noise1.0x", ("noise", 1.0, None)), ("quant4", ("quant", 4, None))]
    cases_anchor = cases_full[:2]

    results["A_边训边建"] = battery("A(边训边建)", regA, cases_full, do_loc=True)
    results["B_事后重建"] = battery("B(事后重建)", regB, cases_anchor)
    results["C_过期旧档"] = battery("C(训练前旧档)", regC, cases_anchor)

    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=float)
    # 存档案A供打包
    payloadA = {f"L{li}": regA[li].state_dict_plain() for li in LAYERS}
    torch.save(payloadA, os.path.join(HERE, "registry_during_training.pt"))
    print("\n完成。", flush=True)


if __name__ == "__main__":
    main()
