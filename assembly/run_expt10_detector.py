# -*- coding: utf-8 -*-
"""
实验十: 把机制从"修理工"改行当"报警器"——重合度读数能不能可靠区分
正常输入和有问题的输入。

九个实验贯穿的线索: 补全(修理)时灵时不灵,但"当前状态与记忆组的重合度"这个
读数始终精确追踪状态好坏(破坏越重读数越低: 0.90→0.63→0.48→0.39;
词越生僻读数越低: 0.94→0.64→0.57)。检测比修复容易得多——不用改对,只用报对。

设置: 冻结的真实语言字符级模型(实验六) + K=512 大容量登记库(实验八已建逻辑,
这里重建)。每个输入序列产生一个"陌生度分数" = 该序列所有位置重合度的均值
(取负,越高越可疑)。

测试的异常类型(每类 vs 正常文本算区分度 AUC):
  A. 内部故障: 激活被随机砍 10%/30%/50%(模拟硬件/量化损伤)——输入正常,内部坏
  B. 乱序文本: 正常字符按序列内随机重排(字符频率不变,结构没了)
  C. 随机乱码: 词表内均匀随机字符
  D. 换域文本: 大写化(简单分布偏移,最难检出的一档)
样本量: 每类 400 条序列 vs 400 条正常序列(AUC 的标准误 ~1-2%)。

AUC 直白解释: 随机抓一条异常和一条正常,报警器把异常排在前面的概率。
0.5=瞎猜, 0.9+=可用, 0.99=接近完美。
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from models import FullAttentionLM  # noqa: E402
from registry import AssemblyRegistry  # noqa: E402
from run_expt6_realtext import load_corpus, make_batch, DIM, LAYERS, HEADS, SEQ_LEN, DATA_SEED  # noqa: E402

K_BIG, TAU_BIG, AF, FLR = 512, 0.2, 0.5, 0.05
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_SEQ = 400
CORRUPT_SEED = 777000


def auc(pos_scores, neg_scores):
    """正类(异常)分数应更高。返回 AUC 及其近似标准误(Hanley-McNeil)。"""
    p = torch.tensor(pos_scores)
    n = torch.tensor(neg_scores)
    cmp = (p.unsqueeze(1) > n.unsqueeze(0)).float() + 0.5 * (p.unsqueeze(1) == n.unsqueeze(0)).float()
    a = float(cmp.mean())
    n1, n2 = len(p), len(n)
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    se = ((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2)) ** 0.5
    return a, se


def main():
    ids_corpus, stoi, itos = load_corpus()
    V = max(itos.keys()) + 1
    torch.manual_seed(0)
    model = FullAttentionLM(V, DIM, LAYERS, HEADS, max_len=SEQ_LEN)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "expt6_model_seed0.pt")))
    model.eval()

    # 大写映射(换域用)
    upper_map = torch.arange(V)
    for c, i in stoi.items():
        cu = c.upper()
        if cu != c and cu in stoi:
            upper_map[i] = stoi[cu]

    # 重建 K=512 登记库(与实验八相同流程)
    print("重建登记库…", flush=True)
    reg = AssemblyRegistry(DIM * 4, K_BIG, TAU_BIG, FLR, act_frac=AF)
    acts = {}
    hcap = model.layers[1].ffn[1].register_forward_hook(lambda m, i, o: acts.__setitem__(1, o.detach()))
    g = torch.Generator().manual_seed(DATA_SEED + 8888)
    with torch.no_grad():
        for _ in range(150):
            ids = make_batch(ids_corpus, 16, SEQ_LEN, g)
            model(ids)
            flat = acts[1].reshape(-1, DIM * 4)[::8]
            reg.register(reg.extract_sets(flat), values=flat)
    print(f"  组数={reg.n_groups}", flush=True)

    # 内部故障注入开关
    fault = {"drop": 0.0, "gen_seed": 0}

    def fault_hook(_m, _i, out):
        if fault["drop"] <= 0:
            return None
        gg = torch.Generator().manual_seed(CORRUPT_SEED + fault["gen_seed"])
        B, T, U = out.shape
        flat = out.reshape(-1, U)
        sets_pre = reg.extract_sets(flat)
        drop = (torch.rand(flat.shape, generator=gg) < fault["drop"]) & sets_pre
        return (flat * (~drop).float()).view(B, T, U)
    hfault = model.layers[1].ffn[1].register_forward_hook(fault_hook)
    # 注意 hook 顺序: hcap 先注册,捕获的是故障前的激活——报警器要读故障后的,
    # 所以评分时用 fault 之后的第二个捕获钩子
    acts2 = {}
    hcap2 = model.layers[1].ffn[1].register_forward_hook(lambda m, i, o: acts2.__setitem__(1, o.detach()))

    def strangeness(ids_batch, drop=0.0, seed_off=0):
        """每条序列一个分数: 位置平均重合度取负(越高越可疑)。"""
        fault["drop"] = drop
        fault["gen_seed"] = seed_off
        with torch.no_grad():
            model(ids_batch)
        fault["drop"] = 0.0
        flat = acts2[1].reshape(-1, DIM * 4)
        sets = reg.extract_sets(flat)
        _, rho = reg.assign(sets)
        rho = rho.clamp_min(0.0).view(ids_batch.shape[0], -1)
        return (-rho.mean(dim=1)).tolist()

    # 正常参照
    eval_g = torch.Generator().manual_seed(DATA_SEED + 424242)
    normal_scores = []
    normal_batches = []
    for i in range(N_SEQ // 16):
        ids = make_batch(ids_corpus, 16, SEQ_LEN, eval_g)
        normal_batches.append(ids)
        normal_scores += strangeness(ids)

    results = {}
    perm_g = torch.Generator().manual_seed(999)

    # A. 内部故障
    for drop in (0.1, 0.3, 0.5):
        scores = []
        for i, ids in enumerate(normal_batches):
            scores += strangeness(ids, drop=drop, seed_off=i + 1)
        a, se = auc(scores, normal_scores)
        results[f"A_内部故障_drop{drop}"] = {"auc": a, "se": se}
        print(f"A 内部故障 drop={drop}: AUC={a:.3f}±{se:.3f}", flush=True)

    # B. 乱序文本
    scores = []
    for ids in normal_batches:
        shuf = torch.stack([r[torch.randperm(SEQ_LEN, generator=perm_g)] for r in ids])
        scores += strangeness(shuf)
    a, se = auc(scores, normal_scores)
    results["B_乱序文本"] = {"auc": a, "se": se}
    print(f"B 乱序文本: AUC={a:.3f}±{se:.3f}", flush=True)

    # C. 随机乱码
    scores = []
    for _ in range(N_SEQ // 16):
        junk = torch.randint(1, V, (16, SEQ_LEN), generator=perm_g)
        scores += strangeness(junk)
    a, se = auc(scores, normal_scores)
    results["C_随机乱码"] = {"auc": a, "se": se}
    print(f"C 随机乱码: AUC={a:.3f}±{se:.3f}", flush=True)

    # D. 换域(大写化)
    scores = []
    for ids in normal_batches:
        scores += strangeness(upper_map[ids])
    a, se = auc(scores, normal_scores)
    results["D_大写化换域"] = {"auc": a, "se": se}
    print(f"D 大写化换域: AUC={a:.3f}±{se:.3f}", flush=True)

    hfault.remove(); hcap.remove(); hcap2.remove()
    with open(os.path.join(OUT_DIR, "expt10_detector.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("完成。", flush=True)


if __name__ == "__main__":
    main()
