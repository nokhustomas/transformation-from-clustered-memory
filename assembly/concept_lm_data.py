# -*- coding: utf-8 -*-
"""
带隐藏概念类的合成语言。

结构: n_classes 个概念类,每类 members 个"同义词" token。序列由类层面的马尔可夫链
生成——下一个类的分布只取决于当前类;选定类之后,按类内的 zipf 风格词频分布随机
选一个成员 token 发出。

要点:
- "类"是数据的真实生成结构,但从不直接给模型(模型只见 token id)。同类 token 的
  下一 token 分布完全相同(在类粒度上),所以模型要预测得好,最有效的解就是给同类
  token 学出可共享的内部表征——这正是"共激活 unit 组应该对齐概念类"这一假设
  希望利用的压力。
- 类内词频故意做成不均匀(zipf 风格): 高频词提供类结构的主要学习信号,稀有词
  是后续实验(补全是否帮到稀有词)的天然测试对象。
- 转移矩阵故意做成强结构(每个类只有少数几个高概率后继),让"知道当前类"对预测
  有实打实的收益,而不是接近均匀的弱信号。
"""

import torch


def make_concept_spec(n_classes=8, members=8, emit_probs=None):
    V = n_classes * members
    token_class = torch.arange(V) // members  # token id -> 类 id(按块划分,便于人查)

    # 类转移: P(c -> c+1)=0.45, P(c -> c+2)=0.25, P(c -> c)=0.15, 其余类均分 0.15
    T = torch.full((n_classes, n_classes), 0.15 / max(1, n_classes - 3))
    for c in range(n_classes):
        T[c, c] = 0.15
        T[c, (c + 1) % n_classes] = 0.45
        T[c, (c + 2) % n_classes] = 0.25
    T = T / T.sum(dim=-1, keepdim=True)

    if emit_probs is not None:
        # 自定义类内词频(如实验二的"超稀有尾部":最后两个同义词训练中只出现几十
        # 到几百次,制造"学得差、补全才有空间帮到"的对象)
        emit = torch.tensor(emit_probs, dtype=torch.float32)
        assert emit.shape == (members,)
        emit = emit / emit.sum()
    else:
        # 类内 zipf 风格词频(members=8 时): 高频到稀有
        ranks = torch.arange(1, members + 1, dtype=torch.float32)
        emit = (1.0 / ranks)
        emit = emit / emit.sum()  # members=8: [0.368, 0.184, 0.123, 0.092, 0.074, 0.061, 0.053, 0.046]

    return {
        "n_classes": n_classes, "members": members, "vocab_size": V,
        "token_class": token_class, "trans": T, "emit": emit,
    }


def generate_batch(batch_size, seq_len, spec, generator=None):
    """返回 ids [B, T] long, classes [B, T] long(仅评估用,不给模型)。"""
    g = generator
    C, M = spec["n_classes"], spec["members"]
    trans, emit = spec["trans"], spec["emit"]

    classes = torch.zeros(batch_size, seq_len, dtype=torch.long)
    classes[:, 0] = torch.randint(0, C, (batch_size,), generator=g)
    for t in range(1, seq_len):
        probs = trans[classes[:, t - 1]]              # [B, C]
        classes[:, t] = torch.multinomial(probs, 1, generator=g).squeeze(-1)

    member_idx = torch.multinomial(
        emit.unsqueeze(0).expand(batch_size * seq_len, M), 1, generator=g
    ).view(batch_size, seq_len)
    ids = classes * M + member_idx
    return ids, classes


def optimal_loss(spec):
    """贝叶斯最优交叉熵(nats): H(下一类|当前类) + H(类内词频)。
    模型的 loss 逼近这个值即接近学满;仅作参照,不用于训练。"""
    T, emit = spec["trans"], spec["emit"]
    h_trans = -(T * T.clamp_min(1e-12).log()).sum(-1)          # [C] 每类的转移熵
    # 平稳分布: 幂迭代
    pi = torch.full((spec["n_classes"],), 1.0 / spec["n_classes"])
    for _ in range(500):
        pi = pi @ T
    h_emit = -(emit * emit.clamp_min(1e-12).log()).sum()
    return float((pi * h_trans).sum() + h_emit)


if __name__ == "__main__":
    spec = make_concept_spec()
    g = torch.Generator().manual_seed(0)
    ids, classes = generate_batch(4, 32, spec, generator=g)
    print("vocab_size:", spec["vocab_size"])
    print("ids[0]:", ids[0].tolist())
    print("classes[0]:", classes[0].tolist())
    print("类内词频:", [round(float(x), 3) for x in spec["emit"]])
    print("最优 loss (nats):", round(optimal_loss(spec), 4),
          " 均匀猜测 loss:", round(float(torch.log(torch.tensor(64.0))), 4))
