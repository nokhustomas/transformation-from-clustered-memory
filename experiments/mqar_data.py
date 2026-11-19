# -*- coding: utf-8 -*-
"""
MQAR 风格(Multi-Query Associative Recall)合成长程记忆任务的数据生成。

任务设计参考了效率 Transformer / 状态空间模型文献中用于诊断"长程关联召回"能力
的合成任务范式(如 Zoology / Based 等工作使用的 associative recall 任务),
具体到本实验的构造如下:

- 序列前 `insert_region_frac` 比例的区域内,插入 num_pairs 个互不相邻、且不跨
  chunk 边界的 (key, value) 二元组(每对占用相邻两个位置)。
- 序列末尾是一个"查询区"(query block),按随机顺序把 num_pairs 个 key 依次以
  (QUERY, key, value) 三元组形式给出——之所以显式写出 value,是为了配合标准的
  "整体右移一位"语言模型目标(teacher forcing):模型在看到 QUERY 和 key 之后
  (但因果掩码下看不到紧随其后的 value),要预测的下一个 token 就是 value。
- 训练/评测时只在这 num_pairs 个"看到 key 之后预测 value"的位置上计算损失和
  准确率,其余位置(填充用的随机 filler token)不参与——这是该类合成诊断任务的
  标准做法,直接衡量"模型能否把很久以前见过的 key 对应的 value 检索回来"。
"""

import torch


def make_vocab(key_vocab=32, value_vocab=32, filler_vocab=32):
    PAD, QUERY = 0, 1
    filler_start = 2
    key_start = filler_start + filler_vocab
    value_start = key_start + key_vocab
    vocab_size = value_start + value_vocab
    return dict(
        PAD=PAD, QUERY=QUERY,
        filler_start=filler_start, filler_vocab=filler_vocab,
        key_start=key_start, key_vocab=key_vocab,
        value_start=value_start, value_vocab=value_vocab,
        vocab_size=vocab_size,
    )


def generate_batch(batch_size, seq_len, chunk_size, num_pairs, vocab,
                    insert_region_frac=0.6, generator=None):
    """
    返回:
      input_ids:   [B, seq_len]       long
      eval_pos:    [B, num_pairs]     long, 每个查询"看到 key 之后"的位置下标
      eval_target: [B, num_pairs]     long, 该位置应预测的 value token id
    """
    g = generator
    query_block_len = 3 * num_pairs
    query_block_start = seq_len - query_block_len
    insert_region_end = int(seq_len * insert_region_frac)
    assert query_block_start > insert_region_end + 2, "插入区域与查询区域重叠，请调整参数"

    input_ids = torch.randint(
        vocab["filler_start"], vocab["filler_start"] + vocab["filler_vocab"],
        (batch_size, seq_len), generator=g,
    )
    eval_pos = torch.zeros(batch_size, num_pairs, dtype=torch.long)
    eval_target = torch.zeros(batch_size, num_pairs, dtype=torch.long)

    bin_edges = torch.linspace(0, insert_region_end, num_pairs + 1).long()

    for b in range(batch_size):
        keys = torch.randperm(vocab["key_vocab"], generator=g)[:num_pairs] + vocab["key_start"]
        values = torch.randint(
            vocab["value_start"], vocab["value_start"] + vocab["value_vocab"],
            (num_pairs,), generator=g,
        )

        for i in range(num_pairs):
            lo, hi = int(bin_edges[i]), max(int(bin_edges[i]) + 1, int(bin_edges[i + 1]) - 1)
            candidates = [p for p in range(lo, hi) if (p % chunk_size) < chunk_size - 1]
            if not candidates:
                candidates = [lo]
            pick = candidates[torch.randint(0, len(candidates), (1,), generator=g).item()]
            input_ids[b, pick] = keys[i]
            input_ids[b, pick + 1] = values[i]

        order = torch.randperm(num_pairs, generator=g)
        for i, qi in enumerate(order.tolist()):
            qpos = query_block_start + 3 * i
            input_ids[b, qpos] = vocab["QUERY"]
            input_ids[b, qpos + 1] = keys[qi]
            input_ids[b, qpos + 2] = values[qi]
            eval_pos[b, i] = qpos + 1
            eval_target[b, i] = values[qi]

    return input_ids, eval_pos, eval_target


if __name__ == "__main__":
    vocab = make_vocab()
    g = torch.Generator().manual_seed(0)
    ids, pos, tgt = generate_batch(4, 256, 16, 8, vocab, generator=g)
    print("input_ids shape:", ids.shape)
    print("一个样例序列的非 filler 位置:")
    filler_hi = vocab["filler_start"] + vocab["filler_vocab"]
    nz = (ids[0] < vocab["filler_start"]) | (ids[0] >= filler_hi)
    idxs = nz.nonzero(as_tuple=True)[0].tolist()
    print([(i, ids[0, i].item()) for i in idxs])
    print("eval_pos[0]:", pos[0].tolist())
    print("eval_target[0]:", tgt[0].tolist())
