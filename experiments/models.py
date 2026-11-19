# -*- coding: utf-8 -*-
"""
实验用的 4 种模型条件共享的骨架代码。

- NoMemory   : ChunkedLM(block_type="none")    —— 只有 chunk 内局部因果自注意力,下界基线
- FullAttn   : FullAttentionLM                  —— 整条序列全量因果自注意力,上界参照(oracle)
- TopKRaw    : ChunkedLM(block_type="rawtopk")  —— 与 CMA 记忆槽位预算相同的朴素 top-K 原始
               token 保留基线(H2O 式:按重要性分数保留,不做任何聚类/压缩)
- CMA        : ChunkedLM(block_type="cluster")  —— 本报告提出的聚类记忆增强

TopKRaw 与 CMA 复用同一个 cluster_memory.py 中已经过单元测试验证的
ClusteredMemoryCrossAttention,只是背后的记忆库实现不同,这样保证两者除了
"记忆库的组织方式"这一个变量之外,其余结构(自注意力、交叉注意力、FFN、
参数量、训练超参数)完全一致,是公平对照实验的关键。

自注意力使用旋转位置编码(RoPE, Su et al. 2021),而不是可学习的绝对位置嵌入。
原因:MQAR 这类"复现更早出现过的 token"任务,在机制上依赖 Transformer 的
"归纳头(induction head)"电路(Olsson et al., 2022 的可解释性研究表明,这类
电路通常需要至少两层注意力组合而成,且用相对位置信息比绝对位置嵌入更容易学到,
因为"attend to前一个 token""attend to与当前 token 相同的更早位置"这类模式
在相对位置表示下是与绝对位置无关的单一模式,而在可学习的绝对位置嵌入下需要
对每个绝对位置分别学习,样本效率低得多)。本实验前期用可学习绝对位置嵌入 +
1 层注意力校准时,包括全量注意力 oracle 在内的所有条件都完全学不动(准确率
停在随机猜测水平),换成 RoPE + 2 层后才恢复正常学习,这一校准过程和原因在
论文的实验设置部分如实说明。
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cluster_memory import (  # noqa: E402
    ClusterMemoryBank,
    ClusteredMemoryCrossAttention,
    pool_segment,
    token_importance_scores,
)


# ---------------------------------------------------------------------------
# RoPE（旋转位置编码）
# ---------------------------------------------------------------------------

def build_rope_cache(positions, head_dim, base=10000.0):
    """positions: [T] long. 返回 cos, sin: [T, head_dim//2]。"""
    half = head_dim // 2
    freq_idx = torch.arange(half, dtype=torch.float32, device=positions.device)
    inv_freq = 1.0 / (base ** (freq_idx / half))
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)  # [T, half]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """x: [B, H, T, head_dim]; cos, sin: [T, head_dim//2]。偶/奇下标交错配对旋转。"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)


class RotarySelfAttention(nn.Module):
    """带 RoPE 的因果自注意力(替代 nn.MultiheadAttention,以便在内部注入旋转位置编码)。"""

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,T,hd]
        cos, sin = build_rope_cache(positions, self.head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        probs = self.dropout(torch.softmax(scores + causal, dim=-1))
        ctx = torch.einsum("bhqk,bhkd->bhqd", probs, v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(ctx)


# ---------------------------------------------------------------------------
# 记忆库:CMA 用 ClusterMemoryBank（从 cluster_memory.py 复用）；
# 基线用下面的 RawTopKMemoryBank。
# ---------------------------------------------------------------------------

class RawTopKMemoryBank(nn.Module):
    """
    H2O 风格基线:大小固定为 capacity 的原始 token 缓冲区,不做任何聚类/压缩。
    每个槽位就是一个原始 token 的隐状态本身;新 token 到来时若缓冲区已满,
    则与当前缓冲区中重要性分数最低的 token 比较,分数更高才替换,否则丢弃。
    与 ClusterMemoryBank 使用完全相同的重要性打分函数(token_importance_scores),
    确保两者的差异只来自"是否聚类压缩"，而不是打分标准不同。
    """

    def __init__(self, dim: int, capacity: int):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self._initialized = False
        self.step = 0

    def reset(self, batch_size, device, dtype=torch.float32):
        B, C, D = batch_size, self.capacity, self.dim
        self.B = B
        self.slot_kv = torch.zeros(B, C, D, device=device, dtype=dtype)
        self.slot_score = torch.full((B, C), float("-inf"), device=device, dtype=dtype)
        self.slot_mask = torch.zeros(B, C, dtype=torch.bool, device=device)
        self.slot_time = torch.full((B, C), -(10 ** 9), device=device, dtype=torch.long)
        self.step = 0
        self._initialized = True

    def _ensure_init(self, batch_size, device, dtype):
        if not self._initialized or self.B != batch_size or self.slot_kv.device != device:
            self.reset(batch_size, device, dtype)

    @property
    def active_counts(self):
        if not self._initialized:
            return torch.zeros(0, dtype=torch.long)
        return self.slot_mask.sum(dim=-1)

    @torch.no_grad()
    def write(self, segment_repr, segment_tokens, token_scores=None):
        B, T, D = segment_tokens.shape
        device = segment_tokens.device
        self._ensure_init(B, device, segment_tokens.dtype)
        if token_scores is None:
            token_scores = token_importance_scores(segment_tokens)

        for t in range(T):
            tok = segment_tokens[:, t, :]
            sc = token_scores[:, t]

            has_empty = ~self.slot_mask.all(dim=-1)
            empty_idx = torch.argmax((~self.slot_mask).float(), dim=-1)
            min_val, min_idx = self.slot_score.min(dim=-1)
            replace = (~has_empty) & (sc > min_val)
            target_idx = torch.where(has_empty, empty_idx, min_idx)
            do_write = has_empty | replace

            b_range = torch.arange(B, device=device)
            wb, ws = b_range[do_write], target_idx[do_write]
            self.slot_kv[wb, ws] = tok[do_write]
            self.slot_score[wb, ws] = sc[do_write]
            self.slot_mask[wb, ws] = True
            self.slot_time[wb, ws] = self.step

        self.step += 1

    def retrieve(self, query_repr, top_k_sim=None, top_k_recent=None):
        B = query_repr.shape[0]
        self._ensure_init(B, query_repr.device, query_repr.dtype)
        return self.slot_kv, self.slot_mask, None

    def retrieve_per_token(self, query_repr, top_k_sim=None, top_k_recent=0):
        """
        与 retrieve() 的关键区别: retrieve() 无条件返回整个缓冲区(不看 query),
        这一版按每个 query token 与缓冲区内每个原始 token 的余弦相似度做逐 token
        top-k 筛选,使得"检索"这一步真正依赖 query 内容,可以与
        ClusterMemoryBank.retrieve_per_token() 做公平对照(两者都变成"逐 token
        判别式检索",唯一差异仍然只是记忆库内容是聚类压缩过的还是原始 token)。

        query_repr: [B, T, D]
        返回: gathered_kv [B, T, top_k_sim, D], gathered_mask [B, T, top_k_sim], sel(诊断用)
        """
        B, T, D = query_repr.shape
        self._ensure_init(B, query_repr.device, query_repr.dtype)

        q = F.normalize(query_repr, dim=-1)
        keys = F.normalize(self.slot_kv, dim=-1)
        sim = torch.einsum("btd,bcd->btc", q, keys)  # [B, T, capacity]
        sim = sim.masked_fill(~self.slot_mask.unsqueeze(1), float("-inf"))

        k_sim = max(1, min(top_k_sim or self.capacity, self.capacity))
        sel = torch.topk(sim, k=k_sim, dim=-1).indices  # [B, T, k_sim]

        b_idx = torch.arange(B, device=sel.device).view(B, 1, 1).expand(B, T, k_sim)
        gathered_kv = self.slot_kv[b_idx, sel]  # [B, T, k_sim, D]
        gathered_mask = self.slot_mask[b_idx, sel]  # [B, T, k_sim]
        return gathered_kv, gathered_mask, sel


# ---------------------------------------------------------------------------
# 网络模块
# ---------------------------------------------------------------------------

class ChunkSelfAttnFFN(nn.Module):
    """chunk 内(或整条序列上,取决于调用方传入的 x 长度)RoPE 因果自注意力 + FFN。"""

    def __init__(self, dim, num_heads, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.self_attn = RotarySelfAttention(dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Linear(dim * ffn_mult, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, positions):
        sa = self.self_attn(x, positions)
        x = self.norm1(x + sa)
        x = self.norm2(x + self.ffn(x))
        return x


class MemoryAugmentedChunkBlock(nn.Module):
    """chunk 内自注意力 + 记忆交叉注意力(记忆库可插拔) + FFN。"""

    def __init__(self, dim, num_heads, make_bank, ffn_mult=4, dropout=0.0, top_k_sim=8, top_k_recent=0,
                 retrieval_mode="pooled"):
        super().__init__()
        assert retrieval_mode in ("pooled", "pertoken")
        self.self_attn = RotarySelfAttention(dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.memory_attn = ClusteredMemoryCrossAttention(dim, num_heads, dropout)
        self.memory_bank = make_bank()
        self.top_k_sim = top_k_sim
        self.top_k_recent = top_k_recent
        self.retrieval_mode = retrieval_mode
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Linear(dim * ffn_mult, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, positions, write_to_memory=True):
        sa = self.self_attn(x, positions)
        x = self.norm1(x + sa)

        if self.retrieval_mode == "pertoken":
            x, diag = self.memory_attn.forward_per_token(x, self.memory_bank, self.top_k_sim, self.top_k_recent)
        else:
            x, diag = self.memory_attn(x, self.memory_bank, self.top_k_sim, self.top_k_recent)

        if write_to_memory:
            with torch.no_grad():
                seg_repr = pool_segment(x, mode="mean")
                scores = token_importance_scores(x)
            self.memory_bank.write(seg_repr, x.detach(), scores)

        x = self.norm2(x + self.ffn(x))
        return x, diag

    def reset_memory(self, batch_size, device, dtype=torch.float32):
        self.memory_bank.reset(batch_size, device, dtype)


class ChunkedLM(nn.Module):
    """block_type in {"none", "cluster", "rawtopk"}。"""

    def __init__(self, vocab_size, dim, num_layers, num_heads, chunk_size, max_len,
                 block_type="none", memory_kwargs=None, top_k_sim=8, top_k_recent=0, dropout=0.0,
                 retrieval_mode="pooled"):
        super().__init__()
        self.chunk_size = chunk_size
        self.block_type = block_type
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)

        memory_kwargs = memory_kwargs or {}
        layers = []
        for _ in range(num_layers):
            if block_type == "none":
                layers.append(ChunkSelfAttnFFN(dim, num_heads, dropout=dropout))
            elif block_type == "cluster":
                layers.append(MemoryAugmentedChunkBlock(
                    dim, num_heads, lambda mk=memory_kwargs: ClusterMemoryBank(dim, **mk),
                    dropout=dropout, top_k_sim=top_k_sim, top_k_recent=top_k_recent,
                    retrieval_mode=retrieval_mode,
                ))
            elif block_type == "rawtopk":
                layers.append(MemoryAugmentedChunkBlock(
                    dim, num_heads, lambda mk=memory_kwargs: RawTopKMemoryBank(dim, **mk),
                    dropout=dropout, top_k_sim=top_k_sim, top_k_recent=top_k_recent,
                    retrieval_mode=retrieval_mode,
                ))
            else:
                raise ValueError(block_type)
        self.layers = nn.ModuleList(layers)
        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def reset_memory(self, batch_size, device, dtype=torch.float32):
        if self.block_type != "none":
            for l in self.layers:
                l.reset_memory(batch_size, device, dtype)

    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device
        if self.block_type != "none":
            self.reset_memory(B, device, self.tok_emb.weight.dtype)

        # 诊断用: 记录每个 chunk/层的 null_attn_weight_mean,不影响前向计算数值,
        # 默认不读取也没有任何开销上的负面影响(纯 Python list append)。
        self.last_diags = []

        outs = []
        n_chunks = math.ceil(T / self.chunk_size)
        for c in range(n_chunks):
            s, e = c * self.chunk_size, min((c + 1) * self.chunk_size, T)
            ids = input_ids[:, s:e]
            positions = torch.arange(s, e, device=device)
            x = self.dropout(self.tok_emb(ids))
            for li, l in enumerate(self.layers):
                if self.block_type == "none":
                    x = l(x, positions)
                else:
                    x, diag = l(x, positions, write_to_memory=True)
                    self.last_diags.append({"chunk": c, "layer": li, **diag})
            outs.append(x)
        x = torch.cat(outs, dim=1)
        x = self.norm_f(x)
        return self.head(x)


class FullAttentionLM(nn.Module):
    """全量因果自注意力上界参照模型(处理整条序列,无分块、无记忆限制)。"""

    def __init__(self, vocab_size, dim, num_layers, num_heads, max_len, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ChunkSelfAttnFFN(dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device
        positions = torch.arange(T, device=device)
        x = self.dropout(self.tok_emb(input_ids))
        for l in self.layers:
            x = l(x, positions)
        x = self.norm_f(x)
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(condition, vocab_size, dim, num_layers, num_heads, chunk_size, max_len,
                 budget=None, exemplars_per_cluster=2, top_k_recent=0, dropout=0.0,
                 merge_threshold=0.3, capacity_policy="merge"):
    """
    condition in {"nomemory", "fullattn", "topkraw", "cma", "topkraw_pertoken", "cma_pertoken"}
    budget: 记忆槽位总预算(topkraw 的 capacity,或 cma 的 max_clusters * exemplars_per_cluster)，
            nomemory / fullattn 不需要。

    "_pertoken" 两个变体是对"pooled_query 是否是瓶颈"这一问题的后续、更充分的测试:
    - 原始 topkraw / cma 把 top_k_sim 设成等于各自全部容量,检索"筛选"是恒等操作
      (不管 query 是什么,返回的都是全部记忆),池化与逐 token 在这种配置下必然
      没有区别。
    - "_pertoken" 变体把 top_k_sim 降到容量的一半(真正产生筛选压力),并且检索
      查询换成逐 token(而不是整个 chunk 共用一个池化向量),这样"检索能否按需
      区分不同 query"才是一个被真正测试到的变量。
    """
    if condition == "nomemory":
        return ChunkedLM(vocab_size, dim, num_layers, num_heads, chunk_size, max_len, block_type="none",
                          dropout=dropout)
    if condition == "fullattn":
        return FullAttentionLM(vocab_size, dim, num_layers, num_heads, max_len, dropout=dropout)
    if condition == "topkraw":
        assert budget is not None
        return ChunkedLM(vocab_size, dim, num_layers, num_heads, chunk_size, max_len, block_type="rawtopk",
                          memory_kwargs=dict(capacity=budget), top_k_sim=budget, top_k_recent=0, dropout=dropout)
    if condition == "cma":
        assert budget is not None and budget % exemplars_per_cluster == 0
        k_max = budget // exemplars_per_cluster
        return ChunkedLM(vocab_size, dim, num_layers, num_heads, chunk_size, max_len, block_type="cluster",
                          memory_kwargs=dict(max_clusters=k_max, exemplars_per_cluster=exemplars_per_cluster,
                                              merge_threshold=merge_threshold, ema_decay=0.9,
                                              capacity_policy=capacity_policy, dead_ttl=10 ** 6),
                          top_k_sim=k_max, top_k_recent=top_k_recent, dropout=dropout)
    if condition == "topkraw_pertoken":
        assert budget is not None
        k_sim = max(1, budget // 2)
        return ChunkedLM(vocab_size, dim, num_layers, num_heads, chunk_size, max_len, block_type="rawtopk",
                          memory_kwargs=dict(capacity=budget), top_k_sim=k_sim, top_k_recent=0, dropout=dropout,
                          retrieval_mode="pertoken")
    if condition == "cma_pertoken":
        assert budget is not None and budget % exemplars_per_cluster == 0
        k_max = budget // exemplars_per_cluster
        k_sim = max(1, k_max // 2)
        return ChunkedLM(vocab_size, dim, num_layers, num_heads, chunk_size, max_len, block_type="cluster",
                          memory_kwargs=dict(max_clusters=k_max, exemplars_per_cluster=exemplars_per_cluster,
                                              merge_threshold=merge_threshold, ema_decay=0.9,
                                              capacity_policy=capacity_policy, dead_ttl=10 ** 6),
                          top_k_sim=k_sim, top_k_recent=top_k_recent, dropout=dropout,
                          retrieval_mode="pertoken")
    raise ValueError(condition)
