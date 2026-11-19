# -*- coding: utf-8 -*-
"""
聚类记忆增强 Transformer 原型 (Clustered Memory Augmentation, 简称 CMA)
============================================================

设计目标
--------
在标准 Transformer 之外挂载一个"有界大小、可持续流式更新"的外部记忆库,
用 **聚类** 同时解决两件事:

1. 写入侧(记忆组织): 把不断到来的历史 token 表征组织成数量有界(<= K_max)的
   "记忆簇",每簇只保留一个质心(centroid,用于检索的软摘要)和少量代表性
   token(exemplars,用于还原细节的硬样本),从而把 O(T) 的历史压缩到 O(K_max)。
2. 读取侧(记忆检索): 用质心做内容相似度检索(近似最近邻),辅以时间连续性,
   选出与当前查询最相关的若干记忆簇,作为交叉注意力的 K/V。

本文件中每个模块在设计文档(见配套的《聚类记忆增强 Transformer 技术方案》)里都
标注了其借鉴的已发表方法,以及本原型相对已有方法所做的原创整合,核心参考包括:

- Routing Transformer (Roy et al., 2020): 在线 k-means 聚类做稀疏注意力路由。
  本原型借鉴其"用聚类降低注意力/检索复杂度"的思路,但把聚类粒度从逐 token
  改为逐"事件段"(见 SurpriseBoundaryDetector),以缓解其被后续工作(Mixture of
  Sparse Attention, 2025)指出的"在线 k-means 收敛慢"问题。
- Compressive Transformer (Rae et al., 2020): 老化记忆被压缩而不是直接丢弃。
  本原型的簇合并(merge)机制是这个思路在"聚类"场景下的对应实现。
- VQ-VAE / VQ-VAE-2 (van den Oord et al., 2017 / Razavi et al., 2019): 用
  EMA 更新 codebook、straight-through estimator 传梯度、对长期不用的 code
  做随机重置(这里简化为"死簇回收")。本原型的 ClusterMemoryBank.write /
  quantize_ste 直接沿用了这一整套已被证明训练稳定的机制。
- DP-means (Kulis & Jordan, 2012): "与最近质心的相似度低于阈值就新开一簇"
  的动态建簇规则,用于替代需要预先设定簇数的经典 k-means。
- EM-LLM (2024/2025, ICLR 2025): 用"惊异"信号做事件边界检测 + 相似性/时间
  连续性双缓冲检索。本原型的 SurpriseBoundaryDetector 与 retrieve() 的
  (top_k_sim + top_k_recent) 设计借鉴了这一思路,并做了简化。
- ClusterAttn (ACL 2025) / H2O: 用 token 的重要性分数筛选"代表 token"。
  本原型的 exemplar 筛选机制(按 token 范数/可替换为注意力质量打分排序取
  top-E)借鉴了这一思路。
- Memory Layers at Scale (Meta, 2024): product-key 分解降低大规模记忆检索
  复杂度。本原型只实现了 O(K) 的朴素检索,但在设计文档中说明了如何替换成
  product-key 版本以支持 K_max 很大的场景。
- StreamingLLM (Xiao et al., 2023): attention sink 的思路启发了本原型
  ClusteredMemoryCrossAttention 中的可学习 null_kv 兜底槽位。

这是一个可运行、可读的**教学/研究原型**,不是工业级实现:核心状态更新用
Python 循环按 batch 维度处理以保证逻辑清晰、易于核对正确性;大规模部署需要
把这部分改写为向量化 / 自定义 CUDA kernel。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 工具函数
# =============================================================================

def pool_segment(tokens: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """把一个 [..., T, D] 的 token 段压成 [..., D] 的段表征,用于聚类分配/检索。"""
    if mode == "mean":
        return tokens.mean(dim=-2)
    if mode == "last":
        return tokens[..., -1, :]
    raise ValueError(f"未知的 pooling 方式: {mode}")


def token_importance_scores(tokens: torch.Tensor) -> torch.Tensor:
    """
    用 token 隐状态的 L2 范数作为"重要性"代理分数,用来挑选每个簇要保留的
    代表 token(exemplars)。如果调用方能拿到真实的 attention 权重(例如
    ClusterAttn / H2O 的做法:用该 token 收到的注意力质量之和),应优先使用
    真实权重替换这个代理分数,精度会更好。
    """
    return tokens.norm(dim=-1)


# =============================================================================
# M1: 在线事件边界检测(可选模块,思路简化自 EM-LLM 的 Bayesian surprise)
# =============================================================================

class SurpriseBoundaryDetector:
    """
    用"相邻 token 表征的语义突变程度"(1 - 余弦相似度)作为惊异的代理信号
    (原始 EM-LLM 用语言模型的负对数似然,这里为了保持模块与具体 LM head
    解耦,改用表征突变作为近似)。在滑动窗口内维护均值/方差,超过
    mean + gamma * std 的位置被标记为新事件的起点。

    这是一个单序列(无 batch 维)的工具类:批处理时需要在外部对 batch
    维度做循环,因为不同序列检测出的边界数量本来就不同(天然不等长)。
    """

    def __init__(self, window: int = 16, gamma: float = 1.0, min_seg_len: int = 2):
        self.window = window
        self.gamma = gamma
        self.min_seg_len = min_seg_len

    @torch.no_grad()
    def detect(self, hidden_states: torch.Tensor):
        """hidden_states: [T, D]。返回升序 int 列表,每个值是新事件起始下标。"""
        T = hidden_states.shape[0]
        if T < 2:
            return []
        h = F.normalize(hidden_states, dim=-1)
        surprise = 1.0 - (h[1:] * h[:-1]).sum(dim=-1)  # [T-1]

        boundaries = []
        last_boundary = 0
        history = []
        for t in range(surprise.shape[0]):
            s = surprise[t].item()
            if len(history) >= 2:
                mu = sum(history) / len(history)
                var = sum((x - mu) ** 2 for x in history) / len(history)
                threshold = mu + self.gamma * (var ** 0.5)
            else:
                threshold = float("inf")

            pos = t + 1  # surprise[t] 描述的是 hidden_states[t] -> [t+1] 的跳变
            if s > threshold and (pos - last_boundary) >= self.min_seg_len:
                boundaries.append(pos)
                last_boundary = pos
                history = []  # 新事件开始,局部统计窗口重置(信号非平稳,不用全局统计)
            else:
                history.append(s)
                if len(history) > self.window:
                    history.pop(0)
        return boundaries

    def segment(self, hidden_states: torch.Tensor):
        """返回覆盖整个序列、互不重叠的 (start, end) 区间列表。"""
        T = hidden_states.shape[0]
        boundaries = self.detect(hidden_states)
        bounds = [0] + boundaries + [T]
        return list(zip(bounds[:-1], bounds[1:]))


# =============================================================================
# M2 + M3 + M5: 有界聚类记忆库(核心模块)
# =============================================================================

class ClusterMemoryBank(nn.Module):
    """
    维护每个 batch 样本最多 K_max 个"记忆簇"。每簇存储:
      - centroid:  [D]           质心(L2 归一化),用于检索阶段的相似度计算
      - exemplars: [E, D]        簇内挑出的 E 个代表 token(硬样本,用于还原细节)
      - count:     标量           已并入该簇的段数(用作 EMA 权重 / 淘汰打分)
      - last_write_t: 标量        最近一次写入时的全局步数(用于时间连续性检索 / 死簇回收)

    写入规则(DP-means 风格): 新段与最相似的活跃簇的余弦相似度若超过
    merge_threshold,则并入该簇(centroid 用 EMA 更新,exemplars 取新旧并集
    按分数截断);否则新开一簇。簇数已达上限时,优先回收"死簇"(长期未更新),
    否则合并当前最相似的两个活跃簇腾出空位(也可配置为直接淘汰最旧/分数最低
    的簇,见 capacity_policy)。
    """

    def __init__(
        self,
        dim: int,
        max_clusters: int = 64,
        exemplars_per_cluster: int = 4,
        ema_decay: float = 0.99,
        merge_threshold: float = 0.6,
        dead_ttl: int = 512,
        capacity_policy: str = "merge",
        routing_mode: str = "similarity",
    ):
        super().__init__()
        assert capacity_policy in ("merge", "evict_lru", "evict_score")
        assert routing_mode in ("similarity", "random")
        self.dim = dim
        self.K = max_clusters
        self.E = exemplars_per_cluster
        self.ema_decay = ema_decay
        self.merge_threshold = merge_threshold
        self.dead_ttl = dead_ttl
        self.capacity_policy = capacity_policy
        # "similarity"(默认): 论文报告的 CMA,按最近质心相似度路由,超过
        # merge_threshold 才并入,否则开新簇 —— 内容决定分组。
        # "random": 消融对照组,忽略 segment_repr 与质心的相似度,均匀随机
        # 选一个 0..K-1 的槽位分配这个 segment(见 write() 里的分支)。除了
        # "怎么分组"这一件事,EMA 质心更新、代表 token 挑选、检索机制全部
        # 复用同一套代码,不额外引入任何其他变量。用来回答:如果涨点存在,
        # 是不是仅仅因为"分成 K 组、每组留 E 个代表"这个机械结构本身,
        # 换成随机分组也会涨——那样涨的就不是语义,不能算聚类真的有用。
        self.routing_mode = routing_mode

        self._initialized = False
        self.B = 0
        self.step = 0

    # -- 状态管理 -------------------------------------------------------

    def reset(self, batch_size: int, device=None, dtype=torch.float32):
        """清空记忆库并按给定 batch_size 重新分配状态张量(开始处理新的一批流式序列时调用)。"""
        B, K, E, D = batch_size, self.K, self.E, self.dim
        device = device or torch.device("cpu")
        self.B = B
        self.centroids = torch.zeros(B, K, D, device=device, dtype=dtype)
        self.exemplar_kv = torch.zeros(B, K, E, D, device=device, dtype=dtype)
        self.exemplar_score = torch.full((B, K, E), float("-inf"), device=device, dtype=dtype)
        self.exemplar_mask = torch.zeros(B, K, E, dtype=torch.bool, device=device)
        self.active = torch.zeros(B, K, dtype=torch.bool, device=device)
        self.count = torch.zeros(B, K, device=device, dtype=dtype)
        self.last_write_t = torch.full((B, K), -(10 ** 9), device=device, dtype=torch.long)
        self.step = 0
        self._initialized = True

    def _ensure_init(self, batch_size, device, dtype):
        if not self._initialized or self.B != batch_size or self.centroids.device != device:
            self.reset(batch_size, device, dtype)

    @property
    def active_counts(self) -> torch.Tensor:
        """每个 batch 样本当前活跃的簇数,形状 [B]。主要用于诊断/日志。"""
        if not self._initialized:
            return torch.zeros(0, dtype=torch.long)
        return self.active.sum(dim=-1)

    # -- 写入(维护聚类) --------------------------------------------------

    @torch.no_grad()
    def write(self, segment_repr: torch.Tensor, segment_tokens: torch.Tensor, token_scores: torch.Tensor = None):
        """
        segment_repr:   [B, D]    该段的池化表征(会被内部 L2 归一化),用于决定分配到哪个簇
        segment_tokens: [B, T, D] 该段的原始 token 隐状态,候选作为代表 token
        token_scores:   [B, T] 或 None,每个 token 的重要性分数;None 时用 L2 范数代替
        """
        B, T, D = segment_tokens.shape
        device = segment_tokens.device
        self._ensure_init(B, device, segment_tokens.dtype)

        if token_scores is None:
            token_scores = token_importance_scores(segment_tokens)

        repr_n = F.normalize(segment_repr, dim=-1)

        if self.routing_mode == "random":
            # 消融对照:分组标准与内容无关,均匀随机选一个槽位 k in [0, K)。
            # 槽位若还没激活就当成"开新簇"初始化,已激活就当成"并入"更新——
            # 复用与 similarity 分支完全相同的 _init_cluster / _merge_segment_into,
            # 唯一被替换的是"选哪个槽位"这个决策本身,不引入其他差异。
            for b in range(B):
                k = int(torch.randint(0, self.K, (1,)).item())
                if bool(self.active[b, k]):
                    self._merge_segment_into(b, k, repr_n[b], segment_tokens[b], token_scores[b])
                else:
                    self._init_cluster(b, k, repr_n[b], segment_tokens[b], token_scores[b])
            self.step += 1
            return

        sim = torch.einsum("bd,bkd->bk", repr_n, self.centroids)
        sim = sim.masked_fill(~self.active, float("-inf"))

        for b in range(B):
            has_active = bool(self.active[b].any())
            if has_active:
                best_val_t, best_k_t = sim[b].max(dim=0)
                best_val, best_k = float(best_val_t.item()), int(best_k_t.item())
            else:
                best_val, best_k = float("-inf"), -1

            if has_active and best_val > self.merge_threshold:
                self._merge_segment_into(b, best_k, repr_n[b], segment_tokens[b], token_scores[b])
            else:
                slot = self._find_slot_for_new_cluster(b)
                self._init_cluster(b, slot, repr_n[b], segment_tokens[b], token_scores[b])

        self.step += 1

    def _merge_segment_into(self, b, k, repr_vec, tokens, scores):
        c = self.count[b, k]
        new_centroid = self.ema_decay * self.centroids[b, k] + (1 - self.ema_decay) * repr_vec
        self.centroids[b, k] = F.normalize(new_centroid, dim=0)
        self.count[b, k] = c + 1
        self.last_write_t[b, k] = self.step
        self._update_exemplars(b, k, tokens, scores)

    def _init_cluster(self, b, k, repr_vec, tokens, scores):
        self.centroids[b, k] = repr_vec
        self.count[b, k] = 1
        self.last_write_t[b, k] = self.step
        self.active[b, k] = True
        self.exemplar_mask[b, k] = False
        self.exemplar_score[b, k] = float("-inf")
        self.exemplar_kv[b, k] = 0
        self._update_exemplars(b, k, tokens, scores)

    def _update_exemplars(self, b, k, tokens, scores):
        """把新段的 token 与簇里已有的代表 token 合并,按重要性分数取 top-E。"""
        E = self.E
        cand_kv = torch.cat([self.exemplar_kv[b, k], tokens], dim=0)  # [E+T, D]
        cand_score = torch.cat([self.exemplar_score[b, k], scores], dim=0)  # [E+T]
        cand_mask = torch.cat(
            [self.exemplar_mask[b, k], torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)],
            dim=0,
        )
        cand_score = cand_score.masked_fill(~cand_mask, float("-inf"))

        n_valid = int(cand_mask.sum().item())
        k_take = min(E, n_valid)
        top_val, top_idx = torch.topk(cand_score, k=E)
        new_kv = cand_kv[top_idx]
        keep = torch.arange(E, device=tokens.device) < k_take  # topk 降序排列,前 k_take 个才是有效值
        new_kv = torch.where(keep.unsqueeze(-1), new_kv, torch.zeros_like(new_kv))
        new_score = torch.where(keep, top_val, torch.full_like(top_val, float("-inf")))

        self.exemplar_kv[b, k] = new_kv
        self.exemplar_score[b, k] = new_score
        self.exemplar_mask[b, k] = keep

    def _merge_clusters(self, b, i, j):
        """把簇 j 并入簇 i(按 count 加权),并释放槽位 j。"""
        wi, wj = self.count[b, i], self.count[b, j]
        merged = wi * self.centroids[b, i] + wj * self.centroids[b, j]
        self.centroids[b, i] = F.normalize(merged, dim=0)
        self.count[b, i] = wi + wj
        self.last_write_t[b, i] = max(int(self.last_write_t[b, i].item()), int(self.last_write_t[b, j].item()))

        E = self.E
        cand_kv = torch.cat([self.exemplar_kv[b, i], self.exemplar_kv[b, j]], dim=0)  # [2E, D]
        cand_score = torch.cat([self.exemplar_score[b, i], self.exemplar_score[b, j]], dim=0)
        cand_mask = torch.cat([self.exemplar_mask[b, i], self.exemplar_mask[b, j]], dim=0)
        cand_score = cand_score.masked_fill(~cand_mask, float("-inf"))
        n_valid = int(cand_mask.sum().item())
        k_take = min(E, n_valid)
        top_val, top_idx = torch.topk(cand_score, k=E)
        new_kv = cand_kv[top_idx]
        keep = torch.arange(E, device=cand_kv.device) < k_take
        new_kv = torch.where(keep.unsqueeze(-1), new_kv, torch.zeros_like(new_kv))
        new_score = torch.where(keep, top_val, torch.full_like(top_val, float("-inf")))

        self.exemplar_kv[b, i] = new_kv
        self.exemplar_score[b, i] = new_score
        self.exemplar_mask[b, i] = keep

        self._deactivate(b, j)

    def _deactivate(self, b, k):
        self.active[b, k] = False
        self.centroids[b, k] = 0
        self.count[b, k] = 0
        self.last_write_t[b, k] = -(10 ** 9)
        self.exemplar_kv[b, k] = 0
        self.exemplar_score[b, k] = float("-inf")
        self.exemplar_mask[b, k] = False

    def _find_slot_for_new_cluster(self, b):
        inactive = (~self.active[b]).nonzero(as_tuple=True)[0]
        if len(inactive) > 0:
            return int(inactive[0].item())

        age = self.step - self.last_write_t[b]
        dead = self.active[b] & (age > self.dead_ttl)
        if bool(dead.any()):
            dead_ages = age.masked_fill(~dead, -1)
            slot = int(torch.argmax(dead_ages).item())
            self._deactivate(b, slot)
            return slot

        n_active = int(self.active[b].sum().item())
        if self.capacity_policy == "merge" and n_active >= 2:
            c = F.normalize(self.centroids[b], dim=-1)
            sim = c @ c.t()
            valid = self.active[b]
            pair_mask = valid.unsqueeze(0) & valid.unsqueeze(1)
            sim = sim.masked_fill(~pair_mask, -2.0)
            sim.fill_diagonal_(-2.0)
            flat_idx = int(torch.argmax(sim).item())
            i, j = divmod(flat_idx, self.K)
            self._merge_clusters(b, i, j)
            return j

        # 兜底: capacity_policy == evict_* ,或 merge 策略下活跃簇不足 2 个无法合并
        if self.capacity_policy == "evict_score":
            decay = 0.999
            score = self.count[b] * (decay ** age.clamp(min=0).float())
            score = score.masked_fill(~self.active[b], float("inf"))
            slot = int(torch.argmin(score).item())
        else:
            lw = self.last_write_t[b].masked_fill(~self.active[b], 2 ** 62)
            slot = int(torch.argmin(lw).item())
        self._deactivate(b, slot)
        return slot

    # -- 读取(检索) -------------------------------------------------------

    def retrieve(self, query_repr: torch.Tensor, top_k_sim: int = 4, top_k_recent: int = 2):
        """
        query_repr: [B, D] 池化后的查询向量(内部会 L2 归一化)。
        返回:
          mem_kv:   [B, (top_k_sim+top_k_recent)*E, D]  拼接后的候选 K/V(未命中的簇填 0)
          mem_mask: [B, (top_k_sim+top_k_recent)*E]     True 表示该位置是有效 token
          sel:      [B, top_k_sim+top_k_recent]          被选中的簇下标(诊断用)
        """
        B = query_repr.shape[0]
        self._ensure_init(B, query_repr.device, query_repr.dtype)

        q = F.normalize(query_repr, dim=-1)
        sim = torch.einsum("bd,bkd->bk", q, self.centroids)
        sim = sim.masked_fill(~self.active, float("-inf"))

        k_sim = max(1, min(top_k_sim, self.K))
        k_rec = max(0, min(top_k_recent, self.K))

        sim_topk = torch.topk(sim, k=k_sim, dim=-1).indices  # [B, k_sim]
        if k_rec > 0:
            rec_scores = self.last_write_t.to(torch.float32).masked_fill(~self.active, float("-inf"))
            rec_topk = torch.topk(rec_scores, k=k_rec, dim=-1).indices  # [B, k_rec]
            sel = torch.cat([sim_topk, rec_topk], dim=-1)
        else:
            sel = sim_topk

        M = sel.shape[1]
        b_idx = torch.arange(B, device=sel.device).view(B, 1).expand(B, M)
        gathered_kv = self.exemplar_kv[b_idx, sel]  # [B, M, E, D]
        gathered_mask = self.exemplar_mask[b_idx, sel]  # [B, M, E]
        gathered_active = self.active[b_idx, sel]  # [B, M]

        mem_kv = gathered_kv.reshape(B, M * self.E, self.dim)
        mem_mask = (gathered_mask & gathered_active.unsqueeze(-1)).reshape(B, M * self.E)
        return mem_kv, mem_mask, sel

    def retrieve_per_token(self, query_repr: torch.Tensor, top_k_sim: int = 4, top_k_recent: int = 0):
        """
        与 retrieve() 的区别: query_repr 是逐 token 的 [B, T, D](未经 chunk 内均值池化),
        因此每个 token 可以选出各自不同的候选簇集合,而不是整个 chunk 共用一组检索结果。
        新增方法而不是修改 retrieve(),是为了不改变 retrieve() 在已报告实验中的行为
        (保持那批结果可复现);两者共享同一份记忆库状态,只是读取方式不同。

        query_repr: [B, T, D]
        返回:
          mem_kv:   [B, T, (top_k_sim+top_k_recent)*E, D]
          mem_mask: [B, T, (top_k_sim+top_k_recent)*E]
          sel:      [B, T, top_k_sim+top_k_recent]  (诊断用)
        """
        B, T, _ = query_repr.shape
        self._ensure_init(B, query_repr.device, query_repr.dtype)

        q = F.normalize(query_repr, dim=-1)
        sim = torch.einsum("btd,bkd->btk", q, self.centroids)  # [B, T, K]
        sim = sim.masked_fill(~self.active.unsqueeze(1), float("-inf"))

        k_sim = max(1, min(top_k_sim, self.K))
        k_rec = max(0, min(top_k_recent, self.K))

        sim_topk = torch.topk(sim, k=k_sim, dim=-1).indices  # [B, T, k_sim]
        if k_rec > 0:
            rec_scores = self.last_write_t.to(torch.float32).masked_fill(~self.active, float("-inf"))
            rec_topk = torch.topk(rec_scores, k=k_rec, dim=-1).indices  # [B, k_rec]
            rec_topk = rec_topk.unsqueeze(1).expand(B, T, k_rec)  # 同一 chunk 内所有 token 共用最近簇
            sel = torch.cat([sim_topk, rec_topk], dim=-1)
        else:
            sel = sim_topk

        M = sel.shape[-1]
        b_idx = torch.arange(B, device=sel.device).view(B, 1, 1).expand(B, T, M)
        gathered_kv = self.exemplar_kv[b_idx, sel]  # [B, T, M, E, D]
        gathered_mask = self.exemplar_mask[b_idx, sel]  # [B, T, M, E]
        gathered_active = self.active[b_idx, sel]  # [B, T, M]

        mem_kv = gathered_kv.reshape(B, T, M * self.E, self.dim)
        mem_mask = (gathered_mask & gathered_active.unsqueeze(-1)).reshape(B, T, M * self.E)
        return mem_kv, mem_mask, sel

    # -- 可选: 端到端训练用的可微量化路径(参考 VQ-VAE 的 straight-through estimator) --

    def quantize_ste(self, segment_repr: torch.Tensor):
        """
        只读查询,不修改记忆库(真正写入由 write() 完成)。
        前向: 返回命中的质心(若尚无活跃簇,退化为输入本身);
        反向: 梯度经 straight-through estimator 直接传给 segment_repr。
        同时返回一个 VQ-VAE 风格的 commitment loss,可选地加入训练目标,
        鼓励编码器输出向被分配的质心靠拢、减少量化误差。
        """
        B = segment_repr.shape[0]
        self._ensure_init(B, segment_repr.device, segment_repr.dtype)

        with torch.no_grad():
            q = F.normalize(segment_repr, dim=-1)
            sim = torch.einsum("bd,bkd->bk", q, self.centroids)
            sim = sim.masked_fill(~self.active, float("-inf"))
            has_active = self.active.any(dim=-1)
            best_idx = sim.argmax(dim=-1)

        b_idx = torch.arange(B, device=segment_repr.device)
        nearest_centroid = self.centroids[b_idx, best_idx]
        nearest_centroid = torch.where(
            has_active.unsqueeze(-1), nearest_centroid, segment_repr.detach()
        )

        ste_out = segment_repr + (nearest_centroid - segment_repr).detach()
        commitment_loss = F.mse_loss(segment_repr, nearest_centroid.detach())
        return ste_out, best_idx, commitment_loss


# =============================================================================
# M4: 聚类记忆交叉注意力(可插拔层)
# =============================================================================

class ClusteredMemoryCrossAttention(nn.Module):
    """
    把 ClusterMemoryBank.retrieve() 检索出的记忆当作 K/V 做一次标准多头交叉
    注意力,输出与输入残差相加 + LayerNorm。额外维护一个可学习的 null_kv
    "空槽"(类似 StreamingLLM 的 attention sink):即使记忆库还是空的(比如
    流式输入的第一段),softmax 也始终有至少一个有效位置可以关注,不会产生
    NaN,也不需要为"记忆为空"写特判分支。
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.null_kv = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, query_hidden, memory_bank: ClusterMemoryBank, top_k_sim=4, top_k_recent=2, pooled_query=None):
        B, Tq, D = query_hidden.shape
        if pooled_query is None:
            pooled_query = query_hidden.mean(dim=1)

        mem_kv, mem_mask, sel = memory_bank.retrieve(pooled_query, top_k_sim, top_k_recent)

        null_kv = self.null_kv.expand(B, 1, D)
        null_mask = torch.ones(B, 1, dtype=torch.bool, device=mem_mask.device)
        mem_kv_full = torch.cat([mem_kv, null_kv], dim=1)
        mem_mask_full = torch.cat([mem_mask, null_mask], dim=1)

        Q = self.q_proj(query_hidden)
        K = self.k_proj(mem_kv_full)
        V = self.v_proj(mem_kv_full)

        def split_heads(t):
            Bx, Lx, _ = t.shape
            return t.view(Bx, Lx, self.num_heads, self.head_dim).transpose(1, 2)

        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
        attn_scores = torch.einsum("bhqd,bhkd->bhqk", Qh, Kh) / math.sqrt(self.head_dim)
        bias = torch.zeros_like(mem_mask_full, dtype=attn_scores.dtype).masked_fill(~mem_mask_full, float("-inf"))
        attn_scores = attn_scores + bias.view(B, 1, 1, -1)
        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))

        ctx = torch.einsum("bhqk,bhkd->bhqd", attn_probs, Vh).transpose(1, 2).reshape(B, Tq, D)
        out = self.out_proj(ctx)
        result = self.norm(query_hidden + out)

        diagnostics = {
            "selected_clusters": sel,
            "mem_mask": mem_mask,
            "null_attn_weight_mean": attn_probs[..., -1].mean().item(),
        }
        return result, diagnostics

    def forward_per_token(self, query_hidden, memory_bank, top_k_sim=4, top_k_recent=0):
        """
        forward() 的逐 token 检索变体: 检索候选集合按每个 query 位置单独选出
        (memory_bank.retrieve_per_token()),而不是整个 chunk 共用一组池化查询选出
        的候选集合。要求 memory_bank 实现 retrieve_per_token(query[B,T,D], ...)
        并返回 (mem_kv[B,T,M,D], mem_mask[B,T,M], sel)。

        新增方法而不是修改 forward(),原因同 retrieve_per_token(): 保持 forward()
        在已报告实验中的行为不变、可复现。
        """
        B, Tq, D = query_hidden.shape
        mem_kv, mem_mask, sel = memory_bank.retrieve_per_token(query_hidden, top_k_sim, top_k_recent)
        # mem_kv: [B, T, M, D], mem_mask: [B, T, M]

        null_kv = self.null_kv.view(1, 1, 1, D).expand(B, Tq, 1, D)
        null_mask = torch.ones(B, Tq, 1, dtype=torch.bool, device=mem_mask.device)
        mem_kv_full = torch.cat([mem_kv, null_kv], dim=2)  # [B, T, M+1, D]
        mem_mask_full = torch.cat([mem_mask, null_mask], dim=2)  # [B, T, M+1]

        Q = self.q_proj(query_hidden)  # [B, T, D]
        K = self.k_proj(mem_kv_full)  # [B, T, M+1, D]
        V = self.v_proj(mem_kv_full)  # [B, T, M+1, D]

        H, Dh = self.num_heads, self.head_dim
        Qh = Q.view(B, Tq, H, Dh)
        Kh = K.view(B, Tq, -1, H, Dh)
        Vh = V.view(B, Tq, -1, H, Dh)

        attn_scores = torch.einsum("bthd,btmhd->bthm", Qh, Kh) / math.sqrt(Dh)  # [B, T, H, M+1]
        bias = torch.zeros_like(mem_mask_full, dtype=attn_scores.dtype).masked_fill(~mem_mask_full, float("-inf"))
        attn_scores = attn_scores + bias.unsqueeze(2)  # broadcast over heads
        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))  # [B, T, H, M+1]

        ctx = torch.einsum("bthm,btmhd->bthd", attn_probs, Vh).reshape(B, Tq, D)
        out = self.out_proj(ctx)
        result = self.norm(query_hidden + out)

        diagnostics = {
            "selected_clusters": sel,
            "mem_mask": mem_mask,
            "null_attn_weight_mean": attn_probs[..., -1].mean().item(),
        }
        return result, diagnostics


# =============================================================================
# 演示: 可插拔进标准 Transformer block
# =============================================================================

class ClusterMemoryTransformerBlock(nn.Module):
    """
    x -> [因果自注意力 + 残差] -> [聚类记忆交叉注意力(内部含残差)] -> [FFN + 残差]

    面向"分段流式"处理: 每次 forward 传入一个 chunk(而不是整条序列),
    forward 内部默认会把这个 chunk 写回记忆库,模拟长文本/多轮对话下
    逐段处理、记忆持续累积的场景。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        max_clusters: int = 64,
        exemplars_per_cluster: int = 4,
        merge_threshold: float = 0.6,
        capacity_policy: str = "merge",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.memory_attn = ClusteredMemoryCrossAttention(dim, num_heads, dropout)
        self.memory_bank = ClusterMemoryBank(
            dim,
            max_clusters=max_clusters,
            exemplars_per_cluster=exemplars_per_cluster,
            merge_threshold=merge_threshold,
            capacity_policy=capacity_policy,
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, write_to_memory=True, top_k_sim=4, top_k_recent=2, causal=True):
        B, T, D = x.shape
        attn_mask = None
        if causal:
            attn_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        sa_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + sa_out)

        x, mem_diag = self.memory_attn(x, self.memory_bank, top_k_sim, top_k_recent)

        if write_to_memory:
            with torch.no_grad():
                seg_repr = pool_segment(x, mode="mean")
                scores = token_importance_scores(x)
            self.memory_bank.write(seg_repr, x.detach(), scores)

        x = self.norm2(x + self.ffn(x))
        return x, mem_diag
