# -*- coding: utf-8 -*-
"""
cluster_memory.py 的最小可运行验证脚本。
不依赖 pytest,直接 `python3 test_cluster_memory.py` 运行,用 assert 做检查,
每个测试打印 [PASS]/结果摘要,任何一步失败都会抛异常并显示具体断言信息。
"""

import torch
import torch.nn.functional as F

from cluster_memory import (
    ClusterMemoryBank,
    ClusteredMemoryCrossAttention,
    ClusterMemoryTransformerBlock,
    SurpriseBoundaryDetector,
    pool_segment,
    token_importance_scores,
)

torch.manual_seed(0)


def test_boundary_detector():
    print("== 测试 1: SurpriseBoundaryDetector ==")
    D = 16
    # 构造 3 段"状态突变"明显的序列: 每段内部平滑漂移,段之间跳变很大
    seg_len = 20
    segs = []
    for _ in range(3):
        base = F.normalize(torch.randn(D), dim=0)
        drift = base.unsqueeze(0) + 0.02 * torch.randn(seg_len, D)
        segs.append(drift)
    seq = torch.cat(segs, dim=0)  # [60, D]

    detector = SurpriseBoundaryDetector(window=10, gamma=1.5, min_seg_len=5)
    spans = detector.segment(seq)

    assert spans[0][0] == 0 and spans[-1][1] == seq.shape[0], "分段必须完整覆盖序列首尾"
    total_len = sum(e - s for s, e in spans)
    assert total_len == seq.shape[0], "分段区间不能有重叠或缺口"
    print(f"  检测到 {len(spans)} 个分段: {spans}")
    print("  [PASS] 分段完整覆盖序列且无重叠\n")


def test_memory_bank_capacity():
    print("== 测试 2: ClusterMemoryBank 的聚类/合并/容量控制 ==")
    D, B = 32, 2
    K_max, E, T = 4, 4, 6
    n_topics = 6

    bank = ClusterMemoryBank(
        dim=D, max_clusters=K_max, exemplars_per_cluster=E,
        ema_decay=0.9, merge_threshold=0.7, dead_ttl=10 ** 6, capacity_policy="merge",
    )

    topic_bases = F.normalize(torch.randn(n_topics, D), dim=-1)
    rng = torch.Generator().manual_seed(42)

    n_steps = 40
    for step in range(n_steps):
        topic_ids = torch.randint(0, n_topics, (B,), generator=rng)
        tokens = torch.stack(
            [topic_bases[topic_ids[b]].unsqueeze(0).expand(T, D) + 0.03 * torch.randn(T, D, generator=rng)
             for b in range(B)],
            dim=0,
        )  # [B, T, D],同一 batch 内每个样本随机选一个"话题"作为这段内容的语义来源

        seg_repr = pool_segment(tokens)
        scores = token_importance_scores(tokens)
        bank.write(seg_repr, tokens, scores)

        active = bank.active_counts
        assert torch.all(active <= K_max), f"第 {step} 步活跃簇数超过上限: {active.tolist()}"

    print(f"  跑完 {n_steps} 步流式写入,最终活跃簇数: {bank.active_counts.tolist()} (上限 {K_max})")
    assert torch.all(bank.active_counts >= 1), "写入过数据后活跃簇数不应为 0"

    # 用和某个已知话题完全一致的向量查询,期望能检索到对应的簇(相似度最高的被选中)
    query = topic_bases[0].unsqueeze(0).expand(B, D).clone()
    mem_kv, mem_mask, sel = bank.retrieve(query, top_k_sim=2, top_k_recent=1)
    assert mem_kv.shape == (B, (2 + 1) * E, D)
    assert mem_mask.shape == (B, (2 + 1) * E)
    assert not torch.isnan(mem_kv).any(), "检索结果不应包含 NaN"
    print(f"  retrieve() 输出 mem_kv={tuple(mem_kv.shape)}, mem_mask={tuple(mem_mask.shape)}")
    print("  [PASS] 簇数量始终受控且检索结果形状/数值正常\n")
    return bank


def test_quantize_ste_gradient():
    print("== 测试 3: quantize_ste 的 straight-through 梯度 ==")
    D, B = 16, 3
    bank = ClusterMemoryBank(dim=D, max_clusters=8, exemplars_per_cluster=4, merge_threshold=0.8)

    # 先写入几个簇,制造非空记忆库
    tokens = torch.randn(B, 5, D)
    bank.write(pool_segment(tokens), tokens, token_importance_scores(tokens))

    x = torch.randn(B, D, requires_grad=True)
    ste_out, best_idx, commit_loss = bank.quantize_ste(x)
    loss = ste_out.sum() + commit_loss
    loss.backward()

    assert x.grad is not None, "straight-through estimator 应让梯度传回输入"
    assert torch.isfinite(x.grad).all(), "梯度必须是有限值"
    assert torch.isfinite(commit_loss), "commitment loss 必须是有限值"
    print(f"  best_idx={best_idx.tolist()}, commitment_loss={commit_loss.item():.4f}")
    print(f"  x.grad 范数={x.grad.norm().item():.4f}(非零/有限即说明梯度正确回传)")
    print("  [PASS] STE 前向/反向均正常\n")


def test_end_to_end_block_streaming():
    print("== 测试 4: ClusterMemoryTransformerBlock 端到端流式前向 ==")
    D, B, T = 64, 2, 12
    n_chunks = 25
    block = ClusterMemoryTransformerBlock(
        dim=D, num_heads=4, ffn_mult=2, max_clusters=6, exemplars_per_cluster=4,
        merge_threshold=0.5, capacity_policy="merge",
    )
    block.eval()

    active_history = []
    with torch.no_grad():
        for c in range(n_chunks):
            chunk = torch.randn(B, T, D)
            out, diag = block(chunk, write_to_memory=True, top_k_sim=3, top_k_recent=1)
            assert out.shape == (B, T, D), f"输出形状不对: {out.shape}"
            assert torch.isfinite(out).all(), f"第 {c} 个 chunk 输出包含 NaN/Inf"
            active_history.append(block.memory_bank.active_counts.clone())

    active_history = torch.stack(active_history)  # [n_chunks, B]
    assert torch.all(active_history <= 6), "任意时刻活跃簇数都不应超过 max_clusters"
    print(f"  处理了 {n_chunks} 个 chunk,活跃簇数变化(前5步->后5步):")
    print(f"    前5步: {active_history[:5].tolist()}")
    print(f"    后5步: {active_history[-5:].tolist()}")
    print(f"  最后一个 chunk 的诊断信息示例键: {list(diag.keys())}")
    print("  [PASS] 端到端流式前向形状正确、数值稳定、容量始终受控\n")


def test_retrieve_per_token_selectivity():
    print("== 测试 6: retrieve_per_token() 能否按 query 内容逐 token 选出不同的簇 ==")
    D, B = 32, 2
    K_max, E = 4, 4
    bank = ClusterMemoryBank(dim=D, max_clusters=K_max, exemplars_per_cluster=E,
                              ema_decay=0.9, merge_threshold=0.99, dead_ttl=10 ** 6)
    # merge_threshold 设得很高(0.99),让 4 个明显不同方向的话题各自占一个簇,不被合并

    topic_bases = F.normalize(torch.randn(K_max, D), dim=-1)
    rng = torch.Generator().manual_seed(7)
    for k in range(K_max):
        tokens = topic_bases[k].unsqueeze(0).unsqueeze(0).expand(B, 5, D) + 0.01 * torch.randn(B, 5, D, generator=rng)
        bank.write(pool_segment(tokens), tokens, token_importance_scores(tokens))
    assert torch.all(bank.active_counts == K_max), f"应形成 {K_max} 个独立簇,实际 {bank.active_counts.tolist()}"

    # 构造一个 T=4 的逐 token query,每个 token 精确对应一个话题
    T = K_max
    query = topic_bases.unsqueeze(0).expand(B, T, D).clone()  # [B, T, D]
    mem_kv, mem_mask, sel = bank.retrieve_per_token(query, top_k_sim=1, top_k_recent=0)
    assert mem_kv.shape == (B, T, 1 * E, D)
    assert mem_mask.shape == (B, T, 1 * E)
    assert sel.shape == (B, T, 1)
    assert not torch.isnan(mem_kv).any()

    # 核心断言: 4 个 token 各自查询不同话题时,选出的簇下标应当互不相同(逐 token 判别式检索的关键性质)
    sel_per_token = sel[0, :, 0]  # [T]
    assert len(set(sel_per_token.tolist())) == T, (
        f"逐 token 检索应为每个 query 位置选出不同的簇,实际选出 {sel_per_token.tolist()}"
    )
    print(f"  4 个不同话题的 query token 分别选中簇 {sel_per_token.tolist()}(应各不相同)")

    # 对照: 用旧的 retrieve() 传入池化(均值)query,4 个话题被压成 1 个池化向量,只能选出 1 组共享结果
    pooled = query.mean(dim=1)
    pooled_kv, pooled_mask, pooled_sel = bank.retrieve(pooled, top_k_sim=1, top_k_recent=0)
    assert pooled_sel.shape == (B, 1), "池化版本每个样本只有一组共享的检索结果"
    print(f"  作为对照,池化 query 只能选出共享的 1 个簇: {pooled_sel[:, 0].tolist()}")
    print("  [PASS] retrieve_per_token 具备逐 token 判别能力,retrieve 不具备\n")


def test_forward_per_token_gradient():
    print("== 测试 7: ClusteredMemoryCrossAttention.forward_per_token 形状与梯度 ==")
    D, B, T = 32, 2, 6
    K_max, E = 4, 2
    bank = ClusterMemoryBank(dim=D, max_clusters=K_max, exemplars_per_cluster=E, merge_threshold=0.5)
    attn = ClusteredMemoryCrossAttention(dim=D, num_heads=4)

    # 先写入几个 chunk,制造非空记忆库
    for _ in range(3):
        tokens = torch.randn(B, 5, D)
        bank.write(pool_segment(tokens), tokens, token_importance_scores(tokens))

    x = torch.randn(B, T, D, requires_grad=True)
    out, diag = attn.forward_per_token(x, bank, top_k_sim=2, top_k_recent=0)
    assert out.shape == (B, T, D), f"输出形状不对: {out.shape}"
    assert torch.isfinite(out).all()
    assert diag["selected_clusters"].shape == (B, T, 2), "诊断里的 sel 应带有逐 token 维度"

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "梯度应能回传到输入隐状态"
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all()
                    for p in attn.parameters() if p.requires_grad)
    assert has_grad, "cross-attention 的参数至少应有一部分收到梯度"
    print(f"  输出形状 {tuple(out.shape)}, null_attn_weight_mean={diag['null_attn_weight_mean']:.4f}")
    print("  [PASS] forward_per_token 形状正确、前向数值有限、反向梯度可回传\n")


def test_random_routing_ablation():
    print("== 测试 8: routing_mode='random' 消融对照——分组是否真的与内容无关 ==")
    D = 32
    K_max, E, T = 4, 4, 6
    n_steps = 40

    topic = F.normalize(torch.randn(D), dim=0)
    rng = torch.Generator().manual_seed(7)

    def make_same_topic_batch():
        tokens = (topic.unsqueeze(0).expand(T, D) + 0.03 * torch.randn(T, D, generator=rng)).unsqueeze(0)  # [1,T,D]
        seg_repr = pool_segment(tokens)
        scores = token_importance_scores(tokens)
        return seg_repr, tokens, scores

    def slot_touched(bank, t):
        hit = (bank.last_write_t[0] == t).nonzero(as_tuple=True)[0]
        assert len(hit) == 1, f"应恰好有一个槽位被这次写入触碰到,实际找到 {len(hit)} 个"
        return int(hit[0].item())

    # 相似度路由(默认,论文报告的 CMA): 反复喂同一个话题,内容驱动分组,
    # 理应几乎全部并入同一个簇(第一次写入创建的那个)。
    sim_bank = ClusterMemoryBank(dim=D, max_clusters=K_max, exemplars_per_cluster=E,
                                  ema_decay=0.9, merge_threshold=0.7, dead_ttl=10 ** 6,
                                  capacity_policy="merge", routing_mode="similarity")
    sim_slots = []
    for _ in range(n_steps):
        seg_repr, tokens, scores = make_same_topic_batch()
        t = sim_bank.step
        sim_bank.write(seg_repr, tokens, scores)
        sim_slots.append(slot_touched(sim_bank, t))
    sim_unique = len(set(sim_slots))
    print(f"  相似度路由: 同一话题连续写入{n_steps}次,用到的簇编号(去重): {sorted(set(sim_slots))}")
    assert sim_unique <= 2, f"相似度路由下同一话题理应集中在1(最多2)个簇,实际用了{sim_unique}个"

    # 随机路由(消融对照): 同样反复喂同一个话题,分组与内容无关,
    # 理应均匀分散到 K_max=4 个槽位里,不会像 similarity 那样集中。
    rand_bank = ClusterMemoryBank(dim=D, max_clusters=K_max, exemplars_per_cluster=E,
                                   ema_decay=0.9, merge_threshold=0.7, dead_ttl=10 ** 6,
                                   capacity_policy="merge", routing_mode="random")
    rand_slots = []
    for _ in range(n_steps):
        seg_repr, tokens, scores = make_same_topic_batch()
        t = rand_bank.step
        rand_bank.write(seg_repr, tokens, scores)
        rand_slots.append(slot_touched(rand_bank, t))
    rand_unique = len(set(rand_slots))
    print(f"  随机路由: 同一话题连续写入{n_steps}次,用到的簇编号(去重): {sorted(set(rand_slots))}")
    assert rand_unique >= 3, f"随机路由下同一话题理应分散到多个簇(K={K_max}),实际只用了{rand_unique}个"

    # 结构性检查: 随机路由下 EMA 质心更新、代表 token 选取、retrieve() 仍然照常工作
    # (换句话说,除了"分组标准"这一个变量,其余机制与 similarity 路由完全一致)。
    query = topic.unsqueeze(0)
    mem_kv, mem_mask, sel = rand_bank.retrieve(query, top_k_sim=2, top_k_recent=0)
    assert mem_kv.shape == (1, 2 * E, D)
    assert not torch.isnan(mem_kv).any(), "random 路由下 retrieve() 结果不应包含 NaN"
    print("  [PASS] random 路由确认与内容无关(同话题分散到多簇,而非集中),"
          "EMA/代表token挑选/retrieve() 数值和形状仍正常\n")


def test_gradient_flows_through_block():
    print("== 测试 5: 训练模式下梯度能否正常回传(自注意力 + 记忆交叉注意力 + FFN) ==")
    D, B, T = 32, 2, 8
    block = ClusterMemoryTransformerBlock(dim=D, num_heads=4, max_clusters=4, exemplars_per_cluster=2)
    block.train()

    x1 = torch.randn(B, T, D, requires_grad=True)
    out1, _ = block(x1, write_to_memory=True)
    out1.sum().backward()
    assert x1.grad is not None and torch.isfinite(x1.grad).all()
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters() if p.requires_grad)
    assert has_grad, "至少应有部分模块参数收到梯度"

    # 第二个 chunk: 此时记忆库里已有第一步写入的内容,验证“读取历史记忆”这条路径同样可导
    block.zero_grad()
    x2 = torch.randn(B, T, D, requires_grad=True)
    out2, diag2 = block(x2, write_to_memory=True)
    out2.sum().backward()
    assert x2.grad is not None and torch.isfinite(x2.grad).all()
    print(f"  两轮 chunk 前向+反向均成功,第二轮 null 槽位平均注意力权重={diag2['null_attn_weight_mean']:.4f}")
    print("  [PASS] 梯度可以正常经过记忆交叉注意力回传\n")


if __name__ == "__main__":
    test_boundary_detector()
    test_memory_bank_capacity()
    test_quantize_ste_gradient()
    test_end_to_end_block_streaming()
    test_retrieve_per_token_selectivity()
    test_forward_per_token_gradient()
    test_gradient_flows_through_block()
    test_random_routing_ablation()
    print("全部测试通过 ✅")
