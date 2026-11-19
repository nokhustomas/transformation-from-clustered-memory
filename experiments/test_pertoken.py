# -*- coding: utf-8 -*-
"""
逐 token 判别式检索(retrieval_mode="pertoken")新增代码的最小验证脚本。
覆盖 test_cluster_memory.py 没有覆盖到的部分: RawTopKMemoryBank.retrieve_per_token()
以及 build_model("cma_pertoken"/"topkraw_pertoken") 端到端前向+反向。
在投入完整训练跑之前,先用这个脚本确认新代码形状正确、数值有限、梯度能回传。
"""
import torch
import torch.nn.functional as F

from models import RawTopKMemoryBank, build_model
from mqar_data import make_vocab, generate_batch

torch.manual_seed(0)


def test_rawtopk_retrieve_per_token_selectivity():
    print("== 测试 A: RawTopKMemoryBank.retrieve_per_token 逐 token 判别能力 ==")
    D, B, C = 16, 2, 4
    bank = RawTopKMemoryBank(dim=D, capacity=C)
    bank.reset(B, torch.device("cpu"))

    # 写入 4 个方向差异很大的 token,容量刚好等于 4,不会触发替换
    topic_bases = F.normalize(torch.randn(C, D), dim=-1)
    for k in range(C):
        tok = topic_bases[k].unsqueeze(0).expand(B, 1, D).clone()
        bank.write(tok.mean(dim=1), tok, torch.ones(B, 1))
    assert bank.slot_mask.all(), "4 次写入、容量4,应该刚好写满且不发生替换"

    T = C
    query = topic_bases.unsqueeze(0).expand(B, T, D).clone()
    gathered_kv, gathered_mask, sel = bank.retrieve_per_token(query, top_k_sim=1)
    assert gathered_kv.shape == (B, T, 1, D)
    assert gathered_mask.shape == (B, T, 1)
    assert not torch.isnan(gathered_kv).any()

    sel_per_token = sel[0, :, 0]
    assert len(set(sel_per_token.tolist())) == T, (
        f"逐 token 检索应为每个 query 位置选出不同的槽位,实际 {sel_per_token.tolist()}"
    )
    print(f"  4 个不同话题的 query token 分别选中槽位 {sel_per_token.tolist()}(应各不相同)")

    # 对照: 旧的 retrieve() 完全不看 query,始终返回整个缓冲区
    old_kv, old_mask, old_sel = bank.retrieve(query.mean(dim=1))
    assert old_sel is None and old_kv.shape == (B, C, D), "旧 retrieve() 应无条件返回整个缓冲区,不做任何筛选"
    print(f"  作为对照,旧 retrieve() 无条件返回全部 {old_kv.shape[1]} 个槽位,不看 query")
    print("  [PASS] retrieve_per_token 具备逐 token 判别能力,旧 retrieve 不具备\n")


def test_pertoken_models_forward_backward():
    print("== 测试 B: build_model(\"cma_pertoken\"/\"topkraw_pertoken\") 端到端前向+反向 ==")
    vocab = make_vocab(key_vocab=16, value_vocab=16, filler_vocab=32)
    DIM, LAYERS, HEADS, CHUNK, SEQ_LEN, NUM_PAIRS = 64, 2, 4, 32, 192, 4
    gen = torch.Generator().manual_seed(1)
    ids, pos, tgt = generate_batch(4, SEQ_LEN, CHUNK, NUM_PAIRS, vocab, generator=gen)

    for condition, budget in [("topkraw_pertoken", 4), ("cma_pertoken", 4),
                               ("topkraw_pertoken", 8), ("cma_pertoken", 8)]:
        torch.manual_seed(0)
        model = build_model(condition, vocab["vocab_size"], DIM, LAYERS, HEADS, CHUNK, SEQ_LEN, budget=budget)
        logits = model(ids)
        assert logits.shape == (4, SEQ_LEN, vocab["vocab_size"]), f"{condition}: 输出形状不对 {logits.shape}"
        assert torch.isfinite(logits).all(), f"{condition}: 输出包含 NaN/Inf"

        B = ids.shape[0]
        b_idx = torch.arange(B).unsqueeze(1).expand(B, NUM_PAIRS)
        sel_logits = logits[b_idx, pos]
        loss = F.cross_entropy(sel_logits.reshape(-1, sel_logits.shape[-1]), tgt.reshape(-1))
        loss.backward()
        has_grad = any(p.grad is not None and torch.isfinite(p.grad).all()
                        for p in model.parameters() if p.requires_grad)
        assert has_grad, f"{condition}: 至少应有部分参数收到梯度"
        print(f"  {condition:18s} budget={budget:2d}  loss={loss.item():.4f}  "
              f"输出形状={tuple(logits.shape)}  梯度回传正常")
    print("  [PASS] 4 组 (condition, budget) 组合均前向数值有限、反向梯度可回传\n")


if __name__ == "__main__":
    test_rawtopk_retrieve_per_token_selectivity()
    test_pertoken_models_forward_backward()
    print("全部测试通过 ✅")
