# -*- coding: utf-8 -*-
"""AssemblyRegistry 单元测试(脚本式 assert,与仓库其他测试同风格,直接 python3 运行)。"""

import torch

from registry import AssemblyRegistry


def mkset(n_units, idxs):
    s = torch.zeros(n_units, dtype=torch.bool)
    s[list(idxs)] = True
    return s


def test_disjoint_sets_make_two_groups():
    print("== 测试 1: 两个不相交集合 -> 各开一组 ==")
    reg = AssemblyRegistry(n_units=32, k_max=8, jaccard_new=0.3)
    a = mkset(32, range(0, 8))
    b = mkset(32, range(16, 24))
    r1 = reg.register(a.unsqueeze(0))
    r2 = reg.register(b.unsqueeze(0))
    assert reg.n_groups == 2, f"应为 2 组,实际 {reg.n_groups}"
    assert int(r1) != int(r2)
    print(f"  [PASS] n_groups={reg.n_groups}, 分配 {int(r1)} vs {int(r2)}")


def test_identical_sets_merge():
    print("== 测试 2: 相同集合重复登记 -> 并入同一组,count 累计,频率巩固 ==")
    reg = AssemblyRegistry(n_units=32, k_max=8, jaccard_new=0.3)
    a = mkset(32, range(0, 8))
    ks = [int(reg.register(a.unsqueeze(0))) for _ in range(5)]
    assert reg.n_groups == 1, f"应为 1 组,实际 {reg.n_groups}"
    assert len(set(ks)) == 1
    k = ks[0]
    assert int(reg.count[k]) == 5
    assert bool((reg.freq[k][:8] > 0.99).all()), "重复并入后频率应趋近 1"
    print(f"  [PASS] 全部并入组 {k}, count={int(reg.count[k])}")


def test_partial_overlap_merges_when_above_threshold():
    print("== 测试 3: 重合度高于/低于门槛 -> 并组/开新组 ==")
    reg = AssemblyRegistry(n_units=32, k_max=8, jaccard_new=0.3)
    a = mkset(32, range(0, 10))
    b = mkset(32, range(2, 12))   # 与 a 的 Jaccard = 8/12 ≈ 0.67 > 0.3 -> 并
    c = mkset(32, range(20, 30))  # 与 a 的 Jaccard = 0 -> 新组
    reg.register(a.unsqueeze(0))
    kb = int(reg.register(b.unsqueeze(0)))
    kc = int(reg.register(c.unsqueeze(0)))
    assert reg.n_groups == 2, f"应为 2 组,实际 {reg.n_groups}"
    assert kb == 0 and kc != 0
    print(f"  [PASS] b 并入组 {kb}, c 开新组 {kc}")


def test_capacity_merge_most_similar():
    print("== 测试 4: 容量满时合并最相似的两组 ==")
    reg = AssemblyRegistry(n_units=64, k_max=3, jaccard_new=0.5)
    g1 = mkset(64, range(0, 10))
    g2 = mkset(64, range(8, 18))    # 与 g1 Jaccard = 2/18 ≈ 0.11 (最相似的一对)
    g3 = mkset(64, range(40, 50))   # 与 g1/g2 都不相交
    g4 = mkset(64, range(52, 62))   # 新来的,与谁都不相交 -> 触发合并腾位
    for g in (g1, g2, g3):
        reg.register(g.unsqueeze(0))
    assert reg.n_groups == 3
    reg.register(g4.unsqueeze(0))
    assert reg.n_groups == 3, "合并一对 + 新开一组,总数应仍为上限 3"
    assert reg.merge_events == 1
    # 被合并的应是 g1/g2 那一对: 40-50 与 52-62 两组的成员集合应完整存活
    mems = reg.membership
    survive_g3 = any(bool(mems[k][40:50].all()) for k in range(3) if reg.active[k])
    survive_g4 = any(bool(mems[k][52:62].all()) for k in range(3) if reg.active[k])
    assert survive_g3 and survive_g4, "不相交的 g3/g4 不应被选中合并"
    print(f"  [PASS] merge_events={reg.merge_events}, g3/g4 完整存活")


def test_batch_path_consistent_with_sequential():
    print("== 测试 5: 批量登记与逐个登记,分配结果一致 ==")
    n_units = 48
    base_a = list(range(0, 12))
    base_b = list(range(24, 36))
    batch = torch.stack([
        mkset(n_units, base_a),
        mkset(n_units, base_b),
        mkset(n_units, base_a[:11] + [12]),   # 接近 a
        mkset(n_units, base_b[:11] + [37]),   # 接近 b
    ])
    reg1 = AssemblyRegistry(n_units, k_max=8, jaccard_new=0.3)
    reg1.register(torch.stack([mkset(n_units, base_a), mkset(n_units, base_b)]))
    out_batch = reg1.register(batch)

    reg2 = AssemblyRegistry(n_units, k_max=8, jaccard_new=0.3)
    reg2.register(torch.stack([mkset(n_units, base_a), mkset(n_units, base_b)]))
    out_seq = torch.cat([reg2.register(batch[i].unsqueeze(0)) for i in range(4)])

    assert torch.equal(out_batch, out_seq), f"{out_batch} vs {out_seq}"
    assert reg1.n_groups == reg2.n_groups == 2
    print(f"  [PASS] 分配一致: {out_batch.tolist()}, n_groups={reg1.n_groups}")


def test_extract_sets_scale_invariant():
    print("== 测试 6: 集合提取对激活尺度缩放不变;空激活无集合 ==")
    reg = AssemblyRegistry(n_units=16, k_max=4)
    acts = torch.tensor([[0.0, 1.0, 0.6, 0.4, 0.51] + [0.0] * 11])
    s1 = reg.extract_sets(acts)
    s2 = reg.extract_sets(acts * 37.5)
    assert torch.equal(s1, s2), "乘以常数不应改变集合"
    assert s1[0].tolist()[:5] == [False, True, True, False, True]
    z = reg.extract_sets(torch.zeros(1, 16))
    assert not bool(z.any()), "全零激活应得到空集合"
    r = reg.register(z)
    assert int(r) == -1 and reg.n_groups == 0, "空集合不登记"
    print("  [PASS] 尺度不变、阈值边界正确(>50% 严格大于)、空集合被跳过")


def test_assign_readonly():
    print("== 测试 7: assign() 只读,不改状态 ==")
    reg = AssemblyRegistry(n_units=32, k_max=8, jaccard_new=0.3)
    reg.register(mkset(32, range(0, 8)).unsqueeze(0))
    before = reg.state_dict_plain()
    freq_before = before["freq"].clone()
    a, j = reg.assign(torch.stack([mkset(32, range(0, 8)), mkset(32, range(20, 28))]))
    assert int(a[0]) == 0 and int(a[1]) == -1
    assert torch.equal(reg.freq, freq_before) and reg.n_groups == 1
    assert reg.total_registered == 1
    print(f"  [PASS] 命中組 {int(a[0])}, 未命中返回 -1, 状态未变")


def test_value_ema_tracking():
    print("== 测试 8: act_ema 典型激活值随登记同步更新;不传 values 时保持不变 ==")
    reg = AssemblyRegistry(n_units=8, k_max=4, jaccard_new=0.3, freq_lr=0.5)
    s = mkset(8, [0, 1, 2]).unsqueeze(0)
    v1 = torch.tensor([[2.0, 4.0, 6.0, 0, 0, 0, 0, 0]])
    v2 = torch.tensor([[4.0, 8.0, 10.0, 0, 0, 0, 0, 0]])
    reg.register(s, values=v1)
    assert torch.allclose(reg.act_ema[0], v1[0]), "首次登记应直接取值"
    reg.register(s, values=v2)
    expect = 0.5 * v1[0] + 0.5 * v2[0]  # freq_lr=0.5 的 EMA
    assert torch.allclose(reg.act_ema[0], expect), f"{reg.act_ema[0]} vs {expect}"
    before = reg.act_ema[0].clone()
    reg.register(s)  # 不传 values
    assert torch.allclose(reg.act_ema[0], before), "不传 values 不应改动 act_ema"
    print(f"  [PASS] act_ema={reg.act_ema[0].tolist()[:3]} 符合 EMA 预期,向后兼容")


if __name__ == "__main__":
    test_disjoint_sets_make_two_groups()
    test_identical_sets_merge()
    test_partial_overlap_merges_when_above_threshold()
    test_capacity_merge_most_similar()
    test_batch_path_consistent_with_sequential()
    test_extract_sets_scale_invariant()
    test_assign_readonly()
    test_value_ema_tracking()
    print("\n全部测试通过 ✅")
