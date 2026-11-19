# -*- coding: utf-8 -*-
"""
AssemblyRegistry: 共激活 unit 集合(cell assembly)的在线登记器。

用户构想(2026-08-02 原话大意): 预训练时,输入 a 激活强度超过阈值(例:该位置最大
激活的 50%)的 unit 被统一登记为一组;训练产物 = LLM 权重 + unit 集合库;之后输入
b 的激活集合与已登记组按重合度匹配,超过门槛则并入/触发该组。百分比是示意,
具体函数形式是设计自由度。

本模块实现其中"登记"这一半(实验一只测组的真实性,补全机制留到实验二):

- 集合提取: 每个位置的激活向量,取 > act_frac * (该位置最大激活) 的 unit 为
  激活集合。用"占最大值的比例"而不是绝对阈值,是为了对训练过程中激活尺度的
  漂移保持不变性——用户选了"推理期也持续登记",登记规则必须全程自校准,
  不能依赖某个训练阶段校准出的绝对数值。
- 在线聚类(DP-means 风格,集合版): 新集合与所有活跃组的成员集合算 Jaccard
  重合度;最高重合 >= jaccard_new 则并入该组(组的 unit 隶属频率做 EMA 更新),
  否则开新组。组数达到上限 k_max 时,合并当前最相似的两个组腾位。
  这套"够近就并、不够就开新、满了就合并"的逻辑与本仓库 ClusterMemoryBank
  的 DP-means/EMA 机制同构,只是把"连续向量+余弦"换成了"二值集合+Jaccard"。
- 组的成员集合 = 隶属频率 > member_threshold 的 unit。频率是软的(EMA),
  成员集合是硬的(阈值化),检索/比较永远用硬集合,与用户"标记成一组"的
  离散语义一致。

设计约束:
- 全程无随机数(不碰 torch 全局 RNG),同一输入流得到同一登记结果,便于复现
  与单元测试。
- 训练期与推理期走同一条 register() 代码路径(用户明确选择在线登记,
  不做"训练完再离线分析"的特权版本)。只读评估用 assign()(不改状态)。

已知的 v1 简化(如实记录,不假装不存在):
- 批量路径里,同一批中"命中同一组"的多个集合按聚合后的均值做一次等效 EMA,
  与逐个顺序更新在数值上略有差异(顺序更新的后来者会看到前面更新过的频率),
  但两者的不动点相同;未命中的集合仍逐个顺序处理(先开的新组能接住同批里
  相似的后来者)。
- 容量满时按 count 加权合并最相似两组,两组重叠不高时合并后的隶属频率可能
  没有 unit 过 member_threshold(产生临时的空成员组),依赖后续写入把它拉回。
  实验里会记录这种情况的发生次数,不掩盖。
"""

import torch


class AssemblyRegistry:
    def __init__(
        self,
        n_units: int,
        k_max: int = 16,
        jaccard_new: float = 0.3,
        freq_lr: float = 0.05,
        member_threshold: float = 0.5,
        act_frac: float = 0.5,
    ):
        assert 0.0 < act_frac < 1.0
        assert 0.0 < jaccard_new < 1.0
        self.n_units = n_units
        self.K = k_max
        self.jaccard_new = jaccard_new
        self.freq_lr = freq_lr
        self.member_threshold = member_threshold
        self.act_frac = act_frac

        self.freq = torch.zeros(k_max, n_units)           # 每组 unit 的隶属频率(EMA)
        self.act_ema = torch.zeros(k_max, n_units)        # 每组 unit 的典型激活值(EMA,供补全注入用)
        self.active = torch.zeros(k_max, dtype=torch.bool)
        self.count = torch.zeros(k_max, dtype=torch.long)  # 每组累计并入的集合数
        self.created_at = torch.zeros(k_max, dtype=torch.long)
        self.step = 0                                      # register() 调用次数
        self.total_registered = 0                          # 累计登记的集合数
        self.merge_events = 0                              # 容量满触发的组合并次数
        self.empty_member_events = 0                       # 合并后成员集合为空的次数

    # ------------------------------------------------------------------ 集合提取

    def extract_sets(self, acts: torch.Tensor) -> torch.Tensor:
        """acts: [P, U] 激活值(如 FFN 的 GELU 输出)。返回 bool [P, U]。
        负激活视为 0(GELU 的负半轴幅度很小,不参与"高激活"语义)。"""
        a = acts.clamp_min(0.0)
        mx = a.max(dim=-1, keepdim=True).values
        sets = a > (self.act_frac * mx)
        sets = sets & (mx > 0)  # 全零激活的位置没有集合
        return sets

    # ------------------------------------------------------------------ 查询

    @property
    def membership(self) -> torch.Tensor:
        """bool [K, U]: 每组的硬成员集合。非活跃组恒为全 False。"""
        return (self.freq > self.member_threshold) & self.active.unsqueeze(-1)

    @property
    def n_groups(self) -> int:
        return int(self.active.sum())

    def jaccard_to_groups(self, sets: torch.Tensor) -> torch.Tensor:
        """sets: bool [P, U] -> Jaccard [P, K],非活跃组填 -1。"""
        mem = self.membership.float()  # [K, U]
        s = sets.float()
        inter = s @ mem.t()  # [P, K]
        union = s.sum(-1, keepdim=True) + mem.sum(-1).unsqueeze(0) - inter
        j = inter / union.clamp_min(1e-9)
        return j.masked_fill(~self.active.unsqueeze(0), -1.0)

    def assign(self, sets: torch.Tensor):
        """只读分配(不更新任何状态)。返回 (assign [P] long, best_j [P] float)。
        未命中任何组(或空集合)的位置为 -1。"""
        P = sets.shape[0]
        out = torch.full((P,), -1, dtype=torch.long)
        if self.n_groups == 0:
            return out, torch.full((P,), -1.0)
        j = self.jaccard_to_groups(sets)
        best_j, best_k = j.max(-1)
        hit = sets.any(-1) & (best_j >= self.jaccard_new)
        out[hit] = best_k[hit]
        return out, best_j

    # ------------------------------------------------------------------ 登记

    def register(self, sets: torch.Tensor, values: torch.Tensor = None) -> torch.Tensor:
        """在线登记一批集合。sets: bool [P, U]。返回每个位置的组下标 [P](空集合为 -1)。

        values: 可选 float [P, U],该位置的原始激活值。提供时,每组同步维护
        act_ema(组内 unit 的典型激活值,与 freq 用同一 EMA 节奏),供补全机制
        注入"该亮而未亮"的 unit 时取值。不提供时 act_ema 保持不变(向后兼容)。"""
        P = sets.shape[0]
        assign_out = torch.full((P,), -1, dtype=torch.long)
        nonempty = sets.any(-1)

        # 1) 批量路径: 对批开始时的组状态算命中
        if self.n_groups > 0:
            j = self.jaccard_to_groups(sets)
            best_j, best_k = j.max(-1)
            hit = nonempty & (best_j >= self.jaccard_new)
        else:
            hit = torch.zeros(P, dtype=torch.bool)
            best_k = torch.zeros(P, dtype=torch.long)

        for k in best_k[hit].unique().tolist():
            sel = hit & (best_k == k)
            n = int(sel.sum())
            mean_set = sets[sel].float().mean(dim=0)
            eff = 1.0 - (1.0 - self.freq_lr) ** n  # n 次逐个 EMA 的等效步长
            self.freq[k] = (1 - eff) * self.freq[k] + eff * mean_set
            if values is not None:
                self.act_ema[k] = (1 - eff) * self.act_ema[k] + eff * values[sel].mean(dim=0)
            self.count[k] += n
        assign_out[hit] = best_k[hit]

        # 2) 未命中的逐个处理(先开的新组能接住同批里相似的后来者)
        miss_idx = torch.nonzero(nonempty & ~hit).flatten().tolist()
        for p in miss_idx:
            s = sets[p]
            if self.n_groups > 0:
                jj = self.jaccard_to_groups(s.unsqueeze(0))[0]
                bj, bk = jj.max(0)
                if float(bj) >= self.jaccard_new:
                    k = int(bk)
                    self.freq[k] = (1 - self.freq_lr) * self.freq[k] + self.freq_lr * s.float()
                    if values is not None:
                        self.act_ema[k] = (1 - self.freq_lr) * self.act_ema[k] + self.freq_lr * values[p]
                    self.count[k] += 1
                    assign_out[p] = k
                    continue
            k = self._alloc_slot()
            self.freq[k] = s.float()
            if values is not None:
                self.act_ema[k] = values[p].clone()
            self.count[k] = 1
            self.active[k] = True
            self.created_at[k] = self.step
            assign_out[p] = k

        self.step += 1
        self.total_registered += int(nonempty.sum())
        return assign_out

    def _alloc_slot(self) -> int:
        free = torch.nonzero(~self.active).flatten()
        if len(free) > 0:
            return int(free[0])

        # 容量满: 合并成员集合 Jaccard 最高的两组,释放其中一个槽位
        mem = self.membership.float()
        inter = mem @ mem.t()
        sz = mem.sum(-1)
        union = sz.unsqueeze(0) + sz.unsqueeze(1) - inter
        j = inter / union.clamp_min(1e-9)
        j.fill_diagonal_(-1.0)
        flat = int(j.argmax())
        a, b = flat // self.K, flat % self.K

        ca, cb = int(self.count[a]), int(self.count[b])
        w = ca / max(1, ca + cb)
        self.freq[a] = w * self.freq[a] + (1 - w) * self.freq[b]
        self.act_ema[a] = w * self.act_ema[a] + (1 - w) * self.act_ema[b]
        self.count[a] = ca + cb
        self.freq[b].zero_()
        self.act_ema[b].zero_()
        self.active[b] = False
        self.count[b] = 0
        self.merge_events += 1
        if not bool((self.freq[a] > self.member_threshold).any()):
            self.empty_member_events += 1
        return int(b)

    # ------------------------------------------------------------------ 存取

    def state_dict_plain(self):
        return {
            "freq": self.freq, "act_ema": self.act_ema, "active": self.active, "count": self.count,
            "created_at": self.created_at, "step": self.step,
            "total_registered": self.total_registered,
            "merge_events": self.merge_events,
            "empty_member_events": self.empty_member_events,
            "config": {
                "n_units": self.n_units, "k_max": self.K,
                "jaccard_new": self.jaccard_new, "freq_lr": self.freq_lr,
                "member_threshold": self.member_threshold, "act_frac": self.act_frac,
            },
        }
