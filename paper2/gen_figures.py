# -*- coding: utf-8 -*-
"""论文二图表生成。全部数字直接读实验结果文件,不手抄。
配色遵循 dataviz 技能的已验证分类色序: slot1 蓝 #2a78d6, slot2 橙 #eb6834,
slot3 青绿 #1baf7a;顺序固定不轮换;单轴;细线;文字用近黑。"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2 = "#0b0b0b", "#52514e"
GRID = "#e6e6e3"

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "assembly", "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.8, "figure.dpi": 150,
})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Fig 1: 涌现(合成, 3种子)
def fig_emergence():
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for si, s in enumerate((0, 1, 2)):
        suf = "" if s == 0 else f"_seed{s}"
        rows = [json.loads(l) for l in open(os.path.join(RES, f"expt1_emergence{suf}.jsonl"))]
        ck = sorted([r for r in rows if "step" in r], key=lambda r: r["step"])
        steps = [r["step"] for r in ck]
        same = [r["L1"]["raw_jaccard"]["same_class_diff_token"] for r in ck]
        cross = [r["L1"]["raw_jaccard"]["cross_class"] for r in ck]
        ax.plot(steps, same, color=C1, lw=1.6 if s == 0 else 1.0,
                alpha=1.0 if s == 0 else 0.45)
        ax.plot(steps, cross, color=C2, lw=1.6 if s == 0 else 1.0,
                alpha=1.0 if s == 0 else 0.45)
    ax.annotate("same class,\ndifferent token", xy=(3000, 0.84), xytext=(1750, 0.62),
                color=C1, fontsize=9, ha="center")
    ax.annotate("cross class", xy=(3000, 0.06), xytext=(2250, 0.17), color=C2, fontsize=9, ha="center")
    ax.set_xlabel("training step")
    ax.set_ylabel("mean pairwise Jaccard of\nactivation sets (layer 2)")
    ax.set_ylim(0, 1.0)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_emergence.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2: 真实语言容量
def fig_realtext_capacity():
    """容量律: 直接读实验6b的扫描结果, 不手抄。"""
    rows = json.load(open(os.path.join(RES, "expt6b_capacity_sweep.json")))["rows"]

    def series(tau, af):
        r = sorted((x for x in rows if x["tau"] == tau and x["act_frac"] == af),
                   key=lambda x: x["K_max"])
        return [x["K_max"] for x in r], [x["coverage"] for x in r], [x["n_groups"] for x in r]

    k2, c2, g2 = series(0.2, 0.5)
    k3, c3, _ = series(0.3, 0.5)
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(k2, c2, color=C1, lw=1.6, marker="o", ms=4, label="τ = 0.2")
    ax.plot(k3, c3, color=C2, lw=1.6, marker="o", ms=4, label="τ = 0.3")
    sat = next(i for i, (k, g) in enumerate(zip(k2, g2)) if g < k)
    ax.annotate(f"only {g2[sat]} groups\never created", (k2[sat], c2[sat]),
                textcoords="offset points", xytext=(-14, -34), ha="center",
                fontsize=7.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(k2, [str(k) for k in k2])
    ax.set_xlabel("registry capacity K")
    ax.set_ylabel("fraction of positions\nmatched to a group")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_realtext_capacity.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3: 修复率(合成vs真实)
def fig_repair():
    syn = json.load(open(os.path.join(RES, "expt3_stress_seed0.json")))["A_corruption"]
    real = json.load(open(os.path.join(RES, "expt8_realrepair_seed0.json")))
    drops = [0.3, 0.5, 0.7]
    syn_rec = []
    for d in drops:
        a = syn[str(d)]["buckets"]["all"]
        syn_rec.append((a["delta_corrupt_off"] - a["delta_corrupt_on"]) / a["delta_corrupt_off"] * 100)
    real_rec = [real[str(d)]["recovery_frac"] * 100 for d in (0.3, 0.5)]

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(drops, syn_rec, color=C1, lw=1.6, marker="o", ms=4, label="synthetic (8 groups)")
    ax.plot([0.3, 0.5], real_rec, color=C2, lw=1.6, marker="o", ms=4, label="real text (512 groups)")
    for x, y in zip(drops, syn_rec):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color=INK2)
    for x, y in zip([0.3, 0.5], real_rec):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, -12),
                    ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("fraction of active units zeroed")
    ax.set_ylabel("loss damage recovered\nby completion (%)")
    ax.set_xticks(drops)
    ax.set_ylim(-5, 100)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_repair.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4: 检索-需求交叉
def fig_crossover():
    """需求-可靠性交叉: 直接读实验7b按词频档位的组分配准确率, 不手抄。"""
    d = json.load(open(os.path.join(RES, "expt7b_retrieval_by_rank.json")))
    # rank 7 ≈ 11 次出现, rank 6 ≈ 77 次, rank 0 是最高频成员
    pick = {7: 11, 6: 77, 0: 130000}
    rows = sorted(((pick[r["rank"]], r) for r in d["rows"] if r["rank"] in pick))
    occ = [o for o, _ in rows]
    acc = [r["correct_group_acc_among_assigned"] for _, r in rows]
    chance = d["chance_level"]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(occ, acc, color=C1, lw=1.6, marker="o", ms=5)
    ax.axhline(chance, color=INK2, lw=0.8, ls=":")
    ax.annotate("chance (8 classes)", (13, chance), textcoords="offset points",
                xytext=(0, 4), fontsize=7.5, color=INK2)
    for x, y in zip(occ, acc):
        ax.annotate(f"{y*100:.0f}%" if y < 0.999 else "~100%", (x, y),
                    textcoords="offset points", xytext=(14, -4),
                    ha="left", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(occ, ["11", "77", "~130k"])
    ax.set_xlabel("training occurrences of the token")
    ax.set_ylabel("correct-group\nretrieval rate")
    ax.set_ylim(0, 1.1)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_crossover.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5: 哨兵两通道读数
def fig_sentinel():
    v = json.load(open(os.path.join(RES, "expt11b_valuescore.json")))
    order = [("drop30", "zero 30% of\nactive units"), ("noise0.5x", "Gaussian noise\n0.5×std"),
             ("stuck10%", "10% units\nstuck at 0"), ("quant4级", "4-level\nquantization"),
             ("quant8级", "8-level\nquantization")]
    ys = range(len(order))
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for yi, (key, _) in zip(ys, order):
        s, val = v[key]["set"][0], v[key]["val"][0]
        ax.plot([s, val], [yi, yi], color=GRID, lw=1.2, zorder=1)
        ax.scatter([s], [yi], color=C1, s=28, zorder=2)
        ax.scatter([val], [yi], color=C2, s=28, zorder=2)
    ax.scatter([], [], color=C1, s=28, label="which-units reading (ρ)")
    ax.scatter([], [], color=C2, s=28, label="value-mismatch reading")
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    ax.annotate("chance", (0.5, len(order) - 0.4), fontsize=7.5, color=INK2, ha="center",
                textcoords="offset points", xytext=(0, 4))
    ax.set_yticks(list(ys), [lbl for _, lbl in order], fontsize=8)
    ax.set_xlabel("fault-detection AUC (400 vs 400 sequences)")
    ax.set_xlim(0.45, 1.03)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.02, 0.72))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sentinel.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6: 定位矩阵
def fig_localization():
    h = json.load(open(os.path.join(RES, "expt11_harden.json")))
    loc = h["seedA"]["localization"]
    m = [[loc["故障在L0_检测器L0"]["auc"], loc["故障在L0_检测器L1"]["auc"]],
         [loc["故障在L1_检测器L0"]["auc"], loc["故障在L1_检测器L1"]["auc"]]]
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    im = ax.imshow(m, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", ["#cde2fb", "#0d366b"]), vmin=0.5, vmax=1.0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i][j]:.3f}", ha="center", va="center",
                    color="white" if m[i][j] > 0.8 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["detector\nat layer 1", "detector\nat layer 2"], fontsize=8)
    ax.set_yticks([0, 1], ["fault at\nlayer 1", "fault at\nlayer 2"], fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("AUC", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_localization.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 7: 载体(状态 vs 变换)
def fig_carrier():
    """左: 插值扫描(增量是合法通道)。右: 存什么(状态 vs 变换), 两个尺度。"""
    itp = json.load(open(os.path.join(RES, "expt19_transform.json")))
    iv = json.load(open(os.path.join(RES, "expt14_interp.json")))["arms"]
    p13 = json.load(open(os.path.join(RES, "expt13_patching.json")))["arms"]
    ex = json.load(open(os.path.join(RES, "expt15_exemplar.json")))["arms"]
    real = json.load(open(os.path.join(
        os.path.dirname(HERE), "sentinel", "expt20_real_transform.json")))["arms"]

    oracle = abs(p13["patch_ffn_L1"]["rare_delta"])
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.9, 2.8),
                                   gridspec_kw={"width_ratios": [1, 1.35]})

    lams = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    ys = [100 * abs(iv[f"interp_λ={l}"]["delta"]) / oracle for l in lams]
    axl.plot(lams, ys, "-o", color=C1, linewidth=1.2, markersize=4)
    axl.axhline(0, color=INK2, linewidth=0.8)
    axl.set_xlabel("$\\lambda$  (step toward a real same-context state)")
    axl.set_ylabel("% of oracle recovered")
    axl.set_title("Increments are a legitimate channel", fontsize=9)
    style(axl)

    names = ["other-\ncontext", "avg. of\ntwo real", "stored\nexemplar",
             "registry\nmean", "$Ah+b$\ntransform"]
    syn = [iv["patch_other_context"]["delta"], iv["patch_avg_two_real"]["delta"],
           ex["exemplar_patch"]["delta"], iv["patch_registry_ema"]["delta"],
           itp["arms"]["global_linear"]["delta"]]
    syn = [100 * (-v) / oracle for v in syn]
    lo, hi = -260.0, 138.0
    clipped = [max(v, lo + 18) for v in syn]
    cols = [C2 if v < 0 else C3 for v in syn]
    axr.bar(range(len(syn)), clipped, color=cols, width=0.62)
    axr.axhline(0, color=INK2, linewidth=0.8)
    axr.axhline(100, color=C1, linewidth=0.9, linestyle="--")
    axr.text(-0.55, 106, "oracle", color=C1, fontsize=7.5, ha="left")
    axr.set_ylim(lo, hi)
    axr.set_xticks(range(len(names)))
    axr.set_xticklabels(names, fontsize=7.2)
    axr.set_ylabel("% of oracle recovered")
    axr.set_title("Stored states fail; the stored correction does not", fontsize=9)
    for i, v in enumerate(syn):
        off = v < lo + 18
        axr.text(i, (lo + 26) if off else (v + 8 if v > 0 else v - 22),
                 f"{v:.0f}%" + (" $\\downarrow$" if off else ""),
                 ha="center", fontsize=7.5, color=INK)
    axr.text(2.25, 52, f"82M pretrained LM:\ntransform "
                       f"{100*real['transform']['recovery']:.0f}%, "
                       f"oracle {100*real['patch']['recovery']:.0f}%",
             ha="center", fontsize=7.5, color=INK2)
    style(axr)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_carrier.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_emergence()
    fig_realtext_capacity()
    fig_repair()
    fig_crossover()
    fig_sentinel()
    fig_localization()
    fig_carrier()
    print("figures done:", sorted(os.listdir(FIG)))
