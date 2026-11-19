# -*- coding: utf-8 -*-
"""
第三轮实验(任务难度削减)的画图脚本。补充 make_plots.py / make_plots_pertoken.py,
不替代它们——三份图各自忠实反映各自那批实验的真实数据,互相独立。

生成 fig_easytask_bar.{pdf,png}: 4 个 easytask 条件在 B=4 时的对比(3-seed 均值+
误差棒),chance=0.125(easytask 的 value_vocab=8,与原任务的 1/16 不同,画图时
不能共用同一条 chance 线,这也是这批数据单独出一张图、不和原任务/pertoken 那两张
图合并的原因)。
"""
import json
import os
import statistics as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results", "raw_results.jsonl")
FIG_DIR = os.path.join(HERE, "results")

COLOR = {
    "nomemory_easytask": "#999999",
    "fullattn_easytask": "#000000",
    "topkraw_easytask": "#D55E00",
    "cma_easytask": "#009E73",
}
LABEL = {
    "nomemory_easytask": "NoMemory",
    "fullattn_easytask": "FullAttn",
    "topkraw_easytask": "TopKRaw\n(per-token)",
    "cma_easytask": "CMA\n(per-token)",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
})


def load_results():
    rows = []
    with open(RESULTS_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def group(rows, condition, budget):
    return [r for r in rows if r["condition"] == condition and r["budget"] == budget]


def mean_std(xs):
    if len(xs) == 1:
        return xs[0], 0.0
    return stats.mean(xs), stats.stdev(xs)


def main():
    rows = load_results()
    conditions = ["nomemory_easytask", "fullattn_easytask", "topkraw_easytask", "cma_easytask"]
    primary_budget = 4
    means, stds = {}, {}
    lines = []
    for c in conditions:
        b = None if c in ("nomemory_easytask", "fullattn_easytask") else primary_budget
        runs = group(rows, c, b)
        accs = [r["final_eval_acc"] for r in runs]
        m, s = mean_std(accs) if accs else (float("nan"), float("nan"))
        means[c], stds[c] = m, s
        lines.append(f"{c:20s} budget={str(b):4s} n_seeds={len(accs)} acc_mean={m:.4f} acc_std={s:.4f} raw={accs}")
    print("\n".join(lines))

    # B=8 单 seed 参照(不进条形图,单独打印,和原论文 Table sweep 的处理方式一致)
    print("\nB=8 单 seed 参照:")
    for c in ["topkraw_easytask", "cma_easytask"]:
        runs = group(rows, c, 8)
        for r in runs:
            print(f"  {c} budget=8 seed={r['seed']} final_acc={r['final_eval_acc']:.4f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    xs = range(len(conditions))
    bars = ax.bar(xs, [means[c] for c in conditions], yerr=[stds[c] for c in conditions],
                   color=[COLOR[c] for c in conditions], capsize=4, width=0.58)
    chance = 1 / 8
    ax.axhline(chance, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"chance ({chance:.3f})")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABEL[c] for c in conditions], fontsize=9.5)
    ax.set_ylabel("Eval accuracy at query positions")
    ax.set_title("Simplified (easier) content-matching task, $B{=}4$, 3 seeds", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, max(0.5, max(means.values()) + max(stds.values()) + 0.08))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_easytask_bar.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig_easytask_bar.png"))
    plt.close(fig)
    print("\n已写出 fig_easytask_bar.{pdf,png}")


if __name__ == "__main__":
    main()
