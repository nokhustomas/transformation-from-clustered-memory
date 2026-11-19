# -*- coding: utf-8 -*-
"""
后续实验(逐 token 判别式检索)的画图脚本。补充 make_plots.py,不替代它——
两份图各自忠实反映各自那批实验的真实数据。

生成 fig_pertoken_bar.{pdf,png}: 6 个条件在 B=4 时的对比(3-seed 均值+误差棒),
一眼看出"把检索换成真正逐 token 判别式选择"前后几乎没有变化。
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
    "nomemory": "#999999",
    "fullattn": "#000000",
    "topkraw": "#E69F00",
    "cma": "#0072B2",
    "topkraw_pertoken": "#D55E00",
    "cma_pertoken": "#009E73",
}
LABEL = {
    "nomemory": "NoMemory",
    "fullattn": "FullAttn",
    "topkraw": "TopKRaw\n(pooled)",
    "cma": "CMA\n(pooled)",
    "topkraw_pertoken": "TopKRaw\n(per-token)",
    "cma_pertoken": "CMA\n(per-token)",
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
    conditions = ["nomemory", "fullattn", "topkraw", "cma", "topkraw_pertoken", "cma_pertoken"]
    primary_budget = 4
    means, stds = {}, {}
    lines = []
    for c in conditions:
        b = None if c in ("nomemory", "fullattn") else primary_budget
        runs = group(rows, c, b)
        accs = [r["final_eval_acc"] for r in runs]
        m, s = mean_std(accs) if accs else (float("nan"), float("nan"))
        means[c], stds[c] = m, s
        lines.append(f"{c:20s} budget={str(b):4s} n_seeds={len(accs)} acc_mean={m:.4f} acc_std={s:.4f} raw={accs}")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = range(len(conditions))
    bars = ax.bar(xs, [means[c] for c in conditions], yerr=[stds[c] for c in conditions],
                   color=[COLOR[c] for c in conditions], capsize=4, width=0.62)
    chance = 1 / 16
    ax.axhline(chance, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"chance ({chance:.3f})")
    ax.axvline(3.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABEL[c] for c in conditions], fontsize=9)
    ax.set_ylabel("Eval accuracy at query positions")
    ax.set_title("Pooled vs. per-token discriminative retrieval (B=4, 3 seeds)", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, max(0.5, max(means.values()) + max(stds.values()) + 0.08))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_pertoken_bar.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig_pertoken_bar.png"))
    plt.close(fig)
    print("\n已写出 fig_pertoken_bar.{pdf,png}")


if __name__ == "__main__":
    main()
