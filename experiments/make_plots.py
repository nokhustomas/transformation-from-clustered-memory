# -*- coding: utf-8 -*-
"""从 results/raw_results.jsonl 汇总统计并画图,供论文使用。所有数字均来自真实运行记录。"""
import json
import os
import statistics as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results", "raw_results.jsonl")
FIG_DIR = os.path.join(HERE, "results")

# Okabe-Ito 色盲友好配色
COLOR = {
    "nomemory": "#999999",
    "fullattn": "#000000",
    "topkraw": "#E69F00",
    "cma": "#0072B2",
}
LABEL = {
    "nomemory": "NoMemory (lower bound)",
    "fullattn": "FullAttn (oracle)",
    "topkraw": "TopKRaw (H2O-style, matched budget)",
    "cma": "CMA (ours)",
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
    print(f"共读取 {len(rows)} 条 run 记录")

    summary_lines = []

    # ---------- 主实验: budget=4, 4 种条件, 3 seeds ----------
    conditions = ["nomemory", "fullattn", "topkraw", "cma"]
    primary_budget = 4
    means, stds, ns = {}, {}, {}
    for c in conditions:
        b = None if c in ("nomemory", "fullattn") else primary_budget
        runs = group(rows, c, b)
        accs = [r["final_eval_acc"] for r in runs]
        m, s = mean_std(accs) if accs else (float("nan"), float("nan"))
        means[c], stds[c], ns[c] = m, s, len(accs)
        summary_lines.append(f"{c:10s} budget={str(b):4s} n_seeds={len(accs)} "
                              f"acc_mean={m:.4f} acc_std={s:.4f} raw={accs}")

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    xs = range(len(conditions))
    bar_colors = [COLOR[c] for c in conditions]
    bars = ax.bar(xs, [means[c] for c in conditions], yerr=[stds[c] for c in conditions],
                   color=bar_colors, capsize=4, width=0.6)
    chance = 1 / 16
    ax.axhline(chance, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"chance ({chance:.3f})")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABEL[c].split(" (")[0] for c in conditions], rotation=20, ha="right")
    ax.set_ylabel("Eval accuracy at query positions")
    ax.set_title(f"Primary comparison (budget = {primary_budget} tokens, 3 seeds)", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, max(0.5, max(means.values()) + max(stds.values()) + 0.08))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_primary_bar.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig_primary_bar.png"))
    plt.close(fig)

    # ---------- 学习曲线(主实验,取每个 condition 的种子均值) ----------
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for c in conditions:
        b = None if c in ("nomemory", "fullattn") else primary_budget
        runs = group(rows, c, b)
        if not runs:
            continue
        steps = [pt["step"] for pt in runs[0]["curve"]]
        acc_by_step = []
        for i in range(len(steps)):
            vals = [r["curve"][i]["eval_acc"] for r in runs]
            acc_by_step.append(sum(vals) / len(vals))
        ax.plot(steps, acc_by_step, label=LABEL[c], color=COLOR[c], linewidth=2)
    ax.axhline(1 / 16, color="red", linestyle="--", linewidth=1, alpha=0.6, label="chance")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Eval accuracy at query positions")
    ax.set_title(f"Learning curves (budget = {primary_budget}, mean over seeds)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_learning_curves.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig_learning_curves.png"))
    plt.close(fig)

    # ---------- 预算扫描(单 seed, topkraw vs cma, 附 nomemory/fullattn 参照线) ----------
    budgets = [4, 8, 16]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for c in ["topkraw", "cma"]:
        accs = []
        for b in budgets:
            runs = group(rows, c, b)
            accs.append(runs[0]["final_eval_acc"] if runs else float("nan"))
        ax.plot(budgets, accs, marker="o", label=LABEL[c], color=COLOR[c], linewidth=2)

    ax.axhline(means["nomemory"], color=COLOR["nomemory"], linestyle=":", linewidth=1.5,
               label=f"{LABEL['nomemory']} (budget-independent)")
    ax.axhline(means["fullattn"], color=COLOR["fullattn"], linestyle=":", linewidth=1.5,
               label=f"{LABEL['fullattn']} (budget-independent)")
    ax.set_xlabel("Memory budget (token-equivalent slots)")
    ax.set_ylabel("Final eval accuracy")
    ax.set_xticks(budgets)
    ax.set_title("Budget sweep (single seed; illustrative, not a significance claim)", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_budget_sweep.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig_budget_sweep.png"))
    plt.close(fig)

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(os.path.join(FIG_DIR, "summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    # 额外输出一份供论文表格直接抄数的 JSON
    table_data = {
        "primary_budget": primary_budget,
        "primary": {c: {"mean": means[c], "std": stds[c], "n_seeds": ns[c]} for c in conditions},
        "sweep": {
            c: {b: (group(rows, c, b)[0]["final_eval_acc"] if group(rows, c, b) else None) for b in budgets}
            for c in ["topkraw", "cma"]
        },
        "params": {c: (group(rows, c, primary_budget if c in ("topkraw", "cma") else None)[0]["n_params"]
                        if group(rows, c, primary_budget if c in ("topkraw", "cma") else None) else None)
                   for c in conditions},
    }
    with open(os.path.join(FIG_DIR, "table_data.json"), "w") as f:
        json.dump(table_data, f, indent=2, ensure_ascii=False)
    print("\n已写出 fig_primary_bar.{pdf,png}, fig_learning_curves.{pdf,png}, "
          "fig_budget_sweep.{pdf,png}, summary.txt, table_data.json")


if __name__ == "__main__":
    main()
