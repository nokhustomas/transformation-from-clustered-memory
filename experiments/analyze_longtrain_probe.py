# -*- coding: utf-8 -*-
"""
5-seed 复现性分析:对 results/longtrain_probe.jsonl 里 topkraw_pertoken 和
cma_pertoken 各 5 个 seed(0-4)的完整 60-checkpoint 曲线,判断 seed=0 观察到的
"后段(约 step 4300 起)从 chance 抬升到 ~8.6%"是否在其余种子上复现。

方法论要点:前后半段切分点(3000)和精细切分点(4200)是看 seed=0 曲线之后定的,
这次直接原样套用到全部 5 个种子,不为每个种子重新挑一个"看起来最像上翘"的切分点
——否则就是在挑对自己有利的分割方式,不是在测复现性。
"""
import json
import math
import os
import statistics

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "longtrain_probe.jsonl")
CHANCE = 1 / 16  # value_vocab=16
HALF_SPLIT = 3000
REFINED_SPLIT = 4200


def load_results():
    results = {}
    with open(RESULTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            results[(r["condition"], r["seed"])] = r
    return results


def welch_t(a, b):
    """Welch's t-test, returns (t, approx_df). No scipy dependency."""
    na, nb = len(a), len(b)
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, na + nb - 2
    t = (mb - ma) / se
    # Welch-Satterthwaite df
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / den if den > 0 else na + nb - 2
    return t, df


def linreg_slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def analyze_one(r):
    curve = r["curve"]
    accs_all = [c["eval_acc"] for c in curve]
    steps_all = [c["step"] for c in curve]

    front_half = [c["eval_acc"] for c in curve if c["step"] <= HALF_SPLIT]
    back_half = [c["eval_acc"] for c in curve if c["step"] > HALF_SPLIT]
    front_ref = [c["eval_acc"] for c in curve if c["step"] <= REFINED_SPLIT]
    back_ref = [c["eval_acc"] for c in curve if c["step"] > REFINED_SPLIT]

    slope = linreg_slope(steps_all, accs_all)
    t_ref, df_ref = welch_t(front_ref, back_ref)

    return {
        "condition": r["condition"], "seed": r["seed"],
        "final_acc": r["final_eval_acc"],
        "front_half_mean": statistics.mean(front_half), "front_half_std": statistics.stdev(front_half),
        "back_half_mean": statistics.mean(back_half), "back_half_std": statistics.stdev(back_half),
        "front_ref_mean": statistics.mean(front_ref), "front_ref_std": statistics.stdev(front_ref),
        "back_ref_mean": statistics.mean(back_ref), "back_ref_std": statistics.stdev(back_ref),
        "back_minus_front_ref": statistics.mean(back_ref) - statistics.mean(front_ref),
        "slope_per_1000steps": slope * 1000,
        "t_ref": t_ref, "df_ref": df_ref,
    }


def main():
    results = load_results()
    print(f"chance = {CHANCE:.4f}\n")
    for condition in ["topkraw_pertoken", "cma_pertoken"]:
        print(f"===== {condition} =====")
        diffs = []
        for seed in range(5):
            r = results.get((condition, seed))
            if r is None:
                print(f"  seed={seed}: 缺失")
                continue
            a = analyze_one(r)
            diffs.append(a["back_minus_front_ref"])
            print(f"  seed={seed}  final={a['final_acc']:.4f}  "
                  f"半段前={a['front_half_mean']:.4f}(sd={a['front_half_std']:.4f})  "
                  f"半段后={a['back_half_mean']:.4f}(sd={a['back_half_std']:.4f})  "
                  f"| 4200前={a['front_ref_mean']:.4f}  4200后={a['back_ref_mean']:.4f}  "
                  f"差={a['back_minus_front_ref']:+.4f}  t={a['t_ref']:+.2f}(df={a['df_ref']:.0f})  "
                  f"斜率/1000步={a['slope_per_1000steps']:+.4f}")
        pooled_mean_diff = statistics.mean(diffs)
        n_positive = sum(1 for d in diffs if d > 0.01)  # 后段比前段高 >1pp 才算"有抬升迹象"
        print(f"  --> 5 个种子里,后段(>4200步)比前段(<=4200步)高出 >1pp 的种子数: {n_positive}/5")
        print(f"  --> 5 个种子 (后段-前段) 差值的均值: {pooled_mean_diff:+.4f}, "
              f"各差值: {[f'{d:+.4f}' for d in diffs]}")
        print()


if __name__ == "__main__":
    main()
