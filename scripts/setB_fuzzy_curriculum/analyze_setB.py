#!/usr/bin/env python3
"""
Set B analysis — aggregate held-out robustness across the 3 arms x 3 seeds and
deliver the honest verdict (adaptive vs fixed; fuzzy vs crisp).

Reads ONLY from source CSVs (no hand-entered numbers; see memory no-fabricated-result-numbers):
  - held-out eval CSVs:  <eval_dir>/setB_<arm>_seed<seed>.csv
        rows: mu,rep,seed,episodes,ep_len_steps,distance_m,slip_per_100m,falls_per_100m,vel_err
  - per-run decisions:   logs/FuzzySoilFurrowsSetB/setB_<arm>_seed<seed>/decisions.csv

Primary metric  : robustness AUC over the held-out mu sweep (higher = more robust
                  across difficulties). robustness(mu) = mean ep_len / ep_len_cap,
                  averaged over reps; AUC = trapezoid over mu (normalized by mu-range).
Secondary metric: blocks-to-hardest-rung from decisions.csv (curriculum pacing speed).

Verdicts use mean +/- sd across seeds with Welch t (n is small -> report effect size
and overlap honestly; do not overclaim significance at n=3).
"""
from __future__ import annotations
import argparse, csv, glob, math, os, statistics as st
from collections import defaultdict

ARMS = ["fixed", "crisp", "fuzzy"]
EP_LEN_CAP = 1000.0


def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def load_eval(path):
    """Return {mu: [ep_len,...]} from a held-out eval CSV."""
    by_mu = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            mu, el = f(r.get("mu")), f(r.get("ep_len_steps"))
            if mu is not None and el is not None:
                by_mu[mu].append(el)
    return by_mu


def robustness_auc(by_mu):
    """Trapezoid AUC of mean(ep_len)/cap over mu, normalized by mu-range -> [0,1]-ish."""
    mus = sorted(by_mu)
    if len(mus) < 2:
        return None
    ys = [st.mean(by_mu[m]) / EP_LEN_CAP for m in mus]
    area = 0.0
    for i in range(len(mus) - 1):
        area += 0.5 * (ys[i] + ys[i + 1]) * (mus[i + 1] - mus[i])
    return area / (mus[-1] - mus[0])


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    return (ma - mb) / se if se > 0 else None


def blocks_to_hardest(decisions_csv, hardest_mu):
    """First block index whose next_mu reached the hardest rung; None if never."""
    if not os.path.isfile(decisions_csv):
        return None
    with open(decisions_csv) as fh:
        for r in csv.DictReader(fh):
            if f(r.get("next_mu")) is not None and abs(f(r["next_mu"]) - hardest_mu) < 1e-6:
                return int(r["block"]) + 1  # +1: reached AFTER this block
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="logs/FuzzySoilFurrowsSetB/eval")
    ap.add_argument("--runs-root", default="logs/FuzzySoilFurrowsSetB")
    ap.add_argument("--hardest-mu", type=float, default=0.35)
    ap.add_argument("--out", default="logs/FuzzySoilFurrowsSetB/setB_analysis.txt")
    args = ap.parse_args()

    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("== Set B — adaptive-vs-fixed friction curriculum (held-out robustness) ==")
    emit(f"   robustness AUC = trapezoid[ mean(ep_len)/{EP_LEN_CAP:.0f} ] over mu sweep, /mu-range")
    emit(f"   (only numbers read from CSVs this run; missing arms/seeds reported as such)\n")

    auc = {a: [] for a in ARMS}          # arm -> [per-seed AUC]
    pacing = {a: [] for a in ARMS}       # arm -> [per-seed blocks-to-hardest]
    emit(f"{'arm':>7} {'seed':>4} | {'AUC':>7} | {'blocks->hardest':>15} | source")
    for arm in ARMS:
        for seed in (1, 2, 3):
            ev = os.path.join(args.eval_dir, f"setB_{arm}_seed{seed}.csv")
            dec = os.path.join(args.runs_root, f"setB_{arm}_seed{seed}", "decisions.csv")
            if not os.path.isfile(ev):
                emit(f"{arm:>7} {seed:>4} | {'--':>7} | {'--':>15} | MISSING {ev}")
                continue
            a = robustness_auc(load_eval(ev))
            b2h = blocks_to_hardest(dec, args.hardest_mu)
            if a is not None: auc[arm].append(a)
            if b2h is not None: pacing[arm].append(b2h)
            emit(f"{arm:>7} {seed:>4} | {a if a is None else round(a,4):>7} | "
                 f"{str(b2h):>15} | {os.path.basename(ev)}")

    emit("\n== Per-arm summary (mean +/- sd across available seeds) ==")
    emit(f"{'arm':>7} | {'n':>2} | {'AUC mean':>9} | {'AUC sd':>7} | {'blocks->hardest':>15}")
    for arm in ARMS:
        n = len(auc[arm])
        m = round(st.mean(auc[arm]), 4) if n else None
        s = round(st.stdev(auc[arm]), 4) if n > 1 else 0.0
        pac = (f"{round(st.mean(pacing[arm]),2)}" if pacing[arm] else "--")
        emit(f"{arm:>7} | {n:>2} | {str(m):>9} | {str(s):>7} | {pac:>15}")

    emit("\n== Contrasts (honest; n is small) ==")
    if auc["fixed"] and (auc["crisp"] or auc["fuzzy"]):
        adaptive = auc["crisp"] + auc["fuzzy"]
        t = welch_t(adaptive, auc["fixed"])
        emit(f"   adaptive(crisp+fuzzy) vs fixed: "
             f"AUC {round(st.mean(adaptive),4) if adaptive else '--'} vs "
             f"{round(st.mean(auc['fixed']),4)}  Welch t={None if t is None else round(t,2)}")
    if auc["fuzzy"] and auc["crisp"]:
        t = welch_t(auc["fuzzy"], auc["crisp"])
        df = st.mean(auc["fuzzy"]) - st.mean(auc["crisp"])
        emit(f"   fuzzy vs crisp:                 "
             f"AUC {round(st.mean(auc['fuzzy']),4)} vs {round(st.mean(auc['crisp']),4)}  "
             f"diff={round(df,4)}  Welch t={None if t is None else round(t,2)}")
        emit("   (Expected per Set A: fuzzy ~= crisp, since difficulty ~ 1-D friction "
             "with kn inert.)")
    else:
        emit("   fuzzy/crisp not both present yet -> verdict pending.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    emit(f"\n   wrote {args.out}")


if __name__ == "__main__":
    main()
