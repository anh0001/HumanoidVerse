#!/usr/bin/env python3
"""
Load-bearing fuzzy redesign (Set C follow-up). Reads ONLY sweep_rigid.csv; NO GPU.

Set C found the paper's Mamdani index d(mu, support) adds ~nothing over raw mu
(LOO-CV R^2 gain +0.025) EVEN on deformable (DFH) soil where the support axis is
physically real and strongly predictive (raw K gain +0.33). Root cause is STRUCTURAL:
the rule base maps Low-traction->Hard and Mid-traction->Moderate REGARDLESS of support,
so on this grid (mu in {0.50,0.56,0.62}, all in the Low/Mid band) d is blind to K.

This script tests a MINIMAL intervention recommended by Codex: change ONLY the rule
table (+ consequent levels), holding the fuzzification (same mu/kn membership
breakpoints) and centroid defuzzification fixed, so any change in predictive power is
attributable to the rule structure alone. The new rule base is a monotone 2-D
anti-diagonal grid in which support varies in EVERY traction row.

HONEST FRAMING (per Codex): this does NOT claim fuzzy discovers new structure. The fair
baseline / honest ceiling for deterministic clean inputs is raw K; a deterministic fuzzy
transform should at best MATCH most of raw K's signal while staying bounded/interpretable/
monotone -- and ideally be more robust under noisy estimated inputs. So we report:
  (1) headline: LOO-CV dR^2 over Y~mu for {raw K, paper fuzzy_d, fuzzy_d_v2}
      -> "v2 recovers X/0.33 of the support-axis gain the paper's index discarded".
  (2) noise robustness: does v2 keep raw's accuracy AND gain fuzzy's noise-stability?
  (3) secondary sensitivity (v2-calibrated): widen breakpoints to the data span -- a
      SEPARATE intervention, reported only as a sanity check, not the main result.
"""
from __future__ import annotations
import argparse, csv, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze_dfh_fuzzy import (  # noqa: E402
    _rd, _ru, _tri, fuzzy_d, K_to_kn, walking_mask, rankdata, spearman,
    partial_spearman, loo_cv_r2, load,
)

EPS = 1e-9

# ---------------- load-bearing fuzzy (v2): rule-table-only redesign ----------------
# 5-level evenly-spaced consequents (vs paper's 3 singletons {0.2,0.5,0.8}).
_LEVELS = {"VE": 0.1, "E": 0.3, "M": 0.5, "H": 0.7, "VH": 0.9}

# Monotone 2-D rule grid. Rows = traction Low/Mid/High; cols = support Soft/Med/Firm.
# Harder (higher) = Low-traction + Soft-support. Support varies in EVERY row -> the
# structural fix. (Paper grid: L->H,H,H ; M->M,M,M ; H->M,E,E -- support inert in L,M.)
_RULES_V2 = {
    ("L", "S"): "VH", ("L", "M"): "H",  ("L", "F"): "M",
    ("M", "S"): "H",  ("M", "M"): "M",  ("M", "F"): "E",
    ("H", "S"): "M",  ("H", "M"): "E",  ("H", "F"): "VE",
}


def _centroid(rules):
    """Mamdani max-aggregation + singleton centroid defuzzification (paper form)."""
    g = {lvl: 0.0 for lvl in _LEVELS}
    for strength, label in rules:
        if strength > g[label]:
            g[label] = strength
    num = sum(_LEVELS[l] * g[l] for l in _LEVELS)
    den = sum(g[l] for l in _LEVELS) + EPS
    return num / den


def fuzzy_d_v2(mu, kn, mu_bp=(0.45, 0.60, 0.75), kn_bp=(150.0, 300.0, 400.0)):
    """Load-bearing fuzzy difficulty index in [0.1,0.9], higher=harder.
    Same fuzzification breakpoints as the paper; ONLY the rule table differs."""
    a, b, c = mu_bp
    tL, tM, tH = _rd(mu, a, b), _tri(mu, a, b, c), _ru(mu, b, c)
    p, q, r = kn_bp
    sS, sM, sF = _rd(kn, p, q), _tri(kn, p, q, r), _ru(kn, q, r)
    trac = {"L": tL, "M": tM, "H": tH}
    supp = {"S": sS, "M": sM, "F": sF}
    rules = [(min(trac[t], supp[s]), _RULES_V2[(t, s)]) for t in trac for s in supp]
    return _centroid(rules)


# ---------------- helpers ----------------
def zscore(a):
    a = np.asarray(a, float); s = a.std()
    return (a - a.mean()) / (s + EPS)


def ols_fit(X, y):
    X1 = np.c_[np.ones(len(y)), X]
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return coef


def ols_pred(coef, X):
    return np.c_[np.ones(len(X)), X] @ coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/DFH_fuzzy/sweep_rigid.csv")
    ap.add_argument("--sigmas", default="0.0,0.10,0.20,0.40")
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--out", default="logs/DFH_fuzzy/fuzzy_v2_loadbearing.txt")
    args = ap.parse_args()

    lines = []
    def emit(s=""): print(s); lines.append(s)

    rows = load(args.csv)
    walk = [r for r, ok in zip(rows, walking_mask(rows)) if ok]
    emit("== Load-bearing fuzzy redesign (v2): does a monotone 2-D rule table make "
         "fuzzy load-bearing? ==")
    emit(f"   source: {args.csv}  walking cells n={len(walk)}/{len(rows)}")
    emit("   Intervention: rule TABLE only (5-level monotone grid); fuzzification + "
         "defuzzification held fixed.\n")

    def col(key): return np.array([r[key] for r in walk], float)
    mu, K = col("mu"), col("K")
    kn = np.array([K_to_kn(k) for k in K])
    d_paper = np.array([fuzzy_d(m, k) for m, k in zip(mu, kn)])
    d_v2 = np.array([fuzzy_d_v2(m, k) for m, k in zip(mu, kn)])
    Y = (rankdata(col("vel_err")) + rankdata(col("slip_per_100m"))
         + rankdata(col("falls_per_100m")) - rankdata(col("distance_m")))

    # ---- (1) HEADLINE: LOO-CV R^2 gain over Y~mu ----
    emit("-- (1) HEADLINE: LOO-CV R^2 gain over Y~mu (deterministic clean inputs) --")
    r2_mu = loo_cv_r2(Y, mu.reshape(-1, 1))
    g_K = loo_cv_r2(Y, np.c_[mu, K]) - r2_mu
    g_paper = loo_cv_r2(Y, np.c_[mu, d_paper]) - r2_mu
    g_v2 = loo_cv_r2(Y, np.c_[mu, d_v2]) - r2_mu
    emit(f"   Y~mu base CV-R^2 = {r2_mu:+.3f}")
    emit(f"   + raw K        : dR^2 = {g_K:+.3f}   (honest ceiling for clean inputs)")
    emit(f"   + paper fuzzy_d: dR^2 = {g_paper:+.3f}   (support-blind rule base)")
    emit(f"   + fuzzy_d_v2   : dR^2 = {g_v2:+.3f}   (load-bearing rule base)")
    recov = g_v2 / g_K if abs(g_K) > EPS else float("nan")
    emit(f"   >> v2 recovers {g_v2:+.3f}/{g_K:+.3f} = {100*recov:.0f}% of the support-axis "
         f"gain the paper's index discarded.")
    emit(f"   partial Spearman(Y, . | mu):  raw K={partial_spearman(Y,K,mu):+.3f}   "
         f"paper d={partial_spearman(Y,d_paper,mu):+.3f}   v2 d={partial_spearman(Y,d_v2,mu):+.3f}")
    emit(f"   Spearman(d_v2, K) = {spearman(d_v2, K):+.3f} (expect <0: firmer->lower difficulty index)")

    # ---- (2) NOISE ROBUSTNESS: raw vs paper-fuzzy vs v2 under noisy soil sensing ----
    emit("\n-- (2) Noise robustness: predict difficulty from NOISY (mu, log2 K) estimates --")
    emit("   fit each predictor on CLEAN inputs, test on noisy; RMSE lower=better, rho higher=better")
    logK = np.log2(K)
    truth = (zscore(col("vel_err")) + zscore(col("slip_per_100m"))
             + zscore(col("falls_per_100m")) - zscore(col("distance_m")))
    mu_rng, logK_rng = mu.max() - mu.min(), logK.max() - logK.min()
    co_raw = ols_fit(np.c_[mu, logK], truth)
    co_pap = ols_fit(d_paper.reshape(-1, 1), truth)
    co_v2 = ols_fit(d_v2.reshape(-1, 1), truth)
    sigmas = [float(s) for s in args.sigmas.split(",")]
    rng = np.random.default_rng(20260605)
    emit(f"   {'sigma':>6} | {'RMSE raw':>9} {'RMSE paper':>10} {'RMSE v2':>8} | "
         f"{'rho raw':>8} {'rho paper':>9} {'rho v2':>7} | winner")
    res = {}
    for sg in sigmas:
        rr = rp = rv = 0.0; sr = sp = sv = 0.0
        accR = []; accP = []; accV = []; soR = []; soP = []; soV = []
        for _ in range(args.reps):
            mun = mu + rng.normal(0, sg * mu_rng, len(mu))
            lkn = logK + rng.normal(0, sg * logK_rng, len(logK))
            knn = np.array([K_to_kn(2.0 ** lk) for lk in lkn])
            dpn = np.array([fuzzy_d(m, kk) for m, kk in zip(mun, knn)])
            dvn = np.array([fuzzy_d_v2(m, kk) for m, kk in zip(mun, knn)])
            praw = ols_pred(co_raw, np.c_[mun, lkn])
            ppap = ols_pred(co_pap, dpn.reshape(-1, 1))
            pv2 = ols_pred(co_v2, dvn.reshape(-1, 1))
            accR.append(np.sqrt(np.mean((praw - truth) ** 2)))
            accP.append(np.sqrt(np.mean((ppap - truth) ** 2)))
            accV.append(np.sqrt(np.mean((pv2 - truth) ** 2)))
            soR.append(spearman(praw, truth)); soP.append(spearman(ppap, truth)); soV.append(spearman(pv2, truth))
        mr, mp, mv = np.mean(accR), np.mean(accP), np.mean(accV)
        res[sg] = (mr, mp, mv, np.mean(soR), np.mean(soP), np.mean(soV))
        win = min([("raw", mr), ("paper", mp), ("v2", mv)], key=lambda x: x[1])[0]
        emit(f"   {sg:>6.2f} | {mr:>9.3f} {mp:>10.3f} {mv:>8.3f} | "
             f"{np.mean(soR):>8.3f} {np.mean(soP):>9.3f} {np.mean(soV):>7.3f} | {win}")
    s0, sM = sigmas[0], sigmas[-1]
    emit(f"   RMSE rise (sigma {s0}->{sM}):  raw +{res[sM][0]-res[s0][0]:.3f}   "
         f"paper +{res[sM][1]-res[s0][1]:.3f}   v2 +{res[sM][2]-res[s0][2]:.3f}")
    v2_clean_ok = res[s0][2] <= res[s0][1] + 1e-6  # v2 at least as accurate as paper-fuzzy at sigma0
    emit(f"   v2 vs paper-fuzzy at sigma=0 (clean accuracy): "
         f"RMSE {res[s0][2]:.3f} vs {res[s0][1]:.3f} -> {'v2 better/equal' if v2_clean_ok else 'paper better'}")

    # ---- (3) SECONDARY sensitivity: v2-calibrated (widened breakpoints) ----
    emit("\n-- (3) SECONDARY (sensitivity only, NOT the headline): v2-calibrated "
         "breakpoints widened to data span --")
    # widen mu breakpoints to the data span (0.50..0.62) and kn to the K_to_kn span
    mu_lo, mu_hi = mu.min(), mu.max(); mu_bp = (mu_lo, (mu_lo + mu_hi) / 2, mu_hi)
    kn_lo, kn_hi = kn.min(), kn.max(); kn_bp = (kn_lo, (kn_lo + kn_hi) / 2, kn_hi)
    d_v2c = np.array([fuzzy_d_v2(m, k, mu_bp=mu_bp, kn_bp=kn_bp) for m, k in zip(mu, kn)])
    g_v2c = loo_cv_r2(Y, np.c_[mu, d_v2c]) - r2_mu
    emit(f"   breakpoints mu{tuple(round(x,3) for x in mu_bp)}  kn{tuple(round(x,1) for x in kn_bp)}")
    emit(f"   + fuzzy_d_v2-calibrated: dR^2 = {g_v2c:+.3f}  (vs v2 {g_v2:+.3f}, raw K {g_K:+.3f})")
    emit("   (separate intervention: combines rule-table fix + recalibration; shown only "
         "to bound how much breakpoint placement matters.)")

    # ---- verdict ----
    emit("\n-- VERDICT --")
    if g_v2 >= 0.10 and recov >= 0.5:
        emit(f"   LOAD-BEARING: the rule-table-only fix lifts the support-axis gain from "
             f"{g_paper:+.3f} (paper) to {g_v2:+.3f} ({100*recov:.0f}% of raw K's {g_K:+.3f}).")
        emit("   => confirms the paper's negative was STRUCTURAL (support-blind rules), not the sim,")
        emit("      and a monotone rule base makes the fuzzy index load-bearing.")
    else:
        emit(f"   NOT load-bearing even after the fix (v2 gain {g_v2:+.3f}); structural change insufficient.")
    emit("   SCOPE/HONESTY: v2's monotone polarity is informed by Set C's observed firm=easier")
    emit("   trend -> this is a corrected descriptor, NOT a vindication of the paper's mapping, and")
    emit("   on clean inputs fuzzy cannot be expected to BEAT raw K (its ceiling) -- only match it")
    emit("   while staying bounded/interpretable and (claim 2) more noise-stable.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"\n   wrote {args.out}")


if __name__ == "__main__":
    main()
