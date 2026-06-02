#!/usr/bin/env python3
"""
Experiment (1): does the fuzzy soil index earn its keep UNDER IMPRECISE SOIL SENSING?

The descriptor tests fed predictors the EXACT (mu, K) -> raw numbers trivially win.
But a real robot only has NOISY estimates of grip (mu) and firmness (K). Fuzzy logic's
core claim is graceful degradation under imprecise inputs. So: fit each difficulty
predictor on CLEAN soil params, then at TEST time feed NOISY estimates and see whose
difficulty prediction degrades slowest.

Predictors (all map estimated soil params -> difficulty):
  - raw     : OLS difficulty ~ mu + log2(K)            (continuous raw numbers)
  - crisp   : paper rule base with HARD-threshold bins -> {Easy,Mod,Hard} singleton
  - fuzzy   : paper Mamdani (smooth memberships) -> d in [0.2,0.8]   (the paper's pitch)

Target difficulty = z(vel_err)+z(slip)+z(falls)-z(distance) over walking cells (fixed).
Monte-Carlo over input-noise sigma; report RMSE (lower=better) + Spearman(pred,truth)
vs noise. If fuzzy degrades SLOWER than raw -> fuzzy matters when soil can't be measured
exactly. If raw stays best even under heavy noise -> fuzzy still adds nothing.

CPU-only, reuses logs/DFH_fuzzy/sweep_rigid.csv. No GPU.
"""
from __future__ import annotations
import argparse, csv, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze_dfh_fuzzy import fuzzy_d, K_to_kn, walking_mask, rankdata, spearman  # noqa

EPS = 1e-9

# ---- crisp (hard-threshold) analog of the paper index: argmax membership + same rules ----
def crisp_d(mu, kn):
    # traction bins (breakpoints 0.45/0.60/0.75 -> thresholds at midpoints)
    trac = "L" if mu < 0.525 else ("M" if mu < 0.675 else "H")
    # support bins (150/300/400 -> thresholds at 225 / 350)
    sup = "S" if kn < 225 else ("M" if kn < 350 else "H")
    if trac == "L":            cons = 0.8           # Hard
    elif trac == "M":          cons = 0.5           # Moderate
    else:  cons = 0.5 if sup == "S" else 0.2        # High+Soft->Mod, High+Med/Hard->Easy
    return cons


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
    ap.add_argument("--sigmas", default="0.0,0.05,0.10,0.20,0.40")
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--out", default="logs/DFH_fuzzy/fuzzy_noise_analysis.txt")
    args = ap.parse_args()

    rows = []
    for r in csv.DictReader(open(args.csv)):
        d = {}
        for k, v in r.items():
            try: d[k] = float(v)
            except (TypeError, ValueError): d[k] = v
        rows.append(d)
    walk = [r for r, ok in zip(rows, walking_mask(rows)) if ok]

    mu = np.array([r["mu"] for r in walk]); K = np.array([r["K"] for r in walk])
    logK = np.log2(K)
    truth = (zscore([r["vel_err"] for r in walk]) + zscore([r["slip_per_100m"] for r in walk])
             + zscore([r["falls_per_100m"] for r in walk]) - zscore([r["distance_m"] for r in walk]))

    # input ranges (for noise scaling) and clean-fit coefficients
    mu_rng = mu.max() - mu.min(); logK_rng = logK.max() - logK.min()
    d_clean = np.array([fuzzy_d(m, K_to_kn(k)) for m, k in zip(mu, K)])
    c_clean = np.array([crisp_d(m, K_to_kn(k)) for m, k in zip(mu, K)])
    co_raw = ols_fit(np.c_[mu, logK], truth)
    co_fuz = ols_fit(d_clean.reshape(-1, 1), truth)
    co_cri = ols_fit(c_clean.reshape(-1, 1), truth)

    sigmas = [float(s) for s in args.sigmas.split(",")]
    rng = np.random.default_rng(20260602)
    lines = []
    def emit(s=""): print(s); lines.append(s)
    emit("== Experiment (1): fuzzy vs raw vs crisp difficulty predictor UNDER NOISY SOIL SENSING ==")
    emit(f"   source: {args.csv}  walking cells n={len(walk)}  reps/sigma={args.reps}")
    emit("   noise: mu += N(0, sigma*range), log2(K) += N(0, sigma*range). Predict difficulty")
    emit("   from NOISY inputs using CLEAN-fit coefficients. RMSE lower=better.\n")
    emit(f"   {'sigma':>6} | {'RMSE raw':>9} {'RMSE crisp':>10} {'RMSE fuzzy':>10} | "
         f"{'rho raw':>8} {'rho crisp':>9} {'rho fuzzy':>9} | winner")
    results = {}
    for sg in sigmas:
        rr, rc, rf = [], [], []
        sr, sc, sf = [], [], []
        for _ in range(args.reps):
            mun = mu + rng.normal(0, sg * mu_rng, len(mu))
            lkn = logK + rng.normal(0, sg * logK_rng, len(logK))
            Kn = np.power(2.0, lkn)
            kn = np.array([K_to_kn(k) for k in Kn])
            dn = np.array([fuzzy_d(m, kk) for m, kk in zip(mun, kn)])
            cn = np.array([crisp_d(m, kk) for m, kk in zip(mun, kn)])
            praw = ols_pred(co_raw, np.c_[mun, lkn])
            pfuz = ols_pred(co_fuz, dn.reshape(-1, 1))
            pcri = ols_pred(co_cri, cn.reshape(-1, 1))
            rr.append(np.sqrt(np.mean((praw - truth) ** 2)))
            rc.append(np.sqrt(np.mean((pcri - truth) ** 2)))
            rf.append(np.sqrt(np.mean((pfuz - truth) ** 2)))
            sr.append(spearman(praw, truth)); sc.append(spearman(pcri, truth)); sf.append(spearman(pfuz, truth))
        mr, mc, mf = np.mean(rr), np.mean(rc), np.mean(rf)
        win = min([("raw", mr), ("crisp", mc), ("fuzzy", mf)], key=lambda x: x[1])[0]
        results[sg] = dict(rmse=(mr, mc, mf), rho=(np.mean(sr), np.mean(sc), np.mean(sf)))
        emit(f"   {sg:>6.2f} | {mr:>9.3f} {mc:>10.3f} {mf:>10.3f} | "
             f"{np.mean(sr):>8.3f} {np.mean(sc):>9.3f} {np.mean(sf):>9.3f} | {win}")

    # verdict: does fuzzy ever win, and does it degrade slower than raw?
    emit("")
    fuzzy_wins = [sg for sg in sigmas if results[sg]["rmse"][2] < min(results[sg]["rmse"][0], results[sg]["rmse"][1]) - 1e-6]
    # degradation slope (RMSE rise from sigma=0 to max)
    s0, sM = sigmas[0], sigmas[-1]
    dr = results[sM]["rmse"][0] - results[s0]["rmse"][0]
    df = results[sM]["rmse"][2] - results[s0]["rmse"][2]
    emit(f"   RMSE rise (sigma {s0}->{sM}):  raw +{dr:.3f}   fuzzy +{df:.3f}   "
         f"({'fuzzy degrades SLOWER' if df < dr else 'raw degrades slower/equal'})")
    if fuzzy_wins:
        emit(f"   => FUZZY has lowest RMSE at sigma in {fuzzy_wins} -> fuzzy matters under that noise.")
    else:
        emit("   => FUZZY never has the lowest RMSE at any noise level tested.")
    emit("   (crisp = hard-threshold bins = the thing the paper's smooth fuzzy is meant to beat.)")
    emit("   SCOPE: tests THIS paper's mapping + this difficulty target; one policy (rigid_walker).")

    # chart
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        xs = sigmas
        ax[0].plot(xs, [results[s]["rmse"][0] for s in xs], "o-", label="raw (mu,log2 K)", color="tab:green")
        ax[0].plot(xs, [results[s]["rmse"][1] for s in xs], "s--", label="crisp bins", color="tab:orange")
        ax[0].plot(xs, [results[s]["rmse"][2] for s in xs], "^-", label="fuzzy d", color="tab:blue")
        ax[0].set_xlabel("soil-estimate noise sigma (frac of input range)"); ax[0].set_ylabel("difficulty-prediction RMSE")
        ax[0].set_title("(1) Robustness to noisy soil sensing\n(lower = better; flatter = more robust)")
        ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(xs, [results[s]["rho"][0] for s in xs], "o-", label="raw", color="tab:green")
        ax[1].plot(xs, [results[s]["rho"][1] for s in xs], "s--", label="crisp", color="tab:orange")
        ax[1].plot(xs, [results[s]["rho"][2] for s in xs], "^-", label="fuzzy", color="tab:blue")
        ax[1].set_xlabel("soil-estimate noise sigma"); ax[1].set_ylabel("Spearman(pred, true difficulty)")
        ax[1].set_title("(2) Difficulty ordering preserved under noise\n(higher = better)")
        ax[1].legend(); ax[1].grid(alpha=0.3)
        fig.tight_layout(); out_png = args.out.replace(".txt", ".png")
        fig.savefig(out_png, bbox_inches="tight", dpi=130); emit(f"\n   chart -> {out_png}")
    except Exception as e:
        emit(f"   (chart skipped: {e})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"\n   wrote {args.out}")


if __name__ == "__main__":
    main()
