#!/usr/bin/env python3
"""
Experiment (2) CHEAP PROXY (Codex) — does fuzzy's noise-stability yield better CURRICULUM
DECISIONS than raw, under noisy soil sensing? Screens whether GPU training is worth it.

Closed-loop curriculum over the measured DFH grid (rigid_walker sweep) used as the
transition model. The curriculum knows the nominal soil ladder (cells sorted by true
difficulty) but decides WHEN to advance from a NOISY estimate of the current cell's
difficulty. raw and fuzzy arms differ ONLY in the estimate mapping (raw = OLS on noisy
mu,log2K; fuzzy = OLS on fuzzy_d(noisy mu,K)); same ladder, target, noise, scheduler.

Decision each block: est<advance_thr -> go harder; est>regress_thr -> go easier; else hold.
Outcome read from the cell's TRUE difficulty. Metrics (Codex): productive-band fraction,
reversals, max/mean true difficulty reached (camping check), regret vs clean-sensor oracle.
GO-TO-GPU gate at sigma 0.3/0.4: fuzzy beats raw AUC-proxy by >=0.10 AND >=30% fewer
reversals AND not camping AND competitive with oracle.  CPU-only.
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze_dfh_fuzzy import fuzzy_d, K_to_kn, walking_mask  # noqa
EPS = 1e-9


def zscore(a):
    a = np.asarray(a, float); return (a - a.mean()) / (a.std() + EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/DFH_fuzzy/sweep_rigid.csv")
    ap.add_argument("--sigmas", default="0.0,0.2,0.3,0.4")
    ap.add_argument("--rollouts", type=int, default=1500)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--out", default="logs/DFH_fuzzy/proxy_curriculum.txt")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv))]
    for r in rows:
        for k in r:
            try: r[k] = float(r[k])
            except (TypeError, ValueError): pass
    walk = [r for r, ok in zip(rows, walking_mask(rows)) if ok]

    # per-cell true difficulty = mean z-composite over the cell's walking seeds
    cells = {}
    velz = zscore([r["vel_err"] for r in walk]); slz = zscore([r["slip_per_100m"] for r in walk])
    faz = zscore([r["falls_per_100m"] for r in walk]); diz = zscore([r["distance_m"] for r in walk])
    comp = velz + slz + faz - diz
    agg = {}
    for r, c in zip(walk, comp):
        agg.setdefault((r["mu"], r["K"]), []).append(c)
    cell_list = sorted(agg.keys())                  # (mu,K)
    Ytrue = {c: float(np.mean(v)) for c, v in agg.items()}
    # nominal ladder: cells sorted easy->hard by true difficulty (shared by all arms)
    ladder = sorted(cell_list, key=lambda c: Ytrue[c])
    Ylad = np.array([Ytrue[c] for c in ladder])
    # productive band = middle of the true-difficulty range (not-too-easy, not-too-hard)
    plo, phi = np.percentile(Ylad, 40), np.percentile(Ylad, 75)
    target = (plo + phi) / 2.0

    # clean-fit difficulty models (predict Ytrue from soil params)
    mu = np.array([c[0] for c in cell_list]); K = np.array([c[1] for c in cell_list])
    logK = np.log2(K); Yc = np.array([Ytrue[c] for c in cell_list])
    def fit(X, y):
        X1 = np.c_[np.ones(len(y)), X]; b, *_ = np.linalg.lstsq(X1, y, rcond=None); return b
    co_raw = fit(np.c_[mu, logK], Yc)
    d_clean = np.array([fuzzy_d(m, K_to_kn(k)) for m, k in zip(mu, K)])
    co_fuz = fit(d_clean.reshape(-1, 1), Yc)
    mu_rng = mu.max() - mu.min(); logK_rng = logK.max() - logK.min()

    def estimate(arm, cell, rng, sg):
        m0, k0 = cell; lk0 = np.log2(k0)
        mn = m0 + rng.normal(0, sg * mu_rng); lkn = lk0 + rng.normal(0, sg * logK_rng)
        if arm == "oracle":   # clean-sensor raw (no noise)
            return co_raw[0] + co_raw[1]*m0 + co_raw[2]*lk0
        if arm == "raw":
            return co_raw[0] + co_raw[1]*mn + co_raw[2]*lkn
        if arm == "fuzzy":
            kn = K_to_kn(2.0**lkn); return co_fuz[0] + co_fuz[1]*fuzzy_d(mn, kn)
        raise ValueError(arm)

    # scheduler thresholds in difficulty units (around the target band)
    adv_thr = target - 0.15*(phi-plo)    # est below -> advance (seems mastered)
    reg_thr = target + 0.85*(phi-plo)    # est above -> regress (too hard)

    def rollout(arm, rng, sg):
        pos = 0; visited = []; revers = 0; last_dir = 0
        for _ in range(args.blocks):
            cell = ladder[pos]; visited.append(Ytrue[cell])
            if arm == "fixed":
                d = +1 if pos < len(ladder)-1 else 0    # ignore sensing; advance on schedule
            else:
                est = estimate(arm, cell, rng, sg)
                d = +1 if est < adv_thr else (-1 if est > reg_thr else 0)
            npos = min(len(ladder)-1, max(0, pos + d))
            mv = np.sign(npos - pos)
            if mv != 0 and last_dir != 0 and mv != last_dir: revers += 1
            if mv != 0: last_dir = mv
            pos = npos
        v = np.array(visited)
        prod = float(np.mean((v >= plo) & (v <= phi)))      # productive-band fraction
        return dict(prod=prod, rev=revers, maxY=float(v.max()), meanY=float(v.mean()),
                    final=float(v[-1]))

    sigmas = [float(s) for s in args.sigmas.split(",")]
    arms = ["fixed", "oracle", "raw", "fuzzy"]
    lines = []
    def emit(s=""): print(s); lines.append(s)
    emit("== Experiment (2) PROXY: noisy curriculum decisions, fuzzy vs raw (Codex gate) ==")
    emit(f"   grid cells={len(cell_list)} ladder(easy->hard) true-diff range "
         f"[{Ylad.min():.2f},{Ylad.max():.2f}] productive band [{plo:.2f},{phi:.2f}] "
         f"target {target:.2f}; blocks={args.blocks} rollouts={args.rollouts}")
    emit(f"   productive = fraction of blocks in band (higher=better); rev = curriculum "
         f"reversals (lower=better); maxY = hardest soil reached (camping check)\n")
    summary = {}
    for sg in sigmas:
        emit(f"   --- sigma = {sg} ---")
        emit(f"   {'arm':>7} | {'productive':>10} {'reversals':>9} {'maxY':>6} {'meanY':>6} {'finalY':>6}")
        res = {}
        for arm in arms:
            rng = np.random.default_rng(1234 + int(sg*100))
            R = [rollout(arm, rng, sg) for _ in range(args.rollouts if arm not in ("fixed","oracle") else max(1,args.rollouts))]
            agg = {k: float(np.mean([r[k] for r in R])) for k in R[0]}
            res[arm] = agg
            emit(f"   {arm:>7} | {agg['prod']:>10.3f} {agg['rev']:>9.2f} {agg['maxY']:>6.2f} "
                 f"{agg['meanY']:>6.2f} {agg['final']:>6.2f}")
        summary[sg] = res

    # ---- GO/NO-GO gate ----
    emit("\n== GO/NO-GO for GPU (Codex thresholds at sigma 0.3/0.4) ==")
    go = False
    for sg in [s for s in sigmas if s >= 0.3]:
        r, f, o = summary[sg]["raw"], summary[sg]["fuzzy"], summary[sg]["oracle"]
        d_prod = f["prod"] - r["prod"]
        rev_red = (r["rev"] - f["rev"]) / (r["rev"] + EPS)
        camps = f["maxY"] < r["maxY"] - 0.15*(Ylad.max()-Ylad.min())   # fuzzy under-explores vs raw
        competitive = f["prod"] >= o["prod"] - 0.10
        cond = (d_prod >= 0.10) and (rev_red >= 0.30) and (not camps) and competitive
        emit(f"   sigma={sg}: Δproductive(fuzzy-raw)={d_prod:+.3f} (need>=0.10); "
             f"reversal_reduction={rev_red:+.0%} (need>=30%); camps={camps}; "
             f"competitive_w_oracle={competitive} -> {'PASS' if cond else 'fail'}")
        go = go or cond
    emit("")
    if go:
        emit("   => PROXY PASSES at some sigma>=0.3 -> GPU Exp(2) is justified (12-run design).")
    else:
        emit("   => PROXY FAILS -> fuzzy is noise-stable but does NOT make better curriculum")
        emit("      decisions; do NOT spend GPU. Terminal: stability != better decisions.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"\n   wrote {args.out}")


if __name__ == "__main__":
    main()
