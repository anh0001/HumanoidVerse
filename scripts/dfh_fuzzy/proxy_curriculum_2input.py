#!/usr/bin/env python3
"""
2-INPUT fuzzy as a curriculum feature — the one untested path before declaring NO-GO.

The 1-scalar tests (proxy_curriculum_v2.py) showed the fuzzy index, used as a SINGLE
difficulty score, loses to raw [mu, log2K]: it compresses the 2-D surface into one
number, camps / thrashes. This script tests the steelman: use fuzzy as a 2-INPUT
scheduler feature [mu, fuzzy_d] (its proven-strong descriptor form: [mu, d_v2] gave
LOO-CV dR^2 +0.416 vs raw [mu,K] +0.334). Honest caveat (Codex): at that point it is a
learned 2-D controller, not "the paper's fuzzy scalar" -- so a tie with raw proves
nothing. The ONLY interesting win is BEST-OF-BOTH: match raw's accuracy AND gain
fuzzy's noise-stability (fewer reversals / higher productive under noisy sensing),
because fuzzy's saturating memberships damp input noise.

Same DFH-grid proxy + PERCENTILE scheduler (Codex's fair interface) as
proxy_curriculum_v2.py. Arms: oracle, raw[mu,log2K], v2_1in (scalar d_v2),
v2_2in[mu,d_v2], fuzzy2_2in[mu,d_paper]. CPU-only; reuses sweep_rigid.csv.

PRE-REGISTERED (written before running): the 2-input fuzzy is "worth a GPU run" iff, at
BOTH sigma=0.3 and 0.4, v2_2in (1) productive >= raw - 0.03 (matches accuracy), (2) maxY>0
(no camp), (3) >=20% FEWER reversals than raw (the stability payoff), AND (4) strictly
beats v2_1in. Match-only (within +-0.03 prod and +-20% rev) => "equivalent to raw, no
fuzzy-specific value". Worse => fail.
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze_dfh_fuzzy import fuzzy_d, K_to_kn, walking_mask  # noqa: E402
from fuzzy_v2_loadbearing import fuzzy_d_v2  # noqa: E402
EPS = 1e-9


def zscore(a):
    a = np.asarray(a, float); return (a - a.mean()) / (a.std() + EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/DFH_fuzzy/sweep_rigid.csv")
    ap.add_argument("--sigmas", default="0.0,0.2,0.3,0.4")
    ap.add_argument("--rollouts", type=int, default=1500)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--out", default="logs/DFH_fuzzy/proxy_curriculum_2input.txt")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv))]
    for r in rows:
        for k in r:
            try: r[k] = float(r[k])
            except (TypeError, ValueError): pass
    walk = [r for r, ok in zip(rows, walking_mask(rows)) if ok]

    velz = zscore([r["vel_err"] for r in walk]); slz = zscore([r["slip_per_100m"] for r in walk])
    faz = zscore([r["falls_per_100m"] for r in walk]); diz = zscore([r["distance_m"] for r in walk])
    comp = velz + slz + faz - diz
    agg = {}
    for r, c in zip(walk, comp):
        agg.setdefault((r["mu"], r["K"]), []).append(c)
    cell_list = sorted(agg.keys())
    Ytrue = {c: float(np.mean(v)) for c, v in agg.items()}
    ladder = sorted(cell_list, key=lambda c: Ytrue[c])
    Ylad = np.array([Ytrue[c] for c in ladder])
    plo, phi = np.percentile(Ylad, 40), np.percentile(Ylad, 75)

    mu = np.array([c[0] for c in cell_list]); K = np.array([c[1] for c in cell_list])
    logK = np.log2(K); Yc = np.array([Ytrue[c] for c in cell_list])
    def fit(X, y):
        X1 = np.c_[np.ones(len(y)), X]; b, *_ = np.linalg.lstsq(X1, y, rcond=None); return b
    d_clean = np.array([fuzzy_d(m, K_to_kn(k)) for m, k in zip(mu, K)])
    v2_clean = np.array([fuzzy_d_v2(m, K_to_kn(k)) for m, k in zip(mu, K)])
    co_raw = fit(np.c_[mu, logK], Yc)
    co_v2_1 = fit(v2_clean.reshape(-1, 1), Yc)
    co_v2_2 = fit(np.c_[mu, v2_clean], Yc)          # 2-input [mu, d_v2]
    co_fz_2 = fit(np.c_[mu, d_clean], Yc)           # 2-input [mu, d_paper]
    mu_rng = mu.max() - mu.min(); logK_rng = logK.max() - logK.min()

    def estimate(arm, cell, rng, sg):
        m0, k0 = cell; lk0 = np.log2(k0)
        mn = m0 + rng.normal(0, sg * mu_rng); lkn = lk0 + rng.normal(0, sg * logK_rng)
        kn = K_to_kn(2.0**lkn)
        if arm == "oracle":  return co_raw[0] + co_raw[1]*m0 + co_raw[2]*lk0
        if arm == "raw":     return co_raw[0] + co_raw[1]*mn + co_raw[2]*lkn
        if arm == "v2_1in":  return co_v2_1[0] + co_v2_1[1]*fuzzy_d_v2(mn, kn)
        if arm == "v2_2in":  return co_v2_2[0] + co_v2_2[1]*mn + co_v2_2[2]*fuzzy_d_v2(mn, kn)
        if arm == "fuzzy2":  return co_fz_2[0] + co_fz_2[1]*mn + co_fz_2[2]*fuzzy_d(mn, kn)
        raise ValueError(arm)

    arms = ["oracle", "raw", "v2_1in", "v2_2in", "fuzzy2"]
    estC = {a: np.array([estimate(a, c, np.random.default_rng(0), 0.0) for c in ladder]) for a in arms}
    tgt_pct = 0.575
    adv_pct = tgt_pct - 0.15*(0.75-0.40)
    reg_pct = tgt_pct + 0.85*(0.75-0.40)

    def rollout(arm, rng, sg):
        pos = 0; visited = []; revers = 0; last_dir = 0
        for _ in range(args.blocks):
            cell = ladder[pos]; visited.append(Ytrue[cell])
            est = estimate(arm, cell, rng, sg)
            pct = float(np.mean(estC[arm] < est))
            d = +1 if pct < adv_pct else (-1 if pct > reg_pct else 0)
            npos = min(len(ladder)-1, max(0, pos + d))
            mv = np.sign(npos - pos)
            if mv != 0 and last_dir != 0 and mv != last_dir: revers += 1
            if mv != 0: last_dir = mv
            pos = npos
        v = np.array(visited)
        prod = float(np.mean((v >= plo) & (v <= phi)))
        return dict(prod=prod, rev=revers, maxY=float(v.max()), meanY=float(v.mean()), final=float(v[-1]))

    sigmas = [float(s) for s in args.sigmas.split(",")]
    lines = []
    def emit(s=""): print(s); lines.append(s)
    emit("== 2-INPUT fuzzy as a curriculum feature [mu, fuzzy] (percentile scheduler) ==")
    emit(f"   grid cells={len(cell_list)} productive band [{plo:.2f},{phi:.2f}]; "
         f"blocks={args.blocks} rollouts={args.rollouts}")
    emit("   productive=frac in band (higher=better); rev=reversals (lower=better); maxY=camping check\n")
    summary = {}
    for sg in sigmas:
        emit(f"   --- sigma = {sg} ---")
        emit(f"   {'arm':>8} | {'productive':>10} {'reversals':>9} {'maxY':>6} {'meanY':>6} {'finalY':>6}")
        res = {}
        for arm in arms:
            rng = np.random.default_rng(1234 + int(sg*100))
            R = [rollout(arm, rng, sg) for _ in range(args.rollouts)]
            a = {k: float(np.mean([r[k] for r in R])) for k in R[0]}
            res[arm] = a
            emit(f"   {arm:>8} | {a['prod']:>10.3f} {a['rev']:>9.2f} {a['maxY']:>6.2f} "
                 f"{a['meanY']:>6.2f} {a['final']:>6.2f}")
        summary[sg] = res

    # ---- PRE-REGISTERED gate ----
    emit("\n== PRE-REGISTERED gate: is 2-input fuzzy BEST-OF-BOTH (raw accuracy + fuzzy stability)? ==")
    verdicts = {}
    for sg in [s for s in sigmas if s >= 0.3]:
        r, v1, v2, o = (summary[sg]["raw"], summary[sg]["v2_1in"],
                        summary[sg]["v2_2in"], summary[sg]["oracle"])
        match_acc = v2["prod"] >= r["prod"] - 0.03
        not_camp = v2["maxY"] > 0.0
        rev_red = (r["rev"] - v2["rev"]) / (r["rev"] + EPS)
        beats_1in = (v2["prod"] > v1["prod"]) or (v2["rev"] < v1["rev"] - EPS)
        win = match_acc and not_camp and rev_red >= 0.20 and beats_1in
        equiv = match_acc and not_camp and abs(rev_red) < 0.20 and not win
        verdicts[sg] = "WIN" if win else ("EQUIV" if equiv else "FAIL")
        emit(f"   sigma={sg}: prod v2_2in={v2['prod']:.3f} vs raw={r['prod']:.3f} (match>=raw-0.03:{match_acc}); "
             f"maxY={v2['maxY']:.2f}(>0:{not_camp}); rev_red_vs_raw={rev_red:+.0%}(win>=20%); "
             f"beats_1in={beats_1in} -> {verdicts[sg]}")
    emit("")
    vs = set(verdicts.values())
    if verdicts and vs == {"WIN"}:
        emit("   => WIN at sigma 0.3 AND 0.4: 2-input fuzzy matches raw's accuracy AND adds")
        emit("      noise-stability -> a GPU curriculum with [mu, d_v2] is JUSTIFIED.")
    elif "FAIL" not in vs and "EQUIV" in vs:
        emit("   => EQUIVALENT to raw (no fuzzy-specific gain): [mu,d_v2] carries the same info")
        emit("      as [mu,log2K]. Honest -> fuzzy adds interpretability, not performance. GPU NOT justified.")
    else:
        emit("   => FAIL: 2-input fuzzy does not match raw. GPU NOT justified.")
    emit("   SCOPE: [mu,d_v2] is a learned 2-D feature, NOT the paper's fuzzy scalar; offline-ladder proxy.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"\n   wrote {args.out}")


if __name__ == "__main__":
    main()
