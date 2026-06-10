#!/usr/bin/env python3
"""
GPU-gate proxy for the LOAD-BEARING fuzzy (v2) as a curriculum signal.

Set C Exp(2) proxy (proxy_curriculum.py) showed the PAPER fuzzy index, used as a
difficulty-tracking curriculum signal, CAMPS at easy soil and fails the GPU gate
(stability of an uninformative signal). That index was support-blind. This proxy asks
the same question for the redesigned LOAD-BEARING index fuzzy_d_v2 (see
fuzzy_v2_loadbearing.py): used as the difficulty estimate a curriculum advances on,
does v2 TRACK the productive band like raw (no camping) while keeping fuzzy-like
noise-stability (fewer reversals)? If yes, a real GPU curriculum is justified.

Identical scheduler / ladder / target / noise to proxy_curriculum.py; arms differ ONLY
in the difficulty-estimate mapping. Adds a `v2` arm and a v2-specific GO/NO-GO gate.
CPU-only; reuses sweep_rigid.csv.
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
    ap.add_argument("--mode", choices=["magnitude", "percentile"], default="percentile",
                    help="percentile (Codex fair fix): threshold the estimate's RANK in the "
                         "arm's own estimate distribution, removing the 1-scalar magnitude "
                         "offset so the test is purely 'does the ordering induce a good traversal'.")
    ap.add_argument("--out", default="logs/DFH_fuzzy/proxy_curriculum_v2_rank.txt")
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
    target = (plo + phi) / 2.0

    mu = np.array([c[0] for c in cell_list]); K = np.array([c[1] for c in cell_list])
    logK = np.log2(K); Yc = np.array([Ytrue[c] for c in cell_list])
    def fit(X, y):
        X1 = np.c_[np.ones(len(y)), X]; b, *_ = np.linalg.lstsq(X1, y, rcond=None); return b
    co_raw = fit(np.c_[mu, logK], Yc)
    d_clean = np.array([fuzzy_d(m, K_to_kn(k)) for m, k in zip(mu, K)])
    v2_clean = np.array([fuzzy_d_v2(m, K_to_kn(k)) for m, k in zip(mu, K)])
    co_fuz = fit(d_clean.reshape(-1, 1), Yc)
    co_v2 = fit(v2_clean.reshape(-1, 1), Yc)
    mu_rng = mu.max() - mu.min(); logK_rng = logK.max() - logK.min()

    def estimate(arm, cell, rng, sg):
        m0, k0 = cell; lk0 = np.log2(k0)
        mn = m0 + rng.normal(0, sg * mu_rng); lkn = lk0 + rng.normal(0, sg * logK_rng)
        if arm == "oracle":
            return co_raw[0] + co_raw[1]*m0 + co_raw[2]*lk0
        if arm == "raw":
            return co_raw[0] + co_raw[1]*mn + co_raw[2]*lkn
        if arm == "fuzzy":
            kn = K_to_kn(2.0**lkn); return co_fuz[0] + co_fuz[1]*fuzzy_d(mn, kn)
        if arm == "v2":
            kn = K_to_kn(2.0**lkn); return co_v2[0] + co_v2[1]*fuzzy_d_v2(mn, kn)
        raise ValueError(arm)

    # --- magnitude-mode thresholds (difficulty units) ---
    adv_thr = target - 0.15*(phi-plo)
    reg_thr = target + 0.85*(phi-plo)
    # --- percentile-mode: each arm judged on its OWN clean-estimate distribution, so the
    #     1-scalar magnitude offset (which made v2 camp at sigma=0) is removed; we test
    #     purely whether the arm's ORDERING induces a good easy->hard traversal under noise.
    estC = {a: np.array([estimate(a, c, np.random.default_rng(0), 0.0) for c in ladder])
            for a in ("oracle", "raw", "fuzzy", "v2")}
    # productive band [40,75] pct of true difficulty -> target pct 0.575; mirror the
    # magnitude margins in percentile space.
    tgt_pct = 0.575
    adv_pct = tgt_pct - 0.15*(0.75-0.40)   # 0.5225
    reg_pct = tgt_pct + 0.85*(0.75-0.40)   # 0.8725

    def rollout(arm, rng, sg):
        pos = 0; visited = []; revers = 0; last_dir = 0
        for _ in range(args.blocks):
            cell = ladder[pos]; visited.append(Ytrue[cell])
            if arm == "fixed":
                d = +1 if pos < len(ladder)-1 else 0
            elif args.mode == "percentile":
                est = estimate(arm, cell, rng, sg)
                pct = float(np.mean(estC[arm] < est))   # rank of this estimate in arm's own scale
                d = +1 if pct < adv_pct else (-1 if pct > reg_pct else 0)
            else:
                est = estimate(arm, cell, rng, sg)
                d = +1 if est < adv_thr else (-1 if est > reg_thr else 0)
            npos = min(len(ladder)-1, max(0, pos + d))
            mv = np.sign(npos - pos)
            if mv != 0 and last_dir != 0 and mv != last_dir: revers += 1
            if mv != 0: last_dir = mv
            pos = npos
        v = np.array(visited)
        prod = float(np.mean((v >= plo) & (v <= phi)))
        return dict(prod=prod, rev=revers, maxY=float(v.max()), meanY=float(v.mean()),
                    final=float(v[-1]))

    sigmas = [float(s) for s in args.sigmas.split(",")]
    arms = ["fixed", "oracle", "raw", "fuzzy", "v2"]
    lines = []
    def emit(s=""): print(s); lines.append(s)
    emit("== GPU-gate proxy: LOAD-BEARING fuzzy v2 as a curriculum signal (vs raw/paper-fuzzy) ==")
    emit(f"   scheduler mode = {args.mode.upper()}"
         + ("  (Codex fair fix: threshold estimate RANK in each arm's own distribution)"
            if args.mode == "percentile" else "  (absolute difficulty units)"))
    emit(f"   grid cells={len(cell_list)} ladder(easy->hard) true-diff range "
         f"[{Ylad.min():.2f},{Ylad.max():.2f}] productive band [{plo:.2f},{phi:.2f}] "
         f"target {target:.2f}; blocks={args.blocks} rollouts={args.rollouts}")
    emit(f"   productive=frac in band (higher=better); rev=reversals (lower=better); "
         f"maxY=hardest reached (camping check)\n")
    summary = {}
    for sg in sigmas:
        emit(f"   --- sigma = {sg} ---")
        emit(f"   {'arm':>7} | {'productive':>10} {'reversals':>9} {'maxY':>6} {'meanY':>6} {'finalY':>6}")
        res = {}
        for arm in arms:
            rng = np.random.default_rng(1234 + int(sg*100))
            R = [rollout(arm, rng, sg) for _ in range(args.rollouts if arm not in ("fixed","oracle") else max(1,args.rollouts))]
            a = {k: float(np.mean([r[k] for r in R])) for k in R[0]}
            res[arm] = a
            emit(f"   {arm:>7} | {a['prod']:>10.3f} {a['rev']:>9.2f} {a['maxY']:>6.2f} "
                 f"{a['meanY']:>6.2f} {a['final']:>6.2f}")
        summary[sg] = res

    # ---- v2 GO/NO-GO gate ----
    # Hypothesis for v2 (DIFFERENT from paper-fuzzy gate): v2 should give raw-like
    # PROGRESSION (not camp) with fuzzy-like STABILITY. GPU justified if, at noisy
    # sigma>=0.3, v2 is (i) NOT camping (maxY within 0.15*range of raw), (ii) productive
    # competitive with raw (Δ>=-0.05) AND with oracle (Δ>=-0.10), (iii) FEWER reversals
    # than raw (>=20% reduction = the stability payoff), and v2 strictly beats paper-fuzzy
    # on productive (fixes the camping).
    # Pre-registered pass condition (Codex, 2026-06-05): at sigma=0.3 AND 0.4, v2 must
    #   (1) beat paper-fuzzy on productive fraction AND maxY,
    #   (2) not camp: maxY > 0,
    #   (3) be within 0.10 productive fraction of raw OR oracle,
    #   (4) have >=20% fewer reversals than raw.
    emit("\n== v2 GO/NO-GO for GPU (Codex pre-registered, percentile mode; pass needs BOTH sigma 0.3 & 0.4) ==")
    per_sigma_pass = {}
    for sg in [s for s in sigmas if s >= 0.3]:
        r, f, v, o = (summary[sg]["raw"], summary[sg]["fuzzy"],
                      summary[sg]["v2"], summary[sg]["oracle"])
        beats_paper = (v["prod"] > f["prod"]) and (v["maxY"] > f["maxY"])
        not_camp = v["maxY"] > 0.0
        within = (v["prod"] >= r["prod"] - 0.10) or (v["prod"] >= o["prod"] - 0.10)
        rev_red_vs_raw = (r["rev"] - v["rev"]) / (r["rev"] + EPS)
        cond = beats_paper and not_camp and within and rev_red_vs_raw >= 0.20
        per_sigma_pass[sg] = cond
        emit(f"   sigma={sg}: beats_paper={beats_paper} (prod {v['prod']:.3f}>{f['prod']:.3f} & "
             f"maxY {v['maxY']:.2f}>{f['maxY']:.2f}); not_camp={not_camp} (maxY {v['maxY']:.2f}>0); "
             f"within0.10_of_raw|oracle={within} (v2 {v['prod']:.3f}; raw {r['prod']:.3f}, oracle {o['prod']:.3f}); "
             f"rev_red_vs_raw={rev_red_vs_raw:+.0%} (need>=20%) -> {'PASS' if cond else 'fail'}")
    go = all(per_sigma_pass.get(s, False) for s in [0.3, 0.4] if s in per_sigma_pass) and bool(per_sigma_pass)
    emit("")
    if go:
        emit("   => PROXY PASSES at sigma 0.3 AND 0.4 -> load-bearing v2 induces a raw-competitive")
        emit("      easy->hard traversal with added stability; GPU curriculum JUSTIFIED.")
    else:
        emit("   => PROXY FAILS the pre-registered condition -> do NOT spend GPU on the fuzzy")
        emit("      curriculum claim yet (inspect which condition/sigma failed above).")
    emit("   SCOPE: proxy uses the measured DFH grid as transition model + 1-scalar OLS")
    emit("   difficulty estimates; it gates GPU, it is NOT the training result itself.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"\n   wrote {args.out}")


if __name__ == "__main__":
    main()
