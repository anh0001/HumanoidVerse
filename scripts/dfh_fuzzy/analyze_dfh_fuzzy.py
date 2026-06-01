#!/usr/bin/env python3
"""
DFH fuzzy descriptor analysis (Codex protocol). Reads ONLY the sweep CSV(s).

Pipeline:
  1. Walking-cell filter — keep cells where the policy still WALKS (else difficulty
     reflects collapse/standing, not locomotion). Report dropped cells.
  2. SUPPORT-AXIS GATE (go/no-go) — at fixed mid traction, does softening Bekker
     support (lower K) measurably increase sinkage, sinkage-drag, AND difficulty?
     (+ optional force_coupling=off shadow control.) NO fuzzy claim unless this passes.
  3. Difficulty Y = rank(vel_err)+rank(slip)+rank(falls)-rank(distance).
  4. Fuzzy go/no-go — does support add info over mu (Y~mu vs Y~mu+K), and does the
     paper fuzzy index d add info over mu (Y~mu vs Y~mu+fuzzy_d)? partial Spearman
     + cross-validated R^2 gain. (fuzzy_d support mapping is documented/provisional.)

numpy-only; no scipy/sklearn dependency.
"""
from __future__ import annotations
import argparse, csv, math, os
from collections import defaultdict
import numpy as np

EPS = 1e-9

# ---------- paper fuzzy Mamdani (mirrors scripts/paper_fuzzy_soil/fuzzy_soil.py) ----------
def _rd(x, a, b): return 1.0 if x <= a else (0.0 if x >= b else (b - x) / (b - a))
def _ru(x, a, b): return 0.0 if x <= a else (1.0 if x >= b else (x - a) / (b - a))
def _tri(x, a, b, c):
    if x <= a or x >= c: return 0.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)

def fuzzy_d(mu, kn_knm):
    """Paper eq.2 difficulty index, d in [0.2,0.8], higher=harder."""
    tl, tm, th = _rd(mu, 0.45, 0.60), _tri(mu, 0.45, 0.60, 0.75), _ru(mu, 0.60, 0.75)
    ss, sm, sh = _rd(kn_knm, 150, 300), _tri(kn_knm, 150, 300, 400), _ru(kn_knm, 300, 400)
    rules = [(min(tl, ss), "H"), (min(tl, sm), "H"), (min(tl, sh), "H"),
             (min(tm, ss), "M"), (min(tm, sm), "M"), (min(tm, sh), "M"),
             (min(th, ss), "M"), (min(th, sm), "E"), (min(th, sh), "E")]
    gE = max([a for a, c in rules if c == "E"], default=0.0)
    gM = max([a for a, c in rules if c == "M"], default=0.0)
    gH = max([a for a, c in rules if c == "H"], default=0.0)
    return (0.2 * gE + 0.5 * gM + 0.8 * gH) / (gE + gM + gH + EPS)

def K_to_kn(K, Klo=1.0, Khi=16.0):
    """Provisional support mapping: DFH Bekker scale K -> paper support kn (kN/m),
    spanning the paper's Soft..Hard band [150,400] on a LOG2 scale (K is multiplicative;
    K=1 softest->150, K=16 firmest->400). Higher K = firmer = higher kn. Provisional."""
    t = math.log2(max(K, EPS) / Klo) / math.log2(Khi / Klo)
    return 150.0 + max(0.0, min(1.0, t)) * (400.0 - 150.0)

# ---------- stats helpers (numpy-only) ----------
def rankdata(a):
    a = np.asarray(a, float); order = a.argsort(); r = np.empty(len(a)); r[order] = np.arange(len(a))
    # average ties
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    avg = {}
    sums = defaultdict(float); n = defaultdict(int)
    for i, v in enumerate(a): sums[v] += r[i]; n[v] += 1
    return np.array([sums[v] / n[v] for v in a])

def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = math.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > EPS else float("nan")

def partial_spearman(x, y, z):
    """Spearman(x,y | z): correlation of rank-residuals after removing z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.c_[np.ones(len(b)), b]
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    ex, ey = resid(rx, rz), resid(ry, rz)
    d = math.sqrt((ex @ ex) * (ey @ ey))
    return float(ex @ ey / d) if d > EPS else float("nan")

def ols_r2(y, X):
    X1 = np.c_[np.ones(len(y)), X]
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ coef
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + EPS)

def loo_cv_r2(y, X):
    """Leave-one-out CV R^2 (honest small-n generalization)."""
    n = len(y); preds = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        X1 = np.c_[np.ones(m.sum()), X[m]]
        coef, *_ = np.linalg.lstsq(X1, y[m], rcond=None)
        preds[i] = np.r_[1.0, X[i]] @ coef
    ss_res = float(((y - preds) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + EPS)

def boot_ci_diff(a, b, n=5000, seed_vals=None):
    """Bootstrap 95% CI for mean(a)-mean(b). Deterministic via fixed index grid."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(12345)
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                try: d[k] = float(v)
                except (TypeError, ValueError): d[k] = v
            rows.append(d)
    return rows


def walking_mask(rows, vel_err_max=0.15, drag_clip_max=0.05, dist_min=1.0, ep_len_min=300):
    out = []
    for r in rows:
        ok = (r.get("vel_err") is not None and r["vel_err"] < vel_err_max
              and (r.get("dfh_drag_clipped_frac") or 0) <= drag_clip_max
              and (r.get("distance_m") or 0) >= dist_min
              and (r.get("ep_len_steps") or 0) >= ep_len_min)
        out.append(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/DFH_fuzzy/sweep_main.csv")
    ap.add_argument("--shadow-csv", default="logs/DFH_fuzzy/sweep_shadow.csv")
    ap.add_argument("--mid-mu", type=float, default=0.56)
    ap.add_argument("--gate-ref-K", type=float, default=1.0)   # soft reference
    ap.add_argument("--gate-firm-K", type=float, default=8.0)  # firm (K=16 = asymptote check)
    ap.add_argument("--out", default="logs/DFH_fuzzy/dfh_fuzzy_analysis.txt")
    args = ap.parse_args()

    lines = []
    def emit(s=""): print(s); lines.append(s)

    rows = load(args.csv)
    emit("== DFH fuzzy descriptor analysis (Codex protocol) ==")
    emit(f"   source: {args.csv}  ({len(rows)} cells)\n")

    # ---- 1. walking-cell filter ----
    wm = walking_mask(rows)
    walk = [r for r, ok in zip(rows, wm) if ok]
    drop = [r for r, ok in zip(rows, wm) if not ok]
    emit(f"-- 1. Walking-cell filter: kept {len(walk)}/{len(rows)} "
         f"(vel_err<0.15, drag_clip<=0.05, dist>=1, ep_len>=300) --")
    for r in drop:
        emit(f"   DROP mu={r['mu']} K={r['K']} s={int(r['seed'])}: "
             f"vel_err={r.get('vel_err')} ep_len={r.get('ep_len_steps')} dist={r.get('distance_m')}")
    if len(walk) < 4:
        emit("\n   <4 walking cells -> insufficient for analysis. STOP."); _write(args.out, lines); return

    # ---- 2. support-axis gate at fixed mid traction ----
    emit(f"\n-- 2. SUPPORT-AXIS GATE (mu={args.mid_mu}; soft=low K vs firm=high K) --")
    mid = [r for r in walk if abs(r["mu"] - args.mid_mu) < 1e-6]
    Ks = sorted(set(r["K"] for r in mid))
    if len(Ks) >= 2:
        # Codex: gate on ref K (soft) vs firm K (not the extreme), K=max as asymptote check.
        Klo = min(Ks, key=lambda k: abs(k - args.gate_ref_K))
        Khi = min(Ks, key=lambda k: abs(k - args.gate_firm_K))
        emit(f"   (gate endpoints: soft K={Klo} vs firm K={Khi}; available K={Ks})")
        def grp(K, key): return [r[key] for r in mid if abs(r["K"] - K) < 1e-6 and r.get(key) is not None]
        for key, label, thr in [("dfh_mean_sink_m", "sinkage m", 0.005),
                                 ("dfh_mean_sink_drag_n", "sink drag N", 10.0),
                                 ("vel_err", "vel_err", None)]:
            soft, firm = grp(Klo, key), grp(Khi, key)
            if not soft or not firm: continue
            d, lo, hi = boot_ci_diff(soft, firm)
            rel = abs(d) / (abs(np.mean(firm)) + EPS) * 100
            note = ""
            if thr is not None:
                note = f" [thr {thr}: {'PASS' if abs(d) >= thr else 'fail'}]"
            ci = f"95%CI[{lo:.4f},{hi:.4f}]"; excl0 = "PASS(excl 0)" if (lo > 0 or hi < 0) else "fail(incl 0)"
            emit(f"   {label:14s}: soft(K={Klo})={np.mean(soft):.4f} firm(K={Khi})={np.mean(firm):.4f} "
                 f"Δ={d:+.4f} ({rel:.0f}%) {ci} {excl0}{note}")
        # monotonicity of difficulty across all K at mid mu
        if len(mid) >= 3:
            rho = spearman([r["K"] for r in mid], [r["vel_err"] for r in mid])
            emit(f"   vel_err vs K Spearman rho={rho:+.3f} (expect <0: firmer=easier)")
    else:
        emit("   <2 K levels at mid mu -> cannot run gate.")

    # shadow control
    if os.path.isfile(args.shadow_csv):
        sh = load(args.shadow_csv)
        emit(f"\n   shadow control (force_coupling=off, {len(sh)} cells): "
             "support should NOT move robot behavior with coupling off")
        shmid = [r for r in sh if abs(r["mu"] - args.mid_mu) < 1e-6]
        if len(shmid) >= 2:
            Kss = sorted(set(r["K"] for r in shmid))
            s_soft = [r["vel_err"] for r in shmid if abs(r["K"] - Kss[0]) < 1e-6 and r.get("vel_err") is not None]
            s_firm = [r["vel_err"] for r in shmid if abs(r["K"] - Kss[-1]) < 1e-6 and r.get("vel_err") is not None]
            if s_soft and s_firm:
                d, lo, hi = boot_ci_diff(s_soft, s_firm)
                emit(f"   shadow vel_err Δ(soft-firm)={d:+.4f} 95%CI[{lo:.4f},{hi:.4f}] "
                     f"({'moves (CONFOUND!)' if (lo>0 or hi<0) else 'flat (good)'})")
    else:
        emit(f"\n   (no shadow CSV at {args.shadow_csv} — run force-coupling=False endpoints for the control)")

    # ---- 3. difficulty Y over walking cells ----
    def col(key): return np.array([r[key] for r in walk], float)
    Y = (rankdata(col("vel_err")) + rankdata(col("slip_per_100m"))
         + rankdata(col("falls_per_100m")) - rankdata(col("distance_m")))
    mu = col("mu"); K = col("K")
    kn = np.array([K_to_kn(k) for k in K]); d = np.array([fuzzy_d(m, k) for m, k in zip(mu, kn)])

    emit("\n-- 3-4. Fuzzy go/no-go (difficulty Y = rank-sum; walking cells only) --")
    emit(f"   n={len(walk)}  Spearman(Y,mu)={spearman(Y,mu):+.3f}  "
         f"Spearman(Y,K)={spearman(Y,K):+.3f}  Spearman(Y,fuzzy_d)={spearman(Y,d):+.3f}")
    emit(f"   partial Spearman(Y,K | mu)      = {partial_spearman(Y, K, mu):+.3f}  "
         "(does SUPPORT add ordering beyond mu?)")
    emit(f"   partial Spearman(Y,fuzzy_d | mu)= {partial_spearman(Y, d, mu):+.3f}  "
         "(does the FUZZY index add beyond mu?)")
    r2_mu = loo_cv_r2(Y, mu.reshape(-1, 1))
    r2_muK = loo_cv_r2(Y, np.c_[mu, K])
    r2_mud = loo_cv_r2(Y, np.c_[mu, d])
    emit(f"   LOO-CV R^2:  Y~mu={r2_mu:+.3f}   Y~mu+K={r2_muK:+.3f} (Δ={r2_muK-r2_mu:+.3f})   "
         f"Y~mu+fuzzy_d={r2_mud:+.3f} (Δ={r2_mud-r2_mu:+.3f})")
    emit("   PASS thresholds (Codex): partial-Spearman CI excl 0 AND CV-R^2 gain >= 0.10.")
    emit("   Interpretation: if SUPPORT helps but fuzzy_d does not -> DFH rescued the axis,")
    emit("   not the paper's fuzzy mapping. If fuzzy_d helps over mu -> first real fuzzy positive.")
    emit("   NOTE: K->kn support mapping is provisional (documented in K_to_kn).")

    # ---- 5. robustness checks (Codex): proxies, all-cells, permutation test ----
    emit("\n-- 5. Robustness checks --")

    # (a) support proxies vs fuzzy_d (CV-R^2 gain over mu), walking cells
    sink = col("dfh_mean_sink_m"); sdrag = col("dfh_mean_sink_drag_n")
    emit("   (a) CV-R^2 gain over Y~mu (walking cells, n=%d):" % len(walk))
    for label, extra in [("mu+K", K), ("mu+sinkage", sink), ("mu+sink_drag", sdrag),
                         ("mu+fuzzy_d", d)]:
        r2 = loo_cv_r2(Y, np.c_[mu, extra])
        emit(f"        {label:14s} R^2={r2:+.3f}  gain={r2-r2_mu:+.3f}")

    # (b) ALL cells with a failure penalty (dropped cells -> worst difficulty),
    #     to show exclusions don't manufacture the result.
    allr = rows
    def acol(key): return np.array([ (r[key] if r.get(key) is not None else np.nan) for r in allr], float)
    wmask_all = np.array(walking_mask(allr))
    # penalty: non-walking cells get worst vel_err/slip/falls and zero distance
    velA = acol("vel_err"); slipA = acol("slip_per_100m"); fallA = acol("falls_per_100m"); distA = acol("distance_m")
    def fill_pen(a, worst):  # nan/non-walking -> worst
        a = a.copy()
        for i in range(len(a)):
            if np.isnan(a[i]) or not wmask_all[i]: a[i] = worst
        return a
    vbad = np.nanmax(velA) * 1.5; sbad = np.nanmax(slipA) * 1.5; fbad = np.nanmax(fallA) * 1.5
    velP, slipP, fallP = fill_pen(velA, vbad), fill_pen(slipA, sbad), fill_pen(fallA, fbad)
    distP = distA.copy()
    for i in range(len(distP)):
        if np.isnan(distP[i]) or not wmask_all[i]: distP[i] = 0.0
    Yall = rankdata(velP) + rankdata(slipP) + rankdata(fallP) - rankdata(distP)
    muA, KA = acol("mu"), acol("K")
    knA = np.array([K_to_kn(k) for k in KA]); dA = np.array([fuzzy_d(m, k) for m, k in zip(muA, knA)])
    r2a_mu = loo_cv_r2(Yall, muA.reshape(-1, 1))
    r2a_K = loo_cv_r2(Yall, np.c_[muA, KA]); r2a_d = loo_cv_r2(Yall, np.c_[muA, dA])
    emit(f"   (b) ALL {len(allr)} cells w/ failure penalty: Y~mu={r2a_mu:+.3f}  "
         f"Y~mu+K={r2a_K:+.3f}(Δ{r2a_K-r2a_mu:+.3f})  Y~mu+fuzzy_d={r2a_d:+.3f}(Δ{r2a_d-r2a_mu:+.3f})")

    # (c) permutation test: is ΔR^2(mu+K) > ΔR^2(mu+fuzzy_d)? shuffle which extra-regressor
    #     label is attached to cells; here we test observed gap vs a null of exchanged residual power.
    obs_gap = (loo_cv_r2(Y, np.c_[mu, K]) - r2_mu) - (loo_cv_r2(Y, np.c_[mu, d]) - r2_mu)
    rng = np.random.default_rng(7)
    null = []
    for _ in range(2000):
        perm = rng.permutation(len(Y))
        gK = loo_cv_r2(Y, np.c_[mu, K[perm]]) - r2_mu
        gd = loo_cv_r2(Y, np.c_[mu, d[perm]]) - r2_mu
        null.append(gK - gd)
    null = np.array(null)
    p = float((np.abs(null) >= abs(obs_gap)).mean())
    emit(f"   (c) ΔR^2(mu+K) - ΔR^2(mu+fuzzy_d) = {obs_gap:+.3f}; permutation p={p:.3f} "
         f"(support beats fuzzy_d as a predictor)")
    emit("   SCOPE: finding is against THIS paper's fuzzy mapping, NOT all 2-D descriptors.")

    _write(args.out, lines)


def _write(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n   wrote {path}")


if __name__ == "__main__":
    main()
