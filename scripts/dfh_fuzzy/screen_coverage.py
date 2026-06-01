#!/usr/bin/env python3
"""Coverage-screen analysis: walking vs survival coverage per policy x K (Codex)."""
from __future__ import annotations
import csv, glob, os
from collections import defaultdict

WALK = dict(vel_err=0.15, dist=1.0, drag_clip=0.05)       # walking pass
SURV = dict(ep_len_frac=0.90, drag_clip=0.05)             # survival pass (ep cap ~1000)
EP_CAP = 1000.0


def load(p):
    rows = []
    for r in csv.DictReader(open(p)):
        d = {}
        for k, v in r.items():
            try: d[k] = float(v)
            except (TypeError, ValueError): d[k] = v
        rows.append(d)
    return rows


def walks(r):
    return ((r.get("vel_err") or 9) < WALK["vel_err"]
            and (r.get("distance_m") or 0) >= WALK["dist"]
            and (r.get("dfh_drag_clipped_frac") or 1) < WALK["drag_clip"])


def survives(r):
    return ((r.get("ep_len_steps") or 0) >= SURV["ep_len_frac"] * EP_CAP
            and (r.get("dfh_drag_clipped_frac") or 1) < SURV["drag_clip"])


def main():
    lines = []
    def emit(s=""): print(s); lines.append(s)
    emit("== DFH coverage screen (walking vs survival) ==")
    emit("   walking: vel_err<0.15 & dist>=1.0 & drag_clip<0.05")
    emit("   survival: ep_len>=900 & drag_clip<0.05 (balancer = survives but doesn't walk)\n")
    emit(f"{'policy':10s} {'walk/N':>7} {'surv/N':>7} | per-K walking (K=1,2,4,8,16)  | verdict")
    qualified = []
    for path in sorted(glob.glob("logs/DFH_fuzzy/screen_*.csv")):
        name = os.path.basename(path)[len("screen_"):-len(".csv")]
        rows = load(path)
        if not rows:
            emit(f"{name:10s}  (empty)"); continue
        nw = sum(walks(r) for r in rows); ns = sum(survives(r) for r in rows); n = len(rows)
        byK = defaultdict(lambda: [0, 0])
        for r in rows:
            byK[r["K"]][0] += int(walks(r)); byK[r["K"]][1] += 1
        perK = " ".join(f"{byK[k][0]}/{byK[k][1]}" for k in sorted(byK))
        hiK_ok = all(byK.get(k, [0, 1])[0] >= 1 for k in (8.0, 16.0) if k in byK)
        verdict = ("WALKER (descriptor candidate)" if nw >= 12 and hiK_ok
                   else ("balancer" if ns >= 12 else "neither"))
        if nw >= 12 and hiK_ok: qualified.append(name)
        emit(f"{name:10s} {nw:>3}/{n:<3} {ns:>3}/{n:<3} | {perK:28s} | {verdict}")
    emit("")
    if qualified:
        emit(f"QUALIFIED for descriptor test (>=12/15 walk, high-K ok): {qualified}")
        emit("=> rerun that policy at seeds {1,2,3}, then fuzzy-vs-raw descriptor analysis.")
    else:
        emit("NO policy gives clean walking coverage across the support-varying K range.")
        emit("=> TERMINAL (Codex): DFH restores the physical support axis, but behavioral")
        emit("   support-difficulty & fuzzy value cannot be fairly tested without training")
        emit("   a soil-robust walker. Balancers (high survival, low distance) are context only.")
    with open("logs/DFH_fuzzy/coverage_screen.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n   wrote logs/DFH_fuzzy/coverage_screen.txt")


if __name__ == "__main__":
    main()
