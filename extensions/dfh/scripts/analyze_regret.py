"""Aggregate DFH cross-eval results into a paired-improvement report.

Reads tfevents under ``logs/DFHTuned/crosseval/<train>_on_<eval>/`` and emits
one row per (train, eval, seed). Then prints, for each seed:

  * **paired improvement** = ep_len(DFH-trained @ DFH-eval)
    − ep_len(RIGID-trained @ DFH-eval)  -- POSITIVE = DFH policy is BETTER
    on deformable terrain (higher episode length is better). This used to be
    called "regret"; renamed because the sign was the opposite of the usual
    regret convention and reviewers found it confusing (Pass-4 R3 W4).
  * normalized = paired_improvement / ep_len(RIGID-trained @ RIGID-eval)
  * the **FULL vs SHUFFLED** negative-control check on DFH eval, including
    matched-dose diagnostics (total drag, p95 drag, clip fraction, stance
    fraction) so a SHUFFLED degradation caused by a *different dose* rather
    than broken spatial memory is detectable (Pass-4 R3 W1 & W3).
  * a per-seed sign-agreement summary (Pass-4 R3 W2): two seeds only support
    the claim if they agree in sign AND SHUFFLED degrades vs FULL.

Usage:
    python -m extensions.dfh.scripts.analyze_regret \\
        --strength ultra --root logs/DFHTuned --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# Hunter approximate body weight (N) -- used only for context in printouts.
_HUNTER_WEIGHT_N = 620.0

# TB tags pulled per eval run. The dfh_* dose/cap tags were added in Pass-4
# R2/R3 specifically so cap saturation and FULL-vs-SHUFFLED dose matching are
# visible at analysis time without a re-run.
_TB_KEYS = (
    "Episode/mean_episode_length",
    "Train/mean_episode_length",
    "Episode/rew_tracking_lin_vel",
    "Episode/rew_termination",
    "Env/dfh_applied_drag_pct_weight",
    "Env/dfh_total_drag_per_robot_n",
    "Env/dfh_total_drag_per_robot_p95_n",
    "Env/dfh_stance_drag_contact_p95_n",
    "Env/dfh_drag_clipped_frac",
    "Env/dfh_stance_fraction",
)


@dataclass(frozen=True)
class RunMetrics:
    train_id: str
    eval_name: str
    seed: int
    ep_len: float
    track_lv: float
    fall_rate: float
    drag_pct: float
    total_drag_n: float
    total_drag_p95_n: float
    stance_drag_p95_n: float
    clip_frac: float
    stance_frac: float


def _read_scalars(tfev_path: Path, keys: Iterable[str]) -> dict[str, float]:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    ea = EventAccumulator(str(tfev_path), size_guidance={"scalars": 100_000})
    ea.Reload()
    out: dict[str, float] = {}
    for k in keys:
        try:
            scalars = ea.Scalars(k)
        except KeyError:
            continue
        if not scalars:
            continue
        vals = [s.value for s in scalars[-20:]]
        out[k] = sum(vals) / len(vals)
    return out


def collect(root: Path) -> list[RunMetrics]:
    rows: list[RunMetrics] = []
    for eval_dir in sorted((root / "crosseval").glob("*_on_*")):
        name = eval_dir.name  # e.g. DFH_FULL_ultra_s1_on_dfh
        m = re.match(r"(?P<train>.+_s(?P<seed>\d+))_on_(?P<ev>rigid|dfh)$", name)
        if not m:
            continue
        train_id = m.group("train")
        seed = int(m.group("seed"))
        eval_name = m.group("ev")

        tfev = next(eval_dir.rglob("events.out.tfevents.*"), None)
        if tfev is None:
            continue
        s = _read_scalars(tfev, _TB_KEYS)
        ep_len = s.get("Episode/mean_episode_length") or s.get(
            "Train/mean_episode_length", 0.0
        )
        fall_rate = max(
            0.0, min(1.0, -s.get("Episode/rew_termination", 0.0) / 4.0)
        )
        rows.append(
            RunMetrics(
                train_id=train_id,
                eval_name=eval_name,
                seed=seed,
                ep_len=ep_len,
                track_lv=s.get("Episode/rew_tracking_lin_vel", 0.0),
                fall_rate=fall_rate,
                drag_pct=s.get("Env/dfh_applied_drag_pct_weight", 0.0),
                total_drag_n=s.get("Env/dfh_total_drag_per_robot_n", 0.0),
                total_drag_p95_n=s.get(
                    "Env/dfh_total_drag_per_robot_p95_n", 0.0
                ),
                stance_drag_p95_n=s.get(
                    "Env/dfh_stance_drag_contact_p95_n", 0.0
                ),
                clip_frac=s.get("Env/dfh_drag_clipped_frac", 0.0),
                stance_frac=s.get("Env/dfh_stance_fraction", 0.0),
            )
        )
    return rows


def _classify(train_id: str, dfh_prefix: str) -> str:
    if train_id.startswith(dfh_prefix):
        return "dfh_trained"
    if train_id.startswith("RIGID_TUNED_"):
        return "rigid_trained"
    if train_id.startswith("SHUFFLED_"):
        return "shuffled_trained"
    return "other"


def print_report(rows: list[RunMetrics], strength: str) -> None:
    dfh_prefix = f"DFH_FULL_{strength}_"

    by_seed: dict[int, dict[tuple[str, str], RunMetrics]] = {}
    for r in rows:
        seed_d = by_seed.setdefault(r.seed, {})
        seed_d[(_classify(r.train_id, dfh_prefix), r.eval_name)] = r

    print(f"\n=== Paired improvement (strength={strength}) ===")
    print("POSITIVE = DFH-trained policy survives longer on DFH eval than "
          "RIGID-trained. Claim needs: improvement>0 AND SHUFFLED degrades "
          "vs FULL, consistently across seeds.\n")
    header = (
        f"{'seed':<5}{'DFH@DFH':>9}{'RIGID@DFH':>11}{'SHUF@DFH':>10}"
        f"{'improv':>9}{'norm':>8}{'FULL-SHUF':>11}"
    )
    print(header)
    print("-" * len(header))

    improvements: list[float] = []
    full_minus_shuf: list[float] = []
    for seed, d in sorted(by_seed.items()):
        dfh = d.get(("dfh_trained", "dfh"))
        rig_d = d.get(("rigid_trained", "dfh"))
        rig_r = d.get(("rigid_trained", "rigid"))
        shuf = d.get(("shuffled_trained", "dfh"))
        if not (dfh and rig_d and rig_r):
            print(f"  seed {seed}: missing pieces; have {sorted(d)}")
            continue
        improv = dfh.ep_len - rig_d.ep_len
        norm = improv / max(1.0, rig_r.ep_len)
        fms = (dfh.ep_len - shuf.ep_len) if shuf else float("nan")
        improvements.append(improv)
        if shuf:
            full_minus_shuf.append(fms)
        shuf_ep = f"{shuf.ep_len:>10.1f}" if shuf else f"{'n/a':>10}"
        fms_s = f"{fms:>11.1f}" if shuf else f"{'n/a':>11}"
        print(
            f"{seed:<5}{dfh.ep_len:>9.1f}{rig_d.ep_len:>11.1f}{shuf_ep}"
            f"{improv:>9.1f}{norm:>8.3f}{fms_s}"
        )

    # --- Pass-4 R3 W3: FULL vs SHUFFLED matched-dose diagnostics ---
    print("\n=== FULL vs SHUFFLED dose match on DFH eval (must be ~equal; "
          "if dose differs, a SHUFFLED loss is NOT clean evidence) ===")
    dh = (
        f"{'seed':<5}{'cond':<9}{'drag%w':>8}{'totN':>8}{'totp95':>9}"
        f"{'stnp95':>9}{'clip%':>8}{'stnfrac':>9}"
    )
    print(dh)
    print("-" * len(dh))
    for seed, d in sorted(by_seed.items()):
        for cond, key in (("FULL", "dfh_trained"),
                          ("SHUF", "shuffled_trained")):
            r = d.get((key, "dfh"))
            if not r:
                continue
            print(
                f"{seed:<5}{cond:<9}{r.drag_pct * 100:>7.1f}%"
                f"{r.total_drag_n:>8.1f}{r.total_drag_p95_n:>9.1f}"
                f"{r.stance_drag_p95_n:>9.1f}{r.clip_frac * 100:>7.1f}%"
                f"{r.stance_frac:>9.3f}"
            )

    # --- Pass-4 R3 W1: cap-saturation flag ---
    print("\n=== Cap-saturation check (stance_drag_p95 near cap with low "
          "clip% can still mean active-contact saturation) ===")
    for r in rows:
        if r.eval_name != "dfh" or r.stance_drag_p95_n <= 0:
            continue
        flag = " <-- review: stance p95 high vs clip%" if (
            r.stance_drag_p95_n >= 480.0 and r.clip_frac < 0.05
        ) else ""
        print(
            f"  {r.train_id:<22} stance_p95={r.stance_drag_p95_n:>6.1f}N "
            f"clip={r.clip_frac * 100:>4.1f}% "
            f"tot/robot={r.total_drag_n:>5.1f}N "
            f"({r.total_drag_n / _HUNTER_WEIGHT_N * 100:>4.1f}% wt){flag}"
        )

    # --- Pass-4 R3 W2: seed-agreement verdict ---
    print("\n=== Seed-agreement verdict ===")
    if not improvements:
        print("  no complete (seed) tuples yet -- sweep still running")
        return
    same_sign = len({i > 0 for i in improvements}) == 1
    all_pos = all(i > 0 for i in improvements)
    shuf_ok = bool(full_minus_shuf) and all(
        f > 0 for f in full_minus_shuf
    )
    print(f"  improvements: {[round(i, 1) for i in improvements]}")
    print(f"  FULL-SHUFFLED: {[round(f, 1) for f in full_minus_shuf]}")
    print(f"  seeds agree in sign: {same_sign}")
    print(f"  all improvements > 0: {all_pos}")
    print(f"  SHUFFLED degrades vs FULL (all seeds): {shuf_ok}")
    if all_pos and shuf_ok:
        verdict = "POSITIVE pilot signal (extend to seeds 1-3 for claim)"
    elif same_sign and not all_pos:
        verdict = "NEGATIVE/NULL -- claim not supported"
    else:
        verdict = "INCONCLUSIVE -- seeds disagree; need seeds 1-3+"
    print(f"  >>> {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/DFHTuned", type=Path)
    ap.add_argument("--strength", default="ultra")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = collect(args.root)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys())
                               if rows else [f.name for f in
                                             RunMetrics.__dataclass_fields__.values()])
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))

    print_report(rows, args.strength)


if __name__ == "__main__":
    main()
