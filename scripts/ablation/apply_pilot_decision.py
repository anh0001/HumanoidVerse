#!/usr/bin/env python3
"""Apply Codex's 4-branch pilot decision rule to logs/WarmstartPilot/pilot_summary.csv.

Reads per-arm × condition × ckpt_iter rows, takes the LAST (highest-iter)
checkpoint per (arm, condition), applies the rule, prints + writes
refine-logs/PILOT_VERDICT.md with the branch + ready-to-fire next commands.

Branches (Codex thread 019e4287):
  1. A9 ≫ A0/A3/A8 with nonzero locomotion -> relaunch full matrix from warm-start.
  2. ALL arms fail -> build staged terrain curriculum before any matrix.
  3. A0 also solves -> ROA+gait unnecessary; re-scope held-out / strengthen
     z-causality test.
  4. A3 or A8 ≈ A9 -> stack thesis weak; refine before spending seeds.

Run any time after the pilot finishes:
    python3 scripts/ablation/apply_pilot_decision.py
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path
from typing import NamedTuple

CSV = Path("logs/WarmstartPilot/pilot_summary.csv")
OUT = Path("refine-logs/PILOT_VERDICT.md")

# Thresholds (Codex spirit: "nonzero locomotion" = meaningfully above the
# A0_s1 fall-immediately floor of ep_len ~30, fall_rate ~1.0)
NONZERO_EP_LEN = 200          # ~4 s under control dt 0.02
NONZERO_FALL_RATE = 0.80      # less than 80% falls
NEAR_TIE_REL = 0.20           # arms within 20% relative ep_len are "matching"
A9_GAP_REL = 0.30             # A9 must beat baseline by 30% to count as ≫

ARMS = ("A0_baseline", "A3_hist50", "A8_gait", "A9_full")
COND = ("furrows_s3", "soil_chal")


class Cell(NamedTuple):
    ep_len: float
    fall_rate: float
    iter: int


def load_last_per(csv_path: Path) -> dict[tuple[str, str], Cell]:
    """{(arm, condition): Cell at last iter}."""
    rows: dict[tuple[str, str, int], Cell] = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            arm, cond = r["arm"], r["condition"]
            try:
                it = int(r["ckpt_iter"])
                ep = float(r["ep_len_steps"])
                fr = float(r["fall_rate"])
            except (ValueError, KeyError):
                continue
            rows[(arm, cond, it)] = Cell(ep, fr, it)
    last: dict[tuple[str, str], Cell] = {}
    for (arm, cond, it), cell in rows.items():
        key = (arm, cond)
        if key not in last or it > last[key].iter:
            last[key] = cell
    return last


def nonzero(cell: Cell) -> bool:
    return cell.ep_len >= NONZERO_EP_LEN and cell.fall_rate < NONZERO_FALL_RATE


def fmt_table(last: dict[tuple[str, str], Cell]) -> str:
    lines = [f"| arm | iter | {COND[0]} ep_len | {COND[0]} fall_rate "
             f"| {COND[1]} ep_len | {COND[1]} fall_rate |",
             "|---|---|---|---|---|---|"]
    for arm in ARMS:
        c0 = last.get((arm, COND[0]))
        c1 = last.get((arm, COND[1]))
        if not c0 and not c1:
            lines.append(f"| {arm} | — | missing | — | missing | — |")
            continue
        it = c0.iter if c0 else (c1.iter if c1 else "-")
        e0 = f"{c0.ep_len:.1f}" if c0 else "—"
        f0 = f"{c0.fall_rate:.3f}" if c0 else "—"
        e1 = f"{c1.ep_len:.1f}" if c1 else "—"
        f1 = f"{c1.fall_rate:.3f}" if c1 else "—"
        lines.append(f"| {arm} | {it} | {e0} | {f0} | {e1} | {f1} |")
    return "\n".join(lines)


def decide(last: dict[tuple[str, str], Cell]) -> tuple[int, str, list[str]]:
    """(branch_id 1-4, headline, bullet evidence)."""
    # Best held-out per arm (avg over conditions present)
    arm_ep: dict[str, float] = {}
    arm_fr: dict[str, float] = {}
    for arm in ARMS:
        cs = [last.get((arm, c)) for c in COND]
        cs = [c for c in cs if c is not None]
        if not cs:
            continue
        arm_ep[arm] = sum(c.ep_len for c in cs) / len(cs)
        arm_fr[arm] = sum(c.fall_rate for c in cs) / len(cs)

    missing = [a for a in ARMS if a not in arm_ep]
    if missing:
        return (0, f"INCOMPLETE: missing arm results for {missing}",
                ["Wait for pilot to finish all arms before applying rule."])

    a0, a3, a8, a9 = (arm_ep[a] for a in ARMS)
    fr0, fr3, fr8, fr9 = (arm_fr[a] for a in ARMS)

    bullets = [
        f"A0 baseline:  ep_len={a0:.1f}  fall_rate={fr0:.3f}",
        f"A3 hist50:    ep_len={a3:.1f}  fall_rate={fr3:.3f}",
        f"A8 gait:      ep_len={a8:.1f}  fall_rate={fr8:.3f}",
        f"A9 full:      ep_len={a9:.1f}  fall_rate={fr9:.3f}",
    ]

    # Branch 2: all fail
    if not any(arm_ep[a] >= NONZERO_EP_LEN and arm_fr[a] < NONZERO_FALL_RATE
               for a in ARMS):
        return (2, "ALL ARMS FAIL on held-out -> terrain too hard even after "
                   "plane warm-start. Build staged terrain curriculum before "
                   "any matrix.", bullets)

    # Branch 1: A9 nonzero AND A9 beats A0/A3/A8 by ≥A9_GAP_REL
    if (nonzero(Cell(a9, fr9, 0))
            and a9 >= a0 * (1 + A9_GAP_REL)
            and a9 >= a3 * (1 + A9_GAP_REL)
            and a9 >= a8 * (1 + A9_GAP_REL)):
        return (1, "A9 (ROA+gait) ≫ A0/A3/A8. Thesis supported -> RELAUNCH "
                   "FULL MATRIX from warm-start with seeds 1-3 (and add A5 "
                   "to isolate ROA-only).", bullets)

    # Branch 3: A0 also solves it (a0 within near-tie of a9)
    if nonzero(Cell(a0, fr0, 0)) and abs(a0 - a9) / max(a9, 1) <= NEAR_TIE_REL:
        return (3, "A0 baseline also solves it (A0 ≈ A9) -> ROA+gait may not "
                   "be necessary at this difficulty. Re-scope: harder held-out "
                   "terrain, stronger z-causality test, OR pivot the claim.",
                bullets)

    # Branch 4: A3 or A8 matches A9
    a3_match = (nonzero(Cell(a3, fr3, 0)) and abs(a3 - a9) / max(a9, 1)
                <= NEAR_TIE_REL)
    a8_match = (nonzero(Cell(a8, fr8, 0)) and abs(a8 - a9) / max(a9, 1)
                <= NEAR_TIE_REL)
    if a3_match or a8_match:
        culprit = "A3 (memory alone)" if a3_match else "A8 (gait alone)"
        return (4, f"{culprit} ≈ A9 -> full ROA+gait stack thesis weak. "
                   f"Refine before spending seeds (boost ROA coef, longer "
                   f"history, harder DR, OR drop the redundant component).",
                bullets)

    return (0, "INDETERMINATE: A9 nonzero but does not clearly beat baselines "
               "by the gap threshold. Inspect per-condition cells; consider "
               "adding seeds 2-3 before deciding.", bullets)


NEXT_CMDS = {
    1: (
        "# Branch 1: relaunch full matrix from warm-start with seeds 1-3 (+A5).\n"
        "# Each arm needs its own per-seed Stage P + Stage F warm-start chain.\n"
        "# Suggested: extend warmstart_pilot.sh to seed loop, ARMS adds A5_roa.\n"
        "# Quick command:\n"
        "PLANE_ITERS=500 FURROW_ITERS=1500 SAVE_EVERY=250 NUM_ENVS=2048 \\\n"
        "  EVAL_ENVS=256 EVAL_EPS=64 \\\n"
        "  ARMS='A0_baseline A3_hist50 A5_roa A8_gait A9_full' \\\n"
        "  SEEDS='1 2 3' nohup bash scripts/ablation/warmstart_pilot.sh \\\n"
        "  > logs/WarmstartMatrix_wave.log 2>&1 &\n"
        "# (warmstart_pilot.sh currently hardcodes SEED=1; add SEED loop first.)"
    ),
    2: (
        "# Branch 2: build staged terrain curriculum BEFORE any matrix.\n"
        "# Stage 0: plane (already validated by pilot Stage P).\n"
        "# Stage 1: shallow furrows / low DR / large env_spacing.\n"
        "# Stage 2: medium furrows (current furrows_s2).\n"
        "# Stage 3: full furrows + DR (current target).\n"
        "# Promotion criteria: ep_len threshold per stage from CLAUDE.md.\n"
        "# New configs needed under humanoidverse/config/curriculum/."
    ),
    3: (
        "# Branch 3: re-scope claim. Two paths:\n"
        "# a) Make held-out HARDER (terrain_furrows_stage3_full + heavier DR\n"
        "#    + perturbations) so baseline cannot solve it. Re-pilot.\n"
        "# b) Pivot claim to z-causality: show A9's learned latent zhat is\n"
        "#    actually USED by the policy (zero/shuffle zhat at eval ablation).\n"
        "#    Even if A0 ≈ A9 on episode length, mechanistic claim can hold."
    ),
    4: (
        "# Branch 4: stack thesis weak. Tune ROA before seeds:\n"
        "# - bump algo.config.roa.roa_coef (1.0 -> 3.0 or 5.0).\n"
        "# - extend history length (short_history term: 5 -> 20).\n"
        "# - increase DR severity so adaptation actually matters.\n"
        "# Re-pilot A0/A3 (or A8)/A9 to see if A9 separates after tuning."
    ),
}


def main():
    if not CSV.exists():
        print(f"NO CSV YET at {CSV} — pilot not finished.")
        sys.exit(2)
    last = load_last_per(CSV)
    table = fmt_table(last)
    branch, headline, bullets = decide(last)

    md = [
        "# Pilot Verdict (Codex 4-branch decision rule)",
        "",
        f"**Source:** `{CSV}`  ·  **Generated:** by `apply_pilot_decision.py`",
        "",
        "## Held-out results (last ckpt per arm × condition)",
        "",
        table,
        "",
        "## Verdict",
        "",
        f"**Branch {branch}** — {headline}",
        "",
        "### Evidence",
    ] + [f"- `{b}`" for b in bullets] + [
        "",
        "### Thresholds applied",
        f"- nonzero locomotion: ep_len ≥ {NONZERO_EP_LEN} steps "
        f"AND fall_rate < {NONZERO_FALL_RATE}",
        f"- near-tie (Branch 3/4): |Δ ep_len| / A9 ≤ {NEAR_TIE_REL}",
        f"- A9 ≫ baselines (Branch 1): A9 ≥ (1 + {A9_GAP_REL}) × each baseline",
        "",
        "### Ready-to-fire next step",
        "```bash",
        NEXT_CMDS.get(branch, "# no branch — see headline."),
        "```",
    ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[wrote] {OUT}")
    sys.exit(0 if branch in (1, 3) else (1 if branch in (2, 4) else 2))


if __name__ == "__main__":
    main()
