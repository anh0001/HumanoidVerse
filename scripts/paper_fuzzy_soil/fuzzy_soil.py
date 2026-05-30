#!/usr/bin/env python3
"""
Fuzzy soil-state layer from docs/wcci2026_hunter.pdf §III-C (faithful).

Maps per-episode soil params (friction mu_f, normal stiffness kn) to an
interpretable difficulty index d via a Mamdani rule base. DESCRIPTIVE only
(paper: "without modifying the PPO policy architecture or adding new
observations"). "Showing it works" = showing d is a faithful, monotone
descriptor of measured task difficulty (see analyze_fuzzy.py).

Paper spec (verbatim where possible):
  Membership functions (Fig.4), piecewise-linear, 3 terms each:
    Traction(mu_f), breakpoints ~0.45/0.60/0.75:  {Low, Med, High}
    Support(kn, kN/m), breakpoints 150/300/400:    {Soft, Med, Hard}
  Rule base (eq.1), AND = min, aggregation = max over rules per consequent:
    Low traction              -> Hard      (any support)
    High traction + Med/Hard  -> Easy
    everything else           -> Moderate  (med traction, or high-traction+soft)
  Difficulty index (eq.2), singleton defuzzification, d in [0.2, 0.8]:
    d = (0.2*gEasy + 0.5*gMod + 0.8*gHard) / (gEasy + gMod + gHard + eps)
  Convention: HIGHER d = HARDER soil ("lower mu_f and lower kn yield higher d").

Rigid contact note: PhysX rigid ~ kn=1e6 N/m -> Support=Hard (1,others 0). Then
sweeping traction gives (Low,Hard)->Hard, (Med,Hard)->Mod, (High,Hard)->Easy, so
d is monotone in mu even at fixed rigid support. The support AXIS still cannot be
independently validated in PhysX (it's physically inert; see conclusion writeup),
but the index produces a valid difficulty ordering on the traction axis.
"""
from __future__ import annotations

EPS = 1e-6
RIGID_KN_KNM = 1.0e6  # PhysX rigid contact -> effectively infinite stiffness (kN/m)

# difficulty consequent singletons
S_EASY, S_MOD, S_HARD = 0.2, 0.5, 0.8


def _ramp_down(x, a, b):
    if x <= a: return 1.0
    if x >= b: return 0.0
    return (b - x) / (b - a)


def _ramp_up(x, a, b):
    if x <= a: return 0.0
    if x >= b: return 1.0
    return (x - a) / (b - a)


def _tri(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)


# ---- traction memberships (from mu_f), breakpoints 0.45 / 0.60 / 0.75 ----
def trac_low(mu):  return _ramp_down(mu, 0.45, 0.60)
def trac_med(mu):  return _tri(mu, 0.45, 0.60, 0.75)
def trac_high(mu): return _ramp_up(mu, 0.60, 0.75)


# ---- support memberships (from kn in kN/m), breakpoints 150 / 300 / 400 ----
def sup_soft(kn): return _ramp_down(kn, 150.0, 300.0)
def sup_med(kn):  return _tri(kn, 150.0, 300.0, 400.0)
def sup_hard(kn): return _ramp_up(kn, 300.0, 400.0)


def difficulty_index(mu, kn_knm):
    """Paper eq.2. Higher d = harder. kn_knm in kN/m. Returns d in [~0.2, ~0.8]."""
    tl, tm, th = trac_low(mu), trac_med(mu), trac_high(mu)
    ss, sm, sh = sup_soft(kn_knm), sup_med(kn_knm), sup_hard(kn_knm)

    # firing strengths alpha_ij = min(traction_i, support_j)
    # consequents per rule base:
    #   Low traction -> Hard (any support)
    #   High traction + Med/Hard support -> Easy
    #   else -> Moderate  (Med traction any; High+Soft)
    rules = [
        (min(tl, ss), "Hard"), (min(tl, sm), "Hard"), (min(tl, sh), "Hard"),
        (min(tm, ss), "Mod"),  (min(tm, sm), "Mod"),  (min(tm, sh), "Mod"),
        (min(th, ss), "Mod"),  (min(th, sm), "Easy"), (min(th, sh), "Easy"),
    ]
    g_easy = max([a for a, c in rules if c == "Easy"], default=0.0)
    g_mod  = max([a for a, c in rules if c == "Mod"],  default=0.0)
    g_hard = max([a for a, c in rules if c == "Hard"], default=0.0)
    return (S_EASY * g_easy + S_MOD * g_mod + S_HARD * g_hard) / (g_easy + g_mod + g_hard + EPS)


def memberships(mu, kn_knm):
    return {
        "mu": mu, "kn_knm": kn_knm,
        "trac_low": trac_low(mu), "trac_med": trac_med(mu), "trac_high": trac_high(mu),
        "sup_soft": sup_soft(kn_knm), "sup_med": sup_med(kn_knm), "sup_hard": sup_hard(kn_knm),
        "d": difficulty_index(mu, kn_knm),
    }


if __name__ == "__main__":
    print("== Fuzzy difficulty index self-test (paper eq.2) ==")
    print("Convention: higher d = HARDER soil. d in [0.2, 0.8].\n")
    print(f"{'mu':>5} | {'d(rigid/Hard)':>13} | {'d(kn=250/Med)':>13} | {'d(kn=50/Soft)':>13}")
    for mu in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
        print(f"{mu:>5.2f} | {difficulty_index(mu, RIGID_KN_KNM):>13.3f} | "
              f"{difficulty_index(mu, 250.0):>13.3f} | {difficulty_index(mu, 50.0):>13.3f}")
    print("\nAt rigid support, d should fall monotonically as mu rises")
    print("(Hard->Moderate->Easy). That monotone column is what the mu-sweep validates.")
