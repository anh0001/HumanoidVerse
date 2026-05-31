#!/usr/bin/env python3
"""
Set B — curriculum-pacing controllers for the block-chained ground-friction
curriculum (see docs/experiments/fuzzy_setB_plan.md).

Promotes the paper's fuzzy machinery (wcci2026_hunter.pdf §III-C: piecewise-linear
memberships, min-AND, max-aggregation, singleton defuzzification — eq.1-2) from a
PASSIVE soil-difficulty LABEL to an ACTIVE curriculum-pacing FUNCTION that decides
when to advance the global ground-friction rung.

Three controllers, all over the SAME inputs / ladder / max step, so the ONLY
variable is the decision rule (this is the fuzzy-vs-crisp non-negotiable):

  - fixed   : ignore performance; advance one rung every `blocks_per_rung` blocks.
  - crisp   : HARD all-or-nothing threshold. Advance one rung iff mastery m >= T,
              else hold (a classic crisp-bin curriculum gate).
  - fuzzy   : SMOOTH proportional pacing — its actual selling point. A Mamdani over
              mastery m yields an advance score a in [0,1]; we ACCUMULATE it
              (progress += a each block) and advance one rung whenever progress
              crosses 1.0. So high mastery (a~1) advances ~every block, medium
              (a~0.5) ~every other block, low (a~0) never. Same one-rung-per-block
              cap as crisp (a<=1 => progress rises <=1/block).

Why accumulation, not a threshold on a: thresholding a defuzzified scalar on a 1-D
signal is just a relabeled crisp threshold (tautologically ~= crisp). Accumulating
the graded score is the genuine "smooth interpolation vs hard bins" contrast the
fuzzy claim rests on — so fuzzy CAN differ from crisp, and we test whether it helps.

`mastery` m in [0,1] is the per-block performance signal = mean episode length at
the current rung / the episode-length cap. Identical signal for crisp and fuzzy.

WHY mastery (not the paper's soil index d) is the controller input: the paper's d
describes how hard the SOIL is (a function of mu and kn). Advancement needs a
PERFORMANCE signal (advance once the policy has mastered the current rung), which d
does not provide. Because kn is physically inert in PhysX (Set A / Arm D), adding d
as a second controller input would be inert by construction — exactly the property
under test. We therefore drive the controller with the 1-D mastery signal and only
LOG d(mu) of the current rung for interpretability. See plan for the honest scope.

The fuzzy memberships reuse the SAME piecewise-linear shape style as the paper's
traction/support sets (ramps + triangle), so the fuzzy controller is a faithful
transplant of the paper's inference structure onto the pacing task.
"""
from __future__ import annotations
from dataclasses import dataclass, field

EPS = 1e-6

# ---- piecewise-linear membership primitives (same shapes as fuzzy_soil.py) ----
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


# advance-score consequent singletons (Hold / Maybe / Advance)
A_HOLD, A_MAYBE, A_ADVANCE = 0.0, 0.5, 1.0


def fuzzy_advance_score(m: float, lo: float, mid: float, hi: float) -> float:
    """Mamdani over mastery -> advance score in [0,1] (singleton defuzzification).

    Memberships mirror the paper's traction breakpoint style (3 terms, ramp/tri/ramp)
    over breakpoints (lo, mid, hi). These MUST be calibrated to the achievable mastery
    range on this stack (the episode cap >> typical ep_len, so mastery is small);
    centred so the 50%-score crossing sits at `mid` ~ the crisp threshold.
    Rule base (paper's min-AND / max-aggregation structure):
      Low mastery -> Hold ; Med -> Maybe ; High -> Advance
    """
    ml = _ramp_down(m, lo, mid)
    mm = _tri(m, lo, mid, hi)
    mh = _ramp_up(m, mid, hi)
    return (A_HOLD * ml + A_MAYBE * mm + A_ADVANCE * mh) / (ml + mm + mh + EPS)


@dataclass
class CurriculumController:
    """Decides the friction rung index for each block of a block-chained run.

    rungs: friction ladder, EASY->HARD, e.g. [0.80, 0.65, 0.50, 0.35] (mu).
    arm: 'fixed' | 'crisp' | 'fuzzy'.
    total_blocks: number of training blocks in the run.
    ep_len_cap: episode-length cap (steps) used to normalize mastery.
    crisp_threshold: mastery threshold T for the crisp arm.
    fuzzy_threshold: advance-score threshold for the fuzzy arm (default 0.5).
    """
    rungs: list
    arm: str
    total_blocks: int
    ep_len_cap: float = 1000.0              # 20s * 50Hz control (fps200/decim4); from config
    crisp_threshold: float = 0.50           # mastery T; CALIBRATE to pre-flight ep_len scale
    # fuzzy mastery breakpoints (lo, mid, hi); mid ~ crisp_threshold so the two arms
    # gate at a comparable mastery level (isolating shape, not threshold offset).
    fuzzy_lo: float = 0.35
    fuzzy_mid: float = 0.50
    fuzzy_hi: float = 0.65

    rung_idx: int = 0                       # current rung (0 = easiest)
    fuzzy_progress: float = 0.0             # fuzzy arm's accumulated advance score
    history: list = field(default_factory=list)

    def __post_init__(self):
        assert self.arm in ("fixed", "crisp", "fuzzy"), self.arm
        assert len(self.rungs) >= 2
        # fixed arm: advance one rung every blocks_per_rung blocks, spread evenly.
        self._blocks_per_rung = max(1, self.total_blocks // (len(self.rungs) - 1))

    @property
    def current_mu(self) -> float:
        return self.rungs[self.rung_idx]

    def mastery(self, mean_ep_len: float) -> float:
        if mean_ep_len is None:
            return 0.0
        return max(0.0, min(1.0, mean_ep_len / max(self.ep_len_cap, EPS)))

    def decide_next(self, block_idx: int, mean_ep_len: float):
        """Given the just-finished block's metric, decide the rung for the NEXT block.

        Returns a dict record (also appended to self.history). Mutates self.rung_idx.
        At most ONE rung advance per call (shared cap across crisp & fuzzy).
        """
        m = self.mastery(mean_ep_len)
        prev_idx = self.rung_idx
        advanced = False
        score = None

        if self.arm == "fixed":
            # performance-independent: target rung from a fixed linear schedule.
            target = min(len(self.rungs) - 1, (block_idx + 1) // self._blocks_per_rung)
            if target > self.rung_idx:
                self.rung_idx = min(self.rung_idx + 1, target)  # cap 1 rung/block
                advanced = True
        elif self.arm == "crisp":
            # hard all-or-nothing gate at threshold T
            if m >= self.crisp_threshold and self.rung_idx < len(self.rungs) - 1:
                self.rung_idx += 1
                advanced = True
        elif self.arm == "fuzzy":
            # smooth proportional pacing: accumulate the graded advance score
            score = fuzzy_advance_score(m, self.fuzzy_lo, self.fuzzy_mid, self.fuzzy_hi)
            self.fuzzy_progress += score
            if self.fuzzy_progress >= 1.0 and self.rung_idx < len(self.rungs) - 1:
                self.rung_idx += 1
                self.fuzzy_progress -= 1.0     # carry remainder; cap 1 rung/block (score<=1)
                advanced = True

        rec = {
            "block": block_idx,
            "arm": self.arm,
            "mean_ep_len": mean_ep_len,
            "mastery": round(m, 4),
            "prev_rung_idx": prev_idx,
            "prev_mu": self.rungs[prev_idx],
            "fuzzy_score": (round(score, 4) if score is not None else ""),
            "fuzzy_progress": (round(self.fuzzy_progress, 4) if self.arm == "fuzzy" else ""),
            "advanced": advanced,
            "next_rung_idx": self.rung_idx,
            "next_mu": self.current_mu,
        }
        self.history.append(rec)
        return rec


if __name__ == "__main__":
    # Self-test: how each arm paces over 8 blocks given a mastery trajectory.
    rungs = [0.80, 0.65, 0.50, 0.35]
    cap = 1000.0   # 20s * 50Hz (from config)
    # mean_ep_len trajectory in the realistic achievable range (mastery ~0.45-0.55):
    # the regime just below the crisp T=0.50 where smooth (fuzzy) vs hard (crisp) diverge.
    traj = [470, 510, 450, 490, 460, 500, 440, 480]
    print(f"ladder(EASY->HARD) = {rungs}  ep_len_cap={cap}  blocks={len(traj)}")
    print(f"mastery traj = {[round(min(1.0,e/cap),2) for e in traj]}  (crisp T=0.50)")
    for arm in ("fixed", "crisp", "fuzzy"):
        c = CurriculumController(rungs=rungs, arm=arm, total_blocks=len(traj), ep_len_cap=cap)
        path = [c.current_mu]
        for b, el in enumerate(traj):
            rec = c.decide_next(b, el)
            path.append(rec["next_mu"])
        print(f"  {arm:6s}: mu path {path}")
    print("(fixed paces on schedule; crisp stalls if mastery never hits T; "
          "fuzzy creeps forward via accumulated partial scores)")
