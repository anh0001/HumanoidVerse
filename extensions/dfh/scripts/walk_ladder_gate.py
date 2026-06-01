#!/usr/bin/env python
"""Classify a walk-ladder rung's eval log (Codex 019e80a8).

Usage: walk_ladder_gate.py <eval_log>
Exit codes: 0 = PROMOTE, 2 = BORDERLINE, 1 = KILL/parse-fail.

Gates (paired with forward-progress, NOT vel_err alone):
  PROMOTE   : vel_err<=0.13 AND avg_dist>=4.0 AND ep_len>=900 AND falls<=10
  KILL      : vel_err>0.16 OR avg_dist<3.5 OR ep_len<800 OR falls>16
  BORDERLINE: anything in between
"""
import sys, re


def parse(log_path):
    txt = open(log_path, encoding="utf-8", errors="ignore").read()

    def grab(pat):
        m = re.findall(pat, txt)
        return float(m[-1]) if m else None

    return {
        "vel_err": grab(r"Velocity error \(m/s\):\s*([0-9.]+)"),
        "avg_dist": grab(r"Average distance per episode:\s*([0-9.]+)"),
        "ep_len": grab(r"Average episode length:\s*([0-9.]+)"),
        "falls": grab(r"Total falls:\s*([0-9.]+)"),
    }


def main(log_path):
    m = parse(log_path)
    print(f"GATE inputs: {m}")
    if any(v is None for v in m.values()):
        print("GATE: KILL (could not parse all metrics)")
        return 1
    ve, dist, ep, falls = m["vel_err"], m["avg_dist"], m["ep_len"], m["falls"]

    promote = ve <= 0.13 and dist >= 4.0 and ep >= 900 and falls <= 10
    kill = ve > 0.16 or dist < 3.5 or ep < 800 or falls > 16

    if promote:
        print("GATE: PROMOTE")
        return 0
    if kill:
        print(f"GATE: KILL (vel_err={ve} dist={dist} ep={ep} falls={falls})")
        return 1
    print(f"GATE: BORDERLINE (vel_err={ve} dist={dist} ep={ep} falls={falls})")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
