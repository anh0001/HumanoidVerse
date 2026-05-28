#!/usr/bin/env python3
"""Flatten the actor's log_std parameter to a constant value.

Used to warm-start DFH training from the v7 m4250 (rigid-furrow) PPOROA
checkpoint with a less stochastic policy — the trained std is per-joint and
trained-in for rigid contact, which over-explores foot-strikes on the more
compliant DFH terrain.

Default: 0.55 (Codex recipe — see thread 019e5e77-612e-7db2-a354-0966ef762c1a).

Usage:
    python3 extensions/dfh/scripts/patch_actor_std.py \\
        --src logs/FixedStageFv7/.../model_4250.pt \\
        --dst logs/FixedStageFv7/.../model_4250_std055.pt \\
        --std 0.55
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path("/home/anhar/codes/HumanoidVerse/logs/FixedStageFv7/"
                                 "20260525_060629-fixedFv7-locomotion-hunter/model_4250.pt"))
    ap.add_argument("--dst", type=Path,
                    default=Path("/home/anhar/codes/HumanoidVerse/logs/FixedStageFv7/"
                                 "20260525_060629-fixedFv7-locomotion-hunter/model_4250_std055.pt"))
    ap.add_argument("--std", type=float, default=0.55)
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ckpt["actor_model_state_dict"]
    before = sd["std"].clone()
    print(f"std before: min={before.min():.3f} max={before.max():.3f} mean={before.mean():.3f}")
    sd["std"].fill_(args.std)
    print(f"std after : flat {args.std}")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.dst)
    print(f"wrote {args.dst} ({os.path.getsize(args.dst) // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
