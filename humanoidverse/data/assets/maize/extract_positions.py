#!/usr/bin/env python3
"""Extract maize plant (x, y, yaw, model_id) from a virtual_maize_field world.

We discard the z because the original z was the heightmap surface elevation —
in IsaacSim we raycast Z onto the actual terrain mesh at spawn time, just like
HumanoidVerse does for the robot's spawn anchor.

Reads `~/.ros/virtual_maize_field/generated.world` (or path on argv[1]).
Writes `maize_positions.json` next to this script.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DEFAULT_WORLD = Path.home() / ".ros" / "virtual_maize_field" / "generated.world"
OUT_PATH = Path(__file__).resolve().parent / "maize_positions.json"


def main() -> int:
    world = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORLD
    if not world.exists():
        print(f"ERROR: world not found at {world}", file=sys.stderr)
        return 1

    text = world.read_text()
    # Plant includes look like:
    # <include>
    #   <uri>model://maize_01</uri>
    #   <pose>X Y Z R P Yw</pose>
    #   <name>maize_01_NNNN</name>
    #   <static>false</static>
    # </include>
    pattern = re.compile(
        r"<include>\s*<uri>model://(maize_\d+)</uri>\s*"
        r"<pose>([-\d.eE+ ]+)</pose>\s*"
        r"<name>[^<]+</name>",
        re.DOTALL,
    )
    plants = []
    for m in pattern.finditer(text):
        model_id = m.group(1)  # "maize_01" or "maize_02"
        vals = m.group(2).split()
        if len(vals) < 6:
            continue
        x, y = float(vals[0]), float(vals[1])
        yaw = float(vals[5])
        plants.append({"model": model_id, "x": x, "y": y, "yaw": yaw})

    OUT_PATH.write_text(json.dumps(plants, indent=2))
    print(f"wrote {OUT_PATH} ({len(plants)} plants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
