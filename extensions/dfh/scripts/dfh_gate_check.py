#!/usr/bin/env python
"""Generic DFH stage gate check for the S3 autopilot.

Usage: dfh_gate_check.py <run_dir>
Exit 0 = PASS (auto-continue), 1 = STOP (fail or borderline — needs review).

Auto-continue criteria are deliberately a touch lenient vs the per-stage
promotion gates Codex specified, so the chain does not stop on noise-level
misses (the chain has cleanly recovered from every borderline so far). But it
STOPS on any genuine failure signature so we never burn compute on a broken
chain while the user sleeps.

PASS (all must hold, median over last 3 save-points):
  - ep_len   >= 680
  - term     >= -1.80
  - lin_vel  >= 0.95
  - worst-of-last-3 ep_len >= 600   (catch a single collapsing tail)
  - stance_fraction >= 0.04                  (not collapsing to protective gait)
  - stance_drag_contact_mean_n <= 385        (loaded-contact saturation rail)
  - applied_drag_pct_weight <= 0.09

IMPORTANT (Codex thread 019e7ae3): global `dfh_drag_clipped_frac` is averaged
over ALL contacts and HIDES loaded-contact saturation when stance_fraction is
small (~0.03-0.05). It is NOT the gate. The real failure rail is the per-stance
loaded-contact drag mean. k=25/max=520 had clip_frac ~0.012 yet went unstable
because stance_drag_contact_mean_n hit 408N. Do not use clip_frac as the gate.
Also: `dfh_total_drag_per_robot_p95_n` is misnamed (it's per-contact p95) — ignore.
"""
import sys, glob, os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

THRESH = dict(med_ep=680.0, med_term=-1.80, med_lin=0.95, worst_ep=600.0,
              med_stance_frac=0.04, med_stance_drag_mean=385.0, med_drag_pct=0.09)


def main(run_dir):
    evs = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not evs:
        print(f"GATE: NO EVENT FILE in {run_dir}")
        return 1
    ea = EventAccumulator(evs[-1], size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]

    def arr(t):
        s = ea.Scalars(t)
        return np.array([x.step for x in s]), np.array([x.value for x in s])

    if "Train/mean_episode_length" not in tags:
        print(f"GATE: no ep_len scalar yet in {run_dir}")
        return 1

    ep_steps, ep_vals = arr("Train/mean_episode_length")
    _, term = arr("Episode/rew_termination")
    _, lin = arr("Episode/rew_tracking_lin_vel")
    _, snk = arr("Env/dfh_mean_sink_m")
    _, sfr = arr("Env/dfh_stance_fraction")
    _, sdm = arr("Env/dfh_stance_drag_contact_mean_n")
    _, dpw = arr("Env/dfh_applied_drag_pct_weight")

    def near(it_):
        return int(np.argmin(np.abs(ep_steps - it_)))

    saved = sorted(
        int(p.split("model_")[-1].split(".pt")[0])
        for p in glob.glob(os.path.join(run_dir, "model_*.pt"))
    )
    if len(saved) < 3:
        print(f"GATE: <3 ckpts in {run_dir}")
        return 1
    last3 = saved[-3:]
    l3_ep = [ep_vals[near(i)] for i in last3]
    l3_term = [term[near(i)] for i in last3]
    l3_lin = [lin[near(i)] for i in last3]
    l3_snk = [snk[near(i)] for i in last3]
    l3_sfr = [sfr[near(i)] for i in last3]
    l3_sdm = [sdm[near(i)] for i in last3]
    l3_dpw = [dpw[near(i)] for i in last3]

    med_ep = float(np.median(l3_ep))
    med_term = float(np.median(l3_term))
    med_lin = float(np.median(l3_lin))
    worst_ep = float(min(l3_ep))
    med_sfr = float(np.median(l3_sfr))
    med_sdm = float(np.median(l3_sdm))
    med_dpw = float(np.median(l3_dpw))

    print(f"GATE check for {os.path.basename(run_dir)} (last3={last3}):")
    print(f"  ep_len    med={med_ep:.1f} worst={worst_ep:.1f}  {[f'{v:.1f}' for v in l3_ep]}")
    print(f"  term      med={med_term:.2f}              {[f'{v:.2f}' for v in l3_term]}")
    print(f"  lin_vel   med={med_lin:.3f}              {[f'{v:.3f}' for v in l3_lin]}")
    print(f"  stance_fr med={med_sfr:.4f}            {[f'{v:.4f}' for v in l3_sfr]}")
    print(f"  stance_drag_mean_n med={med_sdm:.1f}     {[f'{v:.1f}' for v in l3_sdm]}")
    print(f"  drag_pct  med={med_dpw:.4f}            {[f'{v:.4f}' for v in l3_dpw]}")
    print(f"  sink                                  {[f'{v:.4f}' for v in l3_snk]}")

    checks = {
        "med_ep>=680": med_ep >= THRESH["med_ep"],
        "med_term>=-1.80": med_term >= THRESH["med_term"],
        "med_lin>=0.95": med_lin >= THRESH["med_lin"],
        "worst_ep>=600": worst_ep >= THRESH["worst_ep"],
        "med_stance_frac>=0.04": med_sfr >= THRESH["med_stance_frac"],
        "med_stance_drag_mean<=385": med_sdm <= THRESH["med_stance_drag_mean"],
        "med_drag_pct<=0.09": med_dpw <= THRESH["med_drag_pct"],
    }
    for k, v in checks.items():
        print(f"    {'OK ' if v else 'XX '} {k}")

    if all(checks.values()):
        print("GATE: PASS — auto-continue")
        return 0
    print("GATE: STOP — needs review")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
