#!/usr/bin/env python3
"""Upload the fuzzy-curriculum ablation policies (Set B) to the EXISTING HF repo,
ADDING a fuzzy_curriculum/ folder + a model-card section. Does NOT delete anything."""
from __future__ import annotations
import json, os, shutil, tempfile
from huggingface_hub import HfApi

REPO = "anhrisn/hunter-dfh-locomotion"
ROOT = "logs/FuzzySoilFurrowsSetB"
api = HfApi()

stage = tempfile.mkdtemp(prefix="fuzzy_hf_")
fc = os.path.join(stage, "fuzzy_curriculum"); os.makedirs(fc)
manifest = {"arms": {}, "verdict": "fuzzy ~= crisp ~= fixed (Welch t=0.45, n=3); no fuzzy-specific value"}
for arm in ("fixed", "crisp", "fuzzy"):
    for s in (1, 2, 3):
        m = f"{ROOT}/setB_{arm}_seed{s}/manifest.json"
        d = json.load(open(m)); ck = d["final_ckpt"]
        cfg = os.path.join(os.path.dirname(ck), "config.yaml")
        dst = os.path.join(fc, f"{arm}_seed{s}"); os.makedirs(dst)
        shutil.copy(ck, os.path.join(dst, os.path.basename(ck)))
        if os.path.isfile(cfg):
            shutil.copy(cfg, os.path.join(dst, "config.yaml"))
        manifest["arms"].setdefault(arm, []).append(dict(seed=s, ckpt=os.path.basename(ck),
                                                          mu_path=d.get("mu_path")))
json.dump(manifest, open(os.path.join(fc, "manifest.json"), "w"), indent=2)

open(os.path.join(fc, "README.md"), "w").write(
"""# fuzzy_curriculum — Set B adaptive-vs-fixed friction-curriculum ablation (rigid PhysX)

3 arms x 3 seeds = 9 PPO-ROA Hunter policies, block-chained friction curriculum on
furrowed terrain (rigid PhysX). Arms differ ONLY in how the next soil difficulty is
chosen from a shared mu-ladder:
- `fixed_seed*`  : preset linear schedule (no performance feedback)
- `crisp_seed*`  : hard-threshold gate (advance when mastery >= T)
- `fuzzy_seed*`  : smooth fuzzy-accumulator pacing (the paper-style Mamdani index)

**Result:** held-out robustness AUC fixed 0.643 / crisp 0.680 / fuzzy 0.724 — differences
NOT significant (fuzzy vs crisp Welch t=0.45, n=3). The fuzzy curriculum confers no
demonstrable advantage over a crisp threshold; in rigid PhysX the support axis is inert.
See the repo's DFH Set-C study (`fuzzy_setC_dfh_result.md`) for the deformable-soil test
where the support axis IS physical and fuzzy STILL adds nothing. Each folder ships its
training-time `config.yaml`. These are RIGID-PhysX rigid-furrow policies, distinct from
the DFH `walkers/`.
""")

print("staged:", os.listdir(fc))
# ADD folder to the repo (upload_folder only touches fuzzy_curriculum/, leaves the rest)
api.upload_folder(repo_id=REPO, folder_path=fc, path_in_repo="fuzzy_curriculum",
                  commit_message="add fuzzy_curriculum (Set B adaptive-vs-fixed ablation, 9 policies)")
print("uploaded fuzzy_curriculum/ to", REPO)
shutil.rmtree(stage, ignore_errors=True)
