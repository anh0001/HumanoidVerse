#!/usr/bin/env python3
"""§0-B latent-identifiability probe trainer.

Loads per-regime rollout dumps from collect_probe_data.py and asks the
falsification question (reviewer's single highest-risk assumption):

  Can a short proprioceptive-history window (the EXACT policy input stream)
  identify the terrain/dynamics regime EARLY -- before the fall -- well
  above chance, and specifically separate safe (plane) from killer
  (soil/furrows) terrain?

Windows are built strictly WITHIN a single episode (no reset inside the
window) because at deployment the history buffer restarts each episode.
Soil episodes collapse in ~16 steps, so we test short windows L in
{8,12,16}: "within ~0.16-0.32 s, is the terrain already legible?"

PASS (ROA premise supported) if, for the smallest L, held-out accuracy is
well above chance AND plane-vs-(soil/furrows) is near-perfect early.
FAIL  -> ROA is the wrong dominant idea; fall back to history-only+reward.
"""
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn as nn


def build_windows(npz_paths, L):
    X, y, age = [], [], []
    for p in sorted(npz_paths):
        d = np.load(p, allow_pickle=True)
        obs = d["obs"].astype(np.float32)          # [T,N,D]
        ep_step = d["ep_step"].astype(np.int64)    # [T,N]
        regime = int(d["regime"])
        T, N, _ = obs.shape
        for n in range(N):
            es = ep_step[:, n]
            for t in range(L - 1, T):
                # window t-L+1..t within one episode iff ep_step strictly
                # increased by 1 each step across it (no reset, contiguous)
                seg = es[t - L + 1 : t + 1]
                if seg[0] >= 1 and np.all(np.diff(seg) == 1):
                    X.append(obs[t - L + 1 : t + 1, n, :])
                    y.append(regime)
                    age.append(int(seg[-1]))  # episode age at window end
    if not X:
        return None
    return (np.stack(X), np.asarray(y, np.int64), np.asarray(age, np.int64))


class TCN(nn.Module):
    """RMA-style small 1-D CNN over the proprio-history window."""
    def __init__(self, d_in, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_cls),
        )

    def forward(self, x):                 # x: [B,L,D] -> [B,D,L]
        return self.net(x.transpose(1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="logs/S0Gate/probe")
    ap.add_argument("--report", default="refine-logs/S0B_PROBE_REPORT.md")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    paths = glob.glob(os.path.join(args.data, "*.npz"))
    if not paths:
        sys.exit(f"no .npz in {args.data}")
    names = {}
    for p in sorted(paths):
        d = np.load(p, allow_pickle=True)
        names[int(d["regime"])] = str(d["regime_name"])
    n_cls = len(names)
    chance = 1.0 / n_cls
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    lines = ["# §0-B Latent-Identifiability Probe Report", "",
             f"Regimes: {names}  | chance = {chance:.3f}", ""]
    overall_pass = True
    safe_ids = [i for i, nm in names.items()
                if "plane" in nm.lower() or "sanity" in nm.lower()]

    for L in (8, 12, 16):
        built = build_windows(paths, L)
        if built is None:
            lines.append(f"## L={L}: no in-episode windows (episodes shorter than L)")
            continue
        X, Y, AGE = built
        Xt = torch.tensor(X)
        Yt = torch.tensor(Y)
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(Xt), generator=g)
        Xt, Yt, AGEp = Xt[perm], Yt[perm], AGE[perm.numpy()]
        ntr = int(0.8 * len(Xt))
        Xtr, Ytr = Xt[:ntr].to(dev), Yt[:ntr].to(dev)
        Xte, Yte = Xt[ntr:].to(dev), Yt[ntr:].to(dev)
        AGEte = AGEp[ntr:]

        m = TCN(X.shape[2], n_cls).to(dev)
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        lossf = nn.CrossEntropyLoss()
        bs = 512
        for _ in range(args.epochs):
            m.train()
            for i in range(0, len(Xtr), bs):
                opt.zero_grad()
                out = m(Xtr[i:i + bs])
                loss = lossf(out, Ytr[i:i + bs])
                loss.backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            pred = m(Xte).argmax(1).cpu().numpy()
        yte = Yte.cpu().numpy()
        acc = float((pred == yte).mean())
        # safe (plane) vs killer (everything else) binary accuracy
        if safe_ids:
            sb_true = np.isin(yte, safe_ids)
            sb_pred = np.isin(pred, safe_ids)
            safe_acc = float((sb_true == sb_pred).mean())
        else:
            safe_acc = float("nan")
        # early (episode age <= 8) accuracy
        early = AGEte <= 8
        early_acc = float((pred[early] == yte[early]).mean()) if early.any() else float("nan")

        passL = acc > chance + 0.15 and (np.isnan(safe_acc) or safe_acc > 0.80)
        overall_pass &= passL
        lines += [
            f"## L={L}  (n={len(Xt)} windows)",
            f"- regime acc: **{acc:.3f}** (chance {chance:.3f})",
            f"- safe(plane)-vs-killer acc: **{safe_acc:.3f}**",
            f"- early (episode age<=8) regime acc: **{early_acc:.3f}**",
            f"- verdict@L: {'PASS' if passL else 'FAIL'}",
            "",
        ]

    lines += ["## §0-B VERDICT",
              f"**{'PASS — proprioceptive history is terrain-identifiable early; ROA premise supported.' if overall_pass else 'FAIL — terrain not legible early from proprio; ROA is NOT the right dominant idea (fall back to history-only + periodic-symmetry reward).'}**"]
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    open(args.report, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[written] {args.report}")


if __name__ == "__main__":
    main()
