"""Score how likely each state x[t] is under the model, given x[<t].

For every step t >= 1 of a trajectory this reports:
  log_prob    log p(x[t] | x[<t]) in normalized units (a density, not a probability)
  percentile  fraction of validation transitions with a lower log_prob
              (small = less likely than typical transitions)
  p_value     P(log p(x~) <= log p(x[t])) for x~ sampled from the predicted
              mixture: how often the model's own prediction produces something
              this unlikely (small = surprising given this context)
  worst_dim   state dimension contributing least to log_prob

CLI:
  python score.py --ckpt runs/mine/best.pt --traj states.npy --index 3
Python:
  scorer = Scorer("runs/mine/best.pt")
  out = scorer.score(states)          # states: (T, D) raw (unnormalized) array
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import build_datasets, load_trajectories
from model import CausalStateTransformer, component_log_probs, log_prob, sample, select


class Scorer:
    def __init__(self, ckpt_path, device="cpu", n_mc=1000):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.device = torch.device(device)
        self.model = CausalStateTransformer(**ckpt["model_cfg"]).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.mean, self.std = ckpt["mean"], ckpt["std"]
        self.train_args = ckpt["args"]
        self.seq_len = self.train_args["seq_len"]
        self.n_mc = n_mc
        self.ref = self._reference(ckpt_path)

    def normalize(self, states):
        return ((np.asarray(states, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    @torch.no_grad()
    def _reference(self, ckpt_path):
        """Sorted log p of every validation transition (cached next to the checkpoint)."""
        cache = os.path.splitext(ckpt_path)[0] + "_ref_logp.npy"
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(ckpt_path):
            return np.load(cache)
        a = self.train_args
        _, val_ds = build_datasets(a["data"], a["seq_len"], a["stride"], a["val_frac"], a["seed"])
        val_ds.mean, val_ds.std = self.mean, self.std
        ref = []
        for x, valid in DataLoader(val_ds, batch_size=128):
            x = x.to(self.device)
            lp = log_prob(self.model(x[:, :-1]), x[:, 1:]).cpu()
            ref.append(lp[valid[:, 1:]].numpy())
        ref = np.sort(np.concatenate(ref))
        np.save(cache, ref)
        return ref

    @torch.no_grad()
    def predict(self, x):
        """Mixture params for x[1..T-1] given the preceding states, x (T, D) normalized.

        Context is capped at seq_len (what the model was trained on): steps
        beyond it are predicted from a sliding window of the last seq_len states.
        """
        x = torch.as_tensor(x, device=self.device)
        T, L = len(x), self.seq_len
        if T <= L:
            return select(self.model(x[None, :-1]), 0)
        parts = [select(self.model(x[None, :L]), 0)]  # predicts x[1..L]
        windows = x.unfold(0, L, 1).permute(0, 2, 1)[1 : T - L]  # window i predicts x[i+L]
        for chunk in windows.split(256):
            parts.append(select(self.model(chunk), (slice(None), -1)))
        return tuple(torch.cat(ps) for ps in zip(*parts))

    @torch.no_grad()
    def score(self, states):
        """states: (T, D) raw array. Returns dict of arrays, length T-1, for t = 1..T-1."""
        x = self.normalize(states)
        params = self.predict(x)
        target = torch.as_tensor(x[1:], device=self.device)
        lp = log_prob(params, target)

        # Monte Carlo tail probability under each step's own predicted mixture.
        p_value = torch.empty_like(lp)
        for s in range(0, len(lp), 64):
            sl = slice(s, s + 64)
            p_s = select(params, sl)
            lp_samples = log_prob(p_s, sample(p_s, self.n_mc))  # (n_mc, chunk)
            p_value[sl] = (lp_samples <= lp[sl]).float().mean(0)

        # Per-dim contribution, weighted by each component's posterior responsibility.
        comp = component_log_probs(params, target)  # (T-1, K, D)
        resp = F.softmax(F.log_softmax(params[0], -1) + comp.sum(-1), -1)  # (T-1, K)
        per_dim = (resp.unsqueeze(-1) * comp).sum(-2)  # (T-1, D)

        lp = lp.cpu().numpy()
        return {
            "t": np.arange(1, len(x)),
            "log_prob": lp,
            "percentile": np.searchsorted(self.ref, lp) / len(self.ref),
            "p_value": p_value.cpu().numpy(),
            "worst_dim": per_dim.argmin(-1).cpu().numpy(),
            "per_dim_log_prob": per_dim.cpu().numpy(),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--traj", required=True, help=".npy (T, D) trajectory, or any dataset file with --index")
    p.add_argument("--index", type=int, default=0, help="which trajectory in a multi-trajectory file")
    p.add_argument("--alpha", type=float, default=0.01, help="flag steps with percentile or p_value below this")
    p.add_argument("--n_mc", type=int, default=1000)
    p.add_argument("--all", action="store_true", help="print every step, not just flagged ones")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    trajs = load_trajectories(args.traj)
    states = trajs[args.index]
    scorer = Scorer(args.ckpt, args.device, args.n_mc)
    out = scorer.score(states)

    flagged = (out["percentile"] < args.alpha) | (out["p_value"] < args.alpha)
    print(f"trajectory {args.index}: {len(states)} states, {flagged.sum()} / {len(flagged)} steps flagged (alpha={args.alpha})")
    print(f"validation reference: {len(scorer.ref)} transitions, median log_prob {np.median(scorer.ref):.2f}")
    print(f"{'t':>5s} {'log_prob':>9s} {'pctile':>7s} {'p_value':>8s} {'worst_dim':>9s}")
    for i in range(len(out["t"])):
        if args.all or flagged[i]:
            print(
                f"{out['t'][i]:5d} {out['log_prob'][i]:9.2f} {out['percentile'][i]:7.3f} "
                f"{out['p_value'][i]:8.3f} {out['worst_dim'][i]:9d}" + ("  *" if flagged[i] else "")
            )


if __name__ == "__main__":
    main()
