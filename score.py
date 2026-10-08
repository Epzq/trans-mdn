"""Score how likely each state x[t] is under the model, given x[<t].

For every step t >= 1 of a trajectory this reports:
  log_lik     log-likelihood log p(x[t] | x[<t]) in normalized units (a log density,
              not a log probability: it can be positive and has no fixed scale)
  percentile  fraction of validation transitions with a lower log_lik
              (small = less likely than typical transitions)
  p_value     P(log p(x~) <= log p(x[t])) for x~ sampled from the predicted
              mixture: how often the model's own prediction produces something
              this unlikely (small = surprising given this context)
  worst_dim   state dimension contributing least to log_lik

CLI:
  python score.py --ckpt runs/mine/best.pt --val                        # all held-out trajectories
  python score.py --ckpt runs/mine/best.pt --traj states.npy --index 3  # one trajectory
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

from data import build_datasets, load_source, load_trajectories, noop_start, split_indices
from model import CausalStateTransformer, component_log_probs, log_prob, mc_p_value, select


class Scorer:
    def __init__(self, ckpt_path, device="cpu", n_mc=1000):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.device = torch.device(device)
        self.model = CausalStateTransformer(**ckpt["model_cfg"]).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.mean, self.std = ckpt["mean"], ckpt["std"]
        self.noise_std = ckpt.get("noise_std_raw")  # training tolerance tau (raw units), None if none
        self.train_args = ckpt["args"]
        # Older checkpoints don't store context_len; derive it from the training window.
        self.context_len = self.model.context_len or self.train_args["seq_len"] - 1
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
        _, val_ds = build_datasets(a["data"], a["seq_len"], a["stride"], a["val_frac"], a["seed"], a.get("trim_noops"))
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

        Context is capped at context_len = seq_len - 1, the longest history the
        model was trained on: later steps are predicted from a sliding window of
        the last context_len states.
        """
        x = torch.as_tensor(x, device=self.device)
        T, L = len(x), self.context_len
        if T - 1 <= L:
            return select(self.model(x[None, :-1]), 0)
        parts = [select(self.model(x[None, :L]), 0)]  # predicts x[1..L]
        windows = x.unfold(0, L, 1).permute(0, 2, 1)[1 : T - L]  # window i predicts x[i+L]
        for chunk in windows.split(256):
            parts.append(select(self.model(chunk), (slice(None), -1)))
        return tuple(torch.cat(ps) for ps in zip(*parts))

    @torch.no_grad()
    def score(self, states, trim=True):
        """states: (T, D) raw array. Returns dict of arrays, length T-1, for t = 1..T-1."""
        # Trim leading no-ops the same way training did; t stays in original indices.
        eps = self.train_args.get("trim_noops")
        start = noop_start(np.asarray(states), eps) if (trim and eps is not None) else 0
        x = self.normalize(states[start:])
        if len(x) < 2:
            raise ValueError(f"nothing to score: trajectory has {len(x)} state(s) after trimming no-ops")
        params = self.predict(x)
        target = torch.as_tensor(x[1:], device=self.device)
        lp = log_prob(params, target)

        # Monte Carlo tail probability under each step's own predicted mixture.
        p_value = mc_p_value(params, target, self.n_mc)

        # Per-dim contribution, weighted by each component's posterior responsibility.
        comp = component_log_probs(params, target)  # (T-1, K, D)
        resp = F.softmax(F.log_softmax(params[0], -1) + comp.sum(-1), -1)  # (T-1, K)
        per_dim = (resp.unsqueeze(-1) * comp).sum(-2)  # (T-1, D)

        lp = lp.cpu().numpy()
        return {
            "t": np.arange(start + 1, start + len(x)),
            "trim_start": start,
            "log_lik": lp,
            "percentile": np.searchsorted(self.ref, lp) / len(self.ref),
            "p_value": p_value.cpu().numpy(),
            "worst_dim": per_dim.argmin(-1).cpu().numpy(),
            "per_dim_log_lik": per_dim.cpu().numpy(),
        }


def print_scores(name, states, out, alpha, show_all):
    flagged = (out["percentile"] < alpha) | (out["p_value"] < alpha)
    trimmed = f" ({out['trim_start']} leading no-ops trimmed)" if out["trim_start"] else ""
    print(f"\n{name}: {len(states)} states{trimmed}, {flagged.sum()} / {len(flagged)} steps flagged, "
          f"min log_lik {out['log_lik'].min():.2f}")
    if not (show_all or flagged.any()):
        return flagged
    # State values x[t] in raw units; the worst_dim value is shown in [brackets].
    dims = "".join(f"{f'x{d}':>11s}" for d in range(states.shape[-1]))
    print(f"{'t':>5s} {'log_lik':>9s} {'pctile':>7s} {'p_value':>8s} {'worst_dim':>9s}   {dims}")
    for i in range(len(out["t"])):
        if show_all or flagged[i]:
            t, w = out["t"][i], out["worst_dim"][i]
            vals = "".join(f"[{v:9.4g}]" if d == w else f" {v:9.4g} " for d, v in enumerate(states[t]))
            print(
                f"{t:5d} {out['log_lik'][i]:9.2f} {out['percentile'][i]:7.3f} "
                f"{out['p_value'][i]:8.3f} {w:9d} {'*' if flagged[i] else ' '} {vals}"
            )
    return flagged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val", action="store_true", help="score every trajectory held out during training")
    p.add_argument("--traj", type=str, default=None, help=".npy (T, D) trajectory, or any dataset file with --index")
    p.add_argument("--index", type=int, default=0, help="which trajectory in a multi-trajectory file")
    p.add_argument("--alpha", type=float, default=0.01, help="flag steps with percentile or p_value below this")
    p.add_argument("--n_mc", type=int, default=1000)
    p.add_argument("--all", action="store_true", help="print every step, not just flagged ones")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.val == bool(args.traj):
        p.error("pass exactly one of --val or --traj")

    scorer = Scorer(args.ckpt, args.device, args.n_mc)
    print(f"validation reference: {len(scorer.ref)} transitions, median log_lik {np.median(scorer.ref):.2f}")
    if scorer.noise_std is None:
        print("tolerance: none (trained on clean demonstrations)")
    else:
        print("tolerance tau per dim (raw units): " + " ".join(f"{v:.4g}" for v in scorer.noise_std))

    if args.traj:
        states = load_trajectories(args.traj)[args.index]
        print_scores(f"trajectory {args.index}", states, scorer.score(states), args.alpha, args.all)
        return

    a = scorer.train_args
    trajs = load_source(a["data"], a["seq_len"], a["seed"])
    if len(trajs) < 2:
        p.error("--val needs a multi-trajectory dataset (single-trajectory runs hold out a tail, not whole trajectories)")
    _, val_idx = split_indices(len(trajs), a["val_frac"], a["seed"])
    if len(val_idx) < 10:
        print(f"note: only {len(val_idx)} held-out trajectories, and they are also the percentile reference, "
              "so percentiles compare them mostly to themselves; rely on p_value")

    n_flag = n_steps = 0
    for i in sorted(val_idx):
        a_eps = a.get("trim_noops")
        if len(trajs[i]) - (noop_start(trajs[i], a_eps) if a_eps is not None else 0) < 2:
            continue
        flagged = print_scores(f"trajectory {i}", trajs[i], scorer.score(trajs[i]), args.alpha, args.all)
        n_flag, n_steps = n_flag + flagged.sum(), n_steps + len(flagged)
    print(f"\nheld-out total: {len(val_idx)} trajectories, {n_flag} / {n_steps} steps flagged "
          f"({100 * n_flag / max(n_steps, 1):.2f}%; {100 * args.alpha:g}-{200 * args.alpha:g}% expected by chance "
          f"for normal data, since each of the two tests flags {100 * args.alpha:g}%)")


if __name__ == "__main__":
    main()
