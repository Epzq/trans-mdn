"""Evaluate a checkpoint: next-state NLL, point-prediction MSE, and rollouts.

  python evaluate.py --ckpt runs/default/best.pt --prefix_len 16
"""

import argparse

import torch
from torch.utils.data import DataLoader

from data import StateSequenceDataset, load_trajectories, make_synthetic
from model import CausalStateTransformer, log_prob, mixture_mean


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", type=str, default=None, help="defaults to the training data source")
    p.add_argument("--prefix_len", type=int, default=16, help="observed states before rollout")
    p.add_argument("--max_batches", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = ckpt["args"]
    model = CausalStateTransformer(**ckpt["model_cfg"]).to(args.device).eval()
    model.load_state_dict(ckpt["model"])

    seq_len = targs["seq_len"]
    data_path = args.data or targs["data"]
    # Synthetic eval uses a different seed so it's unseen data.
    trajs = load_trajectories(data_path) if data_path else make_synthetic(seq_len=seq_len, seed=999, num_traj=500)
    ds = StateSequenceDataset(trajs, seq_len, mean=ckpt["mean"], std=ckpt["std"])
    loader = DataLoader(ds, batch_size=64)

    P = args.prefix_len
    H = seq_len - P
    # Sums of squared error (averaged over D) and counts of valid targets.
    # Padded targets are excluded via the valid mask.
    one_step = {"model": 0.0, "copy": 0.0, "linear": 0.0}
    rollout = {"model": 0.0, "copy": 0.0, "linear": 0.0}
    nll_sum, n_one, n_roll = 0.0, 0.0, 0.0
    per_step, per_step_n = torch.zeros(H), torch.zeros(H)

    def sse(err, w):
        return (err.mean(-1) * w).sum().item()

    for i, (x, valid) in enumerate(loader):
        if i >= args.max_batches:
            break
        x_dev = x.to(args.device)

        # One-step (teacher forced): distribution over x[t+1] from x[<=t], t >= 1.
        params = tuple(q.cpu()[:, 1:] for q in model(x_dev[:, :-1]))
        tgt, w = x[:, 2:], valid[:, 2:].float()
        nll_sum += (-log_prob(params, tgt) * w).sum().item()
        one_step["model"] += sse((mixture_mean(params) - tgt) ** 2, w)
        one_step["copy"] += sse((x[:, 1:-1] - tgt) ** 2, w)
        one_step["linear"] += sse((2 * x[:, 1:-1] - x[:, :-2] - tgt) ** 2, w)
        n_one += w.sum().item()

        # Deterministic rollout (highest-weight component mean): observe P
        # states, generate the remaining H. Windows whose real length is <= P
        # have no valid future and contribute nothing.
        gen = model.generate(x_dev[:, :P], H, mode="mode").cpu()[:, P:]
        future, w = x[:, P:], valid[:, P:].float()
        last, vel = x[:, P - 1 : P], x[:, P - 1 : P] - x[:, P - 2 : P - 1]
        steps = torch.arange(1, H + 1).view(1, H, 1)
        err = (gen - future) ** 2
        rollout["model"] += sse(err, w)
        rollout["copy"] += sse((last - future) ** 2, w)
        rollout["linear"] += sse((last + steps * vel - future) ** 2, w)
        n_roll += w.sum().item()
        per_step += (err.mean(-1) * w).sum(0)
        per_step_n += w.sum(0)

    print(f"one-step NLL per timestep (normalized units): {nll_sum / max(n_one, 1):.4f}")
    print("MSE in normalized units (predicting the mean ~ 1.0)")
    print(f"{'':22s} {'model':>8s} {'copy':>8s} {'linear':>8s}")
    print(f"{'one-step (mean)':22s} " + " ".join(f"{one_step[k] / max(n_one, 1):8.4f}" for k in one_step))
    print(f"{f'rollout ({H} steps)':22s} " + " ".join(f"{rollout[k] / max(n_roll, 1):8.4f}" for k in rollout))
    per_step /= per_step_n.clamp(min=1)
    marks = sorted({1, H // 4, H // 2, 3 * H // 4, H} - {0})
    print("model rollout MSE by horizon: " + "  ".join(f"t+{h}: {per_step[h - 1]:.4f}" for h in marks))


if __name__ == "__main__":
    main()
