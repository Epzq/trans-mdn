"""Train a causal state transformer with a mixture density head (next-state NLL).

Examples:
  python train.py                                  # synthetic data, quick sanity run
  python train.py --data states.npy --seq_len 32 --stride 8 --n_components 5 --epochs 100
  python train.py --data states.npy --noise_frac 0.25            # tolerance = 1/4 of a typical step
  python train.py --data states.npy --noise_std 0.002            # tolerance = 0.002 raw units, all dims
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import build_datasets
from model import CausalStateTransformer, log_prob, mc_p_value, nll_loss


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data", type=str, default=None, help=".npy/.npz state file; synthetic if omitted")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--stride", type=int, default=None, help="window stride (default: seq_len)")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--trim_noops", type=float, default=None, metavar="EPS",
                   help="drop leading steps where every dim changes by < EPS (raw units)")
    # tolerance: Gaussian noise (std tau per dim, raw units) added to training windows only
    tol = p.add_mutually_exclusive_group()
    tol.add_argument("--noise_frac", type=float, default=None, metavar="C",
                     help="tau = C * median per-step |change| of each dim in the training data")
    tol.add_argument("--noise_std", type=float, nargs="+", default=None, metavar="TAU",
                     help="tau in raw units: one value for all dims, or one per dim (0 = no noise on that dim)")
    # model
    p.add_argument("--n_components", type=int, default=5, help="mixture components (1 = single Gaussian)")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    # optim
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_mc", type=int, default=200, help="MC samples for the val p<0.01 rate (0 = off)")
    # misc
    p.add_argument("--out_dir", type=str, default="runs/default")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def noise_tolerance(args, train_ds):
    """Per-dim tolerance tau in raw units from --noise_frac / --noise_std, or None."""
    D = train_ds.state_dim
    if args.noise_frac is not None:
        steps = np.concatenate([np.abs(np.diff(t, axis=0)) for t in train_ds.trajectories])
        return (args.noise_frac * np.median(steps, axis=0)).astype(np.float32)
    if args.noise_std is not None:
        tau = np.asarray(args.noise_std, dtype=np.float32)
        if tau.size == 1:
            return np.full(D, tau.item(), dtype=np.float32)
        if tau.size != D:
            raise ValueError(f"--noise_std takes 1 or {D} values (state_dim), got {tau.size}")
        return tau
    return None


@torch.no_grad()
def evaluate(model, loader, device, n_mc=200):
    """Teacher-forced next-state NLL per timestep, and the fraction of valid
    transitions with MC p-value < 0.01 (~1% if calibrated; higher = overconfident)."""
    model.eval()
    total, n_low, count = 0.0, 0.0, 0
    for x, valid in loader:
        x, valid = x.to(device), valid.to(device)
        params, target, w = model(x[:, :-1]), x[:, 1:], valid[:, 1:]
        total += -log_prob(params, target)[w].sum().item()
        if n_mc:
            n_low += (mc_p_value(params, target, n_mc)[w] < 0.01).sum().item()
        count += w.sum().item()
    model.train()
    return total / max(count, 1), (n_low / max(count, 1) if n_mc else float("nan"))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    train_ds, val_ds = build_datasets(args.data, args.seq_len, args.stride, args.val_frac, args.seed, args.trim_noops)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, args.batch_size)
    print(f"train windows: {len(train_ds)}  val windows: {len(val_ds)}  state_dim: {train_ds.state_dim}")

    tau = noise_tolerance(args, train_ds)
    tau_norm = None
    if tau is not None:
        tau_norm = torch.as_tensor(tau / train_ds.std, device=device)
        print("tolerance tau per dim (raw units):  " + " ".join(f"{v:.4g}" for v in tau))
        print("tolerance tau per dim (normalized): " + " ".join(f"{v:.4g}" for v in tau_norm.tolist()))

    model_cfg = dict(
        state_dim=train_ds.state_dim,
        n_components=args.n_components,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_len=max(4096, args.seq_len),
        context_len=args.seq_len - 1,
    )
    model = CausalStateTransformer(**model_cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0, running = time.time(), 0.0
        for x, valid in train_loader:
            x, valid = x.to(device), valid.to(device)
            if tau_norm is not None:
                # Train on demonstrations blurred by tau (history and targets alike);
                # validation stays clean.
                x = x + torch.randn_like(x) * tau_norm * valid.unsqueeze(-1)
            loss = nll_loss(model, x, valid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            running += loss.item()

        train_loss = running / len(train_loader)
        val_loss, val_p01 = evaluate(model, val_loader, device, args.val_mc)
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"val p<0.01 {100 * val_p01:5.2f}% | {time.time() - t0:.1f}s")

        ckpt = {
            "model": model.state_dict(),
            "model_cfg": model_cfg,
            "args": vars(args),
            "mean": train_ds.mean,
            "std": train_ds.std,
            "noise_std_raw": tau,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_p01": val_p01,
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"best val next-state NLL (normalized units, lower is better): {best_val:.4f}")


if __name__ == "__main__":
    main()
