"""State sequence datasets.

Data is a set of continuous state trajectories. Supported on-disk formats:
  - .npy with shape (N, T, D): N trajectories of length T with D-dim states
  - .npy object array of N trajectories, each (T_i, D) (variable lengths)
  - .npy with shape (T_total, D): one long trajectory, cut into windows
  - .npz: every array is treated as one trajectory of shape (T_i, D)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def load_trajectories(path):
    """Return a list of float32 arrays, each (T_i, D)."""
    if path.endswith(".npz"):
        data = np.load(path)
        return [np.asarray(data[k], dtype=np.float32) for k in data.files]
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object:
        # Ragged: object array of N trajectories, each (T_i, D).
        trajs = [np.asarray(t, dtype=np.float32) for t in arr]
        bad = {t.shape for t in trajs if t.ndim != 2 or t.shape[-1] != trajs[0].shape[-1]}
        if bad:
            raise ValueError(f"Each trajectory must be (T_i, D) with the same D; got shapes like {sorted(bad)[:3]}")
        return trajs
    arr = arr.astype(np.float32)
    if arr.ndim == 3:
        return list(arr)
    if arr.ndim == 2:
        return [arr]
    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")


def make_synthetic(num_traj=2000, seq_len=64, state_dim=8, seed=0):
    """Sums of sinusoids with random frequency/phase per dim, plus coupling
    between dims, so future states are predictable from the history."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, seq_len, dtype=np.float32)[None, :, None]
    freq = rng.uniform(0.5, 3.0, size=(num_traj, 1, state_dim)).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, size=(num_traj, 1, state_dim)).astype(np.float32)
    amp = rng.uniform(0.5, 2.0, size=(num_traj, 1, state_dim)).astype(np.float32)
    x = amp * np.sin(freq * t + phase)
    # Fixed coupling so different seeds sample the same distribution.
    mix = np.random.default_rng(12345).normal(0, 0.3, size=(state_dim, state_dim)).astype(np.float32)
    x = x + x @ mix
    x += rng.normal(0, 0.02, size=x.shape).astype(np.float32)
    return list(x)


class StateSequenceDataset(Dataset):
    """Fixed-length windows over state trajectories, z-score normalized.

    Trajectories shorter than seq_len become one right-padded window. Longer
    ones get an extra end-aligned window if the stride doesn't reach the end.
    Items are (x, valid): x (seq_len, D), valid (seq_len,) bool, False = padding.
    """

    def __init__(self, trajectories, seq_len, stride=None, mean=None, std=None):
        self.seq_len = seq_len
        stride = stride or seq_len
        # Need at least 2 states to have one next-state target.
        self.trajectories = [traj for traj in trajectories if len(traj) >= 2]
        if not self.trajectories:
            raise ValueError("No trajectory has length >= 2")

        if mean is None:
            cat = np.concatenate(self.trajectories, axis=0)
            mean, std = cat.mean(0), cat.std(0) + 1e-6
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

        self.index = []
        for i, traj in enumerate(self.trajectories):
            last = max(len(traj) - seq_len, 0)
            starts = list(range(0, last + 1, stride))
            if starts[-1] != last:
                starts.append(last)
            self.index += [(i, s) for s in starts]

    @property
    def state_dim(self):
        return self.trajectories[0].shape[-1]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, s = self.index[idx]
        x = (self.trajectories[i][s : s + self.seq_len] - self.mean) / self.std
        n = len(x)
        out = np.zeros((self.seq_len, x.shape[-1]), dtype=np.float32)
        out[:n] = x
        valid = np.zeros(self.seq_len, dtype=bool)
        valid[:n] = True
        return torch.from_numpy(out), torch.from_numpy(valid)

    def denormalize(self, x):
        mean = torch.as_tensor(self.mean, device=x.device)
        std = torch.as_tensor(self.std, device=x.device)
        return x * std + mean


def load_source(data_path, seq_len, seed=0, synthetic_kwargs=None):
    """Trajectories from a file, or the synthetic set training used if no path."""
    if data_path:
        return load_trajectories(data_path)
    return make_synthetic(seq_len=seq_len, seed=seed, **(synthetic_kwargs or {}))


def split_indices(n, val_frac=0.1, seed=0):
    """(train_idx, val_idx) trajectory indices, as used by build_datasets."""
    order = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(n * val_frac)) if n > 1 else 0
    return order[n_val:], order[:n_val]


def noop_start(traj, eps):
    """Start index after trimming leading no-ops: the last idle state before the
    first move, kept as context. A move is a step where some dim changes by >= eps
    (raw units). A trajectory that never moves keeps only its last state."""
    moving = np.abs(np.diff(traj, axis=0)).max(-1) >= eps
    return int(moving.argmax()) if moving.any() else len(traj) - 1


def trim_noops(trajectories, eps):
    """Trim leading no-ops from every trajectory (no-op if eps is None)."""
    if eps is None:
        return trajectories
    starts = [noop_start(t, eps) for t in trajectories]
    print(f"trim_noops={eps}: removed {sum(starts)} leading no-op states "
          f"from {sum(s > 0 for s in starts)} / {len(trajectories)} trajectories")
    return [t[s:] for t, s in zip(trajectories, starts)]


def build_datasets(data_path, seq_len, stride=None, val_frac=0.1, seed=0, trim_noops_eps=None, synthetic_kwargs=None):
    """Split by trajectory (not by window) so val windows never overlap train."""
    trajectories = trim_noops(load_source(data_path, seq_len, seed, synthetic_kwargs), trim_noops_eps)
    train_idx, val_idx = split_indices(len(trajectories), val_frac, seed)
    val_traj = [trajectories[i] for i in val_idx]
    train_traj = [trajectories[i] for i in train_idx]

    if not val_traj:
        # Single long trajectory: hold out its tail instead.
        full = train_traj[0]
        cut = int(len(full) * (1 - val_frac))
        train_traj, val_traj = [full[:cut]], [full[cut:]]

    train_ds = StateSequenceDataset(train_traj, seq_len, stride)
    val_ds = StateSequenceDataset(val_traj, seq_len, stride, mean=train_ds.mean, std=train_ds.std)
    return train_ds, val_ds
