# Causal State Transformer (Mixture Density)

Minimal causal (GPT-style) distribution modeling of continuous state sequences. A transformer reads states `x_0..x_t` and outputs a K-component Gaussian mixture over `x_{t+1}`. This gives:
- a likelihood for any observed state, `log p(x[t] | x[<t])`
- stochastic or deterministic rollouts from a prefix
- per-step anomaly scores (`score.py`)

## Files

| File | Contents |
|---|---|
| `data.py` | Loads trajectories from `.npy`/`.npz` (fixed or variable length), slices them into windows, pads short ones, normalizes each dimension to zero mean and unit variance, and generates synthetic data |
| `model.py` | `CausalStateTransformer` (mixture head, `generate`), `log_prob`, `sample`, `mixture_mean`, `mixture_mode`, `nll_loss` |
| `train.py` | Training loop with NLL loss and teacher forcing; saves `last.pt` / `best.pt` to `--out_dir` |
| `evaluate.py` | NLL, one-step MSE of the mixture mean, and rollout MSE, compared with copy-last and linear-extrapolation baselines |
| `score.py` | Per-step likelihood of a trajectory: log-likelihood, validation percentile, Monte Carlo p-value, worst dimension |

## Data format

- `.npy` of shape `(N, T, D)`: N trajectories
- `.npy` object array of N trajectories, each `(T_i, D)` (variable lengths)
- `.npy` of shape `(T_total, D)`: one long trajectory, windowed with `--stride`
- `.npz`: each array is one trajectory `(T_i, D)` (lengths can differ)

Train and validation data are split by trajectory, so their windows never overlap. A smaller `--stride` than `--seq_len` gives overlapping windows and more training samples.

Variable lengths are handled without dropping data:
- Trajectories shorter than `--seq_len` become one right-padded window. Trajectories of length 1 are dropped, since they have no next state.
- For longer trajectories, if the stride doesn't land on the end, an extra end-aligned window covers the tail.
- `--trim_noops EPS` drops the idle states at the start of each trajectory. A step is a no-op when every dimension changes by less than `EPS` (raw units). The last idle state is kept as context; mid-episode pauses are left in. The setting is saved in the checkpoint, and `evaluate.py` / `score.py` apply the same trimming. `score.py` still reports `t` in original, untrimmed indices. 
- The dataset returns `(x, valid)`, where `valid` is `False` on padding. The loss and eval metrics ignore padded targets. No attention mask is needed: with causal attention and padding at the end, real timesteps never attend to padded ones.

## Model

**Input**: `x` of shape `(B, T, D)`, normalized states. Position `t` only sees `x[0..t]` (causal mask).

**Output** at each position `t`: a mixture over `x[t+1]`:
```
logits  (B, T, K)      mixture weights π = softmax(logits)
mu      (B, T, K, D)   component means (absolute next state)
log_std (B, T, K, D)   diagonal spreads (unconstrained)
```
`p(x[t+1] | x[≤t]) = Σ_k π_k · N(x[t+1]; μ_k, diag σ_k²)`. Training minimizes the negative log of this density for the real next state. `--n_components 1` gives a single Gaussian per step.

This is the basic mixture density head: a linear layer outputs logits, means and log-stds directly, with default PyTorch initialization. 
## Usage

```bash
# Train (synthetic data if --data is omitted)
python train.py --data /path/to/states.npy --seq_len 32 --stride 8 --n_components 5 --out_dir runs/mine

# Evaluate
python evaluate.py --ckpt runs/mine/best.pt --prefix_len 16

# Score every trajectory held out during training (same data, seed and --val_frac as train.py)
python score.py --ckpt runs/mine/best.pt --val

# Score one trajectory from any file (prints flagged steps; --all prints every step)
python score.py --ckpt runs/mine/best.pt --traj /path/to/states.npy --index 3 --alpha 0.01
```

From Python:
```python
from score import Scorer
scorer = Scorer("runs/mine/best.pt")
out = scorer.score(states)   # states: raw (T, D) array; results are for t = 1..T-1
out["log_lik"], out["percentile"], out["p_value"], out["worst_dim"], out["per_dim_log_lik"]
```
