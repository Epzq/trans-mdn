# Causal State Transformer

Minimal causal (GPT-style) modeling of continuous state sequences: a transformer reads states `x_0..x_t` and predicts `x_{t+1}`, and `generate` rolls out future states from a prefix.

## Files

| File | Contents |
|---|---|
| `data.py` | Loads trajectories from `.npy`/`.npz`, slices them into windows, normalizes each dimension to zero mean and unit variance, and generates synthetic data |
| `model.py` | `CausalStateTransformer` (with `generate`) and `next_state_loss` |
| `train.py` | Training loop with teacher forcing; saves `last.pt` / `best.pt` to `--out_dir` |
| `evaluate.py` | One-step and rollout MSE, compared with copy-last-state and linear-extrapolation baselines |

## Data format

- `.npy` of shape `(N, T, D)`: N trajectories
- `.npy` of shape `(T_total, D)`: one long trajectory, windowed with `--stride`
- `.npz`: each array is one trajectory `(T_i, D)` (lengths can differ)

Train and validation data are split by trajectory, so their windows never overlap. A smaller `--stride` than `--seq_len` gives overlapping windows and more training samples.

Variable lengths are handled without dropping data:
- Trajectories shorter than `--seq_len` become one right-padded window. Trajectories of length 1 are dropped, since they have no next state.
- For longer trajectories, if the stride doesn't land on the end, an extra end-aligned window covers the tail.
- The dataset returns `(x, valid)`, where `valid` is `False` on padding. The loss and eval metrics ignore padded targets. No attention mask is needed: with causal attention and padding at the end, real timesteps never attend to padded ones.

## Model

- One token per timestep: `state → linear → d_model`, plus sinusoidal positions.
- Pre-LN `nn.TransformerEncoder` with a causal attention mask → linear head back to `D`.
- Output at position `t` predicts `x_{t+1}`. The loss is MSE over all positions, in normalized units.
- `model.generate(prefix, n_steps, context_len=None)` produces states autoregressively and feeds each prediction back as input.

## Usage

```bash
# Sanity run on synthetic data
python train.py --out_dir runs/synthetic

# Your own states
python train.py --data /path/to/states.npy --seq_len 32 --stride 8 --out_dir runs/mine

# Evaluate: observe 16 states, roll out the rest of the window
python evaluate.py --ckpt runs/mine/best.pt --prefix_len 16
```

Normalize inputs with the checkpoint's `mean`/`std` before calling the model, and use `dataset.denormalize` to convert outputs back.

## Reference result (synthetic, CPU, ~1 min)

`--d_model 64 --n_layers 3 --epochs 60 --seq_len 64`, held-out data, MSE (normalized):

| | model | copy last | linear extrap. |
|---|---|---|---|
| one-step | 0.0028 | 0.0369 | 0.0042 |
| rollout, 16-state prefix → 48 steps | 1.26 | 2.15 | 33.2 |
| rollout, 40-state prefix → 24 steps | 0.75 | 2.01 | 9.35 |

One-step prediction is accurate. Rollout error grows quickly (t+6: 0.20, t+12: 0.93). This is expected for a deterministic MSE model trained only with teacher forcing: its own errors compound. To improve rollouts, the usual next steps are training on its own rollouts (scheduled sampling or multi-step loss) or a probabilistic output head (Gaussian/GMM, or discretized tokens) so the model represents uncertainty instead of averaging over it.
