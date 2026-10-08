"""Causal state transformer with a mixture density head.

Each timestep is one token. A causal (GPT-style) transformer reads states
x_0..x_t and outputs a K-component diagonal Gaussian mixture over x_{t+1}:

    p(x_{t+1} | x_{<=t}) = sum_k pi_k * N(x_{t+1}; mu_k, diag(sigma_k^2))

Mixture parameters are passed around as a tuple (logits, mu, log_std) with
shapes (..., K), (..., K, D), (..., K, D).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2 * math.pi)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class CausalStateTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        n_components=5,
        d_model=128,
        n_layers=4,
        n_heads=4,
        ff_mult=4,
        dropout=0.0,
        max_len=4096,
        context_len=None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_components = n_components
        self.max_len = max_len
        # Longest history the model was trained on (seq_len - 1: a window of
        # seq_len states gives seq_len - 1 inputs). Positions beyond it are untrained.
        self.context_len = context_len
        self.in_proj = nn.Linear(state_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ff_mult * d_model, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_components * (1 + 2 * state_dim))

    def forward(self, x):
        """x: (B, T, D) states. Returns mixture params where index t is the
        distribution over x[:, t+1]: logits (B,T,K), mu (B,T,K,D), log_std (B,T,K,D)."""
        B, T, _ = x.shape
        K, D = self.n_components, self.state_dim
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=x.device, dtype=x.dtype)
        h = self.encoder(self.pos(self.in_proj(x)), mask=causal, is_causal=True)
        out = self.out_proj(self.norm(h))
        logits = out[..., :K]
        mu, log_std = out[..., K:].view(B, T, K, 2 * D).chunk(2, dim=-1)
        return logits, mu, log_std

    @torch.no_grad()
    def generate(self, prefix, n_steps, mode="sample", context_len=None):
        """Autoregressively roll out n_steps states after prefix (B, T0, D).

        mode="sample": draw from the predicted mixture (stochastic rollouts)
        mode="mode":   mean of the highest-weight component (deterministic)
        Returns (B, T0 + n_steps, D). context_len caps the attended history
        (default: the trained context length).
        """
        context_len = context_len or self.context_len or self.max_len
        x = prefix
        for _ in range(n_steps):
            params = select(self(x[:, -context_len:]), (slice(None), -1))
            nxt = sample(params) if mode == "sample" else mixture_mode(params)
            x = torch.cat([x, nxt[:, None]], dim=1)
        return x


def select(params, idx):
    """Index the leading dims of all mixture params, e.g. select(p, (0, -1))."""
    return tuple(p[idx] for p in params)


def component_log_probs(params, x):
    """Per-component, per-dim Gaussian log densities. x (..., D) -> (..., K, D)."""
    _, mu, log_std = params
    z = (x.unsqueeze(-2) - mu) / log_std.exp()
    return -0.5 * z**2 - log_std - 0.5 * LOG_2PI


def log_prob(params, x):
    """log p(x) under the mixture. x (..., D) -> (...)."""
    logits = params[0]
    comp = component_log_probs(params, x).sum(-1)  # (..., K)
    return torch.logsumexp(F.log_softmax(logits, -1) + comp, dim=-1)


def sample(params, n=None):
    """Draw samples. Returns (..., D), or (n, ..., D) if n is given."""
    logits, mu, log_std = params
    shape = (n,) if n is not None else ()
    k = torch.distributions.Categorical(logits=logits).sample(shape)  # (n?, ...)
    if n is not None:
        mu, log_std = mu.expand(n, *mu.shape), log_std.expand(n, *log_std.shape)
    idx = k[..., None, None].expand(*k.shape, 1, mu.size(-1))
    mu_k = mu.gather(-2, idx).squeeze(-2)
    std_k = log_std.gather(-2, idx).squeeze(-2).exp()
    return mu_k + std_k * torch.randn_like(mu_k)


@torch.no_grad()
def mc_p_value(params, x, n=1000, chunk=100):
    """P(log p(x~) <= log p(x)) for x~ drawn from the mixture, by Monte Carlo.
    Small = x is unusually unlikely under its own prediction. x (..., D) -> (...)."""
    lp = log_prob(params, x)
    count = torch.zeros_like(lp)
    for s in range(0, n, chunk):
        count += (log_prob(params, sample(params, min(chunk, n - s))) <= lp).float().sum(0)
    return count / n


def mixture_mean(params):
    """Expected value of the mixture. (..., D)"""
    logits, mu, _ = params
    return (F.softmax(logits, -1).unsqueeze(-1) * mu).sum(-2)


def mixture_mode(params):
    """Mean of the highest-weight component (a cheap mode estimate). (..., D)"""
    logits, mu, _ = params
    k = logits.argmax(-1)
    return mu.gather(-2, k[..., None, None].expand(*k.shape, 1, mu.size(-1))).squeeze(-2)


def nll_loss(model, x, valid=None):
    """Mean negative log-likelihood of x[t+1] given x[<=t], per timestep.

    valid: optional (B, T) bool, False = right padding. Padding needs no
    attention mask: with causal attention, real steps never see later pads.
    """
    nll = -log_prob(model(x[:, :-1]), x[:, 1:])  # (B, T-1)
    if valid is None:
        return nll.mean()
    w = valid[:, 1:].float()
    return (nll * w).sum() / w.sum().clamp(min=1)
