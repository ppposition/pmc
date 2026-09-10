"""Norm-clamped guidance for SiT and SD3.5."""

import math
from bisect import bisect_right

import torch


def _broadcast_weight(w, ref):
    if isinstance(w, torch.Tensor) and w.dim() >= 1:
        return w.view(-1, *([1] * (ref.dim() - 1))).to(ref.dtype)
    return w


def norm_clamped(
    sigma, gap=None, v_uncond=None, v_cond=None, cfg_scale=4.0, a=0.1, b=0.5, xt=None, gamma=1.1
):
    """w = 1 + max(0, α_max) where ||m_c + α_max · gap_m||² = γ² · ||m_c||².

    γ controls the relaxation: γ=1 means ||m_cfg|| <= ||m_c|| (original);
    γ>1 allows the adjusted estimate to exceed the conditional norm by up to γ×.
    """
    if xt is None or gap is None or v_cond is None:
        return cfg_scale
    c = xt.size(1)
    gap_v = gap[:, :c]
    m_c = xt.float() + (1 - sigma) * v_cond[:, :c].float()
    gap_m = (1 - sigma) * gap_v.float()  # m_c - m_u
    dot = (m_c * gap_m).flatten(1).sum(dim=1)
    gap_norm_sq_raw = gap_m.flatten(1).norm(dim=1).pow(2)  # no epsilon — must track true zero
    m_c_norm_sq = m_c.flatten(1).norm(dim=1).pow(2)
    disc = dot.pow(2) + (gamma**2 - 1.0) * gap_norm_sq_raw * m_c_norm_sq
    safe = gap_norm_sq_raw > 1e-8
    alpha = torch.zeros_like(dot)
    alpha[safe] = (-dot[safe] + torch.sqrt(disc[safe].clamp(min=0.0))) / gap_norm_sq_raw[safe]
    return 1.0 + torch.clamp(alpha, min=0.0, max=cfg_scale - 1.0)


def sd35_norm_clamped(
    sigma, gap=None, v_uncond=None, v_cond=None, cfg_scale=4.0, a=0.1, b=0.5, xt=None, gamma=1.1
):
    """Clamp the SD3 CFG change to the predicted clean latent per sample."""
    if xt is None or gap is None or v_cond is None:
        return cfg_scale
    c = xt.size(1)
    gap_v = gap[:, :c]
    x0_cond = xt[:, :c].float() - sigma * v_cond[:, :c].float()
    delta_x0_unit = -sigma * gap_v.float()
    x0_cond_norm = x0_cond.flatten(1).norm(dim=1)
    delta_x0_unit_norm = delta_x0_unit.flatten(1).norm(dim=1)

    max_alpha = float(cfg_scale - 1.0)
    alpha = torch.full_like(x0_cond_norm, max_alpha)
    safe = delta_x0_unit_norm > 1e-8
    alpha[safe] = (gamma - 1.0) * x0_cond_norm[safe] / delta_x0_unit_norm[safe]
    return 1.0 + torch.clamp(alpha, min=0.0, max=max_alpha)


def make_cfg_forward(model, weight_fn, default_cfg_scale=4.0, gamma=None):
    """Wrap model.forward with per-step CFG scheduling.

    Returns a callable with the same interface as forward_with_cfg: fn(x, t, y, cfg_scale=None).
    gamma (if given) overrides the weight function's default relaxation parameter.
    """

    def forward_cfg_scheduled(x, t, y, cfg_scale=None):
        if cfg_scale is None:
            cfg_scale = default_cfg_scale
        n = len(x) // 2
        sigma = t[0].item() if isinstance(t, torch.Tensor) else t

        out_cond = model.forward(x[:n], t[:n], y[:n])
        out_uncond = model.forward(x[n:], t[n:], y[n:])

        c = x.size(1)
        eps_cond, rest_cond = out_cond[:, :c], out_cond[:, c:]
        eps_uncond, rest_uncond = out_uncond[:, :c], out_uncond[:, c:]

        gap = eps_cond - eps_uncond
        kwargs = dict(gap=gap, v_cond=eps_cond, v_uncond=eps_uncond, cfg_scale=cfg_scale, xt=x[:n])
        if gamma is not None:
            kwargs["gamma"] = gamma
        w = weight_fn(sigma, **kwargs)
        w = _broadcast_weight(w, gap)

        half_eps = eps_uncond + w * gap
        eps = torch.cat([half_eps, half_eps], dim=0)
        rest = torch.cat([rest_cond, rest_uncond], dim=0)
        return torch.cat([eps, rest], dim=1)

    return forward_cfg_scheduled


class TVCFG:
    """Discrete low-high-low CFG, normalized over the sampling interval.

    Pass the actual Euler grid (N + 1 points, in sampling order). SiT time
    increases from noise to data; decreasing grids are supported as well.
    Interval widths are normalized to sum to one, including truncated grids.
    Lookup uses time rather than a call counter so each batch reuses the schedule.
    """

    def __init__(self, times, cfg_scale):
        times = [float(t) for t in times]
        if len(times) < 2 or not all(math.isfinite(t) for t in times):
            raise ValueError("TV-CFG requires at least two finite time points")
        if not math.isfinite(cfg_scale) or cfg_scale < 1:
            raise ValueError("TV-CFG requires a finite cfg_scale >= 1")
        self.direction = 1 if times[-1] > times[0] else -1
        self.times = [self.direction * t for t in times]
        widths = [b - a for a, b in zip(self.times, self.times[1:])]
        if any(dt <= 0 for dt in widths):
            raise ValueError("TV-CFG time points must be strictly monotonic")
        self.num_evaluations = len(widths)
        n_steps = self.num_evaluations
        midpoint = (n_steps + 1) // 2
        raw = [
            1 + 2 * (cfg_scale - 1) / midpoint * (n if n <= midpoint else n_steps - n)
            for n in range(n_steps)
        ]
        self.normalized_widths = [dt / sum(widths) for dt in widths]
        self.normalization = cfg_scale / sum(q * dt for q, dt in zip(raw, self.normalized_widths))
        self.scales = [self.normalization * q for q in raw]

    def __call__(self, time):
        index = bisect_right(self.times, self.direction * float(time)) - 1
        return self.scales[max(0, min(index, self.num_evaluations - 1))]


def make_tv_cfg_forward(model, schedule):
    """Change only the scale of the model's vanilla CFG implementation."""

    def forward(x, t, y, cfg_scale=None):
        return model.forward_with_cfg(x, t, y, cfg_scale=schedule(t[0]))

    return forward
