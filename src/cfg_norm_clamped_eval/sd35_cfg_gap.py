"""
Analyze time-varying CFG weight functions w(t) for SD3.5.

Background:
  - v_t = a_t(μ̄ - x) + b_t x,  a_t = (1-t)/C_t,  b_t = tσ_y²/C_t
  - gap = v_t^C - v_t^all = a_t · (μ̄^C - μ̄^all)
  - For Dirac-like data (σ_y → 0): a_t ≈ 1/(1-t) → ∞ near t=1
  - w(t) should compensate a_t divergence near data

Usage:
    python -m cfg_norm_clamped_eval.sd35_cfg_gap
"""

import os

import matplotlib
import numpy as np
import torch

from .cfg_schedules import sd35_norm_clamped

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from PIL import JpegImagePlugin  # noqa: F401 - register the JPEG encoder for PDF output
from torchvision.utils import save_image

# ============================================================
# Configuration
# ============================================================

HEIGHT = 1024
WIDTH = 1024
NUM_STEPS = 20
CFG_SCALES = [5.0]  # [1.0, 2.0, 4.5, 7.0]
SEED = 43
NUM_SAMPLES = 8
# (short_name, prompt_text)
PROMPTS = [
    # ("dog",        "A dog is shown sitting on rocks by the beach, photorealistic"),
    # ("cat",        "A cat holding a sign that says hello world, photorealistic"),
    # ("landscape",  "A beautiful mountain and river landscape, photorealistic"),
    # ("banana",     "A banana on a white background, photorealistic"),
    ("bedroom", "A bedroom with a large bed sitting under a painting, photorealistic"),
    # ("juice", "A man selling oranges and juice to a bunch of people.")
    # ("people", "a person on a small boat with another boat in the background, photorealistic"),
]
OUTPUT_DIR = "results_sd35"

# Integration method: "euler" (1st-order) or "heun" (2nd-order predictor-corrector).
# Heun at 50 steps ≈ Euler at 500 steps in accuracy.
SCHEDULER = "euler"


# ============================================================
# CFG weight functions
#
# Each receives sigma (float), gap/v_uncond/v_cond (tensors), cfg_scale (float),
# and returns w (float) — the time-varying guidance weight.
# ============================================================


def cfg_weight_01_constant(sigma, gap=None, v_uncond=None, v_cond=None, cfg_scale=4.0, xt=None):
    """w = w₀ — baseline constant guidance."""
    return cfg_scale


def cfg_weight_18_norm_clamped(
    sigma, gap=None, v_uncond=None, v_cond=None, cfg_scale=4.0, a=0.1, b=0.5, xt=None, gamma=1.15
):
    """Limit the CFG-induced change of the predicted clean latent.

    With ``w = 1 + alpha`` under the SD3 flow-matching convention,
    ``x0_cfg - x0_cond = -alpha * sigma * (v_cond - v_uncond)``.
    Choose the largest per-sample alpha satisfying

        ||x0_cfg - x0_cond|| <= (gamma - 1) * ||x0_cond||.

    For example, gamma=1.01 permits at most a 1% relative change.
    """
    return sd35_norm_clamped(
        sigma,
        gap=gap,
        v_uncond=v_uncond,
        v_cond=v_cond,
        cfg_scale=cfg_scale,
        a=a,
        b=b,
        xt=xt,
        gamma=gamma,
    )


CFG_PRESETS = [
    ("01_constant", cfg_weight_01_constant, "w = w₀"),
    (
        "18_norm_clamped",
        cfg_weight_18_norm_clamped,
        "||x0_cfg-x0_cond|| <= (γ-1)||x0_cond||, γ=1.15",
    ),
]

PRESET_COLORS = {"01_constant": "#333333", "18_norm_clamped": "#00ff44"}

# Model setup
# ============================================================


def load_pipe(device):
    """Load SD3.5, return pipe."""
    print("Loading SD3.5 Medium...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    return pipe


def encode_prompt(pipe, prompt, device):
    """Encode a single prompt, return (cond, uncond, cond_pooled, uncond_pooled)."""
    print(f"  Encoding: {prompt}")
    return pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt="",
        negative_prompt_2=None,
        negative_prompt_3=None,
        do_classifier_free_guidance=True,
        device=device,
        num_images_per_prompt=NUM_SAMPLES,
        max_sequence_length=256,
    )


def free_text_encoders(pipe, vae):
    """Free text encoders/tokenizers and move VAE to CPU."""
    del pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3
    del pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3
    torch.cuda.empty_cache()
    vae.to("cpu")
    torch.cuda.empty_cache()
    print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


# ============================================================
# Core integration
# ============================================================


@torch.no_grad()
def _compute_velocity(transformer, xt, t, embeds, pooled):
    """v(x_t, t) — single transformer forward pass."""
    return transformer(
        hidden_states=xt,
        timestep=t,
        encoder_hidden_states=embeds,
        pooled_projections=pooled,
        return_dict=False,
    )[0]


def _record_metrics(gap, v_cond, v_uncond, v_cfg, xt, sigma):
    """Record per-step scalar and per-sample metrics."""
    gap_f = gap.float()
    v_uncond_f = v_uncond.float()

    gap_spatial = torch.norm(gap_f, dim=(2, 3))
    v_uncond_ps = torch.norm(v_uncond_f, dim=(2, 3)).mean(dim=1)

    # --- x0 prediction gap: E[x0|xt,c] - E[x0|xt,null] ---
    # Flow matching: x_t = (1-σ)·x_0 + σ·ε,  v = ε - x_0  =>  x_0 = x_t - σ·v
    x0_cond = xt.float() - sigma * v_cond.float()
    x0_uncond = xt.float() - sigma * v_uncond.float()
    x0_cfg = xt.float() - sigma * v_cfg.float()
    x0_gap = x0_cond - x0_uncond  # = -sigma * gap

    x0_gap_spatial = torch.norm(x0_gap, dim=(2, 3))  # (N, C)
    x0_uncond_spatial = torch.norm(x0_uncond, dim=(2, 3))  # (N, C)

    # --- x0 norms used by the existing per-channel diagnostics ---
    x0_cond_spatial = torch.norm(x0_cond, dim=(2, 3))  # (N, C)
    x0_cond_norm_ps = x0_cond_spatial.mean(dim=1)  # (N,)

    # Full-latent L2 norms. These exactly match the norm constrained by
    # cfg_weight_18_norm_clamped and are used in the cond-vs-CFG plot.
    x0_cond_l2_norm_ps = x0_cond.flatten(1).norm(dim=1)  # (N,)
    x0_cfg_l2_norm_ps = x0_cfg.flatten(1).norm(dim=1)  # (N,)

    x0_gap_norm_ps = x0_gap_spatial.mean(dim=1)  # (N,)
    x0_uncond_norm_ps = x0_uncond_spatial.mean(dim=1)  # (N,)

    # Cosine similarity between v_cond and v_uncond (per sample)
    v_cond_flat = v_cond.float().flatten(1)  # (N, C*H*W)
    v_uncond_flat = v_uncond_f.flatten(1)
    cos_sim = (v_cond_flat * v_uncond_flat).sum(dim=1) / (
        v_cond_flat.norm(dim=1) * v_uncond_flat.norm(dim=1) + 1e-8
    )

    # For x_t = (1-sigma)x_0 + sigma*epsilon and v = epsilon - x_0,
    # score = -epsilon/sigma, hence the conditional score difference is
    # -(1-sigma)/sigma times the velocity difference.
    score_gap = -((1.0 - sigma.float()) / sigma.float().clamp(min=1e-8)) * gap_f
    score_mse_per_sample = score_gap.square().mean(dim=(1, 2, 3))

    return {
        "gap_overall": torch.norm(gap_f, dim=(2, 3)).mean().item(),
        "v_cond_norm": torch.norm(v_cond.float(), dim=(2, 3)).mean().item(),
        "v_uncond_norm": torch.norm(v_uncond_f, dim=(2, 3)).mean().item(),
        "v_cfg_norm": torch.norm(v_cfg.float(), dim=(2, 3)).mean().item(),
        "gap_overall_per_sample": gap_spatial.mean(dim=1).cpu(),
        "gap_norms_per_sample": gap_spatial.cpu(),
        "ratio_per_sample": (gap_spatial.mean(dim=1) / (v_uncond_ps + 1e-8)).cpu(),
        "score_mse_per_sample": score_mse_per_sample.cpu(),
        "cos_sim_per_sample": cos_sim.cpu(),
        # x0 gap metrics
        "x0_gap_overall": x0_gap_norm_ps.mean().item(),
        "x0_gap_rel": (x0_gap_norm_ps / (x0_cond_norm_ps + 1e-8)).mean().item(),
        "x0_gap_per_sample": x0_gap_norm_ps.cpu(),
        "x0_gap_rel_per_sample": (x0_gap_norm_ps / (x0_cond_norm_ps + 1e-8)).cpu(),
        "x0_gap_norms_per_sample": x0_gap_spatial.cpu(),
        "x0_uncond_norms_per_sample": x0_uncond_norm_ps.cpu(),
        "x0_cond_norms_per_sample": x0_cond_norm_ps.cpu(),
        "x0_cond_l2_norms_per_sample": x0_cond_l2_norm_ps.cpu(),
        "x0_cfg_l2_norms_per_sample": x0_cfg_l2_norm_ps.cpu(),
    }


def _broadcast_weight(w, ref_tensor):
    """Reshape w to broadcast with ref_tensor of shape (N, C, H, W).

    Scalars pass through unchanged. Per-sample tensors (N,) are reshaped to (N, 1, 1, 1)
    and cast to ref_tensor.dtype to avoid dtype mismatch.
    """
    if isinstance(w, torch.Tensor) and w.dim() >= 1:
        return w.view(-1, *([1] * (ref_tensor.dim() - 1))).to(ref_tensor.dtype)
    return w


def _scalar_w(w):
    """Extract a scalar from w for logging. Per-sample tensors return the mean."""
    if isinstance(w, torch.Tensor) and w.numel() > 1:
        return w.mean().item()
    return w if isinstance(w, float) else w.item()


def run_trajectory(
    transformer,
    z,
    timesteps,
    sigmas,
    cond_embeds,
    uncond_embeds,
    cond_pooled,
    uncond_pooled,
    weight_fn,
    cfg_scale,
    scheduler="heun",
    verbose=True,
):
    """
    Run one denoising trajectory with time-varying CFG.

    Args:
        scheduler: "euler" — 1st-order Euler steps.
                   "heun"  — 2nd-order predictor-corrector (last step falls back to Euler).

    Returns:
        (xt_final, metrics_list, w_values)
    """
    num_samples = z.shape[0]
    N = len(timesteps)  # sigmas has N+1 elements
    xt = z.clone()

    # Reset stateful weight functions
    for attr in ("_gap_ref", "_gap_max"):
        if hasattr(weight_fn, attr):
            delattr(weight_fn, attr)

    metrics_list = []
    w_values = []

    for step_idx in range(N):
        sigma_val = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]
        dt = sigma_next - sigma_val
        t_batch = timesteps[step_idx].expand(num_samples)

        use_heun = (scheduler == "heun") and (step_idx < N - 1)

        # Velocity at current state
        v_cond = _compute_velocity(transformer, xt, t_batch, cond_embeds, cond_pooled)
        v_uncond = _compute_velocity(transformer, xt, t_batch, uncond_embeds, uncond_pooled)
        gap = v_cond - v_uncond

        w_t = weight_fn(
            sigma_val.item(), gap=gap, v_uncond=v_uncond, v_cond=v_cond, cfg_scale=cfg_scale, xt=xt
        )
        # Reshape per-sample tensor to (N,1,1,1) for broadcasting with (N,C,H,W)
        w_t_bc = _broadcast_weight(w_t, gap)
        v_cfg_current = v_uncond + w_t_bc * gap

        metrics = _record_metrics(gap, v_cond, v_uncond, v_cfg_current, xt, sigma_val)
        metrics["w"] = w_t
        if isinstance(w_t, torch.Tensor):
            if w_t.ndim == 0:
                w_per_sample = w_t.expand(num_samples)
            else:
                w_per_sample = w_t.reshape(num_samples, -1).mean(dim=1)
        else:
            w_per_sample = torch.full((num_samples,), float(w_t), device=xt.device)
        metrics["w_per_sample"] = w_per_sample.detach().cpu()
        metrics_list.append(metrics)
        w_values.append(_scalar_w(w_t))

        if use_heun:
            # Predictor: x̂ = x_t + dt · v_cfg(x_t, t)
            t_next = timesteps[step_idx + 1].expand(num_samples)
            xt_pred = xt + dt * v_cfg_current

            # Velocity at predicted state (t_next)
            v_cond_pred = _compute_velocity(transformer, xt_pred, t_next, cond_embeds, cond_pooled)
            v_uncond_pred = _compute_velocity(
                transformer, xt_pred, t_next, uncond_embeds, uncond_pooled
            )

            gap_pred = v_cond_pred - v_uncond_pred
            w_t_pred = weight_fn(
                sigma_next.item(),
                gap=gap_pred,
                v_uncond=v_uncond_pred,
                v_cond=v_cond_pred,
                cfg_scale=cfg_scale,
                xt=xt_pred,
            )
            w_t_pred_bc = _broadcast_weight(w_t_pred, gap_pred)
            v_cfg_pred = v_uncond_pred + w_t_pred_bc * gap_pred

            # Corrector: x_{t+1} = x_t + dt · (v_c + v̂_c) / 2
            xt = xt + dt * (v_cfg_current + v_cfg_pred) / 2
        else:
            # Euler: x_{t+1} = x_t + dt · v_cfg(x_t, t)
            xt = xt + dt * v_cfg_current

        if verbose and (step_idx + 1) % 10 == 0:
            print(
                f"  Step {step_idx + 1}/{N}, sigma={sigma_val:.4f}, "
                f"w={_scalar_w(w_t):.3f}, gap={metrics['gap_overall']:.4f}"
            )

    return xt, metrics_list, w_values


def stack_metrics(metrics_list):
    """Convert list of per-step metric dicts to numpy arrays for plotting."""
    return {
        "gap_overall": [m["gap_overall"] for m in metrics_list],
        "v_cond_norms": np.array([m["v_cond_norm"] for m in metrics_list]),
        "v_uncond_norms": np.array([m["v_uncond_norm"] for m in metrics_list]),
        "v_cfg_norms": np.array([m["v_cfg_norm"] for m in metrics_list]),
        "w_values": np.array([_scalar_w(m["w"]) for m in metrics_list]),
        "gap_overall_per_sample": torch.stack(
            [m["gap_overall_per_sample"] for m in metrics_list]
        ).numpy(),
        "gap_norms_per_sample": torch.stack(
            [m["gap_norms_per_sample"] for m in metrics_list]
        ).numpy(),
        "ratio_per_sample": torch.stack([m["ratio_per_sample"] for m in metrics_list]).numpy(),
        "score_mse_per_sample": torch.stack(
            [m["score_mse_per_sample"] for m in metrics_list]
        ).numpy(),
        "cfg_scale_trace": torch.stack([m["w_per_sample"] for m in metrics_list]).numpy(),
        "cos_sim_per_sample": torch.stack([m["cos_sim_per_sample"] for m in metrics_list]).numpy(),
        # x0 gap metrics
        "x0_gap_overall": np.array([m["x0_gap_overall"] for m in metrics_list]),
        "x0_gap_rel": np.array([m["x0_gap_rel"] for m in metrics_list]),
        "x0_gap_per_sample": torch.stack([m["x0_gap_per_sample"] for m in metrics_list]).numpy(),
        "x0_gap_rel_per_sample": torch.stack(
            [m["x0_gap_rel_per_sample"] for m in metrics_list]
        ).numpy(),
        "x0_gap_norms_per_sample": torch.stack(
            [m["x0_gap_norms_per_sample"] for m in metrics_list]
        ).numpy(),
        "x0_uncond_norms_per_sample": torch.stack(
            [m["x0_uncond_norms_per_sample"] for m in metrics_list]
        ).numpy(),
        "x0_cond_norms_per_sample": torch.stack(
            [m["x0_cond_norms_per_sample"] for m in metrics_list]
        ).numpy(),
        "x0_cond_l2_norms_per_sample": torch.stack(
            [m["x0_cond_l2_norms_per_sample"] for m in metrics_list]
        ).numpy(),
        "x0_cfg_l2_norms_per_sample": torch.stack(
            [m["x0_cfg_l2_norms_per_sample"] for m in metrics_list]
        ).numpy(),
    }


# ============================================================
# Plotting
# ============================================================


def _style_metric_axis(ax, xlabel, ylabel, title, cfg_scale):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (cfg={cfg_scale})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def _save_metric_figure(draw, save_dir, filename):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    draw(ax)
    fig.tight_layout()
    path = os.path.join(save_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved subplot to {path}")


def plot_per_cfg(
    metrics, xt_final, preset_name, cfg_scale, t_plot, save_dir, latent_channels, num_samples
):
    """Generate the same nine diagnostic plots as analyze_cfg_gap.py."""

    # Unpack metrics
    gap_overall_per_sample = metrics["gap_overall_per_sample"]
    gap_norms_per_sample = metrics["gap_norms_per_sample"]
    ratio_per_sample = metrics["ratio_per_sample"]
    score_mse_per_sample = metrics["score_mse_per_sample"]
    cfg_scale_trace = metrics["cfg_scale_trace"]
    cos_sim_per_sample = metrics["cos_sim_per_sample"]
    # x0 gap metrics
    x0_gap_ps = metrics["x0_gap_per_sample"]
    x0_gap_rel_ps = metrics["x0_gap_rel_per_sample"]
    x0_cond_norms_ps = metrics["x0_cond_norms_per_sample"]
    x0_uncond_norms_ps = metrics["x0_uncond_norms_per_sample"]

    sample_cmap = plt.get_cmap("tab20")
    sample_colors = [sample_cmap(i % 20) for i in range(num_samples)]
    mean_color = "#111111"

    fig, axes = plt.subplots(5, 2, figsize=(13, 26))

    def draw_samples(ax, data, mean_label):
        for i in range(num_samples):
            ax.plot(t_plot, data[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7)
        ax.plot(t_plot, data.mean(axis=1), color=mean_color, linewidth=2.0, label=mean_label)

    # 1) Velocity gap
    ax = axes[0, 0]
    draw_samples(ax, gap_overall_per_sample, r"$\|v_{cond} - v_{uncond}\|$ (mean)")
    _style_metric_axis(
        ax, "Time t (0=noise, 1=data)", "Velocity norm", "Velocity gap components", cfg_scale
    )

    # 2) Relative guidance strength
    ax = axes[0, 1]
    draw_samples(ax, ratio_per_sample, r"Gap / $\|v_{uncond}\|$ (mean)")
    _style_metric_axis(
        ax,
        "Time t (0=noise, 1=data)",
        r"Gap / $\|v_{uncond}\|$",
        "Relative guidance strength",
        cfg_scale,
    )

    # 3) Score MSE
    ax = axes[1, 0]
    draw_samples(ax, score_mse_per_sample, r"$\mathbb{E}[(s_{cond} - s_{uncond})^2]$ (mean)")
    _style_metric_axis(
        ax,
        "Time t (0=noise, 1=data)",
        r"$\mathbb{E}[(s_{cond} - s_{uncond})^2]$",
        "Score MSE (cond vs uncond)",
        cfg_scale,
    )

    # 4) Per-channel gap
    ax = axes[1, 1]
    ch_colors = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
    ]
    for ch in range(latent_channels):
        for i in range(num_samples):
            ax.plot(
                t_plot,
                gap_norms_per_sample[:, i, ch],
                color=sample_colors[i],
                linewidth=0.3,
                alpha=0.5,
            )
        ax.plot(
            t_plot,
            gap_norms_per_sample[:, :, ch].mean(axis=1),
            color=ch_colors[ch % 16],
            label=f"Ch {ch}",
            linewidth=1.5,
        )
    ax.set_xlabel("Time t (0=noise, 1=data)")
    ax.set_ylabel(r"$\|v_{cond} - v_{uncond}\|$ (per ch)")
    ax.set_title(f"Per-channel gap norm (cfg={cfg_scale})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 5) E[x0|xt] gap norm (x0 is the data endpoint for SD3.5)
    ax = axes[2, 0]
    draw_samples(ax, x0_gap_ps, r"$\|\mathbb{E}[x_0|x_t,c] - \mathbb{E}[x_0|x_t]\|$ (mean)")
    _style_metric_axis(
        ax,
        "Time t (0=noise, 1=data)",
        r"$\|x_0^{cond} - x_0^{uncond}\|$",
        "E[x0|xt] gap norm",
        cfg_scale,
    )

    # 6) Relative E[x0|xt] gap: gap / ||E[x0|xt, c]||
    ax = axes[2, 1]
    draw_samples(ax, x0_gap_rel_ps, r"$\|x_0^{gap}\| / \|x_0^{cond}\|$ (mean)")
    _style_metric_axis(
        ax, "Time t (0=noise, 1=data)", r"Relative $x_0$ gap", "Relative E[x0|xt] gap", cfg_scale
    )

    # 7) Cosine similarity between v_cond and v_uncond
    ax = axes[3, 0]
    for i in range(num_samples):
        ax.plot(t_plot, cos_sim_per_sample[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7)
    ax.plot(
        t_plot,
        cos_sim_per_sample.mean(axis=1),
        color=mean_color,
        linewidth=2.0,
        label=r"$\cos(v_{cond}, v_{uncond})$ (mean)",
    )
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time t (0=noise, 1=data)")
    ax.set_ylabel(r"$\cos(v_{cond}, v_{uncond})$")
    ax.set_title(f"Cosine similarity: v_cond vs v_uncond (cfg={cfg_scale})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    axes[3, 1].axis("off")

    # 8) Conditional vs unconditional data-prediction norm
    ax = axes[4, 0]
    for i in range(num_samples):
        ax.plot(t_plot, x0_cond_norms_ps[:, i], color="#d62728", linewidth=0.3, alpha=0.25)
        ax.plot(t_plot, x0_uncond_norms_ps[:, i], color="#1f77b4", linewidth=0.3, alpha=0.25)
    ax.plot(
        t_plot,
        x0_cond_norms_ps.mean(axis=1),
        color="#d62728",
        linewidth=2.5,
        label=r"$\mathbb{E}[x_0|x_t,c]$ (mean)",
    )
    ax.plot(
        t_plot,
        x0_uncond_norms_ps.mean(axis=1),
        color="#1f77b4",
        linewidth=2.5,
        label=r"$\mathbb{E}[x_0|x_t,null]$ (mean)",
    )
    _style_metric_axis(
        ax,
        "Time t (0=noise, 1=data)",
        r"$\|\mathbb{E}[x_0|x_t]\|$",
        "Cond vs Uncond E[x0|xt] norm",
        cfg_scale,
    )

    # 9) Effective CFG scale
    ax = axes[4, 1]
    draw_samples(ax, cfg_scale_trace, "effective cfg_scale (mean)")
    ax.axhline(y=cfg_scale, color="gray", linestyle=":", linewidth=0.8, label="target cfg_scale")
    _style_metric_axis(
        ax, "Time t (0=noise, 1=data)", "CFG scale", "Effective CFG scale over time", cfg_scale
    )

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"gap_{preset_name}_cfg{cfg_scale}.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {plot_path}")

    # Match analyze_cfg_gap.py: also save every metric as an individual PDF.
    specs = [
        (
            "01_velocity_gap",
            lambda ax: (
                draw_samples(ax, gap_overall_per_sample, r"$\|v_{cond} - v_{uncond}\|$ (mean)"),
                _style_metric_axis(
                    ax, "Time t (0=noise, 1=data)", "Velocity norm", "Velocity gap", cfg_scale
                ),
            ),
        ),
        (
            "02_relative_guidance",
            lambda ax: (
                draw_samples(ax, ratio_per_sample, r"Gap / $\|v_{uncond}\|$ (mean)"),
                _style_metric_axis(
                    ax,
                    "Time t (0=noise, 1=data)",
                    r"Gap / $\|v_{uncond}\|$",
                    "Relative guidance strength",
                    cfg_scale,
                ),
            ),
        ),
        (
            "03_score_mse",
            lambda ax: (
                draw_samples(
                    ax, score_mse_per_sample, r"$\mathbb{E}[(s_{cond} - s_{uncond})^2]$ (mean)"
                ),
                _style_metric_axis(
                    ax,
                    "Time t (0=noise, 1=data)",
                    r"$\mathbb{E}[(s_{cond} - s_{uncond})^2]$",
                    "Score MSE (cond vs uncond)",
                    cfg_scale,
                ),
            ),
        ),
        (
            "05_x0_gap",
            lambda ax: (
                draw_samples(ax, x0_gap_ps, r"$\|x_0^{cond}-x_0^{uncond}\|$ (mean)"),
                _style_metric_axis(
                    ax,
                    "Time t (0=noise, 1=data)",
                    r"$\|x_0^{cond}-x_0^{uncond}\|$",
                    "E[x0|xt] gap norm",
                    cfg_scale,
                ),
            ),
        ),
        (
            "06_x0_gap_rel",
            lambda ax: (
                draw_samples(ax, x0_gap_rel_ps, r"$\|x_0^{gap}\|/\|x_0^{cond}\|$ (mean)"),
                _style_metric_axis(
                    ax,
                    "Time t (0=noise, 1=data)",
                    r"Relative $x_0$ gap",
                    "Relative E[x0|xt] gap",
                    cfg_scale,
                ),
            ),
        ),
        (
            "07_cosine_sim",
            lambda ax: (
                draw_samples(ax, cos_sim_per_sample, r"$\cos(v_{cond},v_{uncond})$ (mean)"),
                ax.axhline(0, color="gray", linestyle=":", linewidth=0.8),
                _style_metric_axis(
                    ax,
                    "Time t (0=noise, 1=data)",
                    r"$\cos(v_{cond},v_{uncond})$",
                    "Cosine similarity: v_cond vs v_uncond",
                    cfg_scale,
                ),
            ),
        ),
        (
            "09_cfg_scale",
            lambda ax: (
                draw_samples(ax, cfg_scale_trace, "effective cfg_scale (mean)"),
                ax.axhline(
                    cfg_scale, color="gray", linestyle=":", linewidth=0.8, label="target cfg_scale"
                ),
                _style_metric_axis(
                    ax,
                    "Time t (0=noise, 1=data)",
                    "CFG scale",
                    "Effective CFG scale over time",
                    cfg_scale,
                ),
            ),
        ),
    ]
    for name, draw in specs:
        _save_metric_figure(draw, save_dir, f"{name}_cfg{cfg_scale}.pdf")

    def draw_channels(ax):
        for ch in range(latent_channels):
            ax.plot(
                t_plot,
                gap_norms_per_sample[:, :, ch].mean(axis=1),
                color=ch_colors[ch % len(ch_colors)],
                label=f"Ch {ch}",
                linewidth=2.0,
            )
        _style_metric_axis(
            ax,
            "Time t (0=noise, 1=data)",
            r"$\|v_{cond}-v_{uncond}\|$ (per ch)",
            "Per-channel gap norm",
            cfg_scale,
        )

    _save_metric_figure(draw_channels, save_dir, f"04_per_channel_gap_cfg{cfg_scale}.pdf")

    def draw_x0_norms(ax):
        for i in range(num_samples):
            ax.plot(t_plot, x0_cond_norms_ps[:, i], color="#d62728", linewidth=0.3, alpha=0.25)
            ax.plot(t_plot, x0_uncond_norms_ps[:, i], color="#1f77b4", linewidth=0.3, alpha=0.25)
        ax.plot(
            t_plot,
            x0_cond_norms_ps.mean(axis=1),
            color="#d62728",
            linewidth=2.5,
            label=r"$\mathbb{E}[x_0|x_t,c]$ (mean)",
        )
        ax.plot(
            t_plot,
            x0_uncond_norms_ps.mean(axis=1),
            color="#1f77b4",
            linewidth=2.5,
            label=r"$\mathbb{E}[x_0|x_t,null]$ (mean)",
        )
        _style_metric_axis(
            ax,
            "Time t (0=noise, 1=data)",
            r"$\|\mathbb{E}[x_0|x_t]\|$",
            "Cond vs Uncond E[x0|xt] norm",
            cfg_scale,
        )

    _save_metric_figure(draw_x0_norms, save_dir, f"08_x0_norm_cond_vs_uncond_cfg{cfg_scale}.pdf")


def plot_cross_preset_summary(all_results, t_plot, cfg_scales, output_dir):
    """Generate cross-preset comparison plots."""
    print(f"\n{'=' * 60}")
    print("Cross-preset summary")
    print(f"{'=' * 60}")

    # Use the actual cfg_scales that were run (capped at 4 to fit layout)
    ref_cfgs = [c for c in cfg_scales if c in [1.0, 2.0, 4.5, 7.0]]
    if not ref_cfgs:
        ref_cfgs = cfg_scales[:4]
    n_refs = len(ref_cfgs)
    fig, axes = plt.subplots(n_refs, 3, figsize=(18, 6 * n_refs))
    if n_refs == 1:
        axes = axes.reshape(1, -1)

    for row, ref_cfg in enumerate(ref_cfgs):
        # w(t) comparison
        ax = axes[row, 0]
        for pname, _, plabel in CFG_PRESETS:
            if ref_cfg in all_results.get(pname, {}):
                wv = all_results[pname][ref_cfg]["w_values"]
                ax.plot(t_plot, wv, color=PRESET_COLORS[pname], label=plabel, linewidth=1.8)
        ax.set_xlabel("t (0=noise, 1=data)")
        ax.set_ylabel(r"$w(t)$")
        ax.set_title(f"w(t) comparison (cfg={ref_cfg})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Effective guidance: w(t) * ||gap||
        ax = axes[row, 1]
        for pname, _, plabel in CFG_PRESETS:
            if ref_cfg in all_results.get(pname, {}):
                wv = all_results[pname][ref_cfg]["w_values"]
                gap = np.array(all_results[pname][ref_cfg]["gap_overall"])
                ax.plot(t_plot, wv * gap, color=PRESET_COLORS[pname], label=plabel, linewidth=1.8)
        ax.set_xlabel("t (0=noise, 1=data)")
        ax.set_ylabel(r"$w(t) \cdot \|gap\|$")
        ax.set_title(f"Effective guidance w·||gap|| (cfg={ref_cfg})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ||v_cfg|| comparison
        ax = axes[row, 2]
        for pname, _, plabel in CFG_PRESETS:
            if ref_cfg in all_results.get(pname, {}):
                ax.plot(
                    t_plot,
                    all_results[pname][ref_cfg]["v_cfg_norms"],
                    color=PRESET_COLORS[pname],
                    label=plabel,
                    linewidth=1.8,
                )
        ax.set_xlabel("t (0=noise, 1=data)")
        ax.set_ylabel(r"$\|v_{cfg}\|$")
        ax.set_title(f"||v_cfg|| comparison (cfg={ref_cfg})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    summary_path = os.path.join(output_dir, "cross_preset_summary.pdf")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cross-preset summary to {summary_path}")

    # w(t) × CFG scales grid
    n_presets = len(CFG_PRESETS)
    n_rows = (n_presets + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else axes
    for idx, (pname, _, plabel) in enumerate(CFG_PRESETS):
        ax = axes[idx]
        for cfg in cfg_scales:
            if cfg in all_results.get(pname, {}):
                ax.plot(
                    t_plot, all_results[pname][cfg]["w_values"], label=f"cfg={cfg}", linewidth=1.5
                )
        ax.set_xlabel("t (0=noise, 1=data)")
        ax.set_ylabel(r"$w(t)$")
        ax.set_title(f"{pname}: {plabel}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    grid_path = os.path.join(output_dir, "cross_preset_w_grid.pdf")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved w(t) grid to {grid_path}")


def print_summary_table(all_results):
    """Print per-preset summary statistics."""
    print(f"\n{'=' * 80}")
    print(f"{'Preset':<25} {'CFG':>5} {'max_gap':>10} {'mean_gap':>10} {'max_w':>8} {'mean_w':>8}")
    print(f"{'=' * 80}")
    for pname, _, _plabel in CFG_PRESETS:
        for cfg in CFG_SCALES:
            if cfg in all_results.get(pname, {}):
                go = np.array(all_results[pname][cfg]["gap_overall"])
                wv = all_results[pname][cfg]["w_values"]
                print(
                    f"{pname:<25} {cfg:>5} {go.max():>10.4f} {go.mean():>10.4f} "
                    f"{wv.max():>8.3f} {wv.mean():>8.3f}"
                )


# ============================================================
# Image decoding
# ============================================================


def decode_and_save(xt, vae, device, save_dir, preset_name, cfg_scale):
    """Decode latents with VAE and save grid + individual images."""
    with torch.no_grad():
        vae.to(device)
        latents_decoded = (xt / vae.config.scaling_factor) + vae.config.shift_factor
        samples = vae.decode(latents_decoded.to(torch.bfloat16)).sample
        vae.to("cpu")
        torch.cuda.empty_cache()

    samples_clamped = torch.clamp(samples, -1, 1)
    samples_01 = (samples_clamped + 1) / 2

    # Grid: keep the PDF output and also save a PNG copy for convenient preview.
    grid_stem = os.path.join(save_dir, f"samples_{preset_name}_cfg{cfg_scale}")
    for extension in ("pdf", "png"):
        grid_path = f"{grid_stem}.{extension}"
        save_image(samples_01, grid_path, nrow=4, normalize=False, value_range=(0, 1))
        print(f"  Saved grid to {grid_path}")

    # Individual images
    indiv_dir = os.path.join(save_dir, "images")
    os.makedirs(indiv_dir, exist_ok=True)
    for i in range(samples_01.shape[0]):
        image_index = i + 8
        save_image(
            samples_01[i],
            os.path.join(indiv_dir, f"{image_index:04d}.pdf"),
            normalize=False,
            value_range=(0, 1),
        )
    print(f"  Saved {samples_01.shape[0]} individual images to {indiv_dir}")


# ============================================================
# Main
# ============================================================


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    torch.set_grad_enabled(False)

    # --- Sigma schedule (Euler discretization; integration method is separate) ---
    sched_config = FlowMatchEulerDiscreteScheduler.load_config(
        "stabilityai/stable-diffusion-3.5-medium", subfolder="scheduler"
    )
    euler_sched = FlowMatchEulerDiscreteScheduler.from_config(sched_config)
    euler_sched.set_timesteps(NUM_STEPS, device=device)
    timesteps = euler_sched.timesteps
    sigmas = euler_sched.sigmas
    t_plot = 1 - sigmas[: len(timesteps)].cpu().numpy()  # 0=noise, 1=data

    # --- Load model once ---
    pipe = load_pipe(device)
    transformer = pipe.transformer
    vae = pipe.vae

    # --- Encode all prompts ---
    prompt_embeds = {}
    for name, text in PROMPTS:
        prompt_embeds[name] = encode_prompt(pipe, text, device)

    free_text_encoders(pipe, vae)

    latent_channels = vae.config.latent_channels
    latent_h, latent_w = HEIGHT // 8, WIDTH // 8
    print(f"Steps: {NUM_STEPS}, scheduler: {SCHEDULER}")
    print(f"Latent shape: ({latent_channels}, {latent_h}, {latent_w})")

    # --- Run: for each prompt × preset × cfg_scale ---
    for prompt_name, _prompt_text in PROMPTS:
        cond_embeds, uncond_embeds, cond_pooled, uncond_pooled = prompt_embeds[prompt_name]
        prompt_dir = os.path.join(OUTPUT_DIR, prompt_name)
        all_results = {}

        print(f"\n{'#' * 60}")
        print(f"Prompt: {prompt_name}")
        print(f"{'#' * 60}")

        # Initial noise (same for all presets under this prompt)
        z = torch.randn(
            NUM_SAMPLES, latent_channels, latent_h, latent_w, device=device, dtype=torch.bfloat16
        )

        for preset_name, weight_fn, preset_label in CFG_PRESETS:
            print(f"\n{'=' * 60}")
            print(f"Preset: {preset_name}  ({preset_label})")
            print(f"{'=' * 60}")

            preset_results = {}

            for cfg in CFG_SCALES:
                tag = f"cfg{cfg}"
                save_dir = os.path.join(prompt_dir, preset_name, tag)
                os.makedirs(save_dir, exist_ok=True)
                print(f"\n--- CFG_SCALE = {cfg} ---")

                xt_final, metrics_list, _w_values = run_trajectory(
                    transformer,
                    z,
                    timesteps,
                    sigmas,
                    cond_embeds,
                    uncond_embeds,
                    cond_pooled,
                    uncond_pooled,
                    weight_fn,
                    cfg,
                    scheduler=SCHEDULER,
                    verbose=True,
                )

                metrics = stack_metrics(metrics_list)
                metrics["xt_final"] = xt_final.clone()
                preset_results[cfg] = metrics

                plot_per_cfg(
                    metrics,
                    xt_final,
                    preset_name,
                    cfg,
                    t_plot,
                    save_dir,
                    latent_channels,
                    NUM_SAMPLES,
                )
                decode_and_save(xt_final, vae, device, save_dir, preset_name, cfg)

            all_results[preset_name] = preset_results

        # Cross-preset summaries (per prompt)
        plot_cross_preset_summary(all_results, t_plot, CFG_SCALES, prompt_dir)
        print_summary_table(all_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
