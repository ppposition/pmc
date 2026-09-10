"""
Analyze how the gap between conditional and unconditional velocity (v_cond - v_uncond)
evolves over sampling time in classifier-free guidance, across multiple CFG scales.

Usage:
    python -m cfg_norm_clamped_eval.analyze_cfg_gap
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL

from .cfg_schedules import _broadcast_weight, norm_clamped
from .download import find_model
from .models import SiT_models
from .transport import create_transport

# --- config ---
IMAGE_SIZE = 256
NUM_STEPS = 50
SEED = 20
CLASS_ID = 300
NUM_SAMPLES = 32
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 20

CFG_SCHEDULES = {
    "norm_clamped": norm_clamped,
}


def _enlarge_axis_text(ax):
    """Use consistently larger axis labels and tick labels for metric plots."""
    ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)


# --- helper: save individual subplots as PDFs ---
def _save_subplot(
    ts_np,
    data_ps,
    sample_colors,
    mean_color,
    cfg,
    save_dir,
    name,
    title,
    xlabel,
    ylabel,
    mean_label,
    plot_fn=None,
    extra_fn=None,
):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i in range(data_ps.shape[1]):
        if plot_fn:
            plot_fn(ax, i)
        else:
            ax.plot(ts_np, data_ps[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7)
    ax.plot(ts_np, data_ps.mean(axis=1), color=mean_color, linewidth=2.0, label=mean_label)
    if extra_fn:
        extra_fn(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _enlarge_axis_text(ax)
    ax.set_title(f"{title} (cfg={cfg})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, f"{name}_cfg{cfg}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved subplot to {path}")


def _save_subplot_ch(
    ts_np, gap_norms_ps, ch_colors, ch_labels, cfg, save_dir, name, title, xlabel, ylabel
):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for ch in range(4):
        ax.plot(
            ts_np,
            gap_norms_ps[:, :, ch].mean(axis=1),
            color=ch_colors[ch],
            label=ch_labels[ch],
            linewidth=2.0,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _enlarge_axis_text(ax)
    ax.set_title(f"{title} (cfg={cfg})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, f"{name}_cfg{cfg}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved subplot to {path}")


def _save_subplot_dual(
    ts_np, data1_ps, data2_ps, cfg, save_dir, name, title, xlabel, ylabel, label1, label2
):
    cond_color = "#d62728"
    uncond_color = "#1f77b4"
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i in range(data1_ps.shape[1]):
        ax.plot(ts_np, data1_ps[:, i], color=cond_color, linewidth=0.3, alpha=0.25)
        ax.plot(ts_np, data2_ps[:, i], color=uncond_color, linewidth=0.3, alpha=0.25)
    ax.plot(ts_np, data1_ps.mean(axis=1), color=cond_color, linewidth=2.5, label=label1)
    ax.plot(ts_np, data2_ps.mean(axis=1), color=uncond_color, linewidth=2.5, label=label2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _enlarge_axis_text(ax)
    ax.set_title(f"{title} (cfg={cfg})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, f"{name}_cfg{cfg}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved subplot to {path}")


def _save_subplot_cfg_trace(
    ts_np, cfg_trace_ps, sample_colors, mean_color, cfg, save_dir, name, title, xlabel, ylabel
):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i in range(cfg_trace_ps.shape[1]):
        ax.plot(ts_np, cfg_trace_ps[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7)
    ax.plot(
        ts_np,
        cfg_trace_ps.mean(axis=1),
        color=mean_color,
        linewidth=2.0,
        label="effective cfg_scale (mean)",
    )
    ax.axhline(y=cfg, color="gray", linestyle=":", linewidth=0.8, label="target cfg_scale")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _enlarge_axis_text(ax)
    ax.set_title(f"{title} (cfg={cfg})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, f"{name}_cfg{cfg}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved subplot to {path}")


def main():
    """Analyze guidance trajectories using shared initial noise."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-scale", type=float, nargs="+", default=[1.0])
    parser.add_argument("--output-root", default="results_imagenet256")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda:0 when available, otherwise cpu)",
    )

    parser.add_argument(
        "--cfg-schedule",
        "--cfg_schedule",
        dest="cfg_schedule",
        nargs="?",
        const="norm_clamped",
        choices=CFG_SCHEDULES,
        default=None,
        help="CFG schedule (bare flag defaults to norm_clamped); omit for constant CFG",
    )
    args = parser.parse_args()

    CFG_SCALES = args.cfg_scale
    CFG_SCHEDULE = args.cfg_schedule
    SCHEDULE_NAME = CFG_SCHEDULE or "constant"
    if any(cfg < 1.0 for cfg in CFG_SCALES):
        parser.error("--cfg-scale must be >= 1.0")
    if not 0 <= CLASS_ID < 1000:
        parser.error("CLASS_ID must be between 0 and 999")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Use one class for every noise sample so trajectories are directly comparable.
    CLASS_LABELS = np.full(NUM_SAMPLES, CLASS_ID, dtype=int)
    torch.set_grad_enabled(False)
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # --- load model ---
    latent_size = IMAGE_SIZE // 8
    model = SiT_models["SiT-XL/2"](
        input_size=latent_size,
        num_classes=1000,
        learn_sigma=(IMAGE_SIZE == 256),
    ).to(device)

    state_dict = find_model("pretrained_models/SiT-XL-2-256.pt")
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")

    # --- load VAE ---
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # --- transport (Linear path, velocity prediction) ---
    transport = create_transport("Linear", "velocity", None, None, None)

    # --- prepare initial noise and labels (shared across all CFG scales) ---
    z = torch.randn(NUM_SAMPLES, 4, latent_size, latent_size, device=device)
    y_cond = torch.tensor(CLASS_LABELS[:NUM_SAMPLES], device=device)
    y_null = torch.tensor([1000] * NUM_SAMPLES, device=device)

    # --- time grid ---
    t0, t1 = transport.check_interval(
        transport.train_eps,
        transport.sample_eps,
        sde=False,
        eval=True,
        reverse=False,
    )
    # NUM_STEPS is the number of Euler intervals.  The old code used NUM_STEPS
    # grid points and nevertheless took NUM_STEPS updates, so the last update
    # stepped past t1 by one dt.
    time_edges = torch.linspace(t0, t1, NUM_STEPS + 1, device=device)
    ts = time_edges[:-1]
    dt = time_edges[1] - time_edges[0]
    ts_np = ts.cpu().numpy()
    get_score = transport.path_sampler.get_score_from_velocity

    # --- run trajectory for each CFG scale ---
    results = {}  # cfg_scale -> {gap_norms_per_ch, v_cond_norms, score_mse, xt_final}

    for cfg in CFG_SCALES:
        cfg_class_dir = f"cfg_{cfg}_class_{CLASS_ID}"
        if CFG_SCHEDULE is None:
            save_dir = os.path.join(args.output_root, cfg_class_dir)
        else:
            save_dir = os.path.join(args.output_root, f"schedule_{CFG_SCHEDULE}", cfg_class_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n=== CFG_SCALE = {cfg}, SCHEDULE = {SCHEDULE_NAME} ===")
        with open(os.path.join(save_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cfg_scale": cfg,
                    "cfg_schedule": SCHEDULE_NAME,
                    "num_steps": NUM_STEPS,
                    "num_samples": NUM_SAMPLES,
                    "seed": SEED,
                    "class_id": CLASS_ID,
                },
                f,
                indent=2,
            )
        cfg_scale_trace = []
        gap_norms_per_ch = []
        gap_overalls = []
        v_cond_norms = []
        v_uncond_norms_list = []
        score_mse = []
        ratio = []
        v_cfg_vs_cond_norms = []
        v_cfg_vs_uncond_norms = []
        gap_overall_per_sample = []
        score_mse_per_sample = []
        ratio_per_sample = []
        gap_norms_per_sample = []
        # --- x1 gap metrics ---
        x1_gap_overall = []
        x1_gap_rel = []
        x1_gap_per_sample = []
        x1_gap_rel_per_sample = []
        x1_gap_norms_per_sample = []
        x1_cond_norms_per_sample = []
        x1_uncond_norms_per_sample = []
        cos_sim_per_sample = []
        xt = z.clone()

        for step_idx in range(NUM_STEPS):
            t_val = ts[step_idx]
            t_batch = torch.full((NUM_SAMPLES,), t_val, device=device)

            v_cond = model(xt, t_batch, y_cond)
            v_uncond = model(xt, t_batch, y_null)

            gap = v_cond - v_uncond

            gap_norm_ch = torch.norm(gap, dim=(2, 3)).mean(dim=0)
            gap_norms_per_ch.append(gap_norm_ch.cpu())

            s_cond = get_score(v_cond, xt, t_batch)
            s_uncond = get_score(v_uncond, xt, t_batch)
            score_mse.append(torch.mean((s_cond - s_uncond) ** 2).cpu().item())

            v_cond_norm = torch.norm(v_cond, dim=(2, 3)).mean().item()
            v_cond_norms.append(v_cond_norm)

            v_uncond_norm = torch.norm(v_uncond, dim=(2, 3)).mean().item()
            v_uncond_norms_list.append(v_uncond_norm)

            sigma_val = t_val  # noise level, 0→1
            if CFG_SCHEDULE:
                weight_fn = CFG_SCHEDULES[CFG_SCHEDULE]
                w = weight_fn(
                    sigma_val,
                    gap=gap,
                    v_cond=v_cond,
                    v_uncond=v_uncond,
                    cfg_scale=cfg,
                    xt=xt,
                    a=0.2,
                    b=0.8,
                )
                w_bc = _broadcast_weight(w, gap)
                v_cfg = v_uncond + w_bc * gap
                if isinstance(w, torch.Tensor):
                    if w.ndim == 0:
                        w_per_sample = w.expand(NUM_SAMPLES)
                    else:
                        w_per_sample = w.reshape(NUM_SAMPLES, -1).mean(dim=1)
                else:
                    w_per_sample = torch.full((NUM_SAMPLES,), float(w), device=device)
                cfg_scale_trace.append(w_per_sample.detach().cpu())
            else:
                v_cfg = v_uncond + cfg * gap
                cfg_scale_trace.append(torch.full((NUM_SAMPLES,), float(cfg)))
            gap_overall = torch.norm(gap, dim=(2, 3)).mean().item()
            gap_overalls.append(gap_overall)
            ratio.append(gap_overall / (v_uncond_norm + 1e-8))
            v_cfg_vs_cond = torch.norm(v_cfg - v_cond, dim=(2, 3)).mean().item()
            v_cfg_vs_uncond = torch.norm(v_cfg - v_uncond, dim=(2, 3)).mean().item()
            v_cfg_vs_cond_norms.append(v_cfg_vs_cond)
            v_cfg_vs_uncond_norms.append(v_cfg_vs_uncond)

            # per-sample metrics (for individual line plots)
            gap_spatial = torch.norm(gap, dim=(2, 3))  # (N, 4)
            gap_overall_per_sample.append(gap_spatial.mean(dim=1).cpu())  # (N,)
            gap_norms_per_sample.append(gap_spatial.cpu())  # (N, 4)

            v_uncond_spatial = torch.norm(v_uncond, dim=(2, 3))  # (N, 4)
            v_uncond_ps = v_uncond_spatial.mean(dim=1)  # (N,)
            ratio_per_sample.append((gap_spatial.mean(dim=1) / (v_uncond_ps + 1e-8)).cpu())

            s_diff_sq = (s_cond - s_uncond) ** 2
            score_mse_per_sample.append(s_diff_sq.mean(dim=(1, 2, 3)).cpu())  # (N,)

            # --- x1 prediction gap: E[x1|xt,c] - E[x1|xt,null] ---
            # Linear path: x_t = (1-t)·x0 + t·x1,  v = x1 - x0  =>  x1 = x_t + (1-t)·v
            t_exp = t_batch.view(-1, 1, 1, 1)
            x1_cond = xt + (1 - t_exp) * v_cond
            x1_uncond = xt + (1 - t_exp) * v_uncond
            x1_gap = x1_cond - x1_uncond  # = (1-t) * gap

            x1_gap_spatial = torch.norm(x1_gap, dim=(2, 3))  # (N, 4)
            x1_cond_spatial = torch.norm(x1_cond, dim=(2, 3))  # (N, 4)
            x1_uncond_spatial = torch.norm(x1_uncond, dim=(2, 3))  # (N, 4)

            x1_gap_norm_ps = x1_gap_spatial.mean(dim=1)  # (N,)
            x1_cond_norm_ps = x1_cond_spatial.mean(dim=1)  # (N,)
            x1_uncond_norm_ps = x1_uncond_spatial.mean(dim=1)  # (N,)
            x1_gap_overall.append(x1_gap_norm_ps.mean().item())
            x1_gap_rel.append((x1_gap_norm_ps / (x1_cond_norm_ps + 1e-8)).mean().item())
            x1_gap_per_sample.append(x1_gap_norm_ps.cpu())
            x1_gap_rel_per_sample.append((x1_gap_norm_ps / (x1_cond_norm_ps + 1e-8)).cpu())
            x1_gap_norms_per_sample.append(x1_gap_spatial.cpu())
            x1_cond_norms_per_sample.append(x1_cond_norm_ps.cpu())
            x1_uncond_norms_per_sample.append(x1_uncond_norm_ps.cpu())

            # Cosine similarity between v_cond and v_uncond (per sample)
            v_cond_flat = v_cond.flatten(1)  # (N, C*H*W)
            v_uncond_flat = v_uncond.flatten(1)
            cos_sim = (v_cond_flat * v_uncond_flat).sum(dim=1) / (
                v_cond_flat.norm(dim=1) * v_uncond_flat.norm(dim=1) + 1e-8
            )
            cos_sim_per_sample.append(cos_sim.cpu())

            xt = xt + v_cfg * dt

            if (step_idx + 1) % 100 == 0:
                g = gap_norm_ch.norm().item()
                print(f"  Step {step_idx + 1}/{NUM_STEPS}, t={t_val:.4f}, gap_norm={g:.4f}")

        gap_norms_per_ch = torch.stack(gap_norms_per_ch).numpy()
        v_cond_norms = np.array(v_cond_norms)
        v_uncond_norms = np.array(v_uncond_norms_list)
        score_mse = np.array(score_mse)
        v_cfg_vs_cond_arr = np.array(v_cfg_vs_cond_norms)
        v_cfg_vs_uncond_arr = np.array(v_cfg_vs_uncond_norms)
        cfg_scale_trace = torch.stack(cfg_scale_trace).numpy()  # (T, N)

        gap_overall_per_sample = torch.stack(gap_overall_per_sample).numpy()  # (T, N)
        score_mse_per_sample = torch.stack(score_mse_per_sample).numpy()
        ratio_per_sample = torch.stack(ratio_per_sample).numpy()
        gap_norms_per_sample = torch.stack(gap_norms_per_sample).numpy()  # (T, N, 4)

        x1_gap_overall = np.array(x1_gap_overall)
        x1_gap_rel = np.array(x1_gap_rel)
        x1_gap_per_sample = torch.stack(x1_gap_per_sample).numpy()  # (T, N)
        x1_gap_rel_per_sample = torch.stack(x1_gap_rel_per_sample).numpy()  # (T, N)
        x1_gap_norms_per_sample = torch.stack(x1_gap_norms_per_sample).numpy()  # (T, N, 4)
        x1_cond_norms_per_sample = torch.stack(x1_cond_norms_per_sample).numpy()  # (T, N)
        x1_uncond_norms_per_sample = torch.stack(x1_uncond_norms_per_sample).numpy()  # (T, N)
        cos_sim_per_sample = torch.stack(cos_sim_per_sample).numpy()  # (T, N)

        results[cfg] = {
            "gap_norms_per_ch": gap_norms_per_ch,
            "v_cond_norms": v_cond_norms,
            "v_uncond_norms": v_uncond_norms,
            "score_mse": score_mse,
            "gap_overall": gap_overalls,
            "v_cfg_vs_cond": v_cfg_vs_cond_arr,
            "v_cfg_vs_uncond": v_cfg_vs_uncond_arr,
            "ratio": ratio,
            "cfg_scale_trace": cfg_scale_trace,
            "x1_gap_overall": x1_gap_overall,
            "x1_gap_rel": x1_gap_rel,
            "x1_gap_per_sample": x1_gap_per_sample,
            "x1_gap_rel_per_sample": x1_gap_rel_per_sample,
            "x1_gap_norms_per_sample": x1_gap_norms_per_sample,
            "x1_cond_norms_per_sample": x1_cond_norms_per_sample,
            "x1_uncond_norms_per_sample": x1_uncond_norms_per_sample,
            "cos_sim_per_sample": cos_sim_per_sample,
            "xt_final": xt.clone(),
        }

        # --- plot for this CFG ---
        fig, axes = plt.subplots(5, 2, figsize=(13, 26))

        # Distinct color per sample (up to 20), mean line in black
        sample_cmap = plt.get_cmap("tab20")
        sample_colors = [sample_cmap(i % 20) for i in range(NUM_SAMPLES)]
        mean_color = "#111111"

        cos_sim_ps = results[cfg]["cos_sim_per_sample"]
        # Unpack x1 metrics
        x1_gap_ps = results[cfg]["x1_gap_per_sample"]
        x1_gap_rel_ps = results[cfg]["x1_gap_rel_per_sample"]
        x1_cond_norms_ps = results[cfg]["x1_cond_norms_per_sample"]
        x1_uncond_norms_ps = results[cfg]["x1_uncond_norms_per_sample"]

        # 1) Velocity gap: per-sample (thin, distinct colors) + mean (thick, black)
        ax = axes[0, 0]
        for i in range(NUM_SAMPLES):
            ax.plot(
                ts_np,
                gap_overall_per_sample[:, i],
                color=sample_colors[i],
                linewidth=0.3,
                alpha=0.7,
            )
        ax.plot(
            ts_np,
            gap_overall_per_sample.mean(axis=1),
            color=mean_color,
            linewidth=2.0,
            label=r"$\|v_{cond} - v_{uncond}\|$ (mean)",
        )
        ax.set_xlabel("Time t")
        ax.set_ylabel("Velocity norm")
        ax.set_title(f"Velocity gap components (cfg={cfg})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 2) Relative guidance strength: per-sample (thin) + mean (thick)
        ax = axes[0, 1]
        for i in range(NUM_SAMPLES):
            ax.plot(ts_np, ratio_per_sample[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7)
        ax.plot(
            ts_np,
            ratio_per_sample.mean(axis=1),
            color=mean_color,
            linewidth=2.0,
            label=r"Gap / $\|v_{\text{uncond}}\|$ (mean)",
        )
        ax.set_xlabel("Time t")
        ax.set_ylabel(r"Gap / $\|v_{\text{uncond}}\|$")
        ax.set_title("Relative guidance strength")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 3) Score MSE: per-sample (thin) + mean (thick)
        ax = axes[1, 0]
        for i in range(NUM_SAMPLES):
            ax.plot(
                ts_np, score_mse_per_sample[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7
            )
        ax.plot(
            ts_np,
            score_mse_per_sample.mean(axis=1),
            color=mean_color,
            linewidth=2.0,
            label=r"$\mathbb{E}[(s_{\text{cond}} - s_{\text{uncond}})^2]$ (mean)",
        )
        ax.set_xlabel("Time t")
        ax.set_ylabel(r"$\mathbb{E}[(s_{\text{cond}} - s_{\text{uncond}})^2]$")
        ax.set_title("Score MSE (cond vs uncond)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 4) Per-channel gap norm: per-sample (thin, distinct colors) + mean per channel (thick)
        ax = axes[1, 1]
        ch_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
        ch_labels = ["Channel 0", "Channel 1", "Channel 2", "Channel 3"]
        for ch in range(4):
            for i in range(NUM_SAMPLES):
                ax.plot(
                    ts_np,
                    gap_norms_per_sample[:, i, ch],
                    color=sample_colors[i],
                    linewidth=0.3,
                    alpha=0.5,
                )
            ax.plot(
                ts_np,
                gap_norms_per_sample[:, :, ch].mean(axis=1),
                color=ch_colors[ch],
                label=ch_labels[ch],
                linewidth=2.0,
            )
        ax.set_xlabel("Time t")
        ax.set_ylabel(r"$\|v_{\text{cond}} - v_{\text{uncond}}\|$ (per ch)")
        ax.set_title(f"Per-channel gap norm (cfg={cfg})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 5) E[x1|xt] gap norm: per-sample (thin) + mean (thick)
        ax = axes[2, 0]
        for i in range(NUM_SAMPLES):
            ax.plot(ts_np, x1_gap_ps[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7)
        ax.plot(
            ts_np,
            x1_gap_ps.mean(axis=1),
            color=mean_color,
            linewidth=2.0,
            label=r"$\|\mathbb{E}[x_1|x_t,c] - \mathbb{E}[x_1|x_t]\|$ (mean)",
        )
        ax.set_xlabel("Time t")
        ax.set_ylabel(r"$\|x_1^{cond} - x_1^{uncond}\|$")
        ax.set_title(f"E[x1|xt] gap norm (cfg={cfg})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 6) Relative E[x1|xt] gap: gap / ||E[x1|xt, null]||
        ax = axes[2, 1]
        for i in range(NUM_SAMPLES):
            ax.plot(ts_np, x1_gap_rel_ps[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7)
        ax.plot(
            ts_np,
            x1_gap_rel_ps.mean(axis=1),
            color=mean_color,
            linewidth=2.0,
            label=r"$\|x_1^{gap}\| / \|x_1^{cond}\|$ (mean)",
        )
        ax.set_xlabel("Time t")
        ax.set_ylabel(r"Relative $x_1$ gap")
        ax.set_title(f"Relative E[x1|xt] gap (cfg={cfg})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 7) Cosine similarity between v_cond and v_uncond
        ax = axes[3, 0]
        for i in range(NUM_SAMPLES):
            ax.plot(ts_np, cos_sim_ps[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7)
        ax.plot(
            ts_np,
            cos_sim_ps.mean(axis=1),
            color=mean_color,
            linewidth=2.0,
            label=r"$\cos(v_{cond}, v_{uncond})$ (mean)",
        )
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Time t")
        ax.set_ylabel(r"$\cos(v_{cond}, v_{uncond})$")
        ax.set_title(f"Cosine similarity: v_cond vs v_uncond (cfg={cfg})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 8) E[x1|xt] norm: conditional vs unconditional (side by side comparison)
        ax = axes[4, 0]
        cond_color = "#d62728"
        uncond_color = "#1f77b4"
        for i in range(NUM_SAMPLES):
            ax.plot(ts_np, x1_cond_norms_ps[:, i], color=cond_color, linewidth=0.3, alpha=0.25)
            ax.plot(ts_np, x1_uncond_norms_ps[:, i], color=uncond_color, linewidth=0.3, alpha=0.25)
        ax.plot(
            ts_np,
            x1_cond_norms_ps.mean(axis=1),
            color=cond_color,
            linewidth=2.5,
            label=r"$\mathbb{E}[x_1|x_t,c]$ (mean)",
        )
        ax.plot(
            ts_np,
            x1_uncond_norms_ps.mean(axis=1),
            color=uncond_color,
            linewidth=2.5,
            label=r"$\mathbb{E}[x_1|x_t,null]$ (mean)",
        )
        ax.set_xlabel("Time t")
        ax.set_ylabel(r"$\|\mathbb{E}[x_1|x_t]\|$")
        ax.set_title(f"Cond vs Uncond E[x1|xt] norm (cfg={cfg})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 9) Effective CFG scale over time
        ax = axes[4, 1]
        for i in range(NUM_SAMPLES):
            ax.plot(ts_np, cfg_scale_trace[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7)
        ax.plot(
            ts_np,
            cfg_scale_trace.mean(axis=1),
            color=mean_color,
            linewidth=2.0,
            label="effective cfg_scale (mean)",
        )
        ax.axhline(y=cfg, color="gray", linestyle=":", linewidth=0.8, label="target cfg_scale")
        ax.set_xlabel("Time t")
        ax.set_ylabel("CFG scale")
        ax.set_title(f"Effective CFG scale over time (cfg={cfg})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        for ax in axes.flat:
            _enlarge_axis_text(ax)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"gap_cfg{cfg}.pdf")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved gap plot to {plot_path}")

        # --- individual subplot PDFs ---
        _save_subplot(
            ts_np,
            gap_overall_per_sample,
            sample_colors,
            mean_color,
            cfg,
            save_dir,
            "01_velocity_gap",
            "Velocity gap",
            "Time t",
            "Velocity norm",
            r"$\|v_{cond} - v_{uncond}\|$ (mean)",
            plot_fn=lambda ax, i: ax.plot(
                ts_np,
                gap_overall_per_sample[:, i],
                color=sample_colors[i],
                linewidth=0.3,
                alpha=0.7,
            ),
        )

        _save_subplot(
            ts_np,
            ratio_per_sample,
            sample_colors,
            mean_color,
            cfg,
            save_dir,
            "02_relative_guidance",
            "Relative guidance strength",
            "Time t",
            r"Gap / $\|v_{uncond}\|$",
            r"Gap / $\|v_{uncond}\|$ (mean)",
            plot_fn=lambda ax, i: ax.plot(
                ts_np, ratio_per_sample[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7
            ),
        )

        _save_subplot(
            ts_np,
            score_mse_per_sample,
            sample_colors,
            mean_color,
            cfg,
            save_dir,
            "03_score_mse",
            "Score MSE (cond vs uncond)",
            "Time t",
            r"$\mathbb{E}[(s_{cond} - s_{uncond})^2]$",
            r"$\mathbb{E}[(s_{cond} - s_{uncond})^2]$ (mean)",
            plot_fn=lambda ax, i: ax.plot(
                ts_np, score_mse_per_sample[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7
            ),
        )

        # Per-channel gap norm (different style: per-channel lines, not per-sample thin)
        _save_subplot_ch(
            ts_np,
            gap_norms_per_sample,
            ch_colors,
            ch_labels,
            cfg,
            save_dir,
            "04_per_channel_gap",
            f"Per-channel gap norm (cfg={cfg})",
            "Time t",
            r"$\|v_{cond} - v_{uncond}\|$ (per ch)",
        )

        _save_subplot(
            ts_np,
            x1_gap_ps,
            sample_colors,
            mean_color,
            cfg,
            save_dir,
            "05_x1_gap",
            "E[x1|xt] gap norm",
            "Time t",
            r"$\|x_1^{cond} - x_1^{uncond}\|$",
            r"$\|\mathbb{E}[x_1|x_t,c] - \mathbb{E}[x_1|x_t]\|$ (mean)",
            plot_fn=lambda ax, i: ax.plot(
                ts_np, x1_gap_ps[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7
            ),
        )

        _save_subplot(
            ts_np,
            x1_gap_rel_ps,
            sample_colors,
            mean_color,
            cfg,
            save_dir,
            "06_x1_gap_rel",
            "Relative E[x1|xt] gap",
            "Time t",
            r"Relative $x_1$ gap",
            r"$\|x_1^{gap}\| / \|x_1^{cond}\|$ (mean)",
            plot_fn=lambda ax, i: ax.plot(
                ts_np, x1_gap_rel_ps[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7
            ),
        )

        _save_subplot(
            ts_np,
            cos_sim_ps,
            sample_colors,
            mean_color,
            cfg,
            save_dir,
            "07_cosine_sim",
            "Cosine similarity: v_cond vs v_uncond",
            "Time t",
            r"$\cos(v_{cond}, v_{uncond})$",
            r"$\cos(v_{cond}, v_{uncond})$ (mean)",
            plot_fn=lambda ax, i: ax.plot(
                ts_np, cos_sim_ps[:, i], color=sample_colors[i], linewidth=0.3, alpha=0.7
            ),
            extra_fn=lambda ax: ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8),
        )

        # Cond vs Uncond E[x1|xt] norm
        _save_subplot_dual(
            ts_np,
            x1_cond_norms_ps,
            x1_uncond_norms_ps,
            cfg,
            save_dir,
            "08_x1_norm_cond_vs_uncond",
            "Cond vs Uncond E[x1|xt] norm",
            "Time t",
            r"$\|\mathbb{E}[x_1|x_t]\|$",
            r"$\mathbb{E}[x_1|x_t,c]$ (mean)",
            r"$\mathbb{E}[x_1|x_t,null]$ (mean)",
        )

        # Effective CFG scale
        _save_subplot_cfg_trace(
            ts_np,
            cfg_scale_trace,
            sample_colors,
            mean_color,
            cfg,
            save_dir,
            "09_cfg_scale",
            "Effective CFG scale over time",
            "Time t",
            "CFG scale",
        )

        # --- decode & save final samples (one PDF per sample) ---
        with torch.no_grad():
            samples_decoded = vae.decode(xt / 0.18215).sample
        grid = samples_decoded.cpu()
        grid = (grid + 1) / 2  # [-1, 1] -> [0, 1]
        grid = grid.clamp(0, 1)
        for idx in range(grid.shape[0]):
            fig, ax = plt.subplots(figsize=(3, 3))
            img = grid[idx].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis("off")
            samples_path = os.path.join(save_dir, f"sample_{idx:02d}_cfg{cfg}.pdf")
            plt.savefig(samples_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved sample {idx} to {samples_path}")

        # --- grid overview ---
        n = grid.shape[0]
        nrow = 4
        ncol = (n + nrow - 1) // nrow
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 2.5))
        axes = np.atleast_2d(axes)
        for idx in range(n):
            r, c = idx // ncol, idx % ncol
            img = grid[idx].permute(1, 2, 0).numpy()
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
        for idx in range(n, axes.size):
            r, c = idx // ncol, idx % ncol
            axes[r, c].axis("off")
        grid_path = os.path.join(save_dir, f"grid_cfg{cfg}.pdf")
        plt.savefig(grid_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved grid to {grid_path}")

    # --- summary ---
    print("\n=== Summary ===")
    for cfg in CFG_SCALES:
        go = np.array(results[cfg]["gap_overall"])
        max_idx = go.argmax()
        cfg_trace = np.array(results[cfg]["cfg_scale_trace"])
        print(
            f"cfg={cfg}: max_gap={go.max():.4f} @ t={ts_np[max_idx]:.4f}, "
            f"mean_gap={go.mean():.4f}, cfg_scale_range=[{cfg_trace.min():.4f}, {cfg_trace.max():.4f}]"
        )


if __name__ == "__main__":
    main()
