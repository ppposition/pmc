"""Shared SiT generation used by the visualization and FID entry points."""

import math
import os
from time import time

import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from tqdm import tqdm

from .cfg_schedules import TVCFG, make_cfg_forward, make_tv_cfg_forward, norm_clamped
from .download import find_model
from .models import SiT_models
from .transport import Sampler, create_transport

CFG_SCHEDULES = {"norm_clamped": norm_clamped}


def add_generation_arguments(parser):
    # Model & sampling
    parser.add_argument("--model", type=str, default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument(
        "--cfg-schedule",
        choices=["constant", *CFG_SCHEDULES, "tv_cfg"],
        default=None,
        help="CFG schedule; tv_cfg is normalized triangular guidance (Euler only); omit for constant CFG",
    )
    parser.add_argument(
        "--cfg-gamma",
        type=float,
        default=None,
        help="Relaxation gamma for norm_clamped (None = schedule default 1.1)",
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=50,
        help="Number of ODE time points (Euler NFE = this value minus 1)",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="euler",
        help="ODE solver: 'euler' for fixed-step (faster), 'dopri5' for adaptive (more accurate)",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--reverse", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--ckpt", type=str, default=None)

    # Transport
    parser.add_argument("--path-type", type=str, default="Linear")
    parser.add_argument("--prediction", type=str, default="velocity")

    # Generation
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size per GPU step (CFG doubling happens internally)",
    )
    parser.add_argument("--out-dir", type=str, default="fid_outputs")

    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="ImageNet class index; omit to cycle through all classes",
    )


def validate_generation_arguments(args):
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("--num-samples and --batch-size must be positive")
    if args.num_classes <= 0 or args.num_sampling_steps < 2:
        raise ValueError("--num-classes must be positive and --num-sampling-steps must be >= 2")
    if args.image_size <= 0 or args.image_size % 8:
        raise ValueError("--image-size must be positive and divisible by 8")
    if getattr(args, "class_id", None) is not None and not 0 <= args.class_id < args.num_classes:
        raise ValueError("--class-id must be in [0, num_classes)")
    if not math.isfinite(args.cfg_scale) or args.cfg_scale < 1:
        raise ValueError("--cfg-scale must be finite and >= 1")
    if args.cfg_gamma is not None and (not math.isfinite(args.cfg_gamma) or args.cfg_gamma < 1):
        raise ValueError("--cfg-gamma must be finite and >= 1")

    if args.cfg_schedule == "tv_cfg" and args.sampling_method != "euler":
        raise ValueError("TV-CFG requires --sampling-method euler")


@torch.no_grad()
def generate_images(args):
    """Write exactly num_samples PNGs; callers manage output-directory reuse."""
    validate_generation_arguments(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # ---- config ----
    torch.manual_seed(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    print(f"Device: {device}")

    # ---- load model ----
    latent_size = args.image_size // 8
    learn_sigma = args.image_size == 256
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    ).to(device)
    ckpt = find_model(args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt")
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # ---- load VAE ----
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    # ---- transport & sampler ----
    transport = create_transport(
        path_type=args.path_type,
        prediction=args.prediction,
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
    )
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=args.sampling_method,
        num_steps=args.num_sampling_steps,
        atol=args.atol,
        rtol=args.rtol,
        reverse=args.reverse,
    )

    # ---- generate images ----
    os.makedirs(args.out_dir, exist_ok=True)
    num_samples = args.num_samples
    batch_size = args.batch_size
    latent_scale = 0.18215  # VAE scaling factor for SD

    # Assign one class per sample, cycling through 1000 classes
    if getattr(args, "class_id", None) is None:
        class_labels = torch.arange(num_samples, device=device) % args.num_classes
    else:
        class_labels = torch.full((num_samples,), args.class_id, device=device)

    total_batches = (num_samples + batch_size - 1) // batch_size
    start_time = time()

    weight_fn = CFG_SCHEDULES.get(args.cfg_schedule)
    model_fn = (
        model.forward_with_cfg
        if weight_fn is None
        else make_cfg_forward(model, weight_fn, args.cfg_scale, args.cfg_gamma)
    )

    if args.cfg_schedule == "tv_cfg":
        # The bound integrator.sample method exposes the actual Euler grid.
        schedule = TVCFG(sample_fn.__self__.t, args.cfg_scale)
        model_fn = make_tv_cfg_forward(model, schedule)
        average = sum(w * dt for w, dt in zip(schedule.scales, schedule.normalized_widths))
        print(
            f"TV-CFG: NFE={schedule.num_evaluations}, A={schedule.normalization:.8f}, "
            f"scale range=[{min(schedule.scales):.6f}, {max(schedule.scales):.6f}], "
            f"time-weighted average={average:.6f}"
        )

    for batch_idx in tqdm(range(total_batches), desc="Generating"):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)
        n = end - start

        if getattr(args, "per_sample_seed", False):
            z = torch.stack(
                [
                    torch.randn(
                        4,
                        latent_size,
                        latent_size,
                        device=device,
                        generator=torch.Generator(device=device).manual_seed(args.seed + idx),
                    )
                    for idx in range(start, end)
                ]
            )
        else:
            z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = class_labels[start:end]

        # CFG: double the batch
        z_cfg = torch.cat([z, z], dim=0)
        y_null = torch.full((n,), args.num_classes, device=device)
        y_cfg = torch.cat([y, y_null], dim=0)
        model_kwargs = dict(y=y_cfg, cfg_scale=args.cfg_scale)

        samples = sample_fn(z_cfg, model_fn, **model_kwargs)[-1]
        samples, _ = samples.chunk(2, dim=0)
        samples = vae.decode(samples / latent_scale).sample

        # Clamp and save as PNG in [0, 1]
        samples = torch.clamp(samples, -1, 1)
        samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]

        for i in range(n):
            idx = start + i
            save_image(samples[i], os.path.join(args.out_dir, f"{idx:06d}.png"))

    elapsed = time() - start_time
    print(
        f"Generated {num_samples} images in {elapsed:.1f}s ({elapsed / num_samples:.2f}s per image)"
    )
    # Release model memory before returning to visualization or evaluation.
    del model, vae, samples, z, z_cfg
    torch.cuda.empty_cache()
