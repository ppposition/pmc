# CFG / Norm-Clamped Evaluation

Generation, evaluation, and guidance-trajectory visualization for **SiT on
ImageNet** and **Stable Diffusion 3.5 Medium on COCO**. The project compares
constant classifier-free guidance (CFG) with norm-clamped guidance. SiT generation
also supports a time-varying, normalized triangular schedule (`tv_cfg`).

The SiT model and transport implementation are derived from SiT. Model weights,
datasets, and experiment outputs are not included.

## Project layout

```text
.
├── src/cfg_norm_clamped_eval/
│   ├── generate_imagenet.py    # Standalone SiT generation with gallery
│   ├── generate_sd35.py        # Standalone SD3.5 text-to-image generation
│   ├── gallery.py              # HTML gallery and paginated PNG contact sheets
│   ├── sit_sampling.py         # Shared SiT generation engine
│   ├── cfg_schedules.py          # Shared SiT / SD3.5 guidance schedules
│   ├── models.py                # SiT model definitions
│   ├── transport/               # Transport paths and ODE / SDE samplers
│   ├── download.py              # SiT checkpoint loading and downloading
│   ├── download_coco.py         # COCO data preparation
│   ├── download_imagenet_val.py # ImageNet validation data preparation
│   ├── generate_coco_sd35.py    # Resumable generation using a fixed COCO subset
│   ├── eval_fid.py              # SiT generation, Clean-FID, saturation, recall
│   ├── eval_imagenet.py         # ADM FID, sFID, IS, precision, and recall
│   ├── eval_metrics.py          # COCO / general image metrics
│   ├── eval_image_reward.py     # ImageReward compatibility and scoring
│   ├── eval_t2i.py              # Batch text-to-image evaluation
│   ├── analyze_cfg_gap.py       # SiT guidance trajectory visualization
│   └── sd35_cfg_gap.py          # SD3.5 guidance trajectory visualization
├── scripts/                    # Experiment batch launchers
├── tests/                      # CPU guidance regression tests
├── pyproject.toml              # Package metadata and development tooling
├── uv.lock                     # Locked runtime and development dependencies
├── .python-version             # Python version used by uv
├── requirements.txt            # Legacy pip entry point
├── requirements-imagenet.txt   # Legacy pip entry point with TensorFlow
└── LICENSE.txt
```

## Environment management with uv

Dependencies are maintained in `pyproject.toml` and pinned in `uv.lock`.
`.python-version` selects Python 3.11. From the project root:

```bash
uv sync --locked

# Optional: ADM-compatible ImageNet metrics, including TensorFlow
uv sync --locked --extra imagenet
```

`uv` creates `.venv`, installs the project in editable mode, and includes the
`dev` dependency group by default. No manual environment activation is needed.
Use `uv run --locked` for commands below. For intentional dependency changes, use
`uv add PACKAGE` or edit `pyproject.toml`, then run `uv lock` and `uv sync`.
The requirements files remain compatibility entry points for pip; they reference
the package instead of maintaining separate dependency lists.

The lockfile uses PyPI's PyTorch and torchvision distributions. GPU execution
requires a compatible NVIDIA driver. If your machine needs a specific PyTorch
CUDA build, configure the corresponding uv package source and regenerate the
lockfile; a subsequent `uv sync` reconciles the environment with that lockfile.

SD3.5 generation requires a CUDA GPU, model access, and Hugging Face
authentication. Set `HF_ENDPOINT` if your environment uses a mirror; the code does
not override it. SiT downloads its default SiT-XL/2 256×256 checkpoint when needed;
use `--ckpt` to select a local checkpoint. Models and datasets are downloaded only
when the relevant tools run.

Shell launchers in `scripts/` use `uv run --locked` and change to the project
root before running; their relative input and output paths are rooted there.
Set `UV=/path/to/uv` if needed. Direct module commands resolve relative paths
against the current working directory.

### Manually downloaded SiT weights

When running from the project root, place the pretrained SiT-XL/2 256×256
checkpoint in `pretrained_models/`:

```text
pretrained_models/
└── SiT-XL-2-256x256.pt
```

The original download filename, `SiT-XL-2-256.pt`, is also accepted in that
directory; no renaming is required. With either file present, the default SiT
commands load it locally. If both are present, `SiT-XL-2-256x256.pt` takes
precedence. If neither is present, the checkpoint is downloaded automatically.

To keep weights elsewhere, pass the checkpoint file path explicitly:

```bash
uv run --locked python -m cfg_norm_clamped_eval.generate_imagenet \
  --ckpt /path/to/SiT-XL-2-256.pt \
  --num-samples 16 --out-dir outputs/preview_sit_local
```

The default `pretrained_models/` directory is relative to the current working
directory. These paths apply to the SiT checkpoint; its VAE is loaded separately
through Hugging Face.

## Generate images for visual inspection

Both standalone generators accept `--num-samples` and generate exactly that many
individual PNGs, including a partial final batch. They do not require reference
datasets and do not compute metrics. Use a new or empty `--out-dir` for each run;
these preview commands refuse to overwrite or mix existing runs. COCO benchmark
generation retains its separate resumable workflow below.

### ImageNet / SiT

Generate 16 images from ImageNet class index 300:

```bash
uv run --locked python -m cfg_norm_clamped_eval.generate_imagenet \
  --num-samples 16 --class-id 300 --seed 42 --batch-size 4 \
  --cfg-scale 2.0 --cfg-schedule norm_clamped --cfg-gamma 1.1 \
  --out-dir outputs/preview_sit_norm_clamped
```

Use `--cfg-schedule constant` or `--cfg-schedule tv_cfg` to compare methods.
Omitting `--class-id` cycles through class indices 0–999. The default checkpoint
uses ImageNet's 1,000-class index ordering, rather than text labels. `--ckpt`,
`--gpu`, `--image-size`, and the other SiT sampling flags are available.
`--num-sampling-steps 50` means 50 time points / 49 Euler integration steps.
TV-CFG requires `--sampling-method euler` (the default).

### SD3.5 from text prompts

Generate 12 images from one prompt:

```bash
uv run --locked python -m cfg_norm_clamped_eval.generate_sd35 \
  --prompt "A red panda sitting on a tree branch, wildlife photography" \
  --num-samples 12 --seed 42 --batch-size 2 \
  --cfg-scale 5.0 --cfg-schedule norm-clamped --cfg-gamma 1.1 \
  --height 512 --width 512 --num-steps 20 \
  --out-dir outputs/preview_sd35_norm_clamped
```

Alternatively, use `--prompt-file prompts.txt` with one nonempty UTF-8 prompt per
line. Prompts cycle in file order until `--num-samples` images are generated.
For example, two prompts and five samples produce A, B, A, B, A. The output
`prompts.txt` contains one prompt per generated image in filename order.
Use `--cfg-schedule constant` for standard CFG, and `--scheduler heun` for Heun.
No COCO data is needed for this entry point.

### Output files

```text
outputs/<run>/
├── images/              # Exactly N original PNGs: 000000.png, 000001.png, ...
├── preview/             # Paginated contact sheets: grid_0000.png, ...
├── index.html           # Open directly in a browser; click images for originals
├── run_config.json      # Model, guidance, resolution, and sampling settings
├── samples.jsonl        # Filename, sample index, seed, and class ID / prompt
└── prompts.txt          # SD3.5 only: aligned prompts for later evaluation
```

Open `index.html` directly in a browser, or open `preview/grid_0000.png` in your
IDE. No server or network connection is needed to view the results. The gallery
uses the original PNGs; contact sheets use aspect-preserving thumbnails.
`--grid-columns 4 --grid-page-size 64 --thumbnail-size 256` controls the preview
layout. Larger sample sets are split across contact sheets to bound memory use.

Each sample uses seed `seed + index`. Keep the seed, class/prompt order, model,
and sampling settings fixed when comparing guidance methods. Initial noise is
independent of batch size; GPU floating-point execution may still cause small
output differences across batch sizes or hardware. The existing FID benchmark
retains its original batched random-number stream.

The shell equivalents forward all command-line arguments:

```bash
bash scripts/generate_imagenet.sh --num-samples 8 --class-id 300 --out-dir outputs/sit_8
bash scripts/generate_sd35.sh --prompt "A cat" --num-samples 8 --out-dir outputs/sd35_8
```

For later metric evaluation, pass the **`images/` subdirectory**, so contact
sheets are excluded, for example:

```bash
uv run --locked python -m cfg_norm_clamped_eval.eval_fid \
  --compute-fid-only --out-dir outputs/preview_sit_norm_clamped/images \
  --ref-dir data/imagenet_val --skip-diversity
```

## SD3.5 / COCO

Prepare COCO, then run both guidance variants:

```bash
uv run --locked python -m cfg_norm_clamped_eval.download_coco --data-dir coco_data
CFG_SCHEDULE=constant bash scripts/run_coco_sd35_cfg_sweep.sh
CFG_SCHEDULE=norm-clamped bash scripts/run_coco_sd35_cfg_sweep.sh
```

The launcher selects the first caption for each of 30,000 distinct COCO val2014
images, with selection seed 42 and noise seed 10000. Defaults are 512×512, Euler
sampling with 20 steps, gamma 1.2, and CFG scales 4.5, 5.0, 6.0, and 7.0. Both
methods share the manifest and noise seeds; their outputs use separate directories
under `coco_data/sd35_coco30k`.

Settings at the top of the launcher can be overridden through environment variables:

```bash
GPU=0 BATCH_SIZE=2 CFG_SCALES="4.5 5.0" \
  CFG_SCHEDULE=norm-clamped bash scripts/run_coco_sd35_cfg_sweep.sh
```

Evaluation uses all val2014 reference images and reports Clean-FID, CLIP ViT-L/14
raw cosine similarity, and ImageReward. Clean-FID values should be compared only
with results using the same feature extraction and preprocessing protocol.

For a smaller generation run:

```bash
uv run --locked python -m cfg_norm_clamped_eval.generate_coco_sd35 \
  --data-dir coco_data --out-dir outputs/sd35/norm_clamped \
  --num-samples 100 --cfg-scale 5.0 --cfg-schedule norm-clamped \
  --cfg-gamma 1.2 --height 512 --width 512 --num-steps 20 --batch-size 2
```

Use `--cfg-schedule constant` for standard CFG. The generator can resume completed
images in an existing run. For trajectory visualization, edit the configuration
at the top of `src/cfg_norm_clamped_eval/sd35_cfg_gap.py`, then run:

```bash
uv run --locked python -m cfg_norm_clamped_eval.sd35_cfg_gap
```

The visualization defaults to gamma **1.15**; the shared SD3.5 guidance function
and standalone generator default to **1.1**. Set gamma explicitly when comparing
experiments. Standalone generation also defaults to 1024×1024 and 40 steps;
the sweep launcher overrides these settings.

## ImageNet / SiT

Place reference images in `data/imagenet_val`, specify an existing directory with
`REF_DIR`, or use the download utility:

```bash
uv run --locked python -m cfg_norm_clamped_eval.download_imagenet_val --out-dir data/imagenet_val
REF_DIR=data/imagenet_val bash scripts/imagenet_generate.sh
bash scripts/imagenet_evaluate.sh
```

The downloader retains original image dimensions unless `--size` is supplied.
Record any resizing or other preprocessing used for reference images.

The generation launcher runs **constant**, **norm_clamped** (gamma 1.1), and
**tv_cfg**, with CFG scale 2.0 and 50,000 images at 256×256 per method. It uses
Euler sampling with **50 time points (49 integration steps)** and computes
Clean-FID. `GPU`, `UV`, and `REF_DIR` can be overridden through environment
variables; the ADM launcher also accepts `DEVICE` (default `cuda:0`).

ADM evaluation separately computes FID, sFID, IS, precision, and recall. It
downloads the reference NPZ and Inception graph on first use.
**Clean-FID and ADM FID use different protocols and must be reported separately.**

Individual generation and analysis commands:

```bash
# Constant CFG: omit --cfg-schedule
uv run --locked python -m cfg_norm_clamped_eval.eval_fid \
  --ref-dir data/imagenet_val --out-dir outputs/imagenet/constant \
  --cfg-scale 2.0 --num-samples 50000 --skip-diversity

# Norm-clamped CFG
uv run --locked python -m cfg_norm_clamped_eval.eval_fid \
  --ref-dir data/imagenet_val --out-dir outputs/imagenet/norm_clamped \
  --cfg-scale 2.0 --cfg-schedule norm_clamped --cfg-gamma 1.1 \
  --num-samples 50000 --skip-diversity

# Normalized triangular CFG (Euler only)
uv run --locked python -m cfg_norm_clamped_eval.eval_fid \
  --ref-dir data/imagenet_val --out-dir outputs/imagenet/tv_cfg \
  --cfg-scale 2.0 --cfg-schedule tv_cfg --sampling-method euler \
  --num-samples 50000 --skip-diversity

uv run --locked python -m cfg_norm_clamped_eval.analyze_cfg_gap --cfg-scale 2.0 --device cuda:0
uv run --locked python -m cfg_norm_clamped_eval.analyze_cfg_gap \
  --cfg-scale 2.0 --cfg-schedule norm_clamped --device cuda:0
```

## Guidance conventions

Standard CFG is `v_uncond + w * (v_cond - v_uncond)`. The two norm-clamped
implementations retain their original, model-specific definitions:

- **SD3.5:** `x0_cond = xt - sigma * v_cond`, with
  `||x0_cfg - x0_cond|| <= (gamma - 1) * ||x0_cond||`.
- **SiT:** `m_cond = xt + (1 - t) * v_cond`, with a quadratic constraint enforcing
  `||m_cfg|| <= gamma * ||m_cond||`.

Norm-clamped weights are clipped to `[1, cfg_scale]`. Use `cfg_scale >= 1` and
`gamma >= 1`. For zero displacement, SD3.5 keeps the requested scale while SiT's
existing convention returns 1. These edge-case behaviors are preserved.

SiT's original `models.py::forward_with_cfg` applies CFG to only the first three
latent channels. The norm-clamped wrapper applies guidance to all latent channels.
Account for this difference when interpreting comparisons.

SiT `tv_cfg` uses the original `forward_with_cfg` channel convention and changes
only its scale over time. Its discrete triangular profile is normalized so that
the interval-width-weighted mean equals `cfg_scale`; instantaneous weights can
exceed that value. It supports Euler sampling only. The ImageNet trajectory
visualizer supports constant and norm-clamped guidance.

## Development

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv run --locked python -m compileall -q src tests
bash -n scripts/imagenet_generate.sh
bash -n scripts/imagenet_evaluate.sh
bash -n scripts/run_coco_sd35_cfg_sweep.sh
bash -n scripts/generate_imagenet.sh
bash -n scripts/generate_sd35.sh
```

CPU tests cover guidance constraints, generation counts, partial batches, prompt
cycling, seed assignment, contact-sheet pagination, and output protection.
Generation orchestration uses lightweight model substitutes in tests; no model
downloads are needed. Full GPU generation and COCO-30K / ImageNet-50K evaluation are
separate integration checks and are not part of this test suite.

Formatting and lint configuration live in `pyproject.toml`; `.editorconfig`
defines basic whitespace conventions. Import ordering that initializes GPU
visibility, plotting backends, or ImageReward compatibility patches is intentional.

## Outputs and licensing

`.gitignore` excludes common checkpoints, datasets, arrays, archives, generated
images, PDFs, logs, caches, virtual environments, and build artifacts. Add custom
output directories to it as needed; Git ignore rules cannot enforce file-size
limits.

SiT-derived code retains its MIT license and copyright notices in `LICENSE.txt`.
Model and dataset usage remains subject to the respective providers' terms.
