"""ADM-compatible evaluation for class-conditional ImageNet generation.

This implements the protocol from OpenAI's guided-diffusion evaluator:

* FID uses the 2048-dimensional ``pool_3:0`` Inception features.
* sFID uses one 2023-dimensional spatial feature per image, obtained from
  ``mixed_6/conv:0[..., :7]`` (17 * 17 * 7).
* Inception Score is computed from the classifier attached to the same
  official Inception graph, in consecutive groups of 5,000 samples.
* Improved Precision/Recall uses squared Euclidean distance, k=3, on the
  same pool_3 features.

Images are passed to the official graph as RGB uint8 values in [0, 255].
There is deliberately no torchvision/Clean-FID preprocessing in this file:
mixing those implementations makes the results incomparable with the ADM,
DiT and SiT ImageNet evaluation tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import site
import sys
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image
from scipy import linalg
from tqdm import tqdm

INCEPTION_V3_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/"
    "jul-2021/ref_batches/classify_image_graph_def.pb"
)
IMAGENET_256_REFERENCE_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/"
    "jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"
)
DEFAULT_REFERENCE_BATCH = "VIRTUAL_imagenet256_labeled.npz"
FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_CUDA_PATH_READY = "SIT_TENSORFLOW_CUDA_PATH_READY"


def _restart_with_tensorflow_cuda_libraries() -> None:
    """Expose pip/uv-installed NVIDIA libraries before TensorFlow is loaded.

    Recent ``tensorflow[and-cuda]`` installations put CUDA libraries under
    ``site-packages/nvidia/*/lib``.  The ELF loader reads ``LD_LIBRARY_PATH``
    when the Python process starts, so changing it immediately before the TF
    import is too late: restart this script once with the correct search path.
    CUDA 13 libraries installed for PyTorch are deliberately excluded because
    TensorFlow 2.21 is built against CUDA 12.x.
    """
    if os.environ.get(_CUDA_PATH_READY) == "1":
        return

    library_dirs: list[str] = []
    roots = [Path(path) for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))
    for root in roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for library_dir in sorted(nvidia_root.glob("*/lib")):
            if library_dir.parent.name == "cu13":
                continue
            library_dirs.append(str(library_dir.resolve()))

    if not library_dirs:
        return
    existing = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(library_dirs + existing))
    environment[_CUDA_PATH_READY] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)


def _tensorflow():
    """Import TensorFlow lazily and provide an actionable dependency error."""
    try:
        import tensorflow.compat.v1 as tf
    except ImportError as exc:
        raise RuntimeError(
            "ADM-compatible evaluation requires TensorFlow. Install it with "
            "`pip install 'tensorflow>=2.12' scipy requests tqdm Pillow`."
        ) from exc
    tf.disable_eager_execution()
    return tf


def image_files(directory: str | Path) -> list[Path]:
    """Return all supported images recursively in deterministic order."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Not an image directory: {root}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f"No images found in {root}")
    return files


def image_batches(files: Sequence[Path], batch_size: int) -> Iterator[np.ndarray]:
    """Yield NHWC RGB uint8 batches without changing pixels or image size."""
    for start in range(0, len(files), batch_size):
        arrays = []
        for path in files[start : start + batch_size]:
            with Image.open(path) as image:
                arrays.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        shapes = {x.shape for x in arrays}
        if len(shapes) != 1:
            raise ValueError(
                f"All images in a batch must have the same resolution; got {sorted(shapes)}"
            )
        yield np.stack(arrays)


def array_batches(array: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    """Yield batches from an official ADM reference/sample array."""
    for start in range(0, len(array), batch_size):
        yield array[start : start + batch_size]


def _download_file(url: str, path: Path, description: str) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    print(f"Downloading {description} to {path} ...")
    try:
        with urllib.request.urlopen(url) as response, open(temporary, "wb") as output:
            total = int(response.headers.get("Content-Length", 0)) or None
            with tqdm.wrapattr(
                response, "read", total=total, desc=path.name, unit="B", unit_scale=True
            ) as source:
                shutil.copyfileobj(source, output)
        if total is not None and temporary.stat().st_size != total:
            raise RuntimeError(
                f"Incomplete download: expected {total} bytes, got {temporary.stat().st_size}"
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _download_inception_model(path: Path) -> None:
    _download_file(INCEPTION_V3_URL, path, "the official ADM Inception graph")


def _update_shapes(pool3) -> None:
    """Apply the batch-shape fix used by the official ADM evaluator."""
    tf = _tensorflow()
    for operation in pool3.graph.get_operations():
        for output in operation.outputs:
            shape = output.get_shape()
            if shape._dims is None:  # pylint: disable=protected-access
                continue
            dims = [None if value == 1 and i == 0 else value for i, value in enumerate(shape)]
            output.__dict__["_shape_val"] = tf.TensorShape(dims)


class ADMEvaluator:
    """Feature extraction and IS using OpenAI's frozen TensorFlow graph."""

    def __init__(self, session, model_path: Path):
        self.tf = _tensorflow()
        self.session = session
        _download_inception_model(model_path)

        with session.graph.as_default():
            self.image_input = self.tf.placeholder(self.tf.float32, shape=[None, None, None, 3])
            self.softmax_input = self.tf.placeholder(self.tf.float32, shape=[None, 2048])
            self.pool_features, self.spatial_features = self._feature_graph(model_path)
            self.softmax = self._softmax_graph(model_path)

    def _read_graph(self, model_path: Path):
        graph_def = self.tf.GraphDef()
        graph_def.ParseFromString(model_path.read_bytes())
        return graph_def

    def _feature_graph(self, model_path: Path):
        prefix = f"features_{random.randrange(2**32)}"
        pool3, spatial = self.tf.import_graph_def(
            self._read_graph(model_path),
            input_map={"ExpandDims:0": self.image_input},
            return_elements=[FID_POOL_NAME, FID_SPATIAL_NAME],
            name=prefix,
        )
        _update_shapes(pool3)
        # This exact slice is what makes the official per-image spatial
        # feature 17 * 17 * 7 = 2023 dimensional.
        return pool3, spatial[..., :7]

    def _softmax_graph(self, model_path: Path):
        prefix = f"softmax_{random.randrange(2**32)}"
        (matmul,) = self.tf.import_graph_def(
            self._read_graph(model_path),
            return_elements=["softmax/logits/MatMul"],
            name=prefix,
        )
        return self.tf.nn.softmax(self.tf.matmul(self.softmax_input, matmul.inputs[1]))

    def warmup(self) -> None:
        list(self.activations([np.zeros((1, 64, 64, 3), dtype=np.uint8)], "warmup"))

    def activations(self, batches: Iterator[np.ndarray], description: str):
        pool, spatial = [], []
        for batch in tqdm(batches, desc=description):
            pool_batch, spatial_batch = self.session.run(
                [self.pool_features, self.spatial_features],
                {self.image_input: batch.astype(np.float32)},
            )
            pool.append(pool_batch.reshape(pool_batch.shape[0], -1))
            spatial.append(spatial_batch.reshape(spatial_batch.shape[0], -1))
        pool_array = np.concatenate(pool, axis=0)
        spatial_array = np.concatenate(spatial, axis=0)
        if pool_array.shape[1] != 2048 or spatial_array.shape[1] != 2023:
            raise RuntimeError(
                "Unexpected official Inception feature shapes: "
                f"pool={pool_array.shape}, spatial={spatial_array.shape}"
            )
        return pool_array, spatial_array

    def inception_score(
        self,
        pool_features: np.ndarray,
        split_size: int = 5000,
        softmax_batch: int = 512,
    ) -> float:
        probabilities = []
        for start in range(0, len(pool_features), softmax_batch):
            probabilities.append(
                self.session.run(
                    self.softmax,
                    {self.softmax_input: pool_features[start : start + softmax_batch]},
                )
            )
        probabilities = np.concatenate(probabilities)
        scores = []
        for start in range(0, len(probabilities), split_size):
            part = probabilities[start : start + split_size]
            marginal = np.mean(part, axis=0, keepdims=True)
            kl = part * (np.log(part) - np.log(marginal))
            scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
        return float(np.mean(scores))


@dataclass(frozen=True)
class FIDStatistics:
    mu: np.ndarray
    sigma: np.ndarray

    @classmethod
    def from_features(cls, features: np.ndarray) -> "FIDStatistics":
        return cls(np.mean(features, axis=0), np.cov(features, rowvar=False))

    def distance(self, other: "FIDStatistics", eps: float = 1e-6) -> float:
        if self.mu.shape != other.mu.shape or self.sigma.shape != other.sigma.shape:
            raise ValueError("FID statistics have incompatible shapes")
        difference = self.mu - other.mu
        covariance_mean, _ = linalg.sqrtm(self.sigma.dot(other.sigma), disp=False)
        if not np.isfinite(covariance_mean).all():
            warnings.warn("Singular covariance product; adding 1e-6 to the diagonal")
            offset = np.eye(self.sigma.shape[0]) * eps
            covariance_mean = linalg.sqrtm((self.sigma + offset).dot(other.sigma + offset))
        if np.iscomplexobj(covariance_mean):
            if not np.allclose(np.diag(covariance_mean).imag, 0, atol=1e-3):
                raise ValueError(
                    f"FID covariance square root has imaginary component "
                    f"{np.max(np.abs(covariance_mean.imag))}"
                )
            covariance_mean = covariance_mean.real
        value = (
            difference.dot(difference)
            + np.trace(self.sigma)
            + np.trace(other.sigma)
            - 2 * np.trace(covariance_mean)
        )
        return float(value)


def _distance_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, matching the official evaluator."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    result = (
        np.sum(a * a, axis=1, keepdims=True)
        - 2.0 * a.dot(b.T)
        + np.sum(b * b, axis=1, keepdims=True).T
    )
    return np.maximum(result, 0.0)


def manifold_radii(features: np.ndarray, k: int, block_size: int) -> np.ndarray:
    """Exact k-NN radii with bounded memory; self-distance occupies index 0."""
    if len(features) <= k:
        raise ValueError(f"Need more than k={k} samples, got {len(features)}")
    radii = np.empty(len(features), dtype=np.float32)
    keep = k + 1
    for row_start in tqdm(range(0, len(features), block_size), desc=f"k-NN radii (k={k})"):
        rows = features[row_start : row_start + block_size]
        nearest = np.full((len(rows), keep), np.inf, dtype=np.float32)
        for col_start in range(0, len(features), block_size):
            distances = _distance_block(rows, features[col_start : col_start + block_size])
            candidates = np.concatenate((nearest, distances), axis=1)
            nearest = np.partition(candidates, keep - 1, axis=1)[:, :keep]
        radii[row_start : row_start + len(rows)] = np.max(nearest, axis=1)
    return radii


def manifold_coverage(
    manifold: np.ndarray,
    radii: np.ndarray,
    queries: np.ndarray,
    block_size: int,
    description: str,
) -> float:
    covered = np.zeros(len(queries), dtype=bool)
    for q_start in tqdm(range(0, len(queries), block_size), desc=description):
        query = queries[q_start : q_start + block_size]
        query_covered = np.zeros(len(query), dtype=bool)
        for m_start in range(0, len(manifold), block_size):
            distances = _distance_block(query, manifold[m_start : m_start + block_size])
            query_covered |= np.any(
                distances <= radii[m_start : m_start + len(distances[0])][None, :],
                axis=1,
            )
            if query_covered.all():
                break
        covered[q_start : q_start + len(query)] = query_covered
    return float(np.mean(covered))


def improved_precision_recall(
    reference: np.ndarray, generated: np.ndarray, k: int, block_size: int
) -> tuple[float, float]:
    reference_radii = manifold_radii(reference, k, block_size)
    generated_radii = manifold_radii(generated, k, block_size)
    precision = manifold_coverage(reference, reference_radii, generated, block_size, "Precision")
    recall = manifold_coverage(generated, generated_radii, reference, block_size, "Recall")
    return precision, recall


def collect_gen_dirs(base_dir: Path) -> list[Path]:
    directories = []
    for path in sorted(base_dir.iterdir()):
        if path.is_dir() and any(p.suffix.lower() in IMAGE_SUFFIXES for p in path.rglob("*")):
            directories.append(path)
    return directories


def save_csv(results: dict[str, dict[str, float]], path: str) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dir", "FID", "sFID", "Precision", "Recall", "IS"])
        for label, metrics in results.items():
            writer.writerow(
                [label]
                + [metrics.get(key, "") for key in ("FID", "sFID", "Precision", "Recall", "IS")]
            )


def save_txt(results: dict[str, dict[str, float]], gen_dir: str, path: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"\n{'=' * 60}\n")
        handle.write(f"Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        handle.write("Protocol: OpenAI ADM/guided-diffusion official Inception\n")
        handle.write(f"Generated images directory: {gen_dir}\n")
        handle.write(f"{'=' * 60}\n")
        for label, metrics in results.items():
            handle.write(f"\n  [{label}]\n")
            for key, value in metrics.items():
                handle.write(f"    {key:<12s} = {value:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official ADM-style ImageNet evaluation (FID, sFID, IS, P/R)"
    )
    parser.add_argument("--gen-dir", required=True, help="Generated RGB images")
    parser.add_argument(
        "--ref-batch",
        default=DEFAULT_REFERENCE_BATCH,
        help="Official ADM ImageNet-256 reference .npz (downloaded automatically)",
    )
    parser.add_argument("--batch", action="store_true", help="Evaluate each gen-dir subdirectory")
    parser.add_argument("--device", default="cuda:0", help="TensorFlow device, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Inception batch size")
    parser.add_argument(
        "--pr-block-size", type=int, default=1024, help="Exact P/R distance block size"
    )
    parser.add_argument("--k", type=int, default=3, help="P/R neighbourhood size (official: 3)")
    parser.add_argument("--inception-model", default="classify_image_graph_def.pb")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-sfid", action="store_true")
    parser.add_argument("--skip-pr", action="store_true")
    parser.add_argument("--skip-is", action="store_true")
    parser.add_argument("--save", help="Write CSV results")
    parser.add_argument("--save-txt", default="imagenet_result.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.pr_block_size <= 0 or args.k <= 0:
        raise ValueError("batch-size, pr-block-size and k must be positive")

    # Configure CUDA before doing any work because adding pip-installed CUDA
    # libraries requires one transparent process restart.
    if args.device.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
        _restart_with_tensorflow_cuda_libraries()

    gen_dirs = collect_gen_dirs(Path(args.gen_dir)) if args.batch else [Path(args.gen_dir)]
    if not gen_dirs:
        raise ValueError(f"No generated image directories found under {args.gen_dir}")

    reference_batch_path = Path(args.ref_batch)
    if reference_batch_path.suffix.lower() != ".npz":
        raise ValueError("--ref-batch must point to the official .npz file, not an image directory")
    _download_file(
        IMAGENET_256_REFERENCE_URL,
        reference_batch_path,
        "the official ImageNet-256 ADM reference batch",
    )
    print(f"Official reference batch: {reference_batch_path}")
    reference_archive = np.load(reference_batch_path)
    required_statistics = {"mu", "sigma", "mu_s", "sigma_s"}
    missing = required_statistics.difference(reference_archive.files)
    if missing:
        reference_archive.close()
        raise ValueError(
            f"Reference batch {reference_batch_path} is missing official statistics: "
            f"{sorted(missing)}"
        )
    reference_fid = FIDStatistics(reference_archive["mu"], reference_archive["sigma"])
    reference_sfid = FIDStatistics(reference_archive["mu_s"], reference_archive["sigma_s"])

    tf = _tensorflow()
    if args.device.startswith("cuda:") and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError(
            "TensorFlow still cannot see a GPU. Install its CUDA dependencies with "
            "`uv pip install --python .venv/bin/python 'tensorflow[and-cuda]==2.21.0'`."
        )
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    if args.device.lower() == "cpu":
        config.device_count["GPU"] = 0

    with tf.Session(config=config) as session:
        evaluator = ADMEvaluator(session, Path(args.inception_model))
        print("Warming up official Inception graph ...")
        evaluator.warmup()
        reference_pool = None
        if not args.skip_pr:
            if "arr_0" not in reference_archive.files:
                raise ValueError(
                    f"Reference batch {reference_batch_path} has no arr_0 images needed for P/R"
                )
            reference_images = reference_archive["arr_0"]
            print(f"Reference images for P/R: {len(reference_images):,}")
            reference_pool, _ = evaluator.activations(
                array_batches(reference_images, args.batch_size),
                "Reference P/R activations",
            )

        all_results: dict[str, dict[str, float]] = {}
        for gen_dir in gen_dirs:
            generated_files = image_files(gen_dir)
            print(f"\nEvaluating {gen_dir} ({len(generated_files):,} images)")
            if len(generated_files) != 50_000:
                warnings.warn(
                    f"Paper-comparable ImageNet metrics use 50,000 generated images; "
                    f"found {len(generated_files):,}."
                )
            generated_pool, generated_spatial = evaluator.activations(
                image_batches(generated_files, args.batch_size),
                f"Generated activations: {gen_dir.name}",
            )

            metrics: dict[str, float] = {}
            if not args.skip_fid:
                metrics["FID"] = FIDStatistics.from_features(generated_pool).distance(reference_fid)
            if not args.skip_sfid:
                metrics["sFID"] = FIDStatistics.from_features(generated_spatial).distance(
                    reference_sfid
                )
            if not args.skip_is:
                metrics["IS"] = evaluator.inception_score(generated_pool)
            if not args.skip_pr:
                metrics["Precision"], metrics["Recall"] = improved_precision_recall(
                    reference_pool, generated_pool, args.k, args.pr_block_size
                )
            all_results[gen_dir.name] = metrics
            print(json.dumps(metrics, indent=2))

    reference_archive.close()

    if args.save:
        save_csv(all_results, args.save)
        print(f"Saved CSV to {args.save}")
    if args.save_txt:
        save_txt(all_results, args.gen_dir, args.save_txt)
        print(f"Appended results to {args.save_txt}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
