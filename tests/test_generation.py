"""Exercise generation orchestration and visualization without model downloads."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from cfg_norm_clamped_eval import generate_imagenet, generate_sd35, sit_sampling
from cfg_norm_clamped_eval.gallery import prepare_output, write_gallery


def test_gallery_pages_and_escaped_prompts(tmp_path):
    root = prepare_output(tmp_path / "run")
    records = []
    for index in range(5):
        file = f"images/{index:06d}.png"
        Image.new("RGB", (80, 40), (index * 40, 0, 0)).save(root / file)
        records.append(dict(index=index, file=file, seed=index, prompt='<script> & "cat"'))
    original = (root / records[0]["file"]).read_bytes()
    write_gallery(root, records, columns=2, page_size=4, thumbnail_size=64)
    assert len(list((root / "images").glob("*.png"))) == 5
    assert len(list((root / "preview").glob("*.png"))) == 2
    with Image.open(root / "preview/grid_0000.png") as grid:
        assert grid.size == (128, 188)
    with Image.open(root / "preview/grid_0001.png") as grid:
        assert grid.size == (64, 94)
    document = (root / "index.html").read_text()
    assert "<script>" not in document
    assert "&lt;script&gt;" in document
    assert document.count('<img loading="lazy"') == 5
    assert (root / records[0]["file"]).read_bytes() == original
    with pytest.raises(ValueError, match="empty directory"):
        prepare_output(root)
    assert (root / records[0]["file"]).read_bytes() == original


def test_sd35_prompt_cycling_and_validation(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a cat\na dog\n", encoding="utf-8")
    args = generate_sd35.build_parser().parse_args(
        ["--prompt-file", str(prompts), "--num-samples", "5", "--seed", "42"]
    )
    records = generate_sd35.prepare_records(args)
    assert [r["prompt"] for r in records] == ["a cat", "a dog", "a cat", "a dog", "a cat"]
    assert [r["seed"] for r in records] == list(range(42, 47))
    prompts.write_text("a cat\n\na dog\n")
    with pytest.raises(ValueError, match="blank lines"):
        generate_sd35.prepare_records(args)


@pytest.mark.parametrize("module", [generate_imagenet, generate_sd35])
def test_generation_entry_points_write_complete_browsable_runs(module, tmp_path, monkeypatch):
    root = tmp_path / module.__name__.split(".")[-1]
    argv = ["generate", "--num-samples", "5", "--batch-size", "2", "--out-dir", str(root)]
    argv += ["--grid-page-size", "4", "--thumbnail-size", "32"]
    if module is generate_sd35:
        argv += ["--prompt", "a cat"]
    else:
        argv += ["--class-id", "7"]

    def fake_generate(args, records=None, *unused):
        for index in range(args.num_samples):
            path = Path(args.out_dir) / (records[index]["file"] if records else f"{index:06d}.png")
            Image.new("RGB", (32, 32)).save(path)

    monkeypatch.setattr(module, "generate_images", fake_generate)
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    records = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    assert len(records) == len(list((root / "images").glob("*.png"))) == 5
    assert len(list((root / "preview").glob("*.png"))) == 2
    assert (root / "index.html").is_file()
    if module is generate_imagenet:
        assert {r["class_id"] for r in records} == {7}
    else:
        assert (root / "prompts.txt").read_text().splitlines() == ["a cat"] * 5
    with pytest.raises(SystemExit):
        module.main()


def test_sit_partial_batches_and_noise_independent_of_batch_size(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def forward_with_cfg(self, *args, **kwargs):
            raise AssertionError("The test sampler should not invoke a real model")

    class VAE(torch.nn.Module):
        def decode(self, samples):
            return SimpleNamespace(sample=samples[:, :3])

    seen = []

    def sample(z, model, y, **kwargs):
        seen.append((z[: len(z) // 2].clone(), y.clone()))
        return [z]

    monkeypatch.setattr(sit_sampling, "SiT_models", {"SiT-XL/2": lambda **kw: Model()})
    monkeypatch.setattr(sit_sampling, "find_model", lambda name: {})
    monkeypatch.setattr(sit_sampling.AutoencoderKL, "from_pretrained", lambda *a, **kw: VAE())
    monkeypatch.setattr(sit_sampling, "create_transport", lambda **kw: None)
    monkeypatch.setattr(
        sit_sampling, "Sampler", lambda transport: SimpleNamespace(sample_ode=lambda **kw: sample)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    noises = []
    for batch_size in [2, 3]:
        seen.clear()
        args = generate_imagenet.build_parser().parse_args(
            [
                "--num-samples",
                "5",
                "--batch-size",
                str(batch_size),
                "--class-id",
                "7",
                "--out-dir",
                str(tmp_path / str(batch_size)),
            ]
        )
        args.per_sample_seed = True
        sit_sampling.generate_images(args)
        assert len(list(Path(args.out_dir).glob("*.png"))) == 5
        assert [len(z) for z, _ in seen] == ([2, 2, 1] if batch_size == 2 else [3, 2])
        for z, labels in seen:
            assert labels.tolist() == [7] * len(z) + [1000] * len(z)
        noises.append(torch.cat([z for z, _ in seen]))
    torch.testing.assert_close(*noises, atol=0, rtol=0)


def test_fid_only_does_not_generate_or_load_models(tmp_path, monkeypatch):
    from cfg_norm_clamped_eval import eval_fid

    def unexpected(*args, **kwargs):
        raise AssertionError("FID-only must not load generation models")

    calls = []
    monkeypatch.setattr(eval_fid, "generate_images", unexpected)
    monkeypatch.setitem(
        sys.modules,
        "cleanfid",
        SimpleNamespace(
            fid=SimpleNamespace(compute_fid=lambda **kwargs: calls.append(kwargs) or 0)
        ),
    )
    args = SimpleNamespace(
        compute_fid_only=True,
        gpu=0,
        out_dir=str(tmp_path),
        ref_dir="reference",
        num_workers=0,
        inception_batch_size=2,
        skip_saturation=True,
        skip_diversity=True,
    )
    eval_fid.main(args)
    assert len(calls) == 1


def test_sd35_partial_batches_and_oom_fallback_preserve_seeds(tmp_path, monkeypatch):
    import diffusers

    from cfg_norm_clamped_eval import generate_coco_sd35

    calls = []

    class Generator:
        def manual_seed(self, seed):
            self.seed = seed
            return self

    class Pipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            instance = cls()
            instance.scheduler = SimpleNamespace(config={})
            return instance

        def to(self, device):
            return self

        def set_progress_bar_config(self, **kwargs):
            pass

        def __call__(self, prompt, generator, **kwargs):
            generators = generator if isinstance(generator, list) else [generator]
            calls.append([g.seed for g in generators])
            if len(calls) == 1:
                # Simulate a failed batch consuming its generators before retry.
                for g in generators:
                    g.seed += 100
                raise torch.cuda.OutOfMemoryError()
            return SimpleNamespace(
                images=[Image.new("RGB", (32, 32), (g.seed, 0, 0)) for g in generators]
            )

    monkeypatch.setattr(diffusers, "StableDiffusion3Pipeline", Pipeline)
    for name in ["FlowMatchEulerDiscreteScheduler", "FlowMatchHeunDiscreteScheduler"]:
        monkeypatch.setattr(diffusers, name, SimpleNamespace(from_config=lambda cfg: None))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda gpu: None)
    monkeypatch.setattr(torch, "Generator", lambda device: Generator())
    args = generate_sd35.build_parser().parse_args(
        [
            "--prompt",
            "a cat",
            "--num-samples",
            "5",
            "--batch-size",
            "2",
            "--seed",
            "10",
            "--out-dir",
            str(tmp_path),
        ]
    )
    args.selection_seed = None
    records = generate_sd35.prepare_records(args)
    generate_coco_sd35.generate_images(args, records)
    assert calls == [[10, 11], [10], [11], [12, 13], [14]]
    assert len(list((tmp_path / "images").glob("*.png"))) == 5
    for record in records:
        with Image.open(tmp_path / record["file"]) as image:
            assert image.getpixel((0, 0))[0] == record["seed"]
    # Existing completed COCO-engine outputs must remain resumable.
    generate_coco_sd35.generate_images(args, records)
    assert len(calls) == 5
