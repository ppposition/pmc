"""
Standalone ImageReward scorer — wraps the `image-reward` package with
full transformers 5.x compatibility via monkey patches.

Usage:
    from .eval_image_reward import compute_image_reward
    result = compute_image_reward("/path/to/images", ["a prompt", ...])
"""

import logging
import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
logging.disable(logging.WARNING)

import torch
import transformers
import transformers.modeling_utils as mu

# ====================================================================
# CRITICAL ORDER: ALL patches to transformers.modeling_utils MUST
# happen BEFORE importing ImageReward, because ImageReward's med.py
# does `from transformers.modeling_utils import (find_pruneable_heads...)`
# at module load time.
# ====================================================================

# 1. apply_chunking_to_forward was moved out of modeling_utils
mu.apply_chunking_to_forward = transformers.apply_chunking_to_forward

# 2. prune_linear_layer was moved to pytorch_utils
mu.prune_linear_layer = transformers.pytorch_utils.prune_linear_layer


# 3. find_pruneable_heads_and_indices was removed entirely
def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    heads = set(heads) - already_pruned_heads
    result = {}
    for head in sorted(heads):
        result[head] = torch.arange(head_size) + head * head_size
    return result


mu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

# 4. PreTrainedModel.all_tied_weights_keys is now _tied_weights_keys
_orig_tie = transformers.PreTrainedModel.tie_weights


def _patched_tie(self, *args, **kwargs):
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = getattr(self, "_tied_weights_keys", None) or {}
    return _orig_tie(self, *args, **kwargs)


transformers.PreTrainedModel.tie_weights = _patched_tie


# 5. get_head_mask was removed from PreTrainedModel (needed by med.py)
def _get_head_mask(self, head_mask, num_hidden_layers):
    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


# 6. invert_attention_mask was removed (needed by med.py)
def _invert_attention_mask(self, encoder_attention_mask):
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    elif encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    else:
        encoder_extended_attention_mask = encoder_attention_mask
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
    return encoder_extended_attention_mask


# Apply to PreTrainedModel so med.py's OWN BertPreTrainedModel inherits them
# (med.py defines its own BertPreTrainedModel ← PreTrainedModel, not using
#  transformers.BertPreTrainedModel — so we patch the common ancestor)
transformers.PreTrainedModel.get_head_mask = _get_head_mask
transformers.PreTrainedModel.invert_attention_mask = _invert_attention_mask

# 7. init_tokenizer — transformers 5.x removed .additional_special_tokens_ids
from transformers import BertTokenizer


def _patched_init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
    return tokenizer


# Override in the image-reward BLIP modules BEFORE importing ImageReward
import ImageReward.models.BLIP.blip as _ir_blip
import ImageReward.models.BLIP.blip_pretrain as _ir_bp

_ir_blip.init_tokenizer = _patched_init_tokenizer
_ir_bp.init_tokenizer = _patched_init_tokenizer

# ImageReward is now loaded and its compatibility patches are in place.

_IMAGE_REWARD_CACHE = {}


def load_image_reward(device: str = "cuda"):
    """Load ImageReward model (cached per device)."""
    cache_key = device
    if cache_key not in _IMAGE_REWARD_CACHE:
        from ImageReward import utils as ir_utils

        model = ir_utils.load("ImageReward-v1.0")
        model.to(device)
        model.eval()
        _IMAGE_REWARD_CACHE[cache_key] = model
    return _IMAGE_REWARD_CACHE[cache_key]


@torch.no_grad()
def compute_image_reward(
    image_dir: str, prompts: list[str], device: str = "cuda", batch_size: int = 32
) -> dict:
    """
    ImageReward: human-preference score for text-to-image generation.

    Scores each (image, prompt) pair. Higher = better alignment with
    human aesthetic/preference judgments. Typical range: -1 to 3.

    Args:
        image_dir: folder of generated images (sorted by filename)
        prompts: one prompt per image (or single prompt for all)
        batch_size: inference batch size

    Returns {"ImageReward_mean": float, "ImageReward_std": float}
    """
    from pathlib import Path

    from PIL import Image
    from tqdm import tqdm

    model = load_image_reward(device)

    img_files = sorted(Path(image_dir).glob("*"))
    img_files = [
        f for f in img_files if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ]
    assert len(img_files) > 0, f"No images found in {image_dir}"

    if len(prompts) == 1:
        prompts = prompts * len(img_files)
    assert len(prompts) == len(img_files), (
        f"Prompt count ({len(prompts)}) != image count ({len(img_files)})"
    )

    all_scores = []
    for i in tqdm(range(0, len(img_files), batch_size), desc="ImageReward"):
        batch_files = img_files[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]

        for fp, prompt in zip(batch_files, batch_prompts):
            img = Image.open(fp).convert("RGB")
            score = model.score(prompt, img)
            all_scores.append(score)

    scores_np = torch.tensor(all_scores).cpu().numpy()
    return {
        "ImageReward_mean": float(scores_np.mean()),
        "ImageReward_std": float(scores_np.std()),
    }
