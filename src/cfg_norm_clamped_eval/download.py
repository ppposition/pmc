# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained SiT models
"""

import os
import subprocess
import urllib.request

import torch
from torchvision.datasets.utils import download_url

pretrained_models = {"SiT-XL-2-256x256.pt"}

# Dropbox download URL for the pre-trained SiT-XL/2 256x256 model
MODEL_URL = (
    "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/"
    "SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=1"
)


def find_model(model_name):
    """
    Finds a pre-trained SiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:
        return download_model(model_name)
    else:
        assert os.path.isfile(model_name), f"Could not find SiT checkpoint at {model_name}"
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        return checkpoint


def _download_with_fallback(url, dest_dir, filename):
    """Try downloading with multiple methods, falling back if one fails."""
    dest_path = os.path.join(dest_dir, filename)

    # Method 1: torchvision's download_url
    try:
        download_url(url, dest_dir, filename=filename)
        if os.path.isfile(dest_path):
            return
    except Exception:
        pass

    # Method 2: urllib with direct download (dl=1)
    try:
        print("Trying direct download with urllib...")
        urllib.request.urlretrieve(url, dest_path)
        if os.path.isfile(dest_path):
            return
    except Exception:
        pass

    # Method 3: wget
    try:
        print("Trying wget...")
        subprocess.run(["wget", "-O", dest_path, url], check=True, capture_output=True)
        if os.path.isfile(dest_path):
            return
    except Exception:
        pass

    # Method 4: curl
    try:
        print("Trying curl...")
        subprocess.run(["curl", "-L", "-o", dest_path, url], check=True, capture_output=True)
        if os.path.isfile(dest_path):
            return
    except Exception:
        pass

    raise RuntimeError(
        f"Failed to download the pre-trained model from {url}. "
        "Please try manually:\n"
        f"  wget -O {dest_path} '{url}'\n"
        "Or place a pre-downloaded checkpoint in the 'pretrained_models/' directory."
    )


def download_model(model_name):
    """
    Downloads a pre-trained SiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f"pretrained_models/{model_name}"

    # Also check for alternative filenames (e.g. user may have uploaded
    # SiT-XL-2-256.pt which is the original Dropbox filename)
    alt_path = f"pretrained_models/{model_name.replace('256x256', '256')}"

    if os.path.isfile(alt_path) and not os.path.isfile(local_path):
        print(f"Found checkpoint at {alt_path}, using it.")
        model = torch.load(alt_path, map_location=lambda storage, loc: storage)
        return model

    if not os.path.isfile(local_path):
        os.makedirs("pretrained_models", exist_ok=True)
        _download_with_fallback(MODEL_URL, "pretrained_models", model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model
