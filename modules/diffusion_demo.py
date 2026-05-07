from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


MODEL_ID = "hf-internal-testing/tiny-stable-diffusion-pipe"
ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = ROOT / "assets" / "hf_runtime_cache"
LOCAL_MODEL_DIR = ROOT / "assets" / "diffusers_tiny_model"


@dataclass(frozen=True)
class DiffusionParams:
    prompt: str
    negative_prompt: str
    steps: int
    seed: int
    guidance_scale: float


def _load_font(size: int, bold: bool = False):
    names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "arial.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        from matplotlib import font_manager

        family = "DejaVu Sans:bold" if bold else "DejaVu Sans"
        path = font_manager.findfont(family, fallback_to_default=True)
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _fallback_image(params: DiffusionParams, size: int = 256) -> Image.Image:
    key = f"{params.prompt}|{params.negative_prompt}|{params.steps}|{params.seed}|{params.guidance_scale}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    digit = next((ch for ch in params.prompt if ch.isdigit()), str(params.seed % 10))
    bg = np.ones((size, size, 3), dtype=np.float32)
    tint = rng.uniform(0.88, 1.0, 3)
    bg *= tint
    paper_noise = rng.normal(0, 0.025 + params.steps * 0.001, bg.shape)
    bg = np.clip(bg + paper_noise, 0, 1)
    img = Image.fromarray(np.uint8(bg * 255), mode="RGB").filter(ImageFilter.GaussianBlur(0.35))
    draw = ImageDraw.Draw(img)
    digit_font = _load_font(int(150 + min(params.guidance_scale, 12) * 3), bold=True)
    label_font = _load_font(14)

    ink = tuple(int(v) for v in rng.integers(20, 90, 3))
    if "watercolor" in params.prompt.lower():
        ink = tuple(int(v) for v in rng.integers(40, 150, 3))
    bbox = draw.textbbox((0, 0), digit, font=digit_font)
    x = (size - (bbox[2] - bbox[0])) // 2 + int(rng.normal(0, 8))
    y = (size - (bbox[3] - bbox[1])) // 2 - 18 + int(rng.normal(0, 8))
    layers = 3 + int(params.guidance_scale // 3)
    for _ in range(layers):
        dx, dy = int(rng.normal(0, 3)), int(rng.normal(0, 3))
        alpha_img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        alpha_draw = ImageDraw.Draw(alpha_img)
        color = ink + (int(rng.integers(70, 135)),)
        alpha_draw.text((x + dx, y + dy), digit, fill=color, font=digit_font)
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.4, 1.6))))
        img = Image.alpha_composite(img.convert("RGBA"), alpha_img).convert("RGB")
    if "blurry" not in params.negative_prompt.lower():
        img = img.filter(ImageFilter.GaussianBlur(0.35))
    label = f"seed {params.seed} | steps {params.steps} | cfg {params.guidance_scale:.1f}"
    draw.rectangle((8, size - 30, size - 8, size - 8), fill=(255, 255, 255))
    draw.text((12, size - 26), label, fill=(20, 24, 35), font=label_font)
    return img


@lru_cache(maxsize=1)
def _load_pipeline():
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download

    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hub_cache = HF_CACHE_DIR / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    if not (LOCAL_MODEL_DIR / "model_index.json").exists():
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(LOCAL_MODEL_DIR),
            local_dir_use_symlinks=False,
            cache_dir=str(hub_cache),
        )
    return DiffusionPipeline.from_pretrained(str(LOCAL_MODEL_DIR), local_files_only=True).to("cpu")


def generate_image(params: DiffusionParams, use_real_model: bool = True) -> tuple[Image.Image, str]:
    if not use_real_model:
        return _fallback_image(params), "fallback"

    try:
        import torch

        pipe = _load_pipeline()
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        generator = torch.Generator(device="cpu").manual_seed(params.seed)
        result = pipe(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt or None,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance_scale,
            generator=generator,
            height=32,
            width=32,
            output_type="pil",
        )
        return result.images[0].resize((256, 256), Image.Resampling.NEAREST), MODEL_ID
    except Exception as exc:
        return _fallback_image(params), f"fallback: {type(exc).__name__}: {str(exc)[:180]}"
