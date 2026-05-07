from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
CACHE_DIR = ASSETS_DIR / "cache"
WEIGHTS_DIR = ASSETS_DIR / "weights"
DEMO_CACHE = CACHE_DIR / "demo_cache.npz"


def _font(size: int = 24):
    for name in ("arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_digit_like(label: int, seed: int, size: int = 28) -> np.ndarray:
    rng = np.random.default_rng(seed)
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    font = _font(int(rng.integers(20, 26)))
    text = str(label % 10)
    bbox = draw.textbbox((0, 0), text, font=font)
    x = int((size - (bbox[2] - bbox[0])) / 2 + rng.normal(0, 1.5))
    y = int((size - (bbox[3] - bbox[1])) / 2 - 2 + rng.normal(0, 1.5))
    draw.text((x, y), text, fill=int(rng.integers(180, 255)), font=font)
    canvas = canvas.rotate(float(rng.normal(0, 8)), resample=Image.Resampling.BILINEAR)
    canvas = canvas.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.15, 0.55))))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr += rng.normal(0, 0.025, arr.shape).astype(np.float32)
    return np.clip(arr, 0.0, 1.0)


def generate_demo_cache(n: int = 500, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.arange(n, dtype=np.int64) % 10
    rng.shuffle(labels)
    images = np.stack([make_digit_like(int(label), seed + i) for i, label in enumerate(labels)])

    centers = np.array(
        [[np.cos(2 * np.pi * k / 10), np.sin(2 * np.pi * k / 10)] for k in range(10)],
        dtype=np.float32,
    )
    latent = centers[labels] * 2.2 + rng.normal(0, 0.28, (n, 2)).astype(np.float32)

    ae_recon = np.stack([_smooth_image(img, 0.82) for img in images])
    vae_recon = np.stack([_smooth_image(img, 0.72) for img in images])
    epochs = np.arange(1, 9, dtype=np.int64)
    loss = np.column_stack(
        [
            epochs,
            0.23 * np.exp(-epochs / 4.0) + 0.035,
            0.31 * np.exp(-epochs / 4.5) + 0.055,
            0.08 * np.exp(-epochs / 5.0) + 0.015,
            0.66 * np.exp(-epochs / 4.8) + 0.11,
            0.74 * np.exp(-epochs / 5.2) + 0.13,
        ]
    ).astype(np.float32)
    return {
        "images": images.astype(np.float32),
        "labels": labels.astype(np.int64),
        "latent": latent.astype(np.float32),
        "ae_recon": ae_recon.astype(np.float32),
        "vae_recon": vae_recon.astype(np.float32),
        "loss": loss,
    }


def _smooth_image(img: np.ndarray, strength: float) -> np.ndarray:
    pil = Image.fromarray(np.uint8(np.clip(img, 0, 1) * 255), mode="L")
    pil = pil.filter(ImageFilter.GaussianBlur(0.45 + (1.0 - strength)))
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return np.clip(strength * arr + (1.0 - strength) * 0.08, 0.0, 1.0)


def ensure_demo_cache() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not DEMO_CACHE.exists():
        cache = generate_demo_cache()
        np.savez_compressed(DEMO_CACHE, **cache)
    return DEMO_CACHE


def load_demo_cache() -> dict[str, np.ndarray]:
    path = ensure_demo_cache()
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_mnist_subset(train: bool = False, limit: int = 2048):
    try:
        from torchvision import datasets, transforms
        import torch
    except Exception as exc:
        raise RuntimeError("torchvision is required to download MNIST.") from exc

    dataset = datasets.MNIST(
        root=str(ROOT / "data"),
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    xs = []
    ys = []
    for i in range(min(limit, len(dataset))):
        x, y = dataset[i]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def decode_latent_numpy(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    angle = np.arctan2(z[..., 1], z[..., 0])
    label = int(np.round(((angle % (2 * np.pi)) / (2 * np.pi)) * 10)) % 10
    radius = float(np.linalg.norm(z))
    img = make_digit_like(label, seed=int(abs(z[0] * 991 + z[1] * 313)) % 10000)
    return np.clip(img * (0.75 + min(radius, 3.0) * 0.08), 0.0, 1.0)
