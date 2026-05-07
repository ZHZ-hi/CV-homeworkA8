from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from modules.data import CACHE_DIR, WEIGHTS_DIR, load_mnist_subset
from modules.models import Autoencoder, VAE, vae_loss


def latent_targets(labels, radius: float, device):
    angles = labels.float() / 10.0 * (2 * torch.pi)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).to(device) * radius


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    images, labels = load_mnist_subset(train=True, limit=args.limit)
    dataset = TensorDataset(images, labels)
    train_len = int(len(dataset) * 0.85)
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    ae = Autoencoder().to(device)
    vae = VAE().to(device)
    ae_opt = optim.Adam(ae.parameters(), lr=args.lr)
    vae_opt = optim.Adam(vae.parameters(), lr=args.lr)
    history = []

    for epoch in range(1, args.epochs + 1):
        ae.train()
        vae.train()
        ae_train = vae_train = kl_train = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            seen += x.size(0)
            ae_opt.zero_grad()
            recon = ae(x)
            ae_loss = torch.nn.functional.mse_loss(recon, x)
            ae_loss.backward()
            ae_opt.step()
            ae_train += ae_loss.item() * x.size(0)

            vae_opt.zero_grad()
            recon, mu, logvar = vae(x)
            loss, _, kl = vae_loss(recon, x, mu, logvar, beta=args.beta)
            if args.latent_guide > 0:
                guide = torch.nn.functional.mse_loss(mu, latent_targets(y, args.latent_radius, device))
                loss = loss + args.latent_guide * guide
            loss.backward()
            vae_opt.step()
            vae_train += loss.item() * x.size(0)
            kl_train += kl.item() * x.size(0)

        ae_val, vae_val = evaluate(ae, vae, val_loader, device, args.beta, args.latent_guide, args.latent_radius)
        history.append([epoch, ae_train / seen, ae_val, kl_train / seen, vae_train / seen, vae_val])
        print(f"epoch {epoch}: ae={history[-1][1]:.4f}/{ae_val:.4f} vae={history[-1][4]:.4f}/{vae_val:.4f}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ae.state_dict(), WEIGHTS_DIR / "autoencoder.pt")
    torch.save(vae.state_dict(), WEIGHTS_DIR / "vae.pt")
    write_cache(ae, vae, images[: args.cache_samples], labels[: args.cache_samples], np.array(history, dtype=np.float32), device)


@torch.no_grad()
def evaluate(ae, vae, loader, device, beta, latent_guide, latent_radius):
    ae.eval()
    vae.eval()
    ae_loss = vae_loss_value = 0.0
    seen = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        seen += x.size(0)
        ae_loss += torch.nn.functional.mse_loss(ae(x), x, reduction="sum").item() / x[0].numel()
        recon, mu, logvar = vae(x)
        loss, _, _ = vae_loss(recon, x, mu, logvar, beta=beta)
        if latent_guide > 0:
            guide = torch.nn.functional.mse_loss(mu, latent_targets(y, latent_radius, device))
            loss = loss + latent_guide * guide
        vae_loss_value += loss.item() * x.size(0)
    return ae_loss / seen, vae_loss_value / seen


@torch.no_grad()
def write_cache(ae, vae, images, labels, history, device):
    ae.eval()
    vae.eval()
    x = images.to(device)
    ae_recon = ae(x).cpu().numpy().squeeze(1)
    vae_recon, mu, _ = vae(x)
    np.savez_compressed(
        CACHE_DIR / "demo_cache.npz",
        images=images.numpy().squeeze(1).astype(np.float32),
        labels=labels.numpy().astype(np.int64),
        latent=mu.cpu().numpy().astype(np.float32),
        ae_recon=ae_recon.astype(np.float32),
        vae_recon=vae_recon.cpu().numpy().squeeze(1).astype(np.float32),
        loss=history,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--cache-samples", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.05, help="KL weight for the 2D VAE latent space.")
    parser.add_argument("--latent-guide", type=float, default=0.0, help="Optional label-guided center loss for clearer demos.")
    parser.add_argument("--latent-radius", type=float, default=5.0)
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
