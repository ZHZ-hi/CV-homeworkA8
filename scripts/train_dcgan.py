from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from modules.data import WEIGHTS_DIR, load_mnist_subset
from modules.models import Discriminator, Generator


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    images, _ = load_mnist_subset(train=True, limit=args.limit)
    images = images * 2 - 1
    loader = DataLoader(TensorDataset(images), batch_size=args.batch_size, shuffle=True, drop_last=True)
    generator = Generator(args.noise_dim).to(device)
    discriminator = Discriminator().to(device)
    g_opt = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        g_running = d_running = 0.0
        for (real,) in loader:
            real = real.to(device)
            batch = real.size(0)
            real_targets = torch.ones(batch, device=device)
            fake_targets = torch.zeros(batch, device=device)

            z = torch.randn(batch, args.noise_dim, device=device)
            fake = generator(z).detach()
            d_opt.zero_grad()
            d_loss = loss_fn(discriminator(real), real_targets) + loss_fn(discriminator(fake), fake_targets)
            d_loss.backward()
            d_opt.step()

            z = torch.randn(batch, args.noise_dim, device=device)
            g_opt.zero_grad()
            g_loss = loss_fn(discriminator(generator(z)), real_targets)
            g_loss.backward()
            g_opt.step()
            g_running += g_loss.item()
            d_running += d_loss.item()
        print(f"epoch {epoch}: g={g_running / len(loader):.4f} d={d_running / len(loader):.4f}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), WEIGHTS_DIR / "dcgan_generator.pt")
    torch.save(discriminator.state_dict(), WEIGHTS_DIR / "dcgan_discriminator.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
