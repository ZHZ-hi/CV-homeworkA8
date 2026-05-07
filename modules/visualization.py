from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


def array_to_image(arr: np.ndarray, scale: int = 6) -> Image.Image:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr.squeeze()
    img = Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255), mode="L")
    return img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)


def error_heatmap(input_img: np.ndarray, recon_img: np.ndarray) -> go.Figure:
    err = np.abs(np.asarray(input_img).squeeze() - np.asarray(recon_img).squeeze())
    fig = px.imshow(err, color_continuous_scale="magma", zmin=0, zmax=max(0.5, float(err.max())))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=260,
        coloraxis_colorbar=dict(title="误差"),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def loss_curve(loss: np.ndarray) -> go.Figure:
    df = pd.DataFrame(
        loss,
        columns=["epoch", "AE 训练", "AE 验证", "VAE KL", "VAE 训练", "VAE 验证"],
    )
    fig = go.Figure()
    for col in ["AE 训练", "AE 验证", "VAE 训练", "VAE 验证", "VAE KL"]:
        fig.add_trace(go.Scatter(x=df["epoch"], y=df[col], mode="lines+markers", name=col))
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=28, b=10),
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend_orientation="h",
    )
    return fig


def latent_scatter(latent: np.ndarray, labels: np.ndarray, color_mode: str) -> go.Figure:
    palette = [
        "#2563eb",
        "#ef4444",
        "#16a34a",
        "#f97316",
        "#7c3aed",
        "#0891b2",
        "#ca8a04",
        "#db2777",
        "#64748b",
        "#111827",
    ]
    fig = go.Figure()
    if color_mode == "按类别":
        for cls in range(10):
            mask = labels == cls
            fig.add_trace(
                go.Scattergl(
                    x=latent[mask, 0],
                    y=latent[mask, 1],
                    mode="markers",
                    name=f"类别 {cls}",
                    marker=dict(size=8, color=palette[cls], opacity=0.76),
                    customdata=np.column_stack([np.where(mask)[0], labels[mask]]),
                    hovertemplate="样本编号=%{customdata[0]}<br>类别=%{customdata[1]}<br>z1=%{x:.2f}<br>z2=%{y:.2f}<extra></extra>",
                )
            )
    else:
        fig.add_trace(
            go.Scattergl(
                x=latent[:, 0],
                y=latent[:, 1],
                mode="markers",
                name="样本编号",
                marker=dict(
                    size=8,
                    color=np.arange(len(labels)),
                    colorscale="Turbo",
                    opacity=0.76,
                    colorbar=dict(title="编号"),
                ),
                customdata=np.column_stack([np.arange(len(labels)), labels]),
                hovertemplate="样本编号=%{customdata[0]}<br>类别=%{customdata[1]}<br>z1=%{x:.2f}<br>z2=%{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=540,
        margin=dict(l=0, r=0, t=12, b=0),
        dragmode=False,
        clickmode="event+select",
        legend_title_text="类别" if color_mode == "按类别" else "样本编号",
        xaxis_title="z1",
        yaxis_title="z2",
    )
    return fig


def image_grid(images: np.ndarray, columns: int = 8, scale: int = 3) -> Image.Image:
    images = np.asarray(images)
    n = len(images)
    rows = int(np.ceil(n / columns))
    cell = 28
    canvas = Image.new("L", (columns * cell, rows * cell), 0)
    for i, img in enumerate(images):
        pil = Image.fromarray(np.uint8(np.clip(np.squeeze(img), 0, 1) * 255), mode="L")
        canvas.paste(pil, ((i % columns) * cell, (i // columns) * cell))
    return canvas.resize((canvas.width * scale, canvas.height * scale), Image.Resampling.NEAREST)


def noise_heatmap(noise: np.ndarray) -> go.Figure:
    shown = np.asarray(noise[: min(len(noise), 16)])
    fig = px.imshow(
        shown,
        color_continuous_scale="RdBu",
        zmin=-3,
        zmax=3,
        aspect="auto",
        labels=dict(x="噪声维度", y="样本", color="z"),
    )
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0))
    return fig


def interpolation_strip(images: list[np.ndarray], scale: int = 4) -> Image.Image:
    return image_grid(np.stack(images), columns=len(images), scale=scale)
