from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_plotly_events import plotly_events

from modules.data import WEIGHTS_DIR, decode_latent_numpy, load_demo_cache
from modules.diffusion_demo import DiffusionParams, generate_image
from modules.visualization import (
    array_to_image,
    error_heatmap,
    image_grid,
    interpolation_strip,
    latent_scatter,
    loss_curve,
    noise_heatmap,
)


ROOT = Path(__file__).resolve().parent


st.set_page_config(page_title="深度生成模型实验平台", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def cached_demo_data(stamp):
    return load_demo_cache()


def cache_stamp() -> float:
    from modules.data import DEMO_CACHE

    return DEMO_CACHE.stat().st_mtime if DEMO_CACHE.exists() else 0.0


def weight_stamp() -> tuple[tuple[str, float], ...]:
    names = [
        "autoencoder.pt",
        "vae.pt",
        "dcgan_generator.pt",
        "dcgan_discriminator.pt",
    ]
    return tuple((name, (WEIGHTS_DIR / name).stat().st_mtime if (WEIGHTS_DIR / name).exists() else 0.0) for name in names)


@st.cache_resource(show_spinner=False)
def load_torch_models(stamp):
    try:
        import torch

        from modules.models import Autoencoder, Discriminator, Generator, VAE
    except Exception as exc:
        return None, f"PyTorch 依赖未安装：{exc}"

    device = torch.device("cpu")
    models = {
        "ae": Autoencoder().to(device).eval(),
        "vae": VAE().to(device).eval(),
        "generator": Generator().to(device).eval(),
        "discriminator": Discriminator().to(device).eval(),
    }
    weight_files = {
        "ae": WEIGHTS_DIR / "autoencoder.pt",
        "vae": WEIGHTS_DIR / "vae.pt",
        "generator": WEIGHTS_DIR / "dcgan_generator.pt",
        "discriminator": WEIGHTS_DIR / "dcgan_discriminator.pt",
    }
    loaded = []
    try:
        for name, path in weight_files.items():
            if path.exists():
                models[name].load_state_dict(torch.load(path, map_location=device))
                loaded.append(name)
        return (models, device, loaded), None
    except Exception as exc:
        return None, f"模型权重加载失败：{exc}"


def infer_reconstruction(models_info, image: np.ndarray, idx: int, cache: dict[str, np.ndarray]):
    if not models_info:
        return cache["ae_recon"][idx], cache["vae_recon"][idx], cache["latent"][idx]
    models, device, loaded = models_info
    if "ae" not in loaded or "vae" not in loaded:
        return cache["ae_recon"][idx], cache["vae_recon"][idx], cache["latent"][idx]

    try:
        import torch

        x = torch.tensor(image, dtype=torch.float32, device=device).view(1, 1, 28, 28)
        with torch.no_grad():
            ae_recon = models["ae"](x).cpu().numpy()[0, 0]
            vae_recon, mu, _ = models["vae"](x)
        return ae_recon, vae_recon.cpu().numpy()[0, 0], mu.cpu().numpy()[0]
    except Exception:
        return cache["ae_recon"][idx], cache["vae_recon"][idx], cache["latent"][idx]


def decode_latent(models_info, z: np.ndarray) -> np.ndarray:
    if models_info:
        models, device, loaded = models_info
        if "vae" in loaded:
            try:
                import torch

                with torch.no_grad():
                    tensor = torch.tensor(z, dtype=torch.float32, device=device).view(1, 2)
                    return models["vae"].decode(tensor).cpu().numpy()[0, 0]
            except Exception:
                pass
    return decode_latent_numpy(z)


def dcgan_generate(models_info, seed: int, sample_count: int, noise_dim: int = 64):
    rng = np.random.default_rng(seed)
    z_np = rng.standard_normal((sample_count, noise_dim)).astype(np.float32)
    if models_info:
        models, device, loaded = models_info
        if "generator" in loaded and "discriminator" in loaded:
            try:
                import torch

                z = torch.tensor(z_np, device=device)
                with torch.no_grad():
                    generated = models["generator"](z)
                    scores = torch.sigmoid(models["discriminator"](generated)).cpu().numpy()
                images = ((generated.cpu().numpy()[:, 0] + 1) / 2).clip(0, 1)
                return images, scores, z_np
            except Exception:
                pass

    images = np.stack([decode_latent_numpy(rng.normal(0, 2.2, size=2)) for _ in range(sample_count)])
    scores = np.clip(0.35 + 0.3 * rng.random(sample_count), 0, 1)
    return images, scores, z_np


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
    h1, h2, h3 { letter-spacing: 0; }
    [data-testid="stMetricValue"] { font-size: 1.35rem; }
    .metric-card {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 14px 16px;
        background: #FFFFFF;
        min-height: 76px;
    }
    .metric-card span {
        display: block;
        color: #6B7280;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    .metric-card strong {
        color: #111827;
        font-size: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


cache = cached_demo_data(cache_stamp())
models_info, model_warning = load_torch_models(weight_stamp())
images = cache["images"]
labels = cache["labels"]
latent = cache["latent"]

st.title("深度生成模型实验平台")
st.caption("MNIST 自编码器、VAE、DCGAN 与文本到图像参数实验")

with st.sidebar:
    st.header("实验设置")
    sample_idx = st.slider("MNIST 样本编号", 0, len(images) - 1, 0)
    random_seed = st.number_input("全局 seed", min_value=0, max_value=999999, value=42, step=1)
    if model_warning:
        st.warning(model_warning)
    elif models_info and models_info[2]:
        st.success("已加载权重：" + "、".join(models_info[2]))
    else:
        st.info("未发现训练权重，当前使用内置缓存演示。")

tab_recon, tab_latent, tab_gan, tab_diffusion = st.tabs(
    ["重构对比", "VAE 潜空间", "DCGAN 生成", "文本到图像"]
)

with tab_recon:
    st.subheader("Autoencoder 与 VAE 重构对比")
    image = images[sample_idx]
    ae_recon, vae_recon, sample_z = infer_reconstruction(models_info, image, sample_idx, cache)
    ae_mse = float(np.mean((image - ae_recon) ** 2))
    vae_mse = float(np.mean((image - vae_recon) ** 2))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(array_to_image(image), caption=f"输入图像 | 标签 {labels[sample_idx]}")
        metric_card("样本 latent", f"({sample_z[0]:.2f}, {sample_z[1]:.2f})")
    with c2:
        st.image(array_to_image(ae_recon), caption="Autoencoder 重构")
        metric_card("AE MSE", f"{ae_mse:.5f}")
    with c3:
        st.image(array_to_image(vae_recon), caption="VAE 重构")
        metric_card("VAE MSE", f"{vae_mse:.5f}")

    h1, h2 = st.columns(2)
    with h1:
        st.caption("Autoencoder 重构误差热力图")
        st.plotly_chart(error_heatmap(image, ae_recon), width="stretch")
    with h2:
        st.caption("VAE 重构误差热力图")
        st.plotly_chart(error_heatmap(image, vae_recon), width="stretch")

    st.plotly_chart(loss_curve(cache["loss"]), width="stretch")

with tab_latent:
    st.subheader("二维潜空间探索")
    left, right = st.columns([1.35, 1])
    with left:
        color_mode = st.radio("散点着色方式", ["按类别", "按样本编号"], horizontal=True)
        st.caption("单击散点会把该位置的 z1/z2 填到右侧，用坐标直接生成图像。")
        clicked = plotly_events(
            latent_scatter(latent, labels, color_mode),
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=540,
            key="latent_click_chart",
        )
        if clicked:
            point = clicked[0]
            click_xy = np.array([float(point["x"]), float(point["y"])], dtype=np.float32)
            nearest = int(np.argmin(np.linalg.norm(latent - click_xy, axis=1)))
            st.session_state["nearest_latent_sample"] = nearest
            st.session_state["latent_x"] = float(click_xy[0])
            st.session_state["latent_y"] = float(click_xy[1])
    with right:
        st.write("直接调整二维潜变量坐标，VAE 解码器会根据坐标生成图像。")
        x_min, x_max = float(latent[:, 0].min() - 0.8), float(latent[:, 0].max() + 0.8)
        y_min, y_max = float(latent[:, 1].min() - 0.8), float(latent[:, 1].max() + 0.8)
        z1 = st.slider("z1", x_min, x_max, float(st.session_state.get("latent_x", latent[sample_idx, 0])), 0.05)
        z2 = st.slider("z2", y_min, y_max, float(st.session_state.get("latent_y", latent[sample_idx, 1])), 0.05)
        nearest_idx = int(np.argmin(np.linalg.norm(latent - np.array([z1, z2], dtype=np.float32), axis=1)))
        st.caption(f"离当前坐标最近的真实样本：#{nearest_idx} | 类别 {labels[nearest_idx]}")
        with st.expander("用真实样本坐标定位"):
            sample_for_latent = st.selectbox(
                "选择一个样本，把它的编码坐标填入 z1/z2",
                np.arange(len(images)),
                index=nearest_idx,
                format_func=lambda i: f"#{i} | 类别 {labels[i]}",
            )
            if st.button("使用该样本坐标"):
                st.session_state["latent_x"] = float(latent[int(sample_for_latent), 0])
                st.session_state["latent_y"] = float(latent[int(sample_for_latent), 1])
                st.rerun()
        generated = decode_latent(models_info, np.array([z1, z2], dtype=np.float32))
        st.image(array_to_image(generated), caption=f"VAE decode({z1:.2f}, {z2:.2f})")

    st.divider()
    st.subheader("潜空间插值")
    a, b, steps = st.columns([1, 1, 1])
    with a:
        start_idx = st.selectbox("起点样本", np.arange(len(images)), index=sample_idx, format_func=lambda i: f"#{i} | 类别 {labels[i]}")
    with b:
        end_idx = st.selectbox("终点样本", np.arange(len(images)), index=min(sample_idx + 37, len(images) - 1), format_func=lambda i: f"#{i} | 类别 {labels[i]}")
    with steps:
        interp_steps = st.slider("插值帧数", 5, 15, 9, 1)
    z_start = latent[int(start_idx)]
    z_end = latent[int(end_idx)]
    strip = []
    for alpha in np.linspace(0, 1, interp_steps):
        strip.append(decode_latent(models_info, (1 - alpha) * z_start + alpha * z_end))
    st.image(interpolation_strip(strip), caption="从起点 latent 线性插值到终点 latent")

with tab_gan:
    st.subheader("轻量 DCGAN 生成实验")
    st.caption("随机噪声 z 输入生成器 G，得到图像 G(z)；判别器 D 再给每张图一个“像真实 MNIST”的概率分数。")
    c1, c2 = st.columns([1, 2])
    with c1:
        gan_seed = st.number_input("GAN seed", min_value=0, max_value=999999, value=int(random_seed), step=1)
        sample_count = st.slider("生成样本数", 8, 32, 16, 8)
        images_gan, scores, noise = dcgan_generate(models_info, int(gan_seed), int(sample_count))
        metric_card("判别器平均分", f"{scores.mean():.3f}")
        st.caption("分数越接近 1，判别器越认为生成图像像真实 MNIST；接近 0 则越像假样本。")
        st.dataframe(
            pd.DataFrame(
                {
                    "统计": ["mean", "std", "min", "max"],
                    "噪声 z": [noise.mean(), noise.std(), noise.min(), noise.max()],
                    "D(G(z))": [scores.mean(), scores.std(), scores.min(), scores.max()],
                }
            ),
            width="stretch",
            hide_index=True,
        )
    with c2:
        st.image(image_grid(images_gan, columns=8), caption="DCGAN 生成样本网格")
        score_df = pd.DataFrame(
            {
                "样本": [f"#{i}" for i in range(len(scores))],
                "D(G(z))": np.round(scores, 4),
            }
        )
        st.dataframe(score_df, width="stretch", hide_index=True)

    st.subheader("生成器输入噪声 z")
    st.plotly_chart(noise_heatmap(noise), width="stretch")

with tab_diffusion:
    st.subheader("文本提示与采样参数实验")
    st.caption("这一页用于观察 prompt、negative prompt、采样步数、seed 和 guidance scale 对文本生成图像结果的影响。")
    d1, d2 = st.columns([0.9, 1.1])
    with d1:
        prompt = st.text_area("Prompt", "a tiny watercolor painting of a handwritten digit, clean background")
        negative_prompt = st.text_input("Negative prompt", "blurry, low contrast")
        steps = st.slider("采样步数", 1, 30, 8)
        guidance = st.slider("Guidance scale", 0.0, 12.0, 4.5, 0.5)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=int(random_seed), step=1)
        backend = st.radio(
            "生成后端",
            ["预生成示例", "diffusers 轻量模型"],
            horizontal=True,
            help="预生成示例不加载模型，稳定展示参数影响；diffusers 会调用本地已下载的轻量测试模型。",
        )
        use_real_model = backend == "diffusers 轻量模型"
        if use_real_model:
            st.warning("真实 diffusers 轻量测试模型已可加载，但输出通常是低分辨率噪声块，仅用于验证推理流程。")
        else:
            st.info("当前使用预生成示例模式，不加载 diffusers，会根据 prompt、seed、steps 和 guidance 生成手写数字风格示例。")
        run = st.button("生成并加入对比", type="primary", width="stretch")
        clear = st.button("清空对比结果", width="stretch")
    with d2:
        if "diffusion_runs" not in st.session_state:
            st.session_state["diffusion_runs"] = []
        if clear:
            st.session_state["diffusion_runs"] = []
        if run:
            params = DiffusionParams(prompt, negative_prompt, int(steps), int(seed), float(guidance))
            with st.spinner("正在生成图像..."):
                img, source = generate_image(params, use_real_model=use_real_model)
            mode_label = "diffusers" if use_real_model else "预生成示例"
            st.session_state["diffusion_runs"].insert(0, {"image": img, "params": params, "source": source, "mode": mode_label})
            st.session_state["diffusion_runs"] = st.session_state["diffusion_runs"][:6]

        if not st.session_state["diffusion_runs"]:
            st.info("设置参数后点击生成，结果会保留在这里用于对比。")
        else:
            for i, item in enumerate(st.session_state["diffusion_runs"], start=1):
                params = item["params"]
                st.image(
                    item["image"],
                    caption=(
                        f"#{i} | mode={item.get('mode', '未知')} | source={item['source']} | seed={params.seed} | "
                        f"steps={params.steps} | guidance={params.guidance_scale:.1f}"
                    ),
                )
                if str(item["source"]).startswith("fallback:"):
                    st.warning(f"真实 diffusers 模型未加载，已使用预生成示例。原因：{item['source']}")
