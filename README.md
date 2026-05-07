# Streamlit 深度生成模型实验平台

这是一个面向图像处理课程作业的 Streamlit Web 应用，统一使用 MNIST 展示 Autoencoder、VAE、DCGAN 和文本到图像扩散模型的交互实验。

## 功能

- Autoencoder 与简化 VAE 的重构对比：输入图像、两种重构、误差热力图、loss 曲线。
- VAE 二维潜空间：按类别或样本编号着色，点击散点样本或手动调整 latent 坐标生成图像。
- 潜空间插值：选择两张样本，展示从起点到终点的线性插值序列。
- DCGAN 生成实验：调整 seed 和样本数量，展示输入噪声统计、判别器分数和生成网格。
- 文本到图像实验：支持“预生成示例”和“diffusers 轻量模型”两种后端，提供 prompt、negative prompt、采样步数、seed、guidance scale 参数对比；真实模型不可用时会显示参数敏感的降级示例，不影响其他页面。

## 本地运行

```bash
pip install -r requirements.txt
python scripts/bootstrap_cache.py
streamlit run app.py
```

## 可选训练

仓库包含已训练好的轻量权重和缓存，部署后可直接演示。若希望重新训练，可运行：

```bash
python scripts/train_autoencoders.py --epochs 8 --limit 6000
python scripts/train_dcgan.py --epochs 5 --limit 6000
```

训练输出会写入：

- `assets/weights/autoencoder.pt`
- `assets/weights/vae.pt`
- `assets/weights/dcgan_generator.pt`
- `assets/weights/dcgan_discriminator.pt`
- `assets/cache/demo_cache.npz`

## Streamlit Cloud 部署

1. 将项目推送到 GitHub。
2. 在 Streamlit Cloud 新建应用。
3. Main file path 填写 `app.py`。
4. Python 版本选择 3.10 或 3.11 更稳妥。
5. 首次启动会安装 `requirements.txt` 中的依赖。
6. 默认“预生成示例”模式不需要下载文本到图像模型；如果勾选 diffusers 轻量模型，应用会尝试下载 `hf-internal-testing/tiny-stable-diffusion-pipe` 到本地缓存，下载失败时自动降级。

## 项目结构

```text
app.py
requirements.txt
.streamlit/config.toml
modules/
  data.py
  diffusion_demo.py
  models.py
  visualization.py
scripts/
  bootstrap_cache.py
  train_autoencoders.py
  train_dcgan.py
assets/
  cache/
  weights/
```
