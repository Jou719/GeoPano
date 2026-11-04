<div align="center">

<h1>GeoPano: Towards Geometrically Accurate 3D Reconstruction from a Single Panorama</h1>

<p>Jing Ou · Zidong Cao · Liaoyuan Fan · Zhuoxiao Li · Shuai Zhang<br>
Yinrui Ren · Dongli Wu · Haoang Li · Hui Xiong · Wufan Zhao</p>

<p>
<img src="https://img.shields.io/badge/NeurIPS-2026-68438C" alt="NeurIPS 2026">
<a href="https://github.com/Jou719/GeoPano"><img src="https://img.shields.io/badge/Code-GitHub-181717" alt="Code"></a>
<a href="https://arxiv.org/abs/2610.21297"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper"></a>
<a href="https://huggingface.co/ouou123/GeoPano"><img src="https://img.shields.io/badge/🤗-Model-yellow" alt="Model weights"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Non--commercial-orange" alt="License"></a>
</p>

<img src="assets/teaser.jpg" alt="GeoPano indoor and outdoor panoramic reconstruction" width="100%">

</div>

GeoPano 从单张 **360° RGB 全景图**恢复几何一致的**彩色三维点云**。
通过感知全景球面几何的注意力机制与混合监督，兼顾局部细节和全局结构。

- 支持室内、室外全景重建。
- 输出稠密 XYZ 点图、径向深度和彩色 PLY 点云。
- 提供交互式 Demo、真实样例和可选去天空功能。

[English](README.md)

## 安装

使用 Linux、Python 3.10 和 NVIDIA GPU。克隆仓库后安装：

```bash
git clone https://github.com/Jou719/GeoPano.git
cd GeoPano
conda create -n geopano python=3.10 -y
conda activate geopano
pip install -r requirements.txt
```

`requirements.txt` 统一安装 PyTorch 2.3.1+cu121、Demo 和 ONNX Runtime 依赖。
NVIDIA 驱动需支持 CUDA 12.1。也可使用 `conda env create -f environment.yml`。
CPU 可使用 `--device cpu --precision fp32`，速度较慢。
无需编译自定义 CUDA 扩展或安装 FlashAttention、xFormers。

## 模型权重

从 [Hugging Face](https://huggingface.co/ouou123/GeoPano/resolve/main/checkpoints/pytorch_model.bin)
下载约 3.7 GiB 的权重，放入 `checkpoints/pytorch_model.bin`。
校验值见 [SHA256SUMS](https://huggingface.co/ouou123/GeoPano/blob/main/checkpoints/SHA256SUMS)。
不需要额外下载 Pi3、VGGT 或 DINOv2 权重。

## 交互式 Demo

```bash
python demo.py --checkpoint checkpoints/pytorch_model.bin
```

打开 `http://127.0.0.1:7860`，选择样例或上传全景，点击 **Run GeoPano**，
即可预览彩色点云并下载 PLY。

去天空时指定兼容的第三方天空分割模型：

```bash
python demo.py --checkpoint checkpoints/pytorch_model.bin \
    --sky-model /path/to/skyseg.onnx
```

检测到模型后默认勾选 **Remove sky points**。预览和下载均使用过滤结果，
原始重建保留。天空分割模型须自行提供并遵守其原始许可。

## 单图与批量重建

```bash
python files_inference.py --input /path/to/panorama.jpg \
    --checkpoint checkpoints/pytorch_model.bin --output outputs

python files_inference.py --input /path/to/panoramas \
    --checkpoint checkpoints/pytorch_model.bin --output outputs/batch

bash scripts/run_demo.sh
```

输入须为 2:1 等距柱状 RGB 全景图，支持 jpg、jpeg、png、bmp。
输入统一缩放为 2880×1440，通过十个 518×518 透视视图处理。
目录不递归扫描，每张全景独立重建。使用 `--overwrite` 覆盖旧结果，
使用 `--save-views` 导出透视视图。详细参数见 [使用指南](docs/inference.md)。

## 输出

结果位于 `<output>/<image_stem>/`：

| 文件 | 内容 |
|---|---|
| `pano_points.ply` | 彩色点云 |
| `pano_points.npy` | float32 XYZ 点图，形状 `(1440, 2880, 3)` |
| `pano_depth.npy` | float32 径向深度，形状 `(1440, 2880)` |
| `pano_mask.png` | 有效点掩码 |
| `pano_depth_color.png` | 深度可视化 |
| `metadata.json` | 尺寸、精度与运行时间 |

深度为点到全景原点的欧氏距离；未经外部尺度标定，不应直接视为米。
Demo 去天空时额外保存 `pano_points_nosky.ply`、`non_sky_mask.png` 和 `sky_filter.json`。

## 引用

```bibtex
@article{ou2026geopano,
  title={GeoPano: Towards Geometrically Accurate 3D Reconstruction from a Single Panorama},
  author={Ou, Jing and Cao, Zidong and Fan, Liaoyuan and Li, Zhuoxiao and Zhang, Shuai and Ren, Yinrui and Wu, Dongli and Li, Haoang and Xiong, Hui and Zhao, Wufan},
  journal={arXiv preprint arXiv:2610.21297},
  year={2026},
  url={https://arxiv.org/abs/2610.21297}
}
```

## 许可与致谢

适用非商业使用限制，见 [LICENSE](LICENSE) 和 [第三方声明](THIRD_PARTY_NOTICES.md)。
示例图像和可选天空模型保留原始许可。
感谢 Pi3、DINOv2 和 utils3d 的作者分享代码与模型。
