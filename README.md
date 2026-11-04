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

GeoPano reconstructs a geometrically consistent **colored 3D point cloud** from a
single **360° RGB panorama**. Panoramic-aware attention guides interactions between
perspective views, while hybrid supervision connects local detail with global
spherical structure.

- **Single-panorama reconstruction** for indoor and outdoor scenes.
- **Dense geometry**: panoramic XYZ point maps, radial depth, and colored PLY export.
- **Interactive demo** with example panoramas and optional sky-point removal.

[中文说明](README_zh.md)

## Contents

- [Installation](#installation)
- [Pretrained model](#pretrained-model)
- [Gradio demo](#gradio-demo)
- [Single and batch reconstruction](#single-and-batch-reconstruction)
- [Outputs](#outputs)
- [Citation](#citation)
- [License](#license)

## Installation

Use **Linux, Python 3.10, and an NVIDIA GPU**. Dependencies include
**PyTorch 2.3.1 + CUDA 12.1**, the browser demo, and ONNX Runtime.
Clone the repository and install the dependencies:

```bash
git clone https://github.com/Jou719/GeoPano.git
cd GeoPano
conda create -n geopano python=3.10 -y
conda activate geopano
pip install -r requirements.txt
```

Alternatively, `conda env create -f environment.yml` installs the same requirements.
The NVIDIA driver must support CUDA 12.1. CPU reconstruction is available with `--device cpu --precision fp32` and is slower. Geometry utilities are bundled; custom
CUDA extensions, FlashAttention, and xFormers are not required.

## Pretrained model

| Model | Download | Size |
|---|---|---|
| GeoPano | [pytorch_model.bin](https://huggingface.co/ouou123/GeoPano/resolve/main/checkpoints/pytorch_model.bin) | 3.7 GiB |

Place the checkpoint at `checkpoints/pytorch_model.bin`, or specify its path with
`--checkpoint`. Verify it against [SHA256SUMS](https://huggingface.co/ouou123/GeoPano/blob/main/checkpoints/SHA256SUMS).
The model does not need separate Pi3, VGGT, or DINOv2 weights.

## Gradio demo

```bash
python demo.py --checkpoint checkpoints/pytorch_model.bin
```

Open `http://127.0.0.1:7860`, select an example or upload a panorama, then click
**Run GeoPano** to inspect the colored point cloud and download its PLY file.

To enable **Remove sky points**, provide a compatible `skyseg.onnx` model:

```bash
python demo.py --checkpoint checkpoints/pytorch_model.bin \
    --sky-model /path/to/skyseg.onnx
```

The checkbox is enabled by default when the sky model is available. Both preview
and download use the filtered cloud; raw predictions are retained. The optional
third-party sky model is supplied separately under its original terms.

## Single and batch reconstruction

```bash
# One panorama
python files_inference.py --input /path/to/panorama.jpg \
    --checkpoint checkpoints/pytorch_model.bin --output outputs

# A directory of panoramas
python files_inference.py --input /path/to/panoramas \
    --checkpoint checkpoints/pytorch_model.bin --output outputs/batch

# All bundled examples
bash scripts/run_demo.sh
```

Inputs must be **2:1 equirectangular RGB panoramas** (`jpg`, `jpeg`, `png`, or `bmp`).
Images are resized to 2880 × 1440 and processed through ten 518 × 518 perspective
views. Directory scanning is non-recursive, and panoramas are reconstructed
independently. Add `--overwrite` to replace existing outputs or `--save-views`
to export perspective views. See [the usage guide](docs/inference.md) for details.

## Outputs

Results are saved under `<output>/<image_stem>/`:

| File | Content |
|---|---|
| `pano_points.ply` | Colored point cloud |
| `pano_points.npy` | Float32 XYZ point map, `(1440, 2880, 3)` |
| `pano_depth.npy` | Float32 radial depth, `(1440, 2880)` |
| `pano_mask.png` | Valid-point mask |
| `pano_depth_color.png` | Depth visualization |
| `metadata.json` | Input dimensions, precision, and runtime |

Depth is Euclidean distance from the panorama origin. No external metric
calibration is applied, so exported values should not be assumed to be metres.
The demo additionally saves `pano_points_nosky.ply`, `non_sky_mask.png`, and
`sky_filter.json` when sky removal is enabled.

## Citation

If you use GeoPano in your research, please cite:

```bibtex
@article{ou2026geopano,
  title={GeoPano: Towards Geometrically Accurate 3D Reconstruction from a Single Panorama},
  author={Ou, Jing and Cao, Zidong and Fan, Liaoyuan and Li, Zhuoxiao and Zhang, Shuai and Ren, Yinrui and Wu, Dongli and Li, Haoang and Xiong, Hui and Zhao, Wufan},
  journal={arXiv preprint arXiv:2610.21297},
  year={2026},
  url={https://arxiv.org/abs/2610.21297}
}
```

## License

Non-commercial restrictions apply. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the applicable terms.
Example images and the optional sky model retain their original source terms.

## Acknowledgments

GeoPano builds on Pi3, DINOv2, and utils3d. We thank their authors for sharing
code and models.
