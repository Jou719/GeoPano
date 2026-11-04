# Inference guide

Run commands from the repository root after following the environment and
checkpoint setup in the root README.

```bash
python files_inference.py --help
bash scripts/run_demo.sh
python files_inference.py --input /path/to/panoramas --output outputs/batch
CUDA_VISIBLE_DEVICES=0 python files_inference.py --input /path/to/pano.jpg --save-views
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--input` | Repository `examples/` | One image or non-recursive directory |
| `--checkpoint` | `checkpoints/pytorch_model.bin` | GeoPano state dictionary |
| `--output` | Repository `outputs/` | Per-image result root |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--precision` | `auto` | `auto`, `fp32`, `fp16`, `bf16` |
| `--save-views` | Off | Also save ten perspective RGB images and PLY clouds |
| `--overwrite` | Off | Permit updates to existing results |

Auto precision selects BF16 on supported CUDA devices, otherwise FP16. CPU uses
FP32. View count and resolution are fixed to match checkpoint attention geometry.
Each input is processed independently. Duplicate filename stems are rejected.
Overwrite replaces generated files; use a new output directory for a clean run.

## Loading outputs

```python
from pathlib import Path
import numpy as np

result = Path('outputs/real_samples/0db658dc49514ca18be7b44ceb363849')
xyz = np.load(result / 'pano_points.npy')
depth = np.load(result / 'pano_depth.npy')
valid = np.isfinite(xyz).all(axis=-1) & (depth > 0)
points = xyz[valid]
assert np.allclose(depth[valid], np.linalg.norm(points, axis=-1))
```

XYZ is float32 with shape `(1440, 2880, 3)`; depth has shape `(1440, 2880)`.
Depth is radial distance, not perspective Z-depth. No external metric calibration
is applied. Invalid values are zero. Coordinates follow the original panorama
geometry convention. The PNG mask uses 255 for valid pixels. Depth visualization
uses a separate min-to-99th-percentile range for each image and is not numeric depth.
Use MeshLab, CloudCompare, or Open3D to open the colored PLY. No mesh or
multi-panorama registration is generated.

## Troubleshooting

- Missing weights: supply the complete GeoPano inference checkpoint. Public
  download the published weights from https://huggingface.co/ouou123/GeoPano/resolve/main/checkpoints/pytorch_model.bin; Pi3/VGGT weights cannot replace it.
- Incompatible checkpoint: loading is strict and strips only `module.` prefixes.
  A complete checkpoint must include the trained attention mask.
- CUDA unavailable: check PyTorch, driver, scheduler allocation, and environment.
  Explicit CPU mode is possible but has not passed full inference validation.
- GPU memory errors: inspect `nvidia-smi` for other jobs. Use mixed precision or
  a GPU with more available memory. Fewer input files do not reduce the peak
  memory of one panorama. Do not arbitrarily change the fixed view count.
- Invalid predictions: try `--precision fp32` if GPU memory permits.
- RoPE/xFormers warnings: pure PyTorch fallback is expected; custom extensions
  are not required.
- Slow first run: geometry caches are created in the source tree and reused.
  Keep the checkout writable and avoid concurrent first runs in one checkout.
- Headless Open3D import errors: install system runtime libraries, such as
  `libgl1` and `libgomp1` on an Ubuntu-compatible machine.
- Outdoor skies: the experimental sky-segmentation branch is not included in
  this release. Sky and distant-region geometry can be unreliable.
