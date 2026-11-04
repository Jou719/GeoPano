# Release validation — 2026-11-04

A fresh Python 3.10 virtual environment was installed with `pip install -r requirements.txt` on Linux. `pip check` reported no broken requirements. This validates the pip setup; creation through Conda was not separately rerun.

Completed checks:

- Gradio 5.35.0 server launches and serves its HTTP configuration in the clean environment.
- Real Demo callback completes reconstruction and RGB sky filtering, returning the filtered PLY for both preview and download.
- Single-panorama CPU FP32 reconstruction, including ten perspective images and ten PLY exports.
- `scripts/run_demo.sh` reconstructs all four real examples using CPU FP32 (about 102–108 seconds per image, excluding model load).
- All four outputs have finite `(1440, 2880, 3)` XYZ and `(1440, 2880)` radial depth; depth equals the XYZ norm; PLY point counts and colors match valid predictions.
- Ten-view unit-sphere fusion, coverage, geometry export and point-evidence checks pass.
- Source syntax and shell syntax checks pass.

A PyTorch 2.3 CPU masked-attention numerical issue was fixed by computing the same masked attention in query chunks on CPU. CUDA execution remains unchanged. All GPUs were occupied during this audit, so CUDA was not rerun; earlier CUDA validation is recorded in the project history. No other GPU jobs were interrupted.

The optional sky segmentation model is supplied separately with `--sky-model`. Its output is a display/export subset; original geometry is retained. CPU Demo runs can use `--device cpu --precision fp32`.
