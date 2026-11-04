# GeoPano checkpoint

Place the GeoPano inference checkpoint here as `pytorch_model.bin`, or pass
`--checkpoint /path/to/pytorch_model.bin`.

The checkpoint is approximately 3.7 GiB and is distributed separately from source.
Download: https://huggingface.co/ouou123/GeoPano/resolve/main/checkpoints/pytorch_model.bin

SHA-256: `fb6be91b354bcfec472d4e47ce86fa6cffa7d50dd3241cc2c3bc36f22e65405c`

Weights are provided for non-commercial research under the included LICENSE and applicable upstream terms. Do not substitute original Pi3 or VGGT weights.

Expected format: a plain PyTorch state dictionary, optionally with `module.` key
prefixes. Loading is strict. It must contain the trained model parameters,
`_cached_perspective_sphere_3d`, and the boolean `token_attn_mask` for ten 518×518
views. The loader accepts tensor-only checkpoints (`weights_only=True`).
No separate DINOv2, VGGT, or sky segmentation checkpoint is needed.

Weights are ignored by Git. Use a model hosting service for distribution.
The source license does not establish a separate license for unpublished weights;
the maintainer must specify their distribution terms when publishing them.
