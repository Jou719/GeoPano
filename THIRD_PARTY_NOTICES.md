# Third-party notices

This release is derived from the local Pi3-based GeoPano research implementation.
Original copyright and license headers in source files are retained.

| Component | Included path | License information |
| --- | --- | --- |
| Pi3-derived model and layers | `pi3/` | Original non-commercial license retained in `LICENSE` |
| DINOv2 | `pi3/models/dinov2/` | Apache-2.0 notices retained in original source headers |
| Naver/CroCo-derived position embeddings | `pi3/models/layers/pos_embed.py` | Source header states CC BY-NC-SA 4.0, non-commercial use only |
| utils3d | `utils3d/` | MIT; full original license in `licenses/utils3d-MIT.txt` |

The panorama geometry and inference architecture are extracted from the supplied
GeoPano research code. This repository does not replace or broaden upstream
permissions. In particular, it must not be presented as an unrestricted MIT or
Apache-licensed project. Review all applicable upstream terms before redistribution
or commercial use.

The four real example panoramas were supplied by the maintainer. Their original
image or dataset terms continue to apply; they are not designated CC0 or licensed
under the source-code license by this release. Model weights and training datasets
are not included.
