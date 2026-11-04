"""Derive agent-facing physical evidence from an existing GeoPano result."""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True,
                        help='One existing outputs/<image_stem> result directory.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace existing evidence files.')
    return parser.parse_args()


def main():
    args = parse_args()
    result = args.result.expanduser().resolve()
    points_path = result / 'pano_points.npy'
    if not points_path.is_file():
        raise ValueError(f'{result} must contain pano_points.npy.')
    output_names = [
        'pano_view_count.npy', 'pano_fusion_disagreement.npy',
        'pano_normals.npy', 'pano_normal_confidence.npy', 'pano_normal_mask.npy',
        'pano_curvature.npy', 'pano_curvature_confidence.npy', 'pano_curvature_mask.npy',
    ]
    existing = [result / name for name in output_names if (result / name).exists()]
    if existing and not args.overwrite:
        raise ValueError(f'Evidence already exists in {result}; use --overwrite.')

    import sys
    sys.path.insert(0, str(ROOT))
    from geopano.evidence import derive_point_evidence

    points = np.load(points_path, allow_pickle=False).astype(np.float32, copy=False)
    valid = np.isfinite(points).all(axis=-1) & (np.linalg.norm(points, axis=-1) > 1e-8)
    evidence = derive_point_evidence(points, valid_mask=valid)
    # A result produced without --save-evidence has no per-view diagnostics. Keep
    # the fields explicit so downstream agents can distinguish missing evidence
    # from a genuine zero disagreement/count.
    view_count = np.zeros(valid.shape, dtype=np.uint8)
    disagreement = np.zeros(valid.shape, dtype=np.float32)
    np.save(result / 'pano_view_count.npy', view_count)
    np.save(result / 'pano_fusion_disagreement.npy', disagreement)
    for name, array in evidence.items():
        np.save(result / f'pano_{name}.npy', array)

    metadata_path = result / 'metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    metadata['evidence'] = {
        'schema_version': 1,
        'files': output_names,
        'confidence_semantics': 'heuristic reliability score in [0,1], not a calibrated probability',
        'normal_convention': 'unit vectors oriented toward the panorama origin; horizontal neighbours wrap',
        'curvature_semantics': 'mean neighbour normal-angle change per 3D distance; inverse uncalibrated scene units',
        'fusion_disagreement_semantics': 'not available because this result predates --save-evidence; stored as zero',
        'scale': 'uncalibrated',
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Wrote evidence to {result}')


if __name__ == '__main__':
    main()
