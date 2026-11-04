"""Reconstruct independent RGB panoramas with a GeoPano checkpoint."""
import argparse
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'examples', help='One 2:1 RGB panorama or a directory (non-recursive).')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/pytorch_model.bin')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--precision', default='auto', choices=['auto', 'fp32', 'fp16', 'bf16'])
    parser.add_argument('--save-views', action='store_true', help='Also export the ten perspective RGB images and PLY files.')
    parser.add_argument('--save-evidence', action='store_true', help='Also export multi-view fusion diagnostics, normals, and curvature estimates.')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing per-image results.')
    return parser.parse_args()


def collect_images(path):
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    if path.is_file():
        images = [path] if path.suffix.lower() in extensions else []
    elif path.is_dir():
        images = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    else:
        raise ValueError(f'Input does not exist: {path}')
    if not images:
        raise ValueError(f'No supported images found in {path}')
    stems = [p.stem for p in images]
    if len(stems) != len(set(stems)):
        raise ValueError('Input files must have unique stems (e.g. room.jpg and room.png would collide).')
    return images


def save_cloud(path, points, colors):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise OSError(f'Could not write {path}')


def main():
    args = parse_args()
    images = collect_images(args.input)
    if not args.checkpoint.is_file():
        raise ValueError(f'Checkpoint not found: {args.checkpoint}. See checkpoints/README.md.')
    for image in images:
        target = args.output / image.stem
        if target.exists() and not args.overwrite:
            raise ValueError(f'Output already exists: {target}. Use --overwrite or choose another --output.')

    import cv2
    import numpy as np
    import torch
    from pi3.models.model import Pi3
    from geopano.panorama import PanoramaToPerspective
    from geopano.panorama_geometry import split_panorama_image
    from geopano.evidence import derive_point_evidence

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Install a CUDA-enabled PyTorch build or explicitly use --device cpu.')
    if args.device == 'cpu' and args.precision not in ('auto', 'fp32'):
        raise ValueError('CPU inference requires --precision auto or fp32.')
    precision = args.precision
    if precision == 'auto':
        precision = ('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if args.device == 'cuda' else 'fp32'
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support BF16. Use --precision fp16 or fp32.')
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[precision]
    print(f'Loading {args.checkpoint} on {args.device} ({precision})', flush=True)
    # Only tensor checkpoints are accepted; never deserialize arbitrary Python objects.
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(weights, dict):
        raise ValueError('Expected a plain GeoPano state_dict checkpoint.')
    weights = {k.removeprefix('module.'): v for k, v in weights.items()}
    model = Pi3().eval()
    model.load_state_dict(weights, strict=True)
    del weights
    model = model.to(args.device)
    projector = PanoramaToPerspective(output_resolution=518, view_type='10views')

    for image in images:
        started = time.perf_counter()
        pano = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if pano is None:
            raise ValueError(f'Unable to decode image: {image}')
        h, w = pano.shape[:2]
        if abs(w / h - 2) > 0.02:
            raise ValueError(f'{image.name}: expected a 2:1 equirectangular panorama, got {w}x{h}.')
        pano = cv2.cvtColor(cv2.resize(pano, (2880, 1440)), cv2.COLOR_BGR2RGB)
        views = split_panorama_image(pano, projector.mapping, True)
        inputs = torch.stack([torch.from_numpy(rgb).permute(2, 0, 1).float() / 255 for rgb in views]).to(args.device)
        with torch.inference_mode(), torch.autocast(device_type=args.device, dtype=dtype, enabled=precision != 'fp32'):
            result = model(inputs[None])
        local = result['local_points'][0].float().cpu().numpy()
        del result, inputs
        world_maps = []
        out = args.output / image.stem
        out.mkdir(parents=True, exist_ok=True)
        for i, points in enumerate(local):
            c2w = np.linalg.inv(projector.np_extrinsics[i])
            world = points @ c2w[:3, :3].T + c2w[:3, 3]
            world_maps.append(world)
            if args.save_views:
                view_dir = out / 'individual_views'
                view_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(view_dir / f'view_{i:02d}.png'), cv2.cvtColor(views[i], cv2.COLOR_RGB2BGR))
                valid = np.isfinite(world).all(-1) & (np.linalg.norm(world, axis=-1) > 0)
                save_cloud(view_dir / f'view_{i:02d}.ply', world[valid], views[i][valid] / 255.0)
        if args.save_evidence:
            points, _, coverage, view_count, fusion_disagreement = projector.fuse_perspective_pointmaps_to_pano(
                world_maps, center_weighted=True, return_diagnostics=True
            )
        else:
            points, _, coverage = projector.fuse_perspective_pointmaps_to_pano(world_maps, center_weighted=True)
        depth = np.linalg.norm(points, axis=-1)
        valid = coverage & np.isfinite(points).all(-1) & np.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise RuntimeError(f'No finite positive-depth predictions for {image.name}. Try --precision fp32.')
        points[~valid] = 0
        depth[~valid] = 0
        np.save(out / 'pano_points.npy', points.astype(np.float32))
        np.save(out / 'pano_depth.npy', depth.astype(np.float32))
        cv2.imwrite(str(out / 'pano_mask.png'), valid.astype(np.uint8) * 255)
        lo, hi = float(depth[valid].min()), float(np.percentile(depth[valid], 99))
        normalized = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        color[~valid] = (0, 0, 128)
        cv2.imwrite(str(out / 'pano_depth_color.png'), color)
        save_cloud(out / 'pano_points.ply', points[valid], pano[valid] / 255.0)
        if args.save_evidence:
            evidence = derive_point_evidence(points, valid_mask=valid, fusion_disagreement=fusion_disagreement)
            np.save(out / 'pano_view_count.npy', view_count)
            np.save(out / 'pano_fusion_disagreement.npy', fusion_disagreement)
            for name, array in evidence.items():
                np.save(out / f'pano_{name}.npy', array)
        metadata = {'input_name': image.name, 'input_size': [h, w], 'panorama_size': [1440, 2880],
                    'views': 10, 'view_resolution': 518, 'precision': precision, 'device': args.device,
                    'valid_pixels': int(valid.sum()), 'elapsed_seconds': time.perf_counter() - started,
                    'depth_visualization_min': lo, 'depth_visualization_p99': hi,
                    'torch_version': torch.__version__}
        if args.save_evidence:
            metadata['evidence'] = {
                'schema_version': 1,
                'files': ['pano_view_count.npy', 'pano_fusion_disagreement.npy',
                          'pano_normals.npy', 'pano_normal_confidence.npy', 'pano_normal_mask.npy',
                          'pano_curvature.npy', 'pano_curvature_confidence.npy', 'pano_curvature_mask.npy'],
                'confidence_semantics': 'heuristic reliability score in [0,1], not a calibrated probability',
                'normal_convention': 'unit vectors oriented toward the panorama origin; horizontal neighbours wrap',
                'curvature_semantics': 'mean neighbour normal-angle change per 3D distance; inverse uncalibrated scene units',
                'fusion_disagreement_semantics': 'weighted RMS 3D spread among contributing perspective predictions',
                'scale': 'uncalibrated',
            }
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'Done: {image.name} -> {out} ({metadata["elapsed_seconds"]:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
