"""RGB sky masking and exact-subset PLY export; raw reconstruction is preserved."""
from pathlib import Path
import json
import numpy as np


def export_without_sky(image_path, output_dir, model_path):
    import cv2
    import onnxruntime as ort
    from PIL import Image
    from files_inference import save_cloud

    output_dir, model_path = Path(output_dir), Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f'Sky segmentation model not found: {model_path}. Supply --sky-model or disable Remove sky points.')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert('RGB').resize((2880, 1440)))
    x = cv2.resize(rgb, (320, 320)).astype(np.float32) / 255
    x = ((x - [.485, .456, .406]) / [.229, .224, .225]).transpose(2, 0, 1)[None].astype(np.float32)
    prediction = session.run(None, {session.get_inputs()[0].name: x})[0].squeeze()
    if prediction.ndim != 2 or not np.isfinite(prediction).all():
        raise ValueError('Unexpected sky segmentation output; expected finite 2D response.')
    response = (prediction - prediction.min()) / max(float(np.ptp(prediction)), 1e-6)
    non_sky = cv2.resize(response, (2880, 1440), interpolation=cv2.INTER_LINEAR) < 32 / 255
    xyz = np.load(output_dir / 'pano_points.npy')
    valid = np.isfinite(xyz).all(-1) & (np.linalg.norm(xyz, axis=-1) > 1e-6)
    keep = valid & non_sky
    if not keep.any():
        raise ValueError('Sky filtering removed every point; disable filtering and inspect the mask.')
    destination = output_dir / 'pano_points_nosky.ply'
    save_cloud(destination, xyz[keep], rgb[keep].astype(np.float32) / 255)
    Image.fromarray(non_sky.astype('uint8') * 255).save(output_dir / 'non_sky_mask.png')
    (output_dir / 'sky_filter.json').write_text(json.dumps({
        'method': 'RGB skyseg ONNX, per-image min-max response, sky threshold 32/255',
        'valid_points': int(valid.sum()), 'retained_points': int(keep.sum()),
        'removed_points': int((valid & ~non_sky).sum()),
        'raw_prediction_modified': False, 'operation': 'exact pointmap vertex subset; no geometry repair'
    }, indent=2))
    return destination
