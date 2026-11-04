"""Check point-derived evidence on a synthetic periodic cylinder."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geopano.evidence import derive_point_evidence

height, width = 96, 192
theta = np.arange(width, dtype=np.float32) * (2 * np.pi / width)
z = np.linspace(-1, 1, height, dtype=np.float32)
tt, zz = np.meshgrid(theta, z)
points = np.stack([np.cos(tt), np.sin(tt), zz], axis=-1).astype(np.float32)
valid = np.ones((height, width), dtype=bool)
# Add a narrow invalid strip to check that holes do not create NaN normals.
valid[40:45, 70:75] = False
points[~valid] = 0
disagreement = np.zeros((height, width), dtype=np.float32)
evidence = derive_point_evidence(points, valid, disagreement)
normal = evidence["normals"]
normal_mask = evidence["normal_mask"]
confidence = evidence["normal_confidence"]
expected = -np.stack([np.cos(tt), np.sin(tt), np.zeros_like(zz)], axis=-1)
interior = normal_mask & valid & (np.abs(zz) < 0.9)
high_confidence = interior & (confidence > 0.8)
cosine = np.sum(normal[high_confidence] * expected[high_confidence], axis=-1)
assert high_confidence.mean() > 0.8, high_confidence.mean()
assert float(np.quantile(cosine, 0.05)) > 0.98, np.quantile(cosine, 0.05)
assert np.isfinite(normal).all() and np.isfinite(confidence).all()
assert np.all((confidence >= 0) & (confidence <= 1))
assert not normal_mask[~valid].any()
curvature = evidence["curvature"]
curvature_mask = evidence["curvature_mask"]
assert curvature_mask.any() and np.isfinite(curvature).all()
assert np.all(curvature[~curvature_mask] == 0)
# The periodic seam should have valid, consistent estimates on both sides.
assert normal_mask[height // 2, 0] and normal_mask[height // 2, -1]
print("PASS: finite camera-facing normals, seam wrapping, hole masks, confidence and curvature proxy")
