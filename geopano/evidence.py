"""Deterministic physical evidence derived from a GeoPano point map.

The functions in this module deliberately expose estimates together with masks
and confidence-like diagnostics.  They do not turn a single-view prediction
into ground truth or a calibrated probability.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def valid_point_mask(points: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Return the finite, non-zero point mask, optionally intersected with ``mask``."""
    points = np.asarray(points)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected a point map with shape (H, W, 3), got {points.shape}.")
    valid = np.isfinite(points).all(axis=-1) & (np.linalg.norm(points, axis=-1) > 1e-8)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != valid.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match point map {valid.shape}.")
        valid &= mask
    return valid


def _neighbor(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift an array, wrapping only the horizontal equirectangular axis."""
    result = np.roll(array, shift=(dy, dx), axis=(0, 1))
    if dy > 0:
        result[:dy] = 0
    elif dy < 0:
        result[dy:] = 0
    return result


def _difference(
    points: np.ndarray,
    valid: np.ndarray,
    minus: np.ndarray,
    minus_valid: np.ndarray,
    plus: np.ndarray,
    plus_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use a centered difference where possible and a one-sided difference at holes."""
    center_valid = valid
    both = center_valid & minus_valid & plus_valid
    plus_only = center_valid & ~both & plus_valid
    minus_only = center_valid & ~both & ~plus_valid & minus_valid
    difference = np.zeros_like(points, dtype=np.float32)
    difference[both] = (plus[both] - minus[both]) * 0.5
    difference[plus_only] = plus[plus_only] - points[plus_only]
    difference[minus_only] = points[minus_only] - minus[minus_only]
    return difference, both | plus_only | minus_only


def estimate_normals(
    points: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    fusion_disagreement: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate camera-facing normals and a diagnostic confidence map.

    The normal is estimated in the equirectangular point-map domain.  Horizontal
    neighbours wrap across the panorama seam; vertical neighbours do not wrap.
    Normals are oriented toward the panorama camera (the origin).  The returned
    confidence combines neighbour support, local triangle consistency, and (when
    supplied) the relative multi-view fusion disagreement.

    Returns ``(normals, confidence, normal_mask)``.  Confidence is in ``[0, 1]``
    and is a reliability heuristic, not a calibrated probability.
    """
    points = np.asarray(points, dtype=np.float32)
    valid = valid_point_mask(points, valid_mask)
    left = _neighbor(points, 0, 1)
    right = _neighbor(points, 0, -1)
    up = _neighbor(points, -1, 0)
    down = _neighbor(points, 1, 0)
    valid_left = _neighbor(valid, 0, 1).astype(bool)
    valid_right = _neighbor(valid, 0, -1).astype(bool)
    valid_up = _neighbor(valid, -1, 0).astype(bool)
    valid_down = _neighbor(valid, 1, 0).astype(bool)

    dx, dx_valid = _difference(points, valid, left, valid_left, right, valid_right)
    dy, dy_valid = _difference(points, valid, up, valid_up, down, valid_down)
    raw = np.cross(dx, dy)
    raw_norm = np.linalg.norm(raw, axis=-1)
    normal_mask = valid & dx_valid & dy_valid & np.isfinite(raw_norm) & (raw_norm > 1e-8)
    normals = np.zeros_like(points, dtype=np.float32)
    normals[normal_mask] = raw[normal_mask] / raw_norm[normal_mask, None]

    # Camera-facing orientation makes the convention independent of the local
    # image winding.  Points lie in an arbitrary uncalibrated scale, so only the
    # direction from the origin is used here.
    toward_camera = -points
    flip = normal_mask & (np.sum(normals * toward_camera, axis=-1) < 0)
    normals[flip] *= -1

    support = (
        valid_left.astype(np.float32)
        + valid_right.astype(np.float32)
        + valid_up.astype(np.float32)
        + valid_down.astype(np.float32)
    ) * 0.25
    confidence = support * normal_mask.astype(np.float32)

    # A second pair of triangles provides a cheap discontinuity diagnostic.  On
    # a smooth patch the two estimates agree; at depth/occlusion boundaries they
    # tend to disagree even when all four neighbours are finite.
    fwd_x = right - points
    fwd_y = down - points
    back_x = points - left
    back_y = points - up
    n_forward = np.cross(fwd_x, fwd_y)
    n_backward = np.cross(back_x, back_y)
    nf = np.linalg.norm(n_forward, axis=-1)
    nb = np.linalg.norm(n_backward, axis=-1)
    triangle_ok = normal_mask & (nf > 1e-8) & (nb > 1e-8)
    cosine = np.zeros_like(confidence)
    cosine[triangle_ok] = np.sum(n_forward[triangle_ok] * n_backward[triangle_ok], axis=-1)
    cosine[triangle_ok] /= nf[triangle_ok] * nb[triangle_ok]
    consistency = np.ones_like(confidence)
    consistency[triangle_ok] = (1.0 + np.clip(cosine[triangle_ok], -1.0, 1.0)) * 0.5
    confidence *= consistency

    if fusion_disagreement is not None:
        disagreement = np.asarray(fusion_disagreement, dtype=np.float32)
        if disagreement.shape != valid.shape:
            raise ValueError(
                f"Fusion disagreement shape {disagreement.shape} does not match point map {valid.shape}."
            )
        depth = np.linalg.norm(points, axis=-1)
        relative = np.divide(
            np.maximum(disagreement, 0), np.maximum(depth, 1e-6),
            out=np.zeros_like(depth, dtype=np.float32), where=np.isfinite(depth),
        )
        confidence *= np.exp(-5.0 * np.clip(relative, 0, 10)).astype(np.float32)

    confidence[~normal_mask] = 0
    return normals, np.clip(confidence, 0, 1).astype(np.float32), normal_mask


def estimate_curvature(
    points: np.ndarray,
    normals: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    normal_confidence: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate curvature from normal variation in inverse scene units.

    This is a stable, inexpensive curvature *proxy*: the mean angular change of
    neighbouring normals divided by the corresponding 3D distance.  It is useful
    for ranking planar regions and likely edges, but is not a principal-curvature
    measurement and must not be treated as metric ground truth.
    """
    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    if points.shape != normals.shape:
        raise ValueError(f"Points {points.shape} and normals {normals.shape} must have the same shape.")
    valid = valid_point_mask(points, valid_mask)
    normal_valid = np.isfinite(normals).all(axis=-1) & (np.linalg.norm(normals, axis=-1) > 1e-8)
    valid &= normal_valid

    curvature_sum = np.zeros(valid.shape, dtype=np.float32)
    curvature_weight = np.zeros(valid.shape, dtype=np.float32)
    for dy, dx in ((0, 1), (0, -1), (-1, 0), (1, 0)):
        neighbour_points = _neighbor(points, dy, dx)
        neighbour_normals = _neighbor(normals, dy, dx)
        neighbour_valid = _neighbor(valid, dy, dx).astype(bool)
        pair = valid & neighbour_valid
        if not np.any(pair):
            continue
        dot = np.sum(normals * neighbour_normals, axis=-1)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        distance = np.linalg.norm(points - neighbour_points, axis=-1)
        usable = pair & np.isfinite(angle) & np.isfinite(distance) & (distance > 1e-8)
        curvature_sum[usable] += (angle[usable] / distance[usable]).astype(np.float32)
        curvature_weight[usable] += 1

    curvature = np.zeros(valid.shape, dtype=np.float32)
    curvature_mask = curvature_weight >= 2
    curvature[curvature_mask] = curvature_sum[curvature_mask] / curvature_weight[curvature_mask]
    confidence = np.clip(curvature_weight * 0.25, 0, 1).astype(np.float32)
    if normal_confidence is not None:
        normal_confidence = np.asarray(normal_confidence, dtype=np.float32)
        if normal_confidence.shape != valid.shape:
            raise ValueError(
                f"Normal confidence shape {normal_confidence.shape} does not match point map {valid.shape}."
            )
        confidence *= np.clip(normal_confidence, 0, 1)
    confidence[~curvature_mask] = 0
    curvature[~curvature_mask] = 0
    return curvature, confidence, curvature_mask


def derive_point_evidence(
    points: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    fusion_disagreement: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Derive all currently supported point-map evidence arrays."""
    normals, normal_confidence, normal_mask = estimate_normals(
        points, valid_mask=valid_mask, fusion_disagreement=fusion_disagreement
    )
    curvature, curvature_confidence, curvature_mask = estimate_curvature(
        points, normals, valid_mask=normal_mask, normal_confidence=normal_confidence
    )
    return {
        "normals": normals,
        "normal_confidence": normal_confidence,
        "normal_mask": normal_mask,
        "curvature": curvature,
        "curvature_confidence": curvature_confidence,
        "curvature_mask": curvature_mask,
    }
