"""Perspective extraction and panorama point-map fusion."""
import os
import numpy as np
from .panorama_geometry import (get_panorama_cameras_8views, get_panorama_cameras_6views,
    get_panorama_cameras_10views, get_panorama_cameras_4views, precompute_mapping_matrices)
class PanoramaToPerspective:

    def __init__(self, output_resolution: int=518, pano_w: int=2880, pano_h: int=1440, view_type: str='8views'):
        self.output_resolution = output_resolution
        self.pano_height = pano_h
        self.pano_width = pano_w
        if view_type == '8views':
            (self.splitted_extrinsics, self.splitted_intrinsics) = get_panorama_cameras_8views()
        elif view_type == '6views':
            (self.splitted_extrinsics, self.splitted_intrinsics) = get_panorama_cameras_6views()
        elif view_type == '10views':
            (self.splitted_extrinsics, self.splitted_intrinsics) = get_panorama_cameras_10views()
        elif view_type == '4views':
            (self.splitted_extrinsics, self.splitted_intrinsics) = get_panorama_cameras_4views()
        else:
            raise ValueError(f'Invalid view type: {view_type}')
        self.np_extrinsics = np.array(self.splitted_extrinsics, dtype=np.float32)
        self.np_intrisics = self.splitted_intrinsics[0] * output_resolution
        self.np_intrisics[2, 2] = 1
        cache_path = os.path.join(os.path.dirname(__file__), 'mapping_10views_518_1440x2880.npz')
        if view_type == '10views' and output_resolution == 518 and (self.pano_width == 2880) and (self.pano_height == 1440) and os.path.exists(cache_path):
            cached = np.load(cache_path)
            self.mapping = [x for x in cached['mapping']]
            self.occupancy_vectors = [x for x in cached['occupancy_vectors']]
            print(f'[PanoramaToPerspective] Loaded mapping cache: {cache_path}')
        else:
            (self.mapping, self.occupancy_vectors) = precompute_mapping_matrices(self.splitted_extrinsics, self.splitted_intrinsics, [self.pano_height, self.pano_width], self.output_resolution)
            if view_type == '10views' and output_resolution == 518 and (self.pano_width == 2880) and (self.pano_height == 1440):
                np.savez_compressed(cache_path, mapping=np.asarray(self.mapping), occupancy_vectors=np.asarray(self.occupancy_vectors))
                print(f'[PanoramaToPerspective] Saved mapping cache: {cache_path}')
        self.pano_pointmap = None

    def pano_pixel_to_world_rays(self):
        (H, W) = (self.pano_height, self.pano_width)
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        (uu, vv) = np.meshgrid(u, v, indexing='xy')
        u_raw = (1.5 * W - uu) % W
        theta = u_raw * (2.0 * np.pi) / W - np.pi
        phi = np.pi / 2.0 - vv * np.pi / H
        x = np.cos(phi) * np.cos(theta)
        y = np.cos(phi) * np.sin(theta)
        z = np.sin(phi)
        rays = np.stack([x, y, z], axis=-1).astype(np.float32)
        return rays

    def bilinear_sample_pointmap(self, point_map: np.ndarray, x: np.ndarray, y: np.ndarray):
        (Hv, Wv) = point_map.shape[:2]
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        inside = (x0 >= 0) & (x1 < Wv) & (y0 >= 0) & (y1 < Hv)
        sampled = np.zeros((len(x), 3), dtype=np.float32)
        valid_out = np.zeros((len(x),), dtype=bool)
        if np.any(inside):
            idx = np.where(inside)[0]
            xv = x[idx]
            yv = y[idx]
            (x0v, x1v) = (x0[idx], x1[idx])
            (y0v, y1v) = (y0[idx], y1[idx])
            wa = (x1v - xv) * (y1v - yv)
            wb = (xv - x0v) * (y1v - yv)
            wc = (x1v - xv) * (yv - y0v)
            wd = (xv - x0v) * (yv - y0v)
            Ia = point_map[y0v, x0v]
            Ib = point_map[y0v, x1v]
            Ic = point_map[y1v, x0v]
            Id = point_map[y1v, x1v]
            norm_a = np.linalg.norm(Ia, axis=-1)
            norm_b = np.linalg.norm(Ib, axis=-1)
            norm_c = np.linalg.norm(Ic, axis=-1)
            norm_d = np.linalg.norm(Id, axis=-1)
            va = np.isfinite(Ia).all(axis=-1) & (norm_a > 1e-05)
            vb = np.isfinite(Ib).all(axis=-1) & (norm_b > 1e-05)
            vc = np.isfinite(Ic).all(axis=-1) & (norm_c > 1e-05)
            vd = np.isfinite(Id).all(axis=-1) & (norm_d > 1e-05)
            wsum = np.zeros_like(wa, dtype=np.float32)
            out = np.zeros((len(idx), 3), dtype=np.float32)
            if np.any(va):
                out[va] += Ia[va] * wa[va, None]
                wsum[va] += wa[va]
            if np.any(vb):
                out[vb] += Ib[vb] * wb[vb, None]
                wsum[vb] += wb[vb]
            if np.any(vc):
                out[vc] += Ic[vc] * wc[vc, None]
                wsum[vc] += wc[vc]
            if np.any(vd):
                out[vd] += Id[vd] * wd[vd, None]
                wsum[vd] += wd[vd]
            good = wsum > 1e-08
            out[good] /= wsum[good, None]
            sampled[idx[good]] = out[good]
            valid_out[idx[good]] = True
        margin = 1.0
        inside_fov = (x > -margin) & (x < Wv - 1 + margin) & (y > -margin) & (y < Hv - 1 + margin)
        bad = ~valid_out & inside_fov
        if np.any(bad):
            xn = np.rint(x[bad]).astype(np.int32)
            yn = np.rint(y[bad]).astype(np.int32)
            xn = np.clip(xn, 0, Wv - 1)
            yn = np.clip(yn, 0, Hv - 1)
            nearest = point_map[yn, xn]
            nearest_norm = np.linalg.norm(nearest, axis=-1)
            nearest_valid = np.isfinite(nearest).all(axis=-1) & (nearest_norm > 1e-05)
            bad_idx = np.where(bad)[0]
            sampled[bad_idx[nearest_valid]] = nearest[nearest_valid]
            valid_out[bad_idx[nearest_valid]] = True
        return (sampled, valid_out)

    def fuse_perspective_pointmaps_to_pano(
        self, pointmaps_world, center_weighted=False, return_diagnostics=False
    ):
        """
        不依赖 mapping_matrices
        直接通过几何把 pano 像素投影到各个透视图，双线性采样 world point map，再多视图平均

        Args:
            pointmaps_world: list[np.ndarray]
                长度为 N_views
                每个元素 shape = [Hv, Wv, 3]
                表示该透视图上每个像素对应的 world point
            center_weighted: bool
                是否按离视图中心距离加权，默认 False（简单平均）

        Returns:
            pano_points: [Hp, Wp, 3]
            legacy_count: [Hp, Wp] integer 0/1 coverage mask (not a view count)
            pano_mask:   [Hp, Wp] bool
            When return_diagnostics=True, also returns the true per-pixel number
            of contributing perspective views and weighted point disagreement.
        """
        (Hp, Wp) = (self.pano_height, self.pano_width)
        rays_world = self.pano_pixel_to_world_rays().reshape(-1, 3)
        P = rays_world.shape[0]
        pano_sum = np.zeros((P, 3), dtype=np.float32)
        pano_wsum = np.zeros((P,), dtype=np.float32)
        if return_diagnostics:
            pano_count = np.zeros((P,), dtype=np.uint8)
            pano_sq_sum = np.zeros((P,), dtype=np.float32)
        K = self.np_intrisics.astype(np.float32)
        (fx, fy) = (K[0, 0], K[1, 1])
        (cx, cy) = (K[0, 2], K[1, 2])
        for (view_idx, pointmap_world) in enumerate(pointmaps_world):
            if pointmap_world is None:
                continue
            (Hv, Wv) = pointmap_world.shape[:2]
            ext = self.np_extrinsics[view_idx].astype(np.float32)
            R = ext[:3, :3]
            rays_cam = rays_world @ R.T
            z = rays_cam[:, 2]
            front = z > 1e-06
            if not np.any(front):
                continue
            rc = rays_cam[front]
            x = fx * (rc[:, 0] / z[front]) + cx
            y = fy * (rc[:, 1] / z[front]) + cy
            (sampled, valid) = self.bilinear_sample_pointmap(pointmap_world, x, y)
            front_idx = np.where(front)[0]
            hit_idx = front_idx[valid]
            if len(hit_idx) == 0:
                continue
            if center_weighted:
                xv = x[valid]
                yv = y[valid]
                nx = (xv - cx) / max(cx, 1e-06)
                ny = (yv - cy) / max(cy, 1e-06)
                r2 = nx * nx + ny * ny
                w = np.exp(-2.0 * r2).astype(np.float32)
            else:
                w = np.ones((len(hit_idx),), dtype=np.float32)
            pano_sum[hit_idx] += sampled[valid] * w[:, None]
            pano_wsum[hit_idx] += w
            if return_diagnostics:
                pano_count[hit_idx] += 1
                pano_sq_sum[hit_idx] += np.sum(sampled[valid] ** 2, axis=-1) * w
        pano_points = np.zeros((P, 3), dtype=np.float32)
        valid_mask = pano_wsum > 1e-08
        pano_points[valid_mask] = pano_sum[valid_mask] / pano_wsum[valid_mask, None]
        pano_points = pano_points.reshape(Hp, Wp, 3)
        pano_mask = valid_mask.reshape(Hp, Wp)
        if return_diagnostics:
            mean_sq_norm = np.zeros((P,), dtype=np.float32)
            mean_sq_norm[valid_mask] = pano_sq_sum[valid_mask] / pano_wsum[valid_mask]
            mean_norm_sq = np.sum(pano_points.reshape(-1, 3) ** 2, axis=-1)
            disagreement = np.sqrt(np.maximum(mean_sq_norm - mean_norm_sq, 0)).reshape(Hp, Wp)
            return (
                pano_points,
                pano_mask.astype(np.int32),
                pano_mask,
                pano_count.reshape(Hp, Wp),
                disagreement.astype(np.float32),
            )
        return (pano_points, pano_mask.astype(np.int32), pano_mask)
