"""Panorama camera geometry from the GeoPano research implementation."""
import os
from pathlib import Path
from typing import *
import itertools
import json
import warnings
import cv2
import numpy as np
from numpy import ndarray
from tqdm import tqdm, trange
from scipy.sparse import csr_array, hstack, vstack
from scipy.ndimage import convolve
from scipy.sparse.linalg import lsmr
from concurrent.futures import ProcessPoolExecutor, as_completed
import utils3d
import time
from utils3d.numpy import intrinsics_from_fov, extrinsics_look_at, icosahedron, image_uv, project_cv, uv_to_pixel

def get_panorama_cameras():
    (vertices, faces) = utils3d.numpy.icosahedron()
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics = utils3d.numpy.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return (extrinsics, [intrinsics] * len(vertices))

def get_panorama_cameras_8views():
    vertices = np.array([[0.5, 0.5, 0], [0, 0.5, 0], [-0.5, 0.5, 0], [-0.5, 0.0, 0], [-0.5, -0.5, 0], [0, -0.5, 0], [0.5, -0.5, 0], [0.5, 0, 0]], dtype=np.float32).reshape((-1, 3))
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics = utils3d.numpy.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return (extrinsics, [intrinsics] * len(vertices))

def get_panorama_cameras_10views():
    """
    全景相机10视角配置
    假设全景图是Z轴向上的约定（常见于室内场景）
    包含8个水平视角 + 上下各1个视角
    """
    face_configs = [([0.5, 0.5, 0], [0, 0, 1]), ([0, 0.5, 0], [0, 0, 1]), ([-0.5, 0.5, 0], [0, 0, 1]), ([-0.5, 0, 0], [0, 0, 1]), ([-0.5, -0.5, 0], [0, 0, 1]), ([0, -0.5, 0], [0, 0, 1]), ([0.5, -0.5, 0], [0, 0, 1]), ([0.5, 0, 0], [0, 0, 1]), ([0, 0, 0.5], [0, 1, 0]), ([0, 0, -0.5], [0, -1, 0])]
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics_list = []
    for (look_at, up) in face_configs:
        look_at_arr = np.array(look_at, dtype=np.float32).reshape(1, 3)
        up_arr = np.array(up, dtype=np.float32)
        ext = utils3d.numpy.extrinsics_look_at([0, 0, 0], look_at_arr, up_arr)
        extrinsics_list.append(ext[0])
    extrinsics = np.array(extrinsics_list, dtype=np.float32)
    return (extrinsics, [intrinsics] * len(extrinsics))

def get_panorama_cameras_6views():
    """
    全景相机6视角配置
    假设全景图是Z轴向上的约定（常见于室内场景）
    """
    face_configs = [([1, 0, 0], [0, 0, 1]), ([-1, 0, 0], [0, 0, 1]), ([0, 1, 0], [0, 0, 1]), ([0, -1, 0], [0, 0, 1]), ([0, 0, 1], [0, 1, 0]), ([0, 0, -1], [-0, -1, 0])]
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics_list = []
    for (look_at, up) in face_configs:
        look_at_arr = np.array(look_at, dtype=np.float32).reshape(1, 3)
        up_arr = np.array(up, dtype=np.float32)
        ext = utils3d.numpy.extrinsics_look_at([0, 0, 0], look_at_arr, up_arr)
        extrinsics_list.append(ext[0])
    extrinsics = np.array(extrinsics_list, dtype=np.float32)
    return (extrinsics, [intrinsics] * len(extrinsics))

def get_panorama_cameras_4views():
    """
    全景相机4视角配置
    前 / 后 / 左 / 右，统一俯仰角向上 10°
    假设全景图是 Z 轴向上的约定
    """
    pitch_deg = 10.0
    pitch = np.deg2rad(pitch_deg)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    face_configs = [([0, cos_pitch, sin_pitch], [0, 0, 1]), ([0, -cos_pitch, sin_pitch], [0, 0, 1]), ([-cos_pitch, 0, sin_pitch], [0, 0, 1]), ([cos_pitch, 0, sin_pitch], [0, 0, 1])]
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(120), fov_y=np.deg2rad(120))
    extrinsics_list = []
    for (look_at, up) in face_configs:
        look_at_arr = np.array(look_at, dtype=np.float32).reshape(1, 3)
        up_arr = np.array(up, dtype=np.float32)
        ext = utils3d.numpy.extrinsics_look_at([0, 0, 0], look_at_arr, up_arr)
        extrinsics_list.append(ext[0])
    extrinsics = np.array(extrinsics_list, dtype=np.float32)
    return (extrinsics, [intrinsics] * len(extrinsics))

def directions_to_spherical_uv(directions: np.ndarray):
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    u = 1 - np.arctan2(directions[..., 1], directions[..., 0]) / (2 * np.pi) % 1.0
    v = np.arccos(directions[..., 2]) / np.pi
    return np.stack([u, v], axis=-1)

def precompute_mapping_matrices(extrinsics: np.ndarray, intrinsics: list[np.ndarray], panorama_size, output_resolution: int):
    """
    预先计算映射矩阵和占据向量
    """
    occupancy_vectors = []
    mapping_matrices = []
    (height, width) = (panorama_size[0], panorama_size[1])
    (H_out, W_out) = (output_resolution, output_resolution)
    uv = utils3d.numpy.image_uv(width=W_out, height=H_out)
    for i in range(len(extrinsics)):
        occupancy_vector = np.zeros(height * width, dtype=bool)
        spherical_uv = directions_to_spherical_uv(utils3d.numpy.unproject_cv(uv, extrinsics=extrinsics[i], intrinsics=intrinsics[i]))
        pixels = utils3d.numpy.uv_to_pixel(spherical_uv, width=width, height=height).astype(np.float32)
        mapping_matrices.append(pixels)
        pixels_flat = pixels.reshape(-1, 2)
        valid_mask = (pixels_flat[:, 0] >= 0) & (pixels_flat[:, 0] < width) & (pixels_flat[:, 1] >= 0) & (pixels_flat[:, 1] < height)
        u = pixels_flat[valid_mask, 0]
        v = pixels_flat[valid_mask, 1]
        u0 = np.floor(u).astype(int)
        v0 = np.floor(v).astype(int)
        offsets = np.arange(-5, 6)
        all_us = []
        all_vs = []
        for du in offsets:
            for dv in offsets:
                temp_u = u0 + du
                temp_v = v0 + dv
                valid_mask = (temp_u >= 0) & (temp_u < width) & (temp_v >= 0) & (temp_v < height)
                all_us.append(temp_u[valid_mask])
                all_vs.append(temp_v[valid_mask])
        all_u = np.concatenate(all_us)
        all_v = np.concatenate(all_vs)
        valid_int = (all_u >= 0) & (all_u < width) & (all_v >= 0) & (all_v < height)
        indices = (all_v[valid_int] * width + all_u[valid_int]).astype(int)
        indices = np.unique(indices)
        occupancy_vector[indices] = True
        occupancy_vectors.append(occupancy_vector)
    return (mapping_matrices, occupancy_vectors)

def split_panorama_image(image: np.ndarray, mapping_matrices: list[np.ndarray], is_rgb):
    """
    使用预计算的映射矩阵快速拆分全景图像
    """
    splitted_images = []
    for pixels in mapping_matrices:
        splitted_image = None
        if is_rgb:
            splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        splitted_images.append(splitted_image)
    return splitted_images
