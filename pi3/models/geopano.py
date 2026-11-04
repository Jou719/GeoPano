"""GeoPano checkpoint-compatible architecture, extracted from the research code."""
import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRopePano, BlockRope
from .layers.attention import FlashAttentionRope, FlashAttentionRopePano
from .layers.transformer_head import TransformerDecoder, LinearPts3d, ContextTransformerDecoder
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from torch.utils.checkpoint import checkpoint
import numpy as np
import cv2
import os
import math
import torch.nn.functional as F
from geopano.panorama_geometry import get_panorama_cameras_8views, get_panorama_cameras_6views, get_panorama_cameras_10views, precompute_mapping_matrices, get_panorama_cameras_8views as get_panorama_cameras_full_8views

def freeze_all_params(modules):
    for module in modules:
        try:
            for (n, param) in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            module.requires_grad = False

class SkipBlock(nn.Module):

    def forward(self, x, *args, **kwargs):
        return x

class Pi3(nn.Module):

    def __init__(self, pos_type='rope100', decoder_size='large', load_vggt=True, freeze_encoder=True, use_global_points=False, train_conf=False, num_dec_blk_not_to_checkpoint=4, ckpt=None):
        super().__init__()
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):
            if RoPE2D is None:
                raise ImportError('Cannot find cuRoPE2D, please install it following the README instructions')
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([BlockRope(dim=dec_embed_dim, num_heads=dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, proj_bias=True, ffn_bias=True, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-06), act_layer=nn.GELU, ffn_layer=Mlp, init_values=0.01, qk_norm=True, attn_class=FlashAttentionRope, rope=self.rope) for _ in range(dec_depth)])
        self.decoder_pano = nn.ModuleList([BlockRopePano(dim=dec_embed_dim, num_heads=dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, proj_bias=True, ffn_bias=True, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-06), act_layer=nn.GELU, ffn_layer=Mlp, init_values=0.01, qk_norm=True, attn_class=FlashAttentionRopePano, rope=self.rope) for _ in range(4)])
        self.dec_embed_dim = dec_embed_dim
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-06)
        self.point_decoder = TransformerDecoder(in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16, out_dim=1024, rope=self.rope)
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
        self.camera_decoder = TransformerDecoder(in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16, out_dim=512, rope=self.rope, use_checkpoint=False)
        self.camera_head = CameraHead(dim=512)
        self.use_global_points = use_global_points
        if use_global_points:
            self.global_points_decoder = ContextTransformerDecoder(in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16, out_dim=1024, rope=self.rope)
            self.global_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('image_mean', image_mean)
        self.register_buffer('image_std', image_std)
        self.train_conf = train_conf
        if train_conf:
            assert ckpt is not None
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)
            freeze_all_params([self.encoder, self.decoder, self.point_decoder, self.point_head, self.camera_decoder, self.camera_head, self.register_token])
            if use_global_points:
                freeze_all_params([self.global_points_decoder, self.global_point_head])
        if freeze_encoder:
            print('Freezing the encoder.')
            freeze_all_params([self.encoder])
        freeze_all_params([self.camera_decoder, self.camera_head])
        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint
        pano_sphere_3d = self._compute_panorama_sphere_3d(pano_height=1440, pano_width=2880)
        self._precompute_perspective_sphere_3d(pano_sphere_3d, view_type='10views', output_resolution=518)
        token_attn_mask = self._build_patch_token_mask_buffer(topk=10, H=1440, W=2880)
        self.token_attn_mask = token_attn_mask

    def _compute_panorama_sphere_3d(self, pano_height, pano_width):
        """预先计算全景图的球面3D坐标"""
        u = torch.arange(pano_width, dtype=torch.float32)
        v = torch.arange(pano_height, dtype=torch.float32)
        (u, v) = torch.meshgrid(u, v, indexing='xy')
        u_raw = (1.5 * pano_width - u) % pano_width
        theta = u_raw * (2 * np.pi) / pano_width - np.pi
        phi = np.pi / 2 - v * np.pi / pano_height
        x = torch.cos(phi) * torch.cos(theta)
        y = torch.cos(phi) * torch.sin(theta)
        z = torch.sin(phi)
        directions = torch.stack([x, y, z], dim=-1)
        directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-08)
        return directions

    def _precompute_perspective_sphere_3d(self, pano_sphere_3d, view_type='10views', output_resolution=518):
        """
        预先计算每个视角的 patch-level 球面3D坐标（token对应的方向向量）
        
        Args:
            view_type: 视角类型 ('8views', '6views', '10views')
            output_resolution: 输出透视图分辨率 H=W=output_resolution
        """
        if view_type == '8views':
            (extrinsics, intrinsics) = get_panorama_cameras_8views()
        elif view_type == '6views':
            (extrinsics, intrinsics) = get_panorama_cameras_6views()
        elif view_type == '10views':
            (extrinsics, intrinsics) = get_panorama_cameras_10views()
        elif view_type == 'full_8views':
            (extrinsics, intrinsics) = get_panorama_cameras_full_8views()
        else:
            raise ValueError(f'Invalid view_type: {view_type}')
        (pano_height, pano_width) = (1440, 2880)
        cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'cache_geopano_patch_sphere_10views_518_1440x2880.pt'))
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location='cpu', weights_only=True)
            if tuple(cached.shape) == (10, 1369, 3):
                self.register_buffer('_cached_perspective_sphere_3d', cached)
                print(f'[Pi3] Loaded patch-level sphere cache: {cache_path}')
                return
        (mapping_matrices, _) = precompute_mapping_matrices(extrinsics, intrinsics, [pano_height, pano_width], output_resolution)
        (patch_h, patch_w) = (output_resolution // self.patch_size, output_resolution // self.patch_size)
        num_patches = patch_h * patch_w
        perspective_patch_3d_list = []
        for view_idx in range(len(mapping_matrices)):
            mapping = mapping_matrices[view_idx]
            mapping_tensor = torch.from_numpy(mapping).float()
            u_coords = mapping_tensor[..., 0]
            v_coords = mapping_tensor[..., 1]
            u_norm = u_coords / (pano_width - 1) * 2 - 1
            v_norm = v_coords / (pano_height - 1) * 2 - 1
            grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
            pano_3d_permuted = pano_sphere_3d.permute(2, 0, 1).unsqueeze(0)
            perspective_3d = torch.nn.functional.grid_sample(pano_3d_permuted, grid, mode='bilinear', padding_mode='border', align_corners=True)
            perspective_3d = perspective_3d.squeeze(0).permute(1, 2, 0)
            perspective_3d = perspective_3d / (perspective_3d.norm(dim=-1, keepdim=True) + 1e-08)
            patch_rays = perspective_3d.reshape(patch_h, self.patch_size, patch_w, self.patch_size, 3).mean(dim=(1, 3))
            patch_rays = patch_rays / (patch_rays.norm(dim=-1, keepdim=True) + 1e-08)
            patch_rays = patch_rays.reshape(num_patches, 3)
            perspective_patch_3d_list.append(patch_rays)
        perspective_patch_3d_all = torch.stack(perspective_patch_3d_list, dim=0)
        self.register_buffer('_cached_perspective_sphere_3d', perspective_patch_3d_all)
        torch.save(perspective_patch_3d_all.cpu(), cache_path)
        print(f'[Pi3] Saved patch-level sphere cache: {cache_path}')
        print(f'[Pi3] Precomputed patch-level perspective sphere 3D coordinates: {perspective_patch_3d_all.shape}')

    def decode(self, hidden, N, H, W):
        (BN, hw, _) = hidden.shape
        B = BN // N
        final_output = []
        hidden = hidden.reshape(B * N, hw, -1)
        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B * N, *self.register_token.shape[-2:])
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]
        pose_inject_blk_idx = 0
        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)
        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        for i in range(len(self.decoder)):
            blk = self.decoder[i]
            if i % 2 == 1 and i < 18:
                continue
            if i % 2 == 0:
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)
            if i >= self.num_dec_blk_not_to_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)
            if i in [21, 25, 29, 33]:
                (patch_h, patch_w) = (H // self.patch_size, W // self.patch_size)
                num_patches = patch_h * patch_w
                tokens_per_view = self.patch_start_idx + num_patches
                hidden_views = hidden.reshape(B, N, tokens_per_view, -1)
                patch_hidden = hidden_views[:, :, self.patch_start_idx:, :]
                patch_hidden = patch_hidden.reshape(B, N * num_patches, -1)
                attn_mask = self.token_attn_mask.unsqueeze(0).to(hidden.device)
                pano_feat = self.decoder_pano[pose_inject_blk_idx](patch_hidden, xpos=None, token_rays=None, attn_mask=attn_mask)
                pano_feat = pano_feat.reshape(B, N, num_patches, -1)
                hidden_views[:, :, self.patch_start_idx:, :] += pano_feat
                hidden = hidden_views.reshape(B, N * tokens_per_view, -1)
                pose_inject_blk_idx += 1
            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))
        return (torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1))

    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std
        (B, N, _, H, W) = imgs.shape
        (patch_h, patch_w) = (H // 14, W // 14)
        imgs = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden['x_norm_patchtokens']
        (hidden, pos) = self.decode(hidden, N, H, W)
        point_hidden = self.point_decoder(hidden, xpos=pos)
        if self.train_conf:
            conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        if self.use_global_points:
            context = hidden.reshape(B, N, patch_h * patch_w + self.patch_start_idx, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B * N, patch_h * patch_w + self.patch_start_idx, -1)
            global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            (xy, z) = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)
            if self.train_conf:
                conf_hidden = conf_hidden.float()
                conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                conf = None
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)
            if self.use_global_points:
                global_point_hidden = global_point_hidden.float()
                global_points = self.global_point_head([global_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                global_points = None
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]
        return dict(points=points, local_points=local_points, conf=conf, camera_poses=camera_poses, global_points=global_points)
