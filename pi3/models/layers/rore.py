from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange


class RoreDotProductAttention(torch.nn.Module):
    """RoRE attention with precomputed RoPE coefficients."""

    def __init__(
        self,
        head_dim: int,
        cameras: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.cameras = cameras
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.freq_base = freq_base
        self.freq_scale = freq_scale

        self.rope = RoRE(num_embeddings=6, embed_dim=head_dim, symmetric=False)

    def forward(
        self,
        q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        rays: torch.Tensor,  # (batch*views, seqlen, 6)
        **kwargs,
    ) -> torch.Tensor:
        
        # _, views, _, _, _ = rays.shape
        # rays = rearrange(
        #         rays, "b v h w c -> (b v) c h w",
        #     )
        # rays_patch = F.interpolate(
        #     rays,
        #     size=(self.patches_x, self.patches_y),
        #     mode='bilinear', align_corners=False
        # )
        # rays_patch = rearrange(
        #         rays_patch, "(b v) c h w -> b (v h w) c",
        #         v = views,
        #     )

        # b = q.shape[0]
        # positions = self.positions.repeat(b, self.cameras, 1).to(q.device)
        # rays = torch.concat((rays, positions), dim=-1)

        q = self.rope(q, rays)
        k = self.rope(k, rays)

        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            **kwargs,
        )

        return out

#----------------------------------------------------------
# RoPEnD: RoPE implementation in 'n' Dims
#----------------------------------------------------------
class RoPEnD(torch.nn.Module):
    """
    RoPE implementation in 'n' Dims, used in ablation study.
    """
        
    def __init__(self, freq=100.0, F0=1.0, num_embeddings=2, scale=1.0, learntscale=False, symmetric=False):
        super().__init__()

        self.F0 = F0
        if symmetric:
            self.num_embeddings = num_embeddings * 2
        else:
            self.num_embeddings = num_embeddings
        self.symmetric = symmetric

        if learntscale:
            self.scale = torch.nn.Parameter(torch.tensor(scale).repeat(self.num_embeddings).float(), requires_grad=True)
            self.base = torch.nn.Parameter(torch.tensor(freq).repeat(self.num_embeddings).float(), requires_grad=True)
        else:
            self.scale = torch.tensor(scale).repeat(self.num_embeddings).cuda()
            self.base = torch.tensor(freq).repeat(self.num_embeddings).cuda()

    def get_cos_sin(self, D, seq, emb_num, device, dtype):
        inv_freq = 1.0 / (self.base[emb_num] ** (torch.arange(0, D, 2).float().to(device) / D)).repeat(seq.shape[0], 1)
        freqs = torch.einsum("bi,bj->bij", seq, inv_freq * self.scale[emb_num]).to(dtype) # Scale: This is temp hack to see if it changes anything
        freqs = torch.cat((freqs, freqs), dim=-1)
        cos = freqs.cos() # (Seq, Dim)
        sin = freqs.sin()
        return cos, sin
        
    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def apply_rope1d(self, tokens, pos1d, cos, sin):
        # assert pos1d.ndim==2
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        return (tokens * cos[..., :tokens.shape[-1]]) + (self.rotate_half(tokens) * sin[..., :tokens.shape[-1]])
        
    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """

        # tokens = tokens.transpose(1, 2)
        assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
        D = tokens.size(3) // 2
        # assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
        # cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
        # split features into two along the feature dimension, and apply rope1d on each half

        split_size = tokens.shape[-1] // self.num_embeddings
        remainder = tokens.shape[-1] % self.num_embeddings

        # Create a list of sizes for each split
        split_sizes = [split_size + 1 if i < remainder else split_size for i in range(self.num_embeddings)]

        # Split the tokens tensor
        token_splits = torch.split(tokens, split_sizes, dim=-1)

        if self.symmetric:
            positions = torch.cat((positions, -positions + 1), dim=-1)

        rotated_tokens = []
        for i in range(self.num_embeddings):
            cos, sin = self.get_cos_sin(D, positions[:, :, i], i, tokens.device, tokens.dtype)
            rotated_tokens.append(self.apply_rope1d(token_splits[i], positions[:,:,i], cos, sin))

        tokens = torch.cat(rotated_tokens, dim=-1)#.transpose(1, 2)
        return tokens
        

# class RoRE(torch.nn.Module):
#     """
#     RoRE ray-based positional encoding.
#     """
        
#     def __init__(self, freq=100.0, F0=1.0, num_embeddings=2, embed_dim=768, symmetric=False):
#         super().__init__()
#         self.base = freq 
#         self.F0 = F0
#         self.symmetric = symmetric
#         if symmetric:
#             self.num_embeddings = num_embeddings * 2
#         else:
#             self.num_embeddings = num_embeddings
#         # inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D)).repeat(seq.shape[0], 1)
#         # inv_freq = torch.rand(embed_dim//2, self.num_embeddings) * 0.5 - 0.25

#         inv_freq = torch.rand(embed_dim//2, self.num_embeddings) * 0.1

#         dim_size = embed_dim//2 // self.num_embeddings
#         for i in range(self.num_embeddings):
#             inv_freq[i*dim_size:(i+1)*dim_size, i] = 1 / (self.base ** (torch.arange(0, dim_size, 2).float() / (embed_dim//2))) * 10
#         # inv_freq[:embed_dim//2//2, 0] = 1 / (self.base ** (torch.arange(0, embed_dim//2, 2).float() / (embed_dim//2)))
#         # inv_freq[embed_dim//2//2:, 1] = 1/ (self.base ** (torch.arange(0, embed_dim//2, 2).float() / (embed_dim//2)))
#         # inv_freq[:embed_dim//2//2, 2] = 1/ (self.base ** (torch.arange(0, embed_dim//2, 2).float() / (embed_dim//2)))
#         # inv_freq[embed_dim//2//2:, 3] = 1/ (self.base ** (torch.arange(0, embed_dim//2, 2).float() / (embed_dim//2)))
#         self.inv_freq = torch.nn.Parameter(inv_freq, requires_grad=True)
#         # torch.nn.init.xavier_uniform_(self.inv_freq, gain=5.0)

#     def get_cos_sin(self, seq, device, dtype):
#         inv_freq = self.inv_freq.to(device).repeat(seq.shape[0], 1, 1)

#         freqs = torch.einsum("bik,bjk->bijk", seq, inv_freq).to(dtype).sum(-1)
#         freqs = torch.cat((freqs, freqs), dim=-1)
#         cos = freqs.cos().to(dtype) # (Seq, Dim)
#         sin = freqs.sin().to(dtype)
#         return cos, sin
        
#     @staticmethod
#     def rotate_half(x):
#         x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
#         return torch.cat((-x2, x1), dim=-1)
#         # x1, x2, x3, x4 = x[..., : x.shape[-1] // 2 // 2], x[..., x.shape[-1] // 2 // 2 : x.shape[-1] // 2], x[..., x.shape[-1] // 2 : x.shape[-1] // 2 // 2 * 3], x[..., x.shape[-1] // 2 // 2 * 3 :]
#         # return torch.cat((-x2, x1, -x4, x3), dim=-1)
        
#     def apply_rope1d(self, tokens, cos, sin):
#         cos = cos[:, None, :, :]
#         sin = sin[:, None, :, :]
#         return (tokens * cos[..., :tokens.shape[-1]]) + (self.rotate_half(tokens) * sin[..., :tokens.shape[-1]])
        
#     def forward(self, tokens, positions):
#         """
#         input:
#             * tokens: batch_size x nheads x ntokens x dim
#             * positions: batch_size x ntokens x 2 (y and x position of each token)
#         output:
#             * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
#         """
#         assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
#         # cos_sum = torch.zeros_like(tokens)
#         # sin_sum = torch.zeros_like(tokens)
#         # for i in range(self.num_embeddings):
#         if self.symmetric:
#             positions = torch.cat((positions, -positions + 1), dim=-1)
        
#         cos, sin = self.get_cos_sin(positions, tokens.device, tokens.dtype)
#             # cos_sum += cos[:, None, :, :]
#             # sin_sum += sin[:, None, :, :]
#         tokens = self.apply_rope1d(tokens, cos, sin)

#         return tokens



def rore_dot_product_attention(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    rays: torch.Tensor,  # (batch, cameras, h, w, 6)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[torch.Tensor] = None,
    coeffs_y: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Similar to torch.nn.functional.scaled_dot_product_attention, but applies RoRE-style
    positional encoding.

    Currently, we assume that the sequence length is equal to:

        cameras * patches_x * patches_y

    And token ordering allows the `(seqlen,)` axis to be reshaped into
    `(cameras, patches_x, patches_y)`.
    """

    # We're going to assume self-attention: all inputs are the same shape.
    (batch, num_heads, seqlen, head_dim) = q.shape
    cameras = rays.shape[1]
    assert q.shape == k.shape == v.shape
    # assert rays.shape == (batch, cameras, 4, 4)
    assert seqlen == cameras * patches_x * patches_y

    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    assert head_dim % 4 == 0



    out = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        **kwargs,
    )

    assert out.shape == (batch, num_heads, seqlen, head_dim)
    return out

class RoRE(torch.nn.Module):
    """
    RoRE ray-based positional encoding (Optimized & Fixed)
    """
    def __init__(self, freq=100.0, F0=1.0, num_embeddings=6, embed_dim=768, symmetric=False):
        super().__init__()
        self.base = freq 
        self.F0 = F0
        self.symmetric = symmetric
        
        # 处理 symmetric 导致的维度翻倍
        self.in_dims = num_embeddings * 2 if symmetric else num_embeddings

        # 1. 初始化频率矩阵 (注意：用 zeros 初始化，而不是 rand，保持确定性)
        inv_freq = torch.zeros(embed_dim // 2, self.in_dims)
        dim_size = (embed_dim // 2) // self.in_dims

        # 2. 修复 arange 长度不匹配的问题，生成完整的频段
        for i in range(self.in_dims):
            # 去掉了步长 2，让生成的频率数量正好等于 dim_size
            bands = 1.0 / (self.base ** (torch.arange(0, dim_size).float() / dim_size))
            inv_freq[i * dim_size : (i + 1) * dim_size, i] = bands * 10.0

        # 3. 核心修复：必须冻结频率！使用 register_buffer 而不是 Parameter
        # 这样反向传播时就不会破坏预设的 3D 射线频率映射
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, seq, dtype):
        # 4. 核心修复：用极速的 matmul 替换极其消耗显存的 einsum
        # seq: [B, N, in_dims], self.inv_freq.T: [in_dims, embed_dim // 2]
        freqs = torch.matmul(seq.to(dtype), self.inv_freq.T.to(dtype)) # [B, N, embed_dim // 2]
        
        freqs = torch.cat((freqs, freqs), dim=-1) # [B, N, embed_dim]
        return freqs.cos(), freqs.sin()
        
    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def apply_rope1d(self, tokens, cos, sin):
        cos = cos.unsqueeze(1) # [B, 1, N, embed_dim]
        sin = sin.unsqueeze(1) # [B, 1, N, embed_dim]
        return (tokens * cos[..., :tokens.shape[-1]]) + (self.rotate_half(tokens) * sin[..., :tokens.shape[-1]])
        
    def forward(self, tokens, positions):
        """
        tokens: [B, num_heads, N, dim]
        positions: [B, N, num_embeddings] (例如 6 维 ray)
        """
        assert tokens.size(-1) % 2 == 0, "Dimension should be a multiple of two"
        
        if self.symmetric:
            positions = torch.cat((positions, -positions + 1.0), dim=-1)
            
        cos, sin = self.get_cos_sin(positions, tokens.dtype)
        tokens = self.apply_rope1d(tokens, cos, sin)

        return tokens


class SphericalRoRE(nn.Module):
    """
    Spherical RoPE: Directly applies Rotary Positional Encoding to (theta, phi)
    """
    def __init__(self, embed_dim=768, theta_base=10000.0):
        super().__init__()
        # 确保 embed_dim 可以被 4 整除 (因为分 theta/phi，且每个都要 rotate_half)
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        
        self.embed_dim = embed_dim
        # 频率向量的长度是特征维度的一半的一半 (即 theta 和 phi 各占一半，每半再切分给 cos/sin)
        self.dim_half_axis = embed_dim // 4  

        # -----------------------------------------------------------------
        # 1. Theta (极角) 频率：使用标准的指数衰减频率 (因为 theta 不跨越周期)
        # -----------------------------------------------------------------
        theta_inv_freq = 1.0 / (theta_base ** (torch.arange(0, self.dim_half_axis).float() / self.dim_half_axis))
        self.register_buffer("theta_inv_freq", theta_inv_freq, persistent=False)

        # -----------------------------------------------------------------
        # 2. Phi (方位角) 频率：【核心】必须使用严格的整数频率！
        # 只有频率 f 为整数，cos(f * (phi + 2pi)) 才能完美等于 cos(f * phi)
        # 从而消除跨越 -pi 和 pi 边界时的不连续性跳变。
        # -----------------------------------------------------------------
        # 生成 1, 2, 3... 的整数频率
        phi_inv_freq = torch.arange(1, self.dim_half_axis + 1).float()
        self.register_buffer("phi_inv_freq", phi_inv_freq, persistent=False)

    def get_cos_sin(self, positions, dtype):
        """
        positions: [B, N, 2] -> 包含 (theta, phi) 且必须是弧度制
        """
        # [B, N, 1]
        theta = positions[..., 0:1].to(dtype)
        phi = positions[..., 1:2].to(dtype)

        # 极速 matmul 计算相位
        # freqs_theta: [B, N, dim_half_axis]
        freqs_theta = torch.matmul(theta, self.theta_inv_freq.unsqueeze(0).to(dtype))
        # freqs_phi: [B, N, dim_half_axis]
        freqs_phi = torch.matmul(phi, self.phi_inv_freq.unsqueeze(0).to(dtype))

        # 将 theta 和 phi 的频率拼接在一起 -> [B, N, embed_dim // 2]
        freqs = torch.cat([freqs_theta, freqs_phi], dim=-1)
        
        # 复制一份以匹配 tokens 的特征维度 -> [B, N, embed_dim]
        # 前半部分将与 tokens 的前一半结合，后半部分与后一半结合
        freqs = torch.cat([freqs, freqs], dim=-1)
        
        return freqs.cos(), freqs.sin()
        
    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def apply_rope1d(self, tokens, cos, sin):
        cos = cos.unsqueeze(1) # [B, 1, N, embed_dim] 适配 num_heads 维度
        sin = sin.unsqueeze(1) # [B, 1, N, embed_dim]
        return (tokens * cos[..., :tokens.shape[-1]]) + (self.rotate_half(tokens) * sin[..., :tokens.shape[-1]])
        
    def forward(self, tokens, positions):
        """
        tokens: [B, num_heads, N, embed_dim]
        positions: [B, N, 2] (theta, phi in radians)
        """
        cos, sin = self.get_cos_sin(positions, tokens.dtype)
        tokens = self.apply_rope1d(tokens, cos, sin)
        return tokens