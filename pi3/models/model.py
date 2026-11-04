"""Inference-only GeoPano model with no VGGT or random parameter initialization."""
import torch
from .geopano import Pi3 as _GeoPanoPi3


def _skip_init(tensor, *args, **kwargs):
    return tensor


class Pi3(_GeoPanoPi3):
    def _build_patch_token_mask_buffer(self, *args, **kwargs):
        return torch.empty((1, 1), dtype=torch.bool)

    def __init__(self, *args, **kwargs):
        kwargs["load_vggt"] = False
        kwargs["ckpt"] = None
        init_names = (
            "constant_", "ones_", "zeros_", "uniform_", "normal_",
            "trunc_normal_", "xavier_uniform_", "xavier_normal_",
            "kaiming_uniform_", "kaiming_normal_",
        )
        original = {name: getattr(torch.nn.init, name) for name in init_names}
        try:
            for name in init_names:
                setattr(torch.nn.init, name, _skip_init)
            super().__init__(*args, **kwargs)
        finally:
            for name, fn in original.items():
                setattr(torch.nn.init, name, fn)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = dict(state_dict)
        if "token_attn_mask" not in state_dict:
            raise ValueError("Checkpoint is missing token_attn_mask; use the GeoPano inference checkpoint.")
        token_attn_mask = state_dict.pop("token_attn_mask")
        if token_attn_mask.shape != (13690, 13690) or token_attn_mask.dtype != torch.bool:
            raise ValueError("Expected a boolean token_attn_mask with shape (13690, 13690).")
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.token_attn_mask = token_attn_mask
        return result
