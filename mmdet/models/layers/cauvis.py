import math

import torch
from torch import Tensor, nn

from mmdet.registry import MODELS


class AuxiliaryBranch(nn.Module):
    def __init__(self,
                 dims,
                 num_layers,
                 min_low_freq_ratio=0.15,
                 max_low_freq_ratio=0.42):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dims, int(dims // 16)),
            nn.Linear(int(dims // 16), dims),
            nn.SiLU(),
        )
        ratios = torch.linspace(min_low_freq_ratio, max_low_freq_ratio,
                                num_layers)
        self.register_buffer('low_freq_ratios', ratios)

    def _get_low_freq_ratio(self, layer):
        layer = min(max(int(layer), 0), self.low_freq_ratios.numel() - 1)
        return float(self.low_freq_ratios[layer].item())

    def _build_center_mask(self, height, width, ratio, device, dtype):
        side_ratio = math.sqrt(ratio)
        keep_h = max(1, int(round(height * side_ratio)))
        keep_w = max(1, int(round(width * side_ratio)))
        h_start = max(0, (height - keep_h) // 2)
        w_start = max(0, (width - keep_w) // 2)

        mask = torch.zeros(height, width, dtype=dtype, device=device)
        mask[h_start:h_start + keep_h, w_start:w_start + keep_w] = 1.0
        return mask

    def _fourier_transform_2d(self, feats, ratio):
        B, L, C = feats.shape
        grid_size = int(math.sqrt(L))
        if grid_size * grid_size != L:
            return None
        if feats.is_cuda and feats.dtype == torch.float16 and (
                grid_size & (grid_size - 1)) != 0:
            return None

        out_dtype = feats.dtype
        grid_feats = feats.float().reshape(B, grid_size, grid_size, C)
        fft_feats = torch.fft.fft2(grid_feats, dim=(1, 2))
        fft_feats = torch.fft.fftshift(fft_feats, dim=(1, 2))
        mask = self._build_center_mask(grid_size, grid_size, ratio,
                                       feats.device, grid_feats.dtype)
        masked_fft = fft_feats * mask[None, :, :, None]
        masked_fft = torch.fft.ifftshift(masked_fft, dim=(1, 2))
        ifft_feats = torch.fft.ifft2(masked_fft, dim=(1, 2)).real
        return ifft_feats.reshape(B, L, C).to(out_dtype)

    def _fourier_transform_1d(self, feats, ratio):
        _, L, _ = feats.shape

        next_power_of_two = 1
        while next_power_of_two < L:
            next_power_of_two *= 2

        out_dtype = feats.dtype
        fft_feats = feats
        if not (feats.is_cuda and feats.dtype == torch.float16):
            fft_feats = feats.float()
        padded_feats = torch.nn.functional.pad(
            fft_feats, (0, 0, 0, next_power_of_two - L))
        fft_feats = torch.fft.fft(padded_feats, dim=1)
        fft_feats = torch.fft.fftshift(fft_feats, dim=1)

        mask = torch.zeros(
            next_power_of_two, dtype=padded_feats.dtype, device=feats.device)
        mask_width = max(1, int(round(next_power_of_two * ratio)))
        mask_start = max(0, (next_power_of_two - mask_width) // 2)
        mask_end = min(next_power_of_two, mask_start + mask_width)
        mask[mask_start:mask_end] = 1.0

        masked_fft = fft_feats * mask[None, :, None]
        masked_fft = torch.fft.ifftshift(masked_fft, dim=1)
        ifft_feats = torch.fft.ifft(masked_fft, dim=1).real
        return ifft_feats[:, :L, :].to(out_dtype)

    def fourier_transform(self, feats, layer):
        ratio = self._get_low_freq_ratio(layer)
        out = self._fourier_transform_2d(feats, ratio)
        if out is not None:
            return out
        return self._fourier_transform_1d(feats, ratio)

    def forward(self, x, layer):
        x = self.mlp(x)
        out = self.fourier_transform(x, layer)
        return out + x


class CausalBranch(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dims, dims), nn.SiLU())

    def forward(self, x):
        return self.mlp(x)


@MODELS.register_module()
class Cauvis(nn.Module):
    def __init__(self,
                 num_layers: int,
                 embed_dims: int,
                 patch_size: int,
                 img_size: int,
                 prompt_init=None,
                 query_dims: int = 256,
                 token_length: int = 100,
                 use_softmax: bool = True,
                 link_token_to_query: bool = True,
                 scale_init: float = 0.001,
                 zero_mlp_delta_f: bool = False,
                 min_low_freq_ratio: float = 0.15,
                 max_low_freq_ratio: float = 0.42) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.min_low_freq_ratio = min_low_freq_ratio
        self.max_low_freq_ratio = max_low_freq_ratio
        self.token_embedding = int(img_size // patch_size) * int(img_size //
                                                                 patch_size)

        self.prompt = nn.Parameter(
            torch.zeros([self.token_length, self.embed_dims]))

        self.mlp_prompt = nn.Linear(self.embed_dims, self.embed_dims)
        self.to_out = nn.Linear(self.embed_dims, self.embed_dims)
        self.alpha = nn.Parameter(torch.tensor(self.scale_init))
        self.beta = nn.Parameter(torch.tensor(self.scale_init))
        self.delta_scale = nn.Parameter(torch.tensor(1.0))

        self.prompt_branch = CausalBranch(self.embed_dims)
        self.aux_branch = AuxiliaryBranch(
            self.embed_dims,
            self.num_layers,
            min_low_freq_ratio=self.min_low_freq_ratio,
            max_low_freq_ratio=self.max_low_freq_ratio)

    def cross_attention(self, x, prompt):
        attn = torch.einsum('bnc,mc->bnm', x, prompt)
        attn = attn * (self.embed_dims**-0.5)
        attn = attn.softmax(-1)
        score = torch.einsum('bnm,mc->bnc', attn, self.mlp_prompt(prompt))
        return self.to_out(score)

    def forward(self,
                feats: Tensor,
                layer: int,
                batch_first=False,
                has_cls_token=True) -> Tensor:
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=1)

        res_prompt = self.cross_attention(feats, self.prompt)
        main = self.prompt_branch(res_prompt) * self.alpha
        aux = self.aux_branch(res_prompt, layer) * self.beta
        feats = feats * self.delta_scale + main + aux

        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=1)
        return feats
