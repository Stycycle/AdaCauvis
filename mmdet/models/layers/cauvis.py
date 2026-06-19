import math

import torch
from torch import Tensor, nn

from mmdet.registry import MODELS


class AuxiliaryBranch(nn.Module):
    def __init__(self,
                 dims,
                 num_layers,
                 min_low_freq_ratio=0.15,
                 max_low_freq_ratio=0.42,
                 use_2d_fft=False,
                 learnable_freq=False,
                 freq_base=0.20,
                 freq_delta_max=0.05,
                 freq_softness=1.0,
                 learnable_ratio=False):
        super().__init__()
        self.use_2d_fft = use_2d_fft
        self.num_layers = num_layers
        self.learnable_freq = learnable_freq
        self.freq_base = freq_base
        self.freq_delta_max = freq_delta_max
        self.freq_softness = freq_softness
        self.learnable_ratio = learnable_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dims, int(dims // 16)),
            nn.Linear(int(dims // 16), dims),
            nn.SiLU(),
        )
        if learnable_ratio == 'range' or learnable_ratio == 'adaptive_range':
            init_min = max(0.01, min(0.99, min_low_freq_ratio))
            init_max = max(0.01, min(0.99, max_low_freq_ratio))
            min_logits = math.log(init_min / (1.0 - init_min))
            max_logits = math.log(init_max / (1.0 - init_max))
            self.min_ratio = nn.Parameter(torch.tensor(min_logits))
            self.max_ratio = nn.Parameter(torch.tensor(max_logits))
        elif learnable_ratio:
            init_ratio = 0.2
            init_logits = math.log(init_ratio / (1.0 - init_ratio))
            ratios = torch.full((num_layers,), init_logits)
            self.low_freq_ratios = nn.Parameter(ratios)
        else:
            ratios = torch.linspace(min_low_freq_ratio, max_low_freq_ratio,
                                    num_layers)
            self.register_buffer('low_freq_ratios', ratios)

        # Learnable bounded per-layer offset: r_l = freq_base + freq_delta_max
        # * tanh(theta_l), so r_l stays in [base - delta_max, base + delta_max].
        if learnable_freq:
            self.freq_delta = nn.Parameter(torch.zeros(num_layers))

    def _get_low_freq_ratio(self, layer):
        layer = min(max(int(layer), 0), self.num_layers - 1)
        if self.learnable_ratio == 'range' or self.learnable_ratio == 'adaptive_range':
            min_r = torch.sigmoid(self.min_ratio)
            max_r = torch.sigmoid(self.max_ratio)
            if self.num_layers <= 1:
                return min_r
            t = layer / (self.num_layers - 1)
            return min_r + t * (max_r - min_r)

        ratio = self.low_freq_ratios[layer]
        if self.learnable_ratio:
            return torch.sigmoid(ratio)
        return float(ratio.item())

    def _get_ratio_tensor(self, layer):
        # Differentiable per-layer ratio centered on freq_base, bounded by tanh.
        layer = min(max(int(layer), 0), self.num_layers - 1)
        delta = self.freq_delta_max * torch.tanh(self.freq_delta[layer])
        return (self.freq_base + delta).clamp(min=1e-6)

    def _build_center_mask(self, height, width, ratio, device, dtype):
        side_ratio = math.sqrt(ratio)
        keep_h = max(1, int(round(height * side_ratio)))
        keep_w = max(1, int(round(width * side_ratio)))
        h_start = max(0, (height - keep_h) // 2)
        w_start = max(0, (width - keep_w) // 2)

        mask = torch.zeros(height, width, dtype=dtype, device=device)
        mask[h_start:h_start + keep_h, w_start:w_start + keep_w] = 1.0
        return mask

    def _soft_profile(self, length, keep_half, device):
        # Smooth centered low-pass profile, differentiable in keep_half.
        coords = torch.arange(length, device=device, dtype=torch.float32)
        center = (length - 1) / 2.0
        dist = (coords - center).abs()
        return torch.sigmoid((keep_half - dist) / self.freq_softness), dist

    def _straight_through_mask(self, hard_mask, soft_mask):
        # Forward uses hard masking; backward follows the soft mask gradient.
        return hard_mask.detach() - soft_mask.detach() + soft_mask

    def _build_soft_mask_2d(self, grid, ratio, device, dtype):
        # Per-axis kept half-width: side = sqrt(ratio) so kept area ~= ratio.
        keep_half = torch.sqrt(ratio.clamp(min=1e-6)) * grid / 2.0
        prof, dist = self._soft_profile(grid, keep_half, device)
        soft_mask = prof[:, None] * prof[None, :]
        hard_prof = (dist <= keep_half).to(soft_mask.dtype)
        hard_mask = hard_prof[:, None] * hard_prof[None, :]
        return self._straight_through_mask(hard_mask, soft_mask).to(dtype)

    def _build_soft_mask_1d(self, length, ratio, device, dtype):
        keep_half = ratio.clamp(min=1e-6) * length / 2.0
        soft_mask, dist = self._soft_profile(length, keep_half, device)
        hard_mask = (dist <= keep_half).to(soft_mask.dtype)
        return self._straight_through_mask(hard_mask, soft_mask).to(dtype)

    def _fourier_transform_2d(self, feats, ratio, force=False):
        B, L, C = feats.shape
        grid_size = int(math.sqrt(L))
        if grid_size * grid_size != L:
            return None
        # cuFFT only supports power-of-two sizes in fp16. The 2D path always
        # computes in fp32 (see the .float() cast below), so when explicitly
        # forced we can safely run it on non-power-of-two grids under AMP.
        if not force and feats.is_cuda and feats.dtype == torch.float16 and (
                grid_size & (grid_size - 1)) != 0:
            return None

        out_dtype = feats.dtype
        grid_feats = feats.float().reshape(B, grid_size, grid_size, C)
        fft_feats = torch.fft.fft2(grid_feats, dim=(1, 2))
        fft_feats = torch.fft.fftshift(fft_feats, dim=(1, 2))
        if torch.is_tensor(ratio):
            mask = self._build_soft_mask_2d(grid_size, ratio, feats.device,
                                            grid_feats.dtype)
        else:
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
        learnable = torch.is_tensor(ratio)
        fft_feats = feats
        # Soft (learnable) masking needs fp32 for stable gradients.
        if learnable or not (feats.is_cuda and feats.dtype == torch.float16):
            fft_feats = feats.float()
        padded_feats = torch.nn.functional.pad(
            fft_feats, (0, 0, 0, next_power_of_two - L))
        fft_feats = torch.fft.fft(padded_feats, dim=1)
        fft_feats = torch.fft.fftshift(fft_feats, dim=1)

        if learnable:
            mask = self._build_soft_mask_1d(next_power_of_two, ratio,
                                            feats.device, padded_feats.dtype)
        else:
            mask = torch.zeros(
                next_power_of_two,
                dtype=padded_feats.dtype,
                device=feats.device)
            mask_width = max(1, int(round(next_power_of_two * ratio)))
            mask_start = max(0, (next_power_of_two - mask_width) // 2)
            mask_end = min(next_power_of_two, mask_start + mask_width)
            mask[mask_start:mask_end] = 1.0

        masked_fft = fft_feats * mask[None, :, None]
        masked_fft = torch.fft.ifftshift(masked_fft, dim=1)
        ifft_feats = torch.fft.ifft(masked_fft, dim=1).real
        return ifft_feats[:, :L, :].to(out_dtype)

    def fourier_transform(self, feats, layer):
        if self.learnable_ratio:
            ratio = self._get_low_freq_ratio(layer)
        elif self.learnable_freq:
            ratio = self._get_ratio_tensor(layer)
        else:
            ratio = self._get_low_freq_ratio(layer)
        out = self._fourier_transform_2d(feats, ratio, force=self.use_2d_fft)
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
                 max_low_freq_ratio: float = 0.42,
                 use_2d_fft: bool = False,
                 learnable_freq: bool = False,
                 freq_base: float = 0.20,
                 freq_delta_max: float = 0.05,
                 freq_softness: float = 1.0,
                 learnable_ratio=False) -> None:
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
        self.use_2d_fft = use_2d_fft
        self.learnable_freq = learnable_freq
        self.freq_base = freq_base
        self.freq_delta_max = freq_delta_max
        self.freq_softness = freq_softness
        self.learnable_ratio = learnable_ratio
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
            max_low_freq_ratio=self.max_low_freq_ratio,
            use_2d_fft=self.use_2d_fft,
            learnable_freq=self.learnable_freq,
            freq_base=self.freq_base,
            freq_delta_max=self.freq_delta_max,
            freq_softness=self.freq_softness,
            learnable_ratio=self.learnable_ratio)

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
