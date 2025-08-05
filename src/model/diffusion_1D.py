# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import torch
from einops import rearrange

from model.base import BaseModule


class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)  # (c, h, w) -> (c, 2h, 2w)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)  # (c, h, w) -> (c, h/2, w/2)


class Rezero(BaseModule):
    """Rezero is a module that wraps a function and scales its output by a learnable parameter g
    initialized to zero as it is used to learn a residual"""

    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(
                dim, dim_out, (1, 3), padding=(0, 1)
            ),  # (dim, h, w) -> (dim_out, h, w)
            ArtChannelsAttention(dim_out, heads=4, dim_head=32),
            torch.nn.GroupNorm(groups, dim_out),  # (dim_out, h, w) -> (dim_out, h, w)
            Mish(),
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask  # (dim, h, w) -> (dim_out, h, w)


class OldBlock(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(OldBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(
                dim, dim_out, 3, padding=1
            ),  # (dim, h, w) -> (dim_out, h, w)
            torch.nn.GroupNorm(groups, dim_out),  # (dim_out, h, w) -> (dim_out, h, w)
            Mish(),
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask  # (dim, h, w) -> (dim_out, h, w)


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)  # (dim, h, w) -> (dim_out, h, w)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)  # (dim_out, 1, 1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class ArtChannelsAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        """
        We have a 2D input tensor with shape (b, dim, n_feats, T)
        where n_feats is 16 art features and T is the sequence length (100 for 2 seconds
        but not always 100... can be shorter in inference).
        We want to apply attention across the channels n_feats.
        Here dim is the sequence length, we want
        """
        super(ArtChannelsAttention, self).__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.hidden_dim = dim_head * heads
        # return
        self.to_qkv = torch.nn.Conv2d(
            dim, self.hidden_dim * 3, (1, 3), padding=(0, 1), bias=False
        )  # each row processed independently
        self.to_out = torch.nn.Conv2d(self.hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)  # (b, hidden_dim * 3, n_feats, T)
        # q, k, v = rearrange(
        #    qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        # )  # (b, heads, dim_head, h*w) * 3
        # k = k.softmax(dim=-1)  # (b, heads, dim_head, h*w)
        # context = torch.einsum(
        #    "bhdn,bhen->bhde", k, v
        # )  # (b, heads, dim_head, dim_head)
        # out = torch.einsum("bhde,bhdn->bhen", context, q)  # (b, heads, dim_head, h*w)
        # out = rearrange(
        #    out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        # )  # (b, heads * dim_heads, h, w)
        # return self.to_out(out)  # (b, heads * dim_heads, h, w) -> (b, dim_in, h, w)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads w h c", heads=self.heads, qkv=3
        )  # (b, heads, T, n_feats, dim_head) * 3
        context = torch.einsum("bhtnd,bhtmd->bhtnm", q, k)  # (b, heads, T, h, h)
        context = torch.softmax(
            context / (self.dim_head**0.5), dim=-1
        )  # (b, heads, T, h, h)
        out = torch.einsum(
            "bhtnm,bhtmd->bhtnd", context, v
        )  # (b, heads, T, h, dim_head)
        out = rearrange(
            out, "b heads w h c -> b (heads c) h w", heads=self.heads, h=h, w=w
        )  # (b, hidden_dim, h, w)
        return self.to_out(out)


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(
            dim, hidden_dim * 3, 1, bias=False
        )  # each row processed independently
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )  # (b, heads, hidden_dim, h*w) * 3
        k = k.softmax(dim=-1)  # (b, heads, hidden_dim, h*w)
        context = torch.einsum(
            "bhdn,bhen->bhde", k, v
        )  # (b, heads, hidden_dim, hidden_dim)
        out = torch.einsum("bhde,bhdn->bhen", context, q)  # (b, heads, hidden_dim, h*w)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )  # (b, heads * dim_heads, h, w)
        return self.to_out(out)  # (b, heads * dim_heads, h, w) -> (b, dim_in, h, w)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GradLogPEstimator2d(BaseModule):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4),
        groups=8,
        n_spks=None,
        spk_emb_dim=64,
        n_feats=80,
        pe_scale=1000,
    ):
        super(GradLogPEstimator2d, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale

        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(
                torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4),
                Mish(),
                torch.nn.Linear(spk_emb_dim * 4, n_feats),
            )
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), Mish(), torch.nn.Linear(dim * 4, dim)
        )

        dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_in))),
                        Upsample(dim_in),
                    ]
                )
            )
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t, spk=None):
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)  # (B, n_feats)

        t = self.time_pos_emb(t, scale=self.pe_scale)  # (B, dim)
        t = self.mlp(t)  # (B, dim)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)  # (B, 2, n_feats, T)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])  # (B, n_feats, T)
            x = torch.stack([mu, x, s], 1)  # (B, 3, n_feats, T)
        mask = mask.unsqueeze(1)  # (B, 1, 1, T)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)  # (B, dim_in, h, w) -> (B, dim_out, h, w)
            x = resnet2(x, mask_down, t)  # (B, dim_out, h, w)
            x = attn(x)  # (B, dim_out, h, w)
            hiddens.append(x)
            x = downsample(
                x * mask_down
            )  # (B, dim_out, h/2, w/2) or (B, dim_out, h, w)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)  # (B, dim_out, n_feats/2^2, T/2^2)
        x = self.mid_attn(x)  # (B, dim_out, n_feats/2^2, T/2^2)
        x = self.mid_block2(x, mask_mid, t)  # (B, dim_out, n_feats/2^2, T/2^2)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)  # (B, 2 * dim_out, h, w)
            x = resnet1(x, mask_up, t)  # (B, 2 * dim_out, h, w) -> (B, dim_in, h, w)
            x = resnet2(x, mask_up, t)  # (B, dim_in, h, w)
            x = attn(x)  # (B, dim_in, h, w)
            x = upsample(x * mask_up)  # (B, dim_in, 2h, 2w)

        x = self.final_block(x, mask)  # (B, dim, n_feats, T) -> (B, dim, n_feats, T)
        output = self.final_conv(x * mask)  # (B, dim, n_feats, T) -> (1, n_feats, T)

        return (output * mask).squeeze(1)  # (B, n_feats, T)


def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init * t + 0.5 * (beta_term - beta_init) * (t**2)
    else:
        noise = beta_init + (beta_term - beta_init) * t
    return noise


class Diffusion1D(BaseModule):
    def __init__(
        self,
        n_feats,
        dim,
        n_spks=1,
        spk_emb_dim=64,
        beta_min=0.05,
        beta_max=20,
        pe_scale=1000,
    ):
        super(Diffusion1D, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.estimator = GradLogPEstimator2d(
            dim, n_spks=n_spks, spk_emb_dim=spk_emb_dim, pe_scale=pe_scale
        )

    def forward_diffusion(self, x0, mask, mu, t):  # 3 (B, n_feats, out_size) (B,)
        time = t.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        cum_noise = get_noise(
            time, self.beta_min, self.beta_max, cumulative=True
        )  # (B, 1, 1)
        mean = x0 * torch.exp(-0.5 * cum_noise) + mu * (
            1.0 - torch.exp(-0.5 * cum_noise)
        )  # rho (B, n_feats, out_size)
        variance = 1.0 - torch.exp(-cum_noise)  # lambda (B, 1, 1)
        z = torch.randn(
            x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False
        )  # (B, n_feats, out_size)
        xt = mean + z * torch.sqrt(variance)  # (B, n_feats, out_size)
        return xt * mask, z * mask  # 2 (B, n_feats, out_size)

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, t, spk)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(
                    z.shape, dtype=z.dtype, device=z.device, requires_grad=False
                )
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, spk))
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk)

    def loss_t(self, x0, mask, mu, t, spk=None):
        xt, z = self.forward_diffusion(x0, mask, mu, t)  # 2 (B, n_feats, out_size)
        time = t.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        cum_noise = get_noise(
            time, self.beta_min, self.beta_max, cumulative=True
        )  # (B, 1, 1)
        noise_estimation = self.estimator(xt, mask, mu, t, spk)
        noise_estimation *= torch.sqrt(
            1.0 - torch.exp(-cum_noise)
        )  # (B, n_feats, out_size) * std_lambda
        loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(mask) * self.n_feats)
        return loss, xt

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        t = torch.rand(
            x0.shape[0], dtype=x0.dtype, device=x0.device, requires_grad=False
        )
        t = torch.clamp(t, offset, 1.0 - offset)  # (B,)
        return self.loss_t(x0, mask, mu, t, spk)
