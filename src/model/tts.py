# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch

from model import monotonic_align
from model.base import BaseModule
from model.text_encoder import IpaTraitEncoder, TextEncoder
from model.diffusion import Diffusion
from model.utils import (
    sequence_mask,
    generate_path,
    duration_loss,
    fix_len_compatibility,
)


class ArtTTS(BaseModule):
    def __init__(
        self,
        n_ipa_feats,
        n_spks,
        spk_emb_dim,
        n_enc_channels,
        filter_channels,
        filter_channels_dp,
        n_heads,
        n_enc_layers,
        enc_kernel,
        enc_dropout,
        window_size,
        n_feats,
        dec_dim,
        beta_min,
        beta_max,
        pe_scale,
    ):
        super(ArtTTS, self).__init__()
        self.n_ipa_feats = n_ipa_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = IpaTraitEncoder(
            n_ipa_feats,
            n_feats,
            n_enc_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_enc_layers,
            enc_kernel,
            enc_dropout,
            window_size,
        )
        self.decoder = Diffusion(
            n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale
        )

    @torch.no_grad()
    def forward(
        self,
        x,
        x_lengths,
        n_timesteps,
        temperature=1.0,
        stoc=False,
        spk=None,
        length_scale=1.0,
        x_durations=None,
    ):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(
            x, x_lengths, spk
        )  # (B, n_feats, T_x), (B, 1, T_x), (B, 1, T_x)

        # If `x_durations` is provided, use it to align text and mel-spectrogram
        if not isinstance(x_durations, type(None)):
            # `x_durations` is a tensor of shape (B, T_x) with durations for each phoneme
            ## PB WITH GETTING EXACTLY THE SAME SIZE AS PHNM EMB... (diphtongues, phnm with various emb lengths...)
            x_durations = self.relocate_input([x_durations])[0]
            w = x_durations.unsqueeze(1) * x_mask  # (B, 1, T_x)
        else:
            w = torch.exp(logw) * x_mask  # (B, 1, T_x)

        w_ceil = torch.ceil(w) * length_scale  # (B, 1, T_x) upper int rounding
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()  # (B,)
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(
            y_max_length
        )  # lower int rounding to be divisible by 2^{num of UNet downsamplings}

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = (
            sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        )  # (B, 1, y_max_length_)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(
            2
        )  # (B, 1, T_x, y_max_length_)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(
            1
        )  # (B, 1, T_x, y_max_length_)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)
        )  # (B, y_max_length_, n_feats)
        mu_y = mu_y.transpose(1, 2)  # (B, n_feats, y_max_length_)
        encoder_outputs = mu_y[:, :, :y_max_length]  # (B, n_feats, y_max_length)

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = (
            mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        )  # (B, n_feats, y_max_length_)
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])
        # shape: (B, n_ipa_feats, T_x), (B,), (B, n_feats, T_y), (B,)
        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(
            x, x_lengths, spk
        )  # (B, n_feats, T_x), (B, 1, T_x), (B, 1, T_x)
        y_max_length = y.shape[-1]  # T_y

        y_mask = (
            sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        )  # (B, 1, T_y)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)  # (B,1, T_x, T_y)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(
                mu_x.shape, dtype=mu_x.dtype, device=mu_x.device
            )  # (B, n_feats, T_x)
            y_square = torch.matmul(factor.transpose(1, 2), y**2)  # (B, T_x)
            y_mu_double = torch.matmul(
                2.0 * (factor * mu_x).transpose(1, 2), y
            )  # (B, T_x, T_y)
            mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)  # (B, T_x, 1)
            log_prior = y_square - y_mu_double + mu_square + const  # (B, T_x, T_y)

            attn = monotonic_align.maximum_path(
                log_prior, attn_mask.squeeze(1)
            )  # (B, T_x, T_y)?
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = (
            torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        )  # (B, 1, T_x)
        dur_loss = duration_loss(logw, logw_, x_lengths)  # (1,)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)  # (B,)
            offset_ranges = list(
                zip([0] * max_offset.shape[0], max_offset.cpu().numpy())
            )  # (B, 2)
            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(y_lengths)  # (B,)

            attn_cut = torch.zeros(
                attn.shape[0],
                attn.shape[1],
                out_size,
                dtype=attn.dtype,
                device=attn.device,
            )  # (B, T_x, out_size)
            y_cut = torch.zeros(
                y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device
            )  # (B, n_feats, out_size)
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)  # (B,)
            ########## modif ##########
            y_cut_mask = (
                sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            )  # (B, 1, max_y_cut_length)
            # y_cut_mask = sequence_mask(y_cut_lengths, out_size).unsqueeze(1).to(y_mask)   # (B, 1, out_size)

            attn = attn_cut  # (B, T_x, out_size)
            y = y_cut  # (B, n_feats, out_size)
            y_mask = y_cut_mask  # (B, 1, max_y_cut_length)

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)
        )  # (B, out_size, n_feats)
        mu_y = mu_y.transpose(1, 2)  # (B, n_feats, out_size)

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)
        # (B, n_feats, out_size), (B, n_feats, out_size), (B, n_feats, out_size), ...

        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        return dur_loss, prior_loss, diff_loss

    @torch.no_grad()
    def lengths_pred(
        self,
        x,
        x_lengths,
        spk=None,
    ):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(
            x, x_lengths, spk
        )  # (B, n_feats, T_x), (B, 1, T_x), (B, 1, T_x)
        w = torch.exp(logw) * x_mask  # (B, 1, T_x)
        return w


class GradTTS(BaseModule):
    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        n_enc_channels,
        filter_channels,
        filter_channels_dp,
        n_heads,
        n_enc_layers,
        enc_kernel,
        enc_dropout,
        window_size,
        n_feats,
        dec_dim,
        beta_min,
        beta_max,
        pe_scale,
    ):
        super(GradTTS, self).__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = TextEncoder(
            n_vocab,
            n_feats,
            n_enc_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_enc_layers,
            enc_kernel,
            enc_dropout,
            window_size,
        )
        self.decoder = Diffusion(
            n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale
        )

    @torch.no_grad()
    def forward(
        self,
        x,
        x_lengths,
        n_timesteps,
        temperature=1.0,
        stoc=False,
        spk=None,
        length_scale=1.0,
    ):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(
            x, x_lengths, spk
        )  # (B, n_feats, T_x), (B, 1, T_x), (B, 1, T_x)

        w = torch.exp(logw) * x_mask  # (B, 1, T_x)

        w_ceil = torch.ceil(w) * length_scale  # (B, 1, T_x) upper int rounding
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()  # (B,)
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(
            y_max_length
        )  # lower int rounding to be divisible by 2^{num of UNet downsamplings}

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = (
            sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        )  # (B, 1, y_max_length_)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(
            2
        )  # (B, 1, T_x, y_max_length_)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(
            1
        )  # (B, 1, T_x, y_max_length_)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)
        )  # (B, y_max_length_, n_feats)
        mu_y = mu_y.transpose(1, 2)  # (B, n_feats, y_max_length_)
        encoder_outputs = mu_y[:, :, :y_max_length]  # (B, n_feats, y_max_length)

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = (
            mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        )  # (B, n_feats, y_max_length_)
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])
        # shape: (B, n_vocab, T_x), (B,), (B, n_feats, T_y), (B,)
        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(
            x, x_lengths, spk
        )  # (B, n_feats, T_x), (B, 1, T_x), (B, 1, T_x)
        y_max_length = y.shape[-1]  # T_y

        y_mask = (
            sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        )  # (B, 1, T_y)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)  # (B,1, T_x, T_y)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(
                mu_x.shape, dtype=mu_x.dtype, device=mu_x.device
            )  # (B, n_feats, T_x)
            y_square = torch.matmul(factor.transpose(1, 2), y**2)  # (B, T_x)
            y_mu_double = torch.matmul(
                2.0 * (factor * mu_x).transpose(1, 2), y
            )  # (B, T_x, T_y)
            mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)  # (B, T_x, 1)
            log_prior = y_square - y_mu_double + mu_square + const  # (B, T_x, T_y)

            attn = monotonic_align.maximum_path(
                log_prior, attn_mask.squeeze(1)
            )  # (B, T_x, T_y)?
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = (
            torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        )  # (B, 1, T_x)
        dur_loss = duration_loss(logw, logw_, x_lengths)  # (1,)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)  # (B,)
            offset_ranges = list(
                zip([0] * max_offset.shape[0], max_offset.cpu().numpy())
            )  # (B, 2)
            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(y_lengths)  # (B,)

            attn_cut = torch.zeros(
                attn.shape[0],
                attn.shape[1],
                out_size,
                dtype=attn.dtype,
                device=attn.device,
            )  # (B, T_x, out_size)
            y_cut = torch.zeros(
                y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device
            )  # (B, n_feats, out_size)
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)  # (B,)

            y_cut_mask = (
                sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            )  # (B, 1, max_y_cut_length)
            # y_cut_mask = sequence_mask(y_cut_lengths, out_size).unsqueeze(1).to(y_mask)   # (B, 1, out_size)

            attn = attn_cut  # (B, T_x, out_size)
            y = y_cut  # (B, n_feats, out_size)
            y_mask = y_cut_mask  # (B, 1, max_y_cut_length)

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)
        )  # (B, out_size, n_feats)
        mu_y = mu_y.transpose(1, 2)  # (B, n_feats, out_size)

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)
        # (B, n_feats, out_size), (B, n_feats, out_size), (B, n_feats, out_size), ...

        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        return dur_loss, prior_loss, diff_loss

    @torch.no_grad()
    def lengths_pred(
        self,
        x,
        x_lengths,
        spk=None,
    ):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(
            x, x_lengths, spk
        )  # (B, n_feats, T_x), (B, 1, T_x), (B, 1, T_x)
        w = torch.exp(logw) * x_mask  # (B, 1, T_x)
        return w
