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

from model_ms.base import BaseModule
from model_ms.phnm_encoder import IpaTraitEncoder
from model_ms.spk_encoder import SpeakerEncodingLayer
from model_ms.diffusion import Diffusion
from model_ms.utils import (
    sequence_mask,
    fix_len_compatibility,
    generate_path,
)


class GradTTArtic(BaseModule):
    """Generative model for articulatory trajectories based on aligned phonemes input.

    Inspired by Grad-TTS paper with 3 major differences:
        1. Uses fixed IPA-ternary phonological traits features phoneme input instead of text input (that is
            partly phonemized to use learnt embeddings on a vocabulary of ARPabet phonemes and latin letters).
            This allow us to expect better generalization to unseen languages.
            This is also a limitation of the model, since it requires phonemes to be provided.
        2. Uses aligned phonemes input to predict articulatory trajectories. Here we bypass the difficulty
            of generating durations that is too much dependent on speaker/language.
            This is also a limitation of the model, since it requires phoneme durations to be provided.
        3. Predicts articulatory trajectories instead of mel-spectrograms.
           Articulatory trajectories are lower dimensional and more interpretable than mel-spectrograms.
           There are also deemed to be universal across languages and speakers.
    """

    def __init__(
        self,
        n_ipa_feats,  # 26, 24 phonological traits + 1 silence dim + 1 phoneme repetition count
        spk_emb_dim,  # 64
        n_enc_channels,  # 192
        filter_channels,  # 768
        filter_channels_dp,  # 256
        n_heads,  # 2
        n_enc_layers,  # 6
        enc_kernel,  # 3
        enc_dropout,  # 0.1
        window_size,  # 4
        n_feats,  # 16 (articulatory features)
        dec_dim,  # 64
        beta_min,  # 0.05
        beta_max,  # 20.0
        pe_scale,  # 1000
        spk_preemb_dim=1024,  # Similar to sparc method from an SSL model
    ):
        super(GradTTArtic, self).__init__()
        self.n_ipa_feats = n_ipa_feats
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

        self.spk_enc = SpeakerEncodingLayer(spk_preemb_dim, spk_emb_dim)

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
            n_feats, dec_dim, spk_emb_dim, beta_min, beta_max, pe_scale
        )

    @torch.no_grad()
    def forward(
        self,
        x,
        x_lengths,
        spk_feats,
        n_timesteps,
        temperature=1.0,
        stoc=False,
        length_scale=1.0,
    ):
        """
        Generates articulatory trajectories from phonemes.

        Returns:
            1. encoder outputs
            2. decoder outputs

        Args:
            x (torch.Tensor): (B, 26, T_x) batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            spk_feats (torch.Tensor): (B, spk_preemb_dim) batch of speaker features.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        spk = self.spk_enc(spk_feats)

        # Get encoder_outputs `mu_x`
        mu_x, x_mask = self.encoder(x, x_lengths, spk)  # (B, n_feats, T_x), (B, 1, T_x)

        # Adjust durations
        x_durations = x[:, -1, :].long()  # (B, T_x) phoneme repetition counts
        x_durations = x_durations.unsqueeze(1) * x_mask  # (B, 1, T_x)
        w_ceil = x_durations * length_scale  # (B, 1, T_x) upper int rounding
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

    def compute_loss(self, x, x_lengths, y, y_lengths, spk_feats, out_size=None):
        """
        Computes 2 losses:
            1. prior loss: loss between mel-spectrogram and encoder outputs.
            2. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.

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

        spk = self.spk_enc(spk_feats)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, x_mask = self.encoder(x, x_lengths, spk)  # (B, n_feats, T_x), (B, 1, T_x)

        x_durations = x[:, -1, :].long()  # (B, T_x) phoneme repetition counts
        w_ceil = x_durations.unsqueeze(1) * x_mask  # (B, 1, T_x)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()  # (B,)
        y_max_length = y.shape[-1]  # T_y

        y_mask = (
            sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        )  # (B, 1, T_y)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)  # (B,1, T_x, T_y)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(
            1
        )  # (B, 1, T_x, T_y)
        attn = attn.squeeze(1).detach()  # (B, T_x, T_y)

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
            # y_cut_mask = (
            #    sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            # )  # (B, 1, max_y_cut_length)
            y_cut_mask = (
                sequence_mask(y_cut_lengths, out_size).unsqueeze(1).to(y_mask)
            )  # (B, 1, out_size)

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

        return prior_loss, diff_loss


class ArticTTS(BaseModule):
    def __init__(
        self,
        generator_ckpt=None,
        generator_configs=None,
    ):
        super(ArticTTS, self).__init__()
