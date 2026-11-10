# -*- coding: utf-8 -*-
"""Standalone DAC (descript-audio-codec) codec for Parler-TTS.

Ported from the vendored ``dac/`` package with the ``audiotools`` dependency
removed (only the decode path is needed for TTS).  Reuses the vendored
``dac.nn.layers`` / ``dac.nn.quantize`` building blocks (torch + einops only),
and reproduces the DACModel wrapper's weight-norm handling so the checkpoint's
``audio_encoder.model.*`` keys (with ``weight_g``/``weight_v``) load strict.
"""
import math

import numpy as np
import torch
from torch import nn

from .dac.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d
from .dac.nn.quantize import ResidualVectorQuantize


def _init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride,
                     padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(input_dim, output_dim, kernel_size=2 * stride,
                              stride=stride, padding=math.ceil(stride / 2)),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, input_channel, channels, rates, d_out: int = 1):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(rates):
            input_dim = channels // 2 ** i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DAC(nn.Module):
    """Architecture-only copy of dac.model.dac.DAC (no audiotools/BaseModel)."""

    def __init__(self, encoder_dim: int = 64, encoder_rates=(2, 4, 8, 8),
                 latent_dim=None, decoder_dim: int = 1536, decoder_rates=(8, 8, 4, 2),
                 n_codebooks: int = 9, codebook_size: int = 1024,
                 codebook_dim=8, quantizer_dropout: bool = False, sample_rate: int = 44100):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.encoder_rates = list(encoder_rates)
        self.decoder_dim = decoder_dim
        self.decoder_rates = list(decoder_rates)
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = int(np.prod(self.encoder_rates))

        self.encoder = _Encoder(encoder_dim, self.encoder_rates, latent_dim)
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim, n_codebooks=n_codebooks, codebook_size=codebook_size,
            codebook_dim=codebook_dim, quantizer_dropout=quantizer_dropout)
        self.decoder = Decoder(latent_dim, decoder_dim, self.decoder_rates)
        self.apply(_init_weights)

    def decode(self, z: torch.Tensor):
        return self.decoder(z)


class _Encoder(nn.Module):
    """Only constructed so that ``audio_encoder.model.encoder.*`` keys load
    strict; never used during TTS inference."""

    def __init__(self, d_model: int = 64, strides=(2, 4, 8, 8), d_latent: int = 64):
        super().__init__()
        block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block += [EncoderBlock(d_model, stride=stride)]
        block += [Snake1d(d_model), WNConv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class DACModelLite(nn.Module):
    """Key-compatible with ``audio_encoder.*`` of the Parler-TTS checkpoint."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.model = DAC(
            n_codebooks=cfg["num_codebooks"],
            latent_dim=cfg["latent_dim"],
            codebook_size=cfg["codebook_size"],
        )
        # reproduce DACModel's weight-norm handling so state-dict keys match
        # (Conv1d weights become weight_g / weight_v pairs)
        self.remove_weight_norm()
        self.apply_weight_norm()

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        def _apply(module):
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                weight_norm(module)
        self.apply(_apply)

    def remove_weight_norm(self):
        def _remove(module):
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.utils.remove_weight_norm(module)
        self.apply(_remove)

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (bsz, num_codebooks, T) with values < codebook_size.
        Returns audio (bsz, 1, samples) in [-1, 1]."""
        z_q, _, _ = self.model.quantizer.from_codes(codes)
        return self.model.decode(z_q)
