from __future__ import annotations

import random

import torch
import torchaudio

from src.features import time_mask_param


def random_gain(waveform: torch.Tensor, min_db: float = -3.0, max_db: float = 3.0) -> torch.Tensor:
    gain_db = random.uniform(min_db, max_db)
    gain = 10.0 ** (gain_db / 20.0)
    out = waveform * gain
    return torch.clamp(out, min=-1.0, max=1.0)


def random_noise(waveform: torch.Tensor, max_std: float = 0.003) -> torch.Tensor:
    std = random.uniform(0.0, max_std)
    return torch.clamp(waveform + torch.randn_like(waveform) * std, min=-1.0, max=1.0)


def random_time_stretch(waveform: torch.Tensor, min_rate: float = 0.95, max_rate: float = 1.05) -> torch.Tensor:
    rate = random.uniform(min_rate, max_rate)
    if abs(rate - 1.0) < 1e-3:
        return waveform
    length = waveform.size(-1)
    new_len = max(8, int(length / rate))
    stretched = torch.nn.functional.interpolate(
        waveform.unsqueeze(0), size=new_len, mode="linear", align_corners=False
    ).squeeze(0)
    if new_len > length:
        return stretched[..., :length]
    if new_len < length:
        pad = torch.zeros(waveform.size(0), length - new_len, device=waveform.device)
        return torch.cat([stretched, pad], dim=-1)
    return stretched


class SpecAugment(torch.nn.Module):
    def __init__(self, freq_mask: int = 8, time_mask_ratio: float = 0.06, n_time_masks: int = 2) -> None:
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask)
        self.time_mask_ratio = time_mask_ratio
        self.n_time_masks = n_time_masks

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [T, F]
        x = feat.transpose(0, 1).unsqueeze(0)
        x = self.freq_mask(x)
        tmask = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param(feat.size(0), self.time_mask_ratio)
        )
        for _ in range(self.n_time_masks):
            x = tmask(x)
        return x.squeeze(0).transpose(0, 1).contiguous()


def maybe_wave_augment(waveform: torch.Tensor, p: float = 0.7) -> torch.Tensor:
    if random.random() > p:
        return waveform
    waveform = random_gain(waveform)
    waveform = random_noise(waveform)
    waveform = random_time_stretch(waveform)
    return waveform
