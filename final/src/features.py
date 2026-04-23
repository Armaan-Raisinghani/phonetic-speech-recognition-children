from __future__ import annotations

import math

import numpy as np
import soundfile as sf
import torch
import torchaudio


class LogMelFrontend(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: float = 20.0,
        f_max: float = 7600.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.mel(waveform)
        spec = self.to_db(spec)
        # [1, n_mels, time] -> [time, n_mels]
        feat = spec.squeeze(0).transpose(0, 1).contiguous()
        feat = cmvn(feat)
        return feat


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.size(0) == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def loudness_normalize(waveform: torch.Tensor, target_dbfs: float = -20.0) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(waveform**2) + 1e-8)
    current_dbfs = 20.0 * torch.log10(rms + 1e-8)
    gain_db = target_dbfs - float(current_dbfs)
    gain = 10.0 ** (gain_db / 20.0)
    out = waveform * gain
    return torch.clamp(out, min=-1.0, max=1.0)


def load_audio(path: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.asarray(samples).T)
    waveform = to_mono(waveform)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    waveform = loudness_normalize(waveform)
    return waveform, sr


def cmvn(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = feat.mean(dim=0, keepdim=True)
    std = feat.std(dim=0, keepdim=True)
    return (feat - mean) / torch.clamp(std, min=eps)


def infer_input_lengths(raw_num_samples: torch.Tensor, hop_length: int = 160) -> torch.Tensor:
    # Approximate frame count after STFT hop processing.
    return torch.ceil(raw_num_samples.float() / hop_length).long()


def time_mask_param(max_time_frames: int, ratio: float = 0.06) -> int:
    return max(1, int(math.ceil(max_time_frames * ratio)))
