#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.train_cldnn_ctc import (
    DurationBucketBatchSampler,
    SpecAugment,
    batch_ctc_loss,
    dataloader_worker_kwargs,
    evaluate_and_write_report,
    load_vocab,
    parse_int_list,
    set_seed,
    trainable_parameter_summary,
    write_phoneme_plots,
    write_training_plots,
)
from src.dataset import PhonemeDataset, ctc_collate
from src.dataset import resolve_audio_path
from src.features import load_audio
from src.text import encode_text


class LegacyCheckpointStub:
    def __init__(self, *args, **kwargs) -> None:
        self.__dict__.update(kwargs)


def install_deepspeech_pytorch_unpickle_stubs() -> None:
    def ensure_module(name: str, package: bool = False) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        module = types.ModuleType(name)
        if package:
            module.__path__ = []
        sys.modules[name] = module
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            setattr(ensure_module(parent_name, package=True), child_name, module)
        return module

    for name in [
        "deepspeech_pytorch",
        "deepspeech_pytorch.configs",
        "hydra_configs",
        "hydra_configs.deepspeech_pytorch",
        "hydra_configs.deepspeech_pytorch.configs",
        "hydra_configs.pytorch_lightning",
        "hydra_configs.pytorch_lightning.callbacks",
        "hydra_configs.pytorch_lightning.trainer",
        "pytorch_lightning",
    ]:
        ensure_module(name, package=True)

    class_names = [
        "CheckpointHandler",
        "FileCheckpointHandler",
        "SpectConfig",
        "AugmentationConfig",
        "DataConfig",
        "BiDirectionalConfig",
        "UniDirectionalConfig",
        "OptimConfig",
        "SGDConfig",
        "AdamConfig",
        "DeepSpeechTrainerConf",
        "DeepSpeechConfig",
        "TrainerConf",
        "ModelCheckpointConf",
        "ModelCheckpoint",
        "RNNType",
        "SpectrogramWindow",
        "TensorBoardLoggerConf",
    ]
    modules = [
        "deepspeech_pytorch.checkpoint",
        "deepspeech_pytorch.configs.train_config",
        "deepspeech_pytorch.configs.lightning_config",
        "deepspeech_pytorch.enums",
        "hydra_configs.deepspeech_pytorch.configs.train_config",
        "hydra_configs.deepspeech_pytorch.configs.lightning_config",
        "hydra_configs.pytorch_lightning.callbacks",
        "hydra_configs.pytorch_lightning.callbacks.model_checkpoint",
        "hydra_configs.pytorch_lightning.trainer",
        "hydra_configs.pytorch_lightning.configs.trainer",
        "pytorch_lightning.callbacks",
    ]
    for module_name in modules:
        module = ensure_module(module_name, package=True)
        for class_name in class_names:
            if not hasattr(module, class_name):
                setattr(module, class_name, type(class_name, (LegacyCheckpointStub,), {}))
    ensure_module("pytorch_lightning", package=True).callbacks = ensure_module("pytorch_lightning.callbacks")


def torch_load_legacy_checkpoint(path: Path) -> dict:
    install_deepspeech_pytorch_unpickle_stubs()
    return torch.load(path, map_location="cpu", weights_only=False)


class SequenceWise(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, feat = x.size()
        x = x.contiguous().view(batch * time, feat)
        x = self.module(x)
        return x.view(batch, time, -1)


class SourceSequenceWise(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, batch = x.size(0), x.size(1)
        x = x.contiguous().view(time * batch, -1)
        x = self.module(x)
        return x.view(time, batch, -1)


class SourceBatchRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, batch_norm: bool) -> None:
        super().__init__()
        self.batch_norm = SourceSequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            bias=True,
        )

    def forward(self, x: torch.Tensor, output_lengths: torch.Tensor) -> torch.Tensor:
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, output_lengths.cpu(), enforce_sorted=False)
        x, _ = self.rnn(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x


class SourceDeepSpeech2PhoneCTC(nn.Module):
    """Shape-compatible with SeanNaren/deepspeech.pytorch LibriSpeech V3 checkpoints."""

    def __init__(self, vocab_size: int, hidden_dim: int, rnn_layers: int, blank_id: int, blank_bias_init: float) -> None:
        super().__init__()
        if hidden_dim != 1024:
            raise ValueError("Source-compatible DeepSpeech2 checkpoint requires --hidden-dim 1024")
        if rnn_layers != 5:
            raise ValueError("Source-compatible DeepSpeech2 checkpoint requires --rnn-layers 5")
        self.conv = nn.Module()
        self.conv.seq_module = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0.0, 20.0, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0.0, 20.0, inplace=True),
        )
        self.rnns = nn.ModuleList(
            [SourceBatchRNN(input_size=1312, hidden_size=hidden_dim, batch_norm=False)]
            + [SourceBatchRNN(input_size=hidden_dim, hidden_size=hidden_dim, batch_norm=True) for _ in range(4)]
        )
        fully_connected = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, vocab_size, bias=False),
        )
        self.fc = nn.Sequential(SourceSequenceWise(fully_connected))
        self.blank_id = int(blank_id)
        self.time_reduction = 2
        self.init_classifier_bias(blank_bias_init)

    def init_classifier_bias(self, blank_bias_init: float) -> None:
        # The source architecture uses a bias-free classifier, so this is a no-op kept for shared loader code.
        return None

    def get_seq_lens(self, input_lens: torch.Tensor) -> torch.Tensor:
        lengths = input_lens
        for module in self.conv.seq_module:
            if isinstance(module, nn.Conv2d):
                lengths = (
                    (lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1)
                    // module.stride[1]
                    + 1
                )
        return torch.clamp(lengths.int(), min=1)

    def forward(self, x: torch.Tensor, input_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [B, T, F] -> [B, 1, F, T], exactly like deepspeech.pytorch.
        x = x.transpose(1, 2).unsqueeze(1)
        out_lens = self.get_seq_lens(input_lens.cpu()).to(input_lens.device)
        x = self.conv.seq_module(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        for rnn in self.rnns:
            x = rnn(x, out_lens)
        x = self.fc(x).transpose(0, 1)
        return x, out_lens


class DeepSpeechSpectrogramFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        window_size: float = 0.02,
        window_stride: float = 0.01,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = int(sample_rate * window_size)
        self.win_length = self.n_fft
        self.hop_length = int(sample_rate * window_stride)
        self.register_buffer("window", torch.hamming_window(self.win_length), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            waveform.squeeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        ).abs()
        spec = torch.log1p(spec)
        spec = (spec - spec.mean()) / torch.clamp(spec.std(), min=1e-6)
        return spec.transpose(0, 1).contiguous()


class DeepSpeechManifestDataset(PhonemeDataset):
    def __init__(self, manifest_path: Path, stoi: dict[str, int], train: bool = False, use_specaugment: bool = False) -> None:
        self.rows = self._read_jsonl(manifest_path)
        self.stoi = stoi
        self.train = train
        self.frontend = DeepSpeechSpectrogramFrontend()
        self.specaug = SpecAugment() if train and use_specaugment else None

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        waveform, _ = load_audio(resolve_audio_path(row))
        feat = self.frontend(waveform)
        if self.specaug is not None:
            feat = self.specaug(feat)
        text = row.get("phonetic_text", "")
        target = torch.tensor(encode_text(text, self.stoi), dtype=torch.long)
        return {
            "utt_id": row["utterance_id"],
            "feature": feat,
            "feature_len": torch.tensor(feat.size(0), dtype=torch.long),
            "target": target,
            "target_len": torch.tensor(target.numel(), dtype=torch.long),
            "label_weight": torch.tensor(float(row.get("label_weight", 1.0)), dtype=torch.float32),
            "text": text,
        }


class DeepSpeech2ConvFrontend(nn.Module):
    def __init__(
        self,
        n_mels: int,
        conv_channels: list[int],
        conv_time_strides: list[int],
        conv_freq_strides: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if not (len(conv_channels) == len(conv_time_strides) == len(conv_freq_strides)):
            raise ValueError("conv channels/time strides/frequency strides must have equal lengths")

        layers: list[nn.Module] = []
        in_ch = 1
        self.time_reduction = 1
        self.time_kernels: list[int] = []
        self.time_strides: list[int] = []
        self.time_paddings: list[int] = []
        for layer_idx, (out_ch, t_stride, f_stride) in enumerate(
            zip(conv_channels, conv_time_strides, conv_freq_strides)
        ):
            kernel = (11, 41) if layer_idx == 0 else (11, 21)
            padding = (kernel[0] // 2, kernel[1] // 2)
            self.time_kernels.append(kernel[0])
            self.time_strides.append(int(t_stride))
            self.time_paddings.append(padding[0])
            layers.extend(
                [
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=kernel,
                        stride=(t_stride, f_stride),
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.Hardtanh(0.0, 20.0, inplace=True),
                    nn.Dropout2d(dropout),
                ]
            )
            self.time_reduction *= int(t_stride)
            in_ch = out_ch

        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, n_mels)
            out = self.net(dummy)
            self.out_dim = int(out.size(1) * out.size(3))

    def output_lengths(self, input_lens: torch.Tensor) -> torch.Tensor:
        lengths = input_lens
        for kernel, stride, padding in zip(self.time_kernels, self.time_strides, self.time_paddings):
            lengths = torch.div(lengths + 2 * padding - (kernel - 1) - 1, stride, rounding_mode="floor") + 1
        return torch.clamp(lengths, min=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, F] -> [B, C, T', F'] -> [B, T', C * F']
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.transpose(1, 2).contiguous()
        return x.flatten(start_dim=2)


class DeepSpeech2PhoneCTC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_mels: int,
        conv_channels: list[int],
        conv_time_strides: list[int],
        conv_freq_strides: list[int],
        conv_dropout: float,
        rnn_type: str,
        hidden_dim: int,
        rnn_layers: int,
        rnn_dropout: float,
        lookahead_context: int,
        proj_dim: int,
        blank_id: int,
        blank_bias_init: float,
    ) -> None:
        super().__init__()
        self.frontend = DeepSpeech2ConvFrontend(
            n_mels=n_mels,
            conv_channels=conv_channels,
            conv_time_strides=conv_time_strides,
            conv_freq_strides=conv_freq_strides,
            dropout=conv_dropout,
        )
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=self.frontend.out_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout if rnn_layers > 1 else 0.0,
        )
        rnn_out_dim = hidden_dim * 2
        self.sequence_norm = SequenceWise(nn.BatchNorm1d(rnn_out_dim))
        self.lookahead = (
            nn.Conv1d(rnn_out_dim, rnn_out_dim, kernel_size=lookahead_context, groups=rnn_out_dim, padding=0)
            if lookahead_context > 0
            else None
        )
        self.proj = nn.Sequential(
            SequenceWise(nn.Linear(rnn_out_dim, proj_dim)),
            nn.Hardtanh(0.0, 20.0, inplace=True),
            nn.Dropout(rnn_dropout),
        )
        self.classifier = nn.Linear(proj_dim, vocab_size)
        self.blank_id = int(blank_id)
        self.time_reduction = int(self.frontend.time_reduction)
        self.init_classifier_bias(blank_bias_init)

    def init_classifier_bias(self, blank_bias_init: float) -> None:
        nn.init.zeros_(self.classifier.bias)
        self.classifier.bias.data[self.blank_id] = float(blank_bias_init)

    def forward(self, x: torch.Tensor, input_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.frontend(x)
        h, _ = self.rnn(h)
        h = self.sequence_norm(h)
        if self.lookahead is not None:
            # Depthwise lookahead convolution over time. Left-pad so output length stays unchanged.
            conv_in = h.transpose(1, 2)
            pad = self.lookahead.kernel_size[0] - 1
            conv_in = torch.nn.functional.pad(conv_in, (pad, 0))
            h = self.lookahead(conv_in).transpose(1, 2)
        logits = self.classifier(self.proj(h))
        out_lens = self.frontend.output_lengths(input_lens)
        out_lens = torch.clamp(out_lens, max=logits.size(1))
        return logits, out_lens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Native PyTorch DeepSpeech2 transfer-learning trainer for child IPA CTC. "
            "Implements the doc idea of dropping output/top source layers for a new alphabet."
        )
    )
    p.add_argument("--train-manifest", type=Path, default=Path("data/manifests/train.jsonl"))
    p.add_argument("--val-manifest", type=Path, default=Path("data/manifests/val.jsonl"))
    p.add_argument("--test-manifest", type=Path, default=Path("data/manifests/test.jsonl"))
    p.add_argument("--vocab", type=Path, default=Path("data/manifests/vocab.json"))
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/pytorch_deepspeech2_transfer"))
    p.add_argument(
        "--source-compatible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the exact SeanNaren/deepspeech.pytorch architecture and spectrogram features.",
    )

    p.add_argument("--init-checkpoint", type=Path, default=None)
    p.add_argument(
        "--drop-source-layers",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="DeepSpeech-style transfer: 1 drops classifier, 2 drops classifier+projection, 3 drops classifier+projection+RNN.",
    )
    p.add_argument("--freeze-encoder-epochs", type=int, default=1)

    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--loader-prefetch-factor", type=int, default=4)
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--torch-num-inter-op-threads", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--new-layer-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--conv-channels", type=parse_int_list, default=[32, 32])
    p.add_argument("--conv-time-strides", type=parse_int_list, default=[2, 1])
    p.add_argument("--conv-freq-strides", type=parse_int_list, default=[2, 2])
    p.add_argument("--conv-dropout", type=float, default=0.05)
    p.add_argument("--rnn-type", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--rnn-layers", type=int, default=5)
    p.add_argument("--rnn-dropout", type=float, default=0.20)
    p.add_argument("--lookahead-context", type=int, default=0)
    p.add_argument("--proj-dim", type=int, default=1024)
    p.add_argument("--blank-bias-init", type=float, default=-2.0)

    p.add_argument("--decode-strategy", choices=["greedy", "beam"], default="greedy")
    p.add_argument("--beam-size", type=int, default=8)
    p.add_argument("--decode-blank-penalty", type=float, default=0.0)
    p.add_argument("--selection-metric", choices=["cer", "per"], default="cer")
    p.add_argument("--phoneme-top-k", type=int, default=25)
    p.add_argument("--phoneme-min-ref-count", type=int, default=5)
    p.add_argument("--use-specaugment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--specaugment-warmup-epochs", type=int, default=2)
    p.add_argument("--sortagrad-epochs", type=int, default=1)
    p.add_argument("--test-each-best", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    return p.parse_args()


def source_skip_prefixes(drop_source_layers: int) -> tuple[str, ...]:
    prefixes: list[str] = []
    if drop_source_layers >= 1:
        prefixes.append("classifier.")
        prefixes.append("fc.0.module.1.")
    if drop_source_layers >= 2:
        prefixes.append("proj.")
        prefixes.append("fc.")
    if drop_source_layers >= 3:
        prefixes.append("rnn.")
        prefixes.append("rnns.")
        prefixes.append("sequence_norm.")
        prefixes.append("lookahead.")
    return tuple(prefixes)


def load_transfer_checkpoint(
    model: nn.Module,
    checkpoint_path: Path | None,
    drop_source_layers: int,
    blank_bias_init: float,
) -> dict[str, object]:
    if checkpoint_path is None:
        return {
            "checkpoint_path": None,
            "loaded_tensors": 0,
            "skipped_tensor_count": 0,
            "drop_source_layers": drop_source_layers,
            "note": "No source checkpoint supplied; training PyTorch DeepSpeech2 from random initialization.",
        }
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing init checkpoint: {checkpoint_path}")

    checkpoint = torch_load_legacy_checkpoint(checkpoint_path)
    source_state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    if any(key.startswith("module.") for key in source_state):
        source_state = {key.removeprefix("module."): value for key, value in source_state.items()}

    skip_prefixes = source_skip_prefixes(drop_source_layers)
    target_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[dict[str, object]] = []
    for key, tensor in source_state.items():
        if key.startswith(skip_prefixes):
            skipped.append({"key": key, "reason": "dropped_source_layer"})
            continue
        if key not in target_state:
            skipped.append({"key": key, "reason": "missing_in_target"})
            continue
        if tuple(tensor.shape) != tuple(target_state[key].shape):
            skipped.append(
                {
                    "key": key,
                    "reason": "shape_mismatch",
                    "source_shape": list(tensor.shape),
                    "target_shape": list(target_state[key].shape),
                }
            )
            continue
        compatible[key] = tensor

    target_state.update(compatible)
    model.load_state_dict(target_state)
    if drop_source_layers >= 1 and hasattr(model, "init_classifier_bias"):
        model.init_classifier_bias(blank_bias_init)

    return {
        "checkpoint_path": str(checkpoint_path),
        "source_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "source_val_cer": checkpoint.get("val_cer") if isinstance(checkpoint, dict) else None,
        "loaded_tensors": len(compatible),
        "skipped_tensors": skipped[:80],
        "skipped_tensor_count": len(skipped),
        "drop_source_layers": drop_source_layers,
    }


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    if isinstance(model, SourceDeepSpeech2PhoneCTC):
        modules = [model.conv, model.rnns]
    else:
        modules = [model.frontend, model.rnn, model.sequence_norm]
        if model.lookahead is not None:
            modules.append(model.lookahead)
    for module in modules:
        for param in module.parameters():
            param.requires_grad = trainable


def configure_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if isinstance(model, SourceDeepSpeech2PhoneCTC):
        new_params = list(model.fc.parameters())
    else:
        new_params = list(model.classifier.parameters()) + list(model.proj.parameters())
    new_param_ids = {id(param) for param in new_params}
    base_params = [param for param in model.parameters() if param.requires_grad and id(param) not in new_param_ids]
    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": args.lr})
    param_groups.append({"params": [p for p in new_params if p.requires_grad], "lr": args.new_layer_lr})
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def make_train_loader(args: argparse.Namespace, train_ds: PhonemeDataset, device: torch.device, shuffle: bool) -> DataLoader:
    worker_kwargs = dataloader_worker_kwargs(args.num_workers, args.loader_prefetch_factor)
    return DataLoader(
        train_ds,
        batch_sampler=DurationBucketBatchSampler(
            durations=[float(row.get("audio_duration_sec", 0.0)) for row in train_ds.rows],
            batch_size=args.batch_size,
            shuffle=shuffle,
        ),
        num_workers=args.num_workers,
        collate_fn=ctc_collate,
        pin_memory=(device.type == "cuda"),
        **worker_kwargs,
    )


def make_eval_loader(args: argparse.Namespace, dataset: PhonemeDataset, device: torch.device) -> DataLoader:
    worker_kwargs = dataloader_worker_kwargs(args.num_workers, args.loader_prefetch_factor)
    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ctc_collate,
        pin_memory=(device.type == "cuda"),
        **worker_kwargs,
    )


def make_config(args: argparse.Namespace, transfer_report: dict[str, object], summary: dict[str, int]) -> dict[str, object]:
    return {
        "experiment": {
            "purpose": "PyTorch DeepSpeech2 transfer learning for child IPA CTC",
            "source": "DeepSpeech-style drop_source_layers transfer semantics implemented natively in PyTorch",
        },
        "transfer": transfer_report,
        "model": {
            "architecture": "pytorch_deepspeech2_phone_ctc",
            "source_compatible": args.source_compatible,
            "feature_type": "deepspeech_log_spectrogram_161" if args.source_compatible else "log_mel_80",
            "n_mels": args.n_mels,
            "conv_channels": args.conv_channels,
            "conv_time_strides": args.conv_time_strides,
            "conv_freq_strides": args.conv_freq_strides,
            "conv_dropout": args.conv_dropout,
            "rnn_type": args.rnn_type,
            "hidden_dim": args.hidden_dim,
            "rnn_layers": args.rnn_layers,
            "rnn_dropout": args.rnn_dropout,
            "lookahead_context": args.lookahead_context,
            "proj_dim": args.proj_dim,
            "blank_bias_init": args.blank_bias_init,
            **summary,
        },
        "train": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_num_threads))
    torch.set_num_threads(max(1, int(args.torch_num_threads)))
    torch.set_num_interop_threads(max(1, int(args.torch_num_inter_op_threads)))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stoi, itos = load_vocab(args.vocab)
    blank_id = stoi["<blank>"]
    dataset_cls = DeepSpeechManifestDataset if args.source_compatible else PhonemeDataset
    train_ds = dataset_cls(args.train_manifest, stoi=stoi, train=True, use_specaugment=False)
    val_ds = dataset_cls(args.val_manifest, stoi=stoi, train=False, use_specaugment=False)
    test_ds = dataset_cls(args.test_manifest, stoi=stoi, train=False, use_specaugment=False)
    val_loader = make_eval_loader(args, val_ds, device)
    test_loader = make_eval_loader(args, test_ds, device)

    if args.source_compatible:
        model = SourceDeepSpeech2PhoneCTC(
            vocab_size=len(stoi),
            hidden_dim=args.hidden_dim,
            rnn_layers=args.rnn_layers,
            blank_id=blank_id,
            blank_bias_init=args.blank_bias_init,
        ).to(device)
    else:
        model = DeepSpeech2PhoneCTC(
            vocab_size=len(stoi),
            n_mels=args.n_mels,
            conv_channels=args.conv_channels,
            conv_time_strides=args.conv_time_strides,
            conv_freq_strides=args.conv_freq_strides,
            conv_dropout=args.conv_dropout,
            rnn_type=args.rnn_type,
            hidden_dim=args.hidden_dim,
            rnn_layers=args.rnn_layers,
            rnn_dropout=args.rnn_dropout,
            lookahead_context=args.lookahead_context,
            proj_dim=args.proj_dim,
            blank_id=blank_id,
            blank_bias_init=args.blank_bias_init,
        ).to(device)
    transfer_report = load_transfer_checkpoint(
        model=model,
        checkpoint_path=args.init_checkpoint,
        drop_source_layers=args.drop_source_layers,
        blank_bias_init=args.blank_bias_init,
    )

    if args.freeze_encoder_epochs > 0 and args.init_checkpoint is not None:
        set_encoder_trainable(model, False)
    summary = trainable_parameter_summary(model)
    config = make_config(args, transfer_report, summary)
    (args.out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "configured",
                "device": str(device),
                "time_reduction": model.time_reduction,
                "num_workers": args.num_workers,
                "decode_strategy": args.decode_strategy,
                **summary,
                "transfer": transfer_report,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    opt = configure_optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    history_path = args.out_dir / "history.jsonl"
    best_metric = float("inf")

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_encoder_epochs + 1:
            set_encoder_trainable(model, True)
            opt = configure_optimizer(model, args)

        train_ds.specaug = SpecAugment() if args.use_specaugment and epoch > args.specaugment_warmup_epochs else None
        train_loader = make_train_loader(args, train_ds, device, shuffle=epoch > args.sortagrad_epochs)
        model.train()
        running_loss = 0.0
        oom_skips = 0
        opt.zero_grad(set_to_none=True)
        prog = tqdm(train_loader, desc=f"pytorch ds2 epoch {epoch}", ncols=120)

        for step_idx, batch in enumerate(prog, start=1):
            try:
                feats = batch["features"].to(device, non_blocking=True)
                feat_lens = batch["feature_lens"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                target_lens = batch["target_lens"].to(device, non_blocking=True)
                label_weights = batch["label_weights"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits, out_lens = model(feats, feat_lens)
                    loss = batch_ctc_loss(logits, out_lens, targets, target_lens, label_weights, blank_id)
                scaler.scale(loss / max(1, args.grad_accum_steps)).backward()
                if step_idx % max(1, args.grad_accum_steps) == 0 or step_idx == len(train_loader):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                running_loss += float(loss.item())
                prog.set_postfix(loss=f"{loss.item():.4f}", oom=oom_skips)
            except torch.OutOfMemoryError:
                oom_skips += 1
                opt.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                prog.set_postfix(loss="oom-skip", oom=oom_skips)
            if args.max_train_batches is not None and step_idx >= args.max_train_batches:
                break

        denom = args.max_train_batches if args.max_train_batches is not None else len(train_loader)
        train_loss = running_loss / max(1, min(len(train_loader), denom))
        val_predictions_path = args.out_dir / f"val_epoch_{epoch:02d}_predictions.jsonl"
        val_report_path = args.out_dir / f"val_epoch_{epoch:02d}_phoneme_report.json"
        val_metrics = evaluate_and_write_report(
            model=model,
            loader=val_loader,
            itos=itos,
            blank_id=blank_id,
            device=device,
            decode_strategy=args.decode_strategy,
            beam_size=args.beam_size,
            decode_blank_penalty=args.decode_blank_penalty,
            predictions_path=val_predictions_path,
            phoneme_report_path=val_report_path,
            phoneme_top_k=args.phoneme_top_k,
            phoneme_min_ref_count=args.phoneme_min_ref_count,
            max_batches=args.max_val_batches,
            progress_desc=f"eval val epoch {epoch}",
        )
        selected_value = float(val_metrics[args.selection_metric])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_per": val_metrics["per"],
            "val_cer": val_metrics["cer"],
            "val_emission_ratio": val_metrics["emission_ratio"],
            "val_avg_blank_posterior": val_metrics["avg_blank_posterior"],
            "val_out_len_lt_target_frac": val_metrics["out_len_lt_target_frac"],
            "selection_metric": args.selection_metric,
            "selection_value": selected_value,
            "oom_skips": oom_skips,
            "specaugment_enabled": train_ds.specaug is not None,
            "encoder_frozen": epoch <= args.freeze_encoder_epochs and args.init_checkpoint is not None,
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with history_path.open("a", encoding="utf-8") as fw:
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_training_plots(history_path, args.out_dir, args.selection_metric)
        write_phoneme_plots(val_report_path, args.out_dir, "latest")

        ckpt = {"model_state": model.state_dict(), "config": config, **row}
        torch.save(ckpt, args.out_dir / "last.pt")
        if selected_value < best_metric:
            best_metric = selected_value
            torch.save(ckpt, args.out_dir / "best.pt")
            (args.out_dir / "best_epoch.txt").write_text(str(epoch), encoding="utf-8")
            shutil.copyfile(val_report_path, args.out_dir / "best_phoneme_report.json")
            shutil.copyfile(val_predictions_path, args.out_dir / "best_predictions.jsonl")
            write_phoneme_plots(args.out_dir / "best_phoneme_report.json", args.out_dir, "best")
            if args.test_each_best:
                test_metrics = evaluate_and_write_report(
                    model=model,
                    loader=test_loader,
                    itos=itos,
                    blank_id=blank_id,
                    device=device,
                    decode_strategy=args.decode_strategy,
                    beam_size=args.beam_size,
                    decode_blank_penalty=args.decode_blank_penalty,
                    predictions_path=args.out_dir / f"test_best_epoch_{epoch:02d}_predictions.jsonl",
                    phoneme_report_path=args.out_dir / f"test_best_epoch_{epoch:02d}_phoneme_report.json",
                    phoneme_top_k=args.phoneme_top_k,
                    phoneme_min_ref_count=args.phoneme_min_ref_count,
                    progress_desc=f"eval test best epoch {epoch}",
                )
                print(json.dumps({"event": "test_at_best", "epoch": epoch, **test_metrics}, ensure_ascii=False), flush=True)
            print(json.dumps({"event": "saved_best", "epoch": epoch, "best_metric": best_metric}), flush=True)

    print(json.dumps({"event": "training_complete", "best_metric": best_metric, "out_dir": str(args.out_dir)}), flush=True)


if __name__ == "__main__":
    main()
