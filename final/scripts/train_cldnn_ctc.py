#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from src.augment import SpecAugment
from src.dataset import PhonemeDataset, ctc_collate
from src.decode import decode_logits
from src.metrics import PhonemeErrorAnalyzer, cer, per
from src.text import decode_ids


class DurationBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        durations: list[float],
        batch_size: int,
        shuffle: bool = True,
        bucket_size_multiplier: int = 20,
    ) -> None:
        self.durations = durations
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.bucket_size = max(self.batch_size, self.batch_size * int(bucket_size_multiplier))

    def __iter__(self):
        indices = list(range(len(self.durations)))
        if self.shuffle:
            random.shuffle(indices)
            pooled: list[int] = []
            for start in range(0, len(indices), self.bucket_size):
                pool = indices[start : start + self.bucket_size]
                pool.sort(key=self.durations.__getitem__)
                pooled.extend(pool)
            indices = pooled
        else:
            indices.sort(key=self.durations.__getitem__)

        batches = [indices[start : start + self.batch_size] for start in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return (len(self.durations) + self.batch_size - 1) // self.batch_size


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "DeepSpeech2/CLDNN-style non-Transformer phone recognizer: "
            "time-preserving CNN frontend + BiRNN + CTC."
        )
    )
    p.add_argument("--train-manifest", type=Path, default=Path("data/manifests/train.jsonl"))
    p.add_argument("--val-manifest", type=Path, default=Path("data/manifests/val.jsonl"))
    p.add_argument("--test-manifest", type=Path, default=Path("data/manifests/test.jsonl"))
    p.add_argument("--vocab", type=Path, default=Path("data/manifests/vocab.json"))
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/cldnn_deepspeech2_ctc"))

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader worker processes. Use 6-12 on a 16-core machine to bypass the GIL for audio/features.",
    )
    p.add_argument(
        "--loader-prefetch-factor",
        type=int,
        default=4,
        help="Batches each worker preloads. Higher uses more RAM but keeps the GPU fed.",
    )
    p.add_argument(
        "--torch-num-threads",
        type=int,
        default=1,
        help="Torch CPU threads per process. Keep at 1 with many DataLoader workers to avoid oversubscription.",
    )
    p.add_argument(
        "--torch-num-inter-op-threads",
        type=int,
        default=1,
        help="Torch inter-op CPU threads. Keep at 1 for GPU training with multiprocessing data loading.",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument(
        "--frontend-type",
        choices=["logmel_cnn", "wav2vec2_cnn"],
        default="logmel_cnn",
        help=(
            "logmel_cnn keeps the original log-Mel + 2D CNN frontend. "
            "wav2vec2_cnn uses only wav2vec2's raw-waveform Conv1d feature encoder, with no Transformer."
        ),
    )
    p.add_argument(
        "--wav2vec2-conv-dim",
        type=int,
        default=512,
        help="Channel size for each wav2vec2-style Conv1d feature-extractor layer.",
    )
    p.add_argument(
        "--wav2vec2-conv-norm",
        choices=["group", "layer"],
        default="group",
        help="Normalization variant for the wav2vec2-style Conv1d frontend.",
    )
    p.add_argument(
        "--wav2vec2-init",
        choices=["none", "torchaudio_base", "torchaudio_asr_base_960h"],
        default="none",
        help=(
            "Optionally initialize only the Conv1d feature extractor from a torchaudio wav2vec2 bundle. "
            "The Transformer, projection, and ASR head are never used."
        ),
    )
    p.add_argument(
        "--freeze-wav2vec2-cnn-epochs",
        type=int,
        default=0,
        help="Freeze the raw-waveform Conv1d frontend for the first N epochs.",
    )
    p.add_argument("--conv-channels", type=parse_int_list, default=[32, 64])
    p.add_argument("--conv-time-strides", type=parse_int_list, default=[1, 1])
    p.add_argument("--conv-freq-strides", type=parse_int_list, default=[2, 2])
    p.add_argument("--conv-dropout", type=float, default=0.1)
    p.add_argument("--rnn-type", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--rnn-layers", type=int, default=4)
    p.add_argument("--rnn-dropout", type=float, default=0.25)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--blank-bias-init", type=float, default=-2.0)

    p.add_argument("--decode-strategy", choices=["greedy", "beam"], default="beam")
    p.add_argument("--beam-size", type=int, default=8)
    p.add_argument(
        "--decode-blank-penalty",
        type=float,
        default=0.0,
        help="Subtract this value from blank logits only during decoding/evaluation.",
    )
    p.add_argument("--selection-metric", choices=["cer", "per"], default="cer")
    p.add_argument("--phoneme-top-k", type=int, default=25)
    p.add_argument("--phoneme-min-ref-count", type=int, default=5)
    p.add_argument("--use-specaugment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--specaugment-warmup-epochs",
        type=int,
        default=3,
        help="Disable SpecAugment for the first N epochs to make early CTC alignment easier.",
    )
    p.add_argument("--test-each-best", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--plots-only", action="store_true", help="Regenerate plots from an existing out-dir, then exit.")
    p.add_argument("--max-train-batches", type=int, default=None, help="Debug only: stop each train epoch early.")
    p.add_argument("--max-val-batches", type=int, default=None, help="Debug only: stop validation early.")
    return p.parse_args()


def load_vocab(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stoi = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in payload["stoi"].items()}
    stoi = {k: int(v) for k, v in stoi.items()}
    itos = {int(k): v for k, v in payload["itos"].items()}
    return stoi, itos


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Conv2dSubsample(nn.Module):
    def __init__(
        self,
        n_mels: int,
        channels: list[int],
        time_strides: list[int],
        freq_strides: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if not (len(channels) == len(time_strides) == len(freq_strides)):
            raise ValueError("conv-channels, conv-time-strides, and conv-freq-strides must have the same length")

        layers: list[nn.Module] = []
        in_ch = 1
        self.time_reduction = 1
        for out_ch, t_stride, f_stride in zip(channels, time_strides, freq_strides):
            if t_stride < 1 or f_stride < 1:
                raise ValueError("Convolution strides must be positive")
            layers.extend(
                [
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=(5, 5),
                        stride=(t_stride, f_stride),
                        padding=(2, 2),
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                ]
            )
            self.time_reduction *= t_stride
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, n_mels)
            out = self.net(dummy)
            self.out_dim = int(out.size(1) * out.size(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, F] -> [B, C, T', F'] -> [B, T', C*F']
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.transpose(1, 2).contiguous()
        return x.flatten(start_dim=2)


class Wav2Vec2ConvFeatureEncoder(nn.Module):
    def __init__(
        self,
        conv_dim: int = 512,
        dropout: float = 0.0,
        norm: str = "group",
    ) -> None:
        super().__init__()
        self.conv_kernels = [10, 3, 3, 3, 3, 2, 2]
        self.conv_strides = [5, 2, 2, 2, 2, 2, 2]
        self.time_reduction = 1
        for stride in self.conv_strides:
            self.time_reduction *= stride

        layers: list[nn.Module] = []
        in_ch = 1
        for layer_idx, (kernel, stride) in enumerate(zip(self.conv_kernels, self.conv_strides)):
            modules: list[nn.Module] = [
                nn.Conv1d(in_ch, conv_dim, kernel_size=kernel, stride=stride, bias=False),
            ]
            if norm == "group":
                if layer_idx == 0:
                    modules.append(nn.GroupNorm(num_groups=conv_dim, num_channels=conv_dim))
            elif norm == "layer":
                modules.append(nn.GroupNorm(num_groups=1, num_channels=conv_dim))
            else:
                raise ValueError(f"Unsupported wav2vec2 conv norm: {norm}")
            modules.extend([nn.GELU(), nn.Dropout(dropout)])
            layers.append(nn.Sequential(*modules))
            in_ch = conv_dim
        self.layers = nn.ModuleList(layers)
        self.out_dim = int(conv_dim)
        self.pretrained_source: str | None = None
        self.pretrained_copied_tensors = 0

    @staticmethod
    def conv_output_length(input_lens: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
        return torch.div(input_lens - kernel_size, stride, rounding_mode="floor") + 1

    def output_lengths(self, input_lens: torch.Tensor) -> torch.Tensor:
        out_lens = input_lens
        for kernel, stride in zip(self.conv_kernels, self.conv_strides):
            out_lens = self.conv_output_length(out_lens, kernel_size=kernel, stride=stride)
        return torch.clamp(out_lens, min=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, samples] -> [B, frames, channels]
        if x.dim() != 2:
            raise ValueError(f"wav2vec2_cnn frontend expects [B, samples], got shape {tuple(x.shape)}")
        x = x.unsqueeze(1)
        min_samples = 400
        if x.size(-1) < min_samples:
            x = F.pad(x, (0, min_samples - x.size(-1)))
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2).contiguous()

    def load_torchaudio_pretrained(self, source: str) -> int:
        if self.out_dim != 512:
            raise ValueError("torchaudio wav2vec2 feature-extractor weights require --wav2vec2-conv-dim 512")
        source = source.strip().lower()
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required for --wav2vec2-init torchaudio_*") from exc

        if source == "torchaudio_base":
            model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        elif source == "torchaudio_asr_base_960h":
            model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
        else:
            raise ValueError(f"Unsupported wav2vec2 pretrained source: {source}")

        src_state = model.feature_extractor.state_dict()
        copied = 0
        with torch.no_grad():
            for idx, layer in enumerate(self.layers):
                conv = layer[0]
                src_weight = src_state.get(f"conv_layers.{idx}.conv.weight")
                if src_weight is None or src_weight.shape != conv.weight.shape:
                    continue
                conv.weight.copy_(src_weight)
                copied += 1
            first_norm = self.layers[0][1] if len(self.layers[0]) > 1 else None
            if isinstance(first_norm, nn.GroupNorm):
                for attr in ("weight", "bias"):
                    src_value = src_state.get(f"conv_layers.0.layer_norm.{attr}")
                    dst_value = getattr(first_norm, attr)
                    if src_value is not None and src_value.shape == dst_value.shape:
                        dst_value.copy_(src_value)
                        copied += 1
        self.pretrained_source = source
        self.pretrained_copied_tensors = copied
        return copied


class CLDNNPhoneCTC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        frontend_type: str,
        n_mels: int,
        conv_channels: list[int],
        conv_time_strides: list[int],
        conv_freq_strides: list[int],
        conv_dropout: float,
        wav2vec2_conv_dim: int,
        wav2vec2_conv_norm: str,
        wav2vec2_init: str,
        rnn_type: str,
        hidden_dim: int,
        rnn_layers: int,
        rnn_dropout: float,
        proj_dim: int,
        blank_id: int,
        blank_bias_init: float,
    ) -> None:
        super().__init__()
        self.frontend_type = frontend_type.strip().lower()
        if self.frontend_type == "logmel_cnn":
            self.frontend = Conv2dSubsample(
                n_mels=n_mels,
                channels=conv_channels,
                time_strides=conv_time_strides,
                freq_strides=conv_freq_strides,
                dropout=conv_dropout,
            )
        elif self.frontend_type == "wav2vec2_cnn":
            self.frontend = Wav2Vec2ConvFeatureEncoder(
                conv_dim=wav2vec2_conv_dim,
                dropout=conv_dropout,
                norm=wav2vec2_conv_norm,
            )
            if wav2vec2_init != "none":
                self.frontend.load_torchaudio_pretrained(wav2vec2_init)
        else:
            raise ValueError(f"Unsupported frontend_type: {frontend_type}")
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=self.frontend.out_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout if rnn_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, proj_dim),
            nn.ReLU(inplace=True),
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
        logits = self.classifier(self.proj(h))
        if hasattr(self.frontend, "output_lengths"):
            out_lens = self.frontend.output_lengths(input_lens)
        else:
            out_lens = torch.clamp(torch.div(input_lens, self.time_reduction, rounding_mode="floor"), min=1)
        out_lens = torch.clamp(out_lens, max=logits.size(1))
        return logits, out_lens

    def set_frontend_trainable(self, trainable: bool) -> None:
        for param in self.frontend.parameters():
            param.requires_grad = bool(trainable)


def trainable_parameter_summary(model: nn.Module) -> dict[str, int]:
    return {
        "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
    }


def batch_ctc_loss(
    logits: torch.Tensor,
    out_lens: torch.Tensor,
    targets: torch.Tensor,
    target_lens: torch.Tensor,
    label_weights: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
    ctc = nn.CTCLoss(blank=blank_id, reduction="none", zero_infinity=True)
    loss_vec = ctc(log_probs, targets, out_lens, target_lens)
    loss_vec = loss_vec / torch.clamp(target_lens.float(), min=1.0)
    norm_w = label_weights / torch.clamp(label_weights.sum(), min=1e-6)
    return (loss_vec * norm_w).sum()


def apply_decode_blank_penalty(logits: torch.Tensor, blank_id: int, penalty: float) -> torch.Tensor:
    if penalty == 0.0:
        return logits
    adjusted = logits.clone()
    adjusted[..., blank_id] -= float(penalty)
    return adjusted


def valid_frame_mask(out_lens: torch.Tensor, max_t: int) -> torch.Tensor:
    positions = torch.arange(max_t, device=out_lens.device).unsqueeze(0)
    return positions < out_lens.unsqueeze(1)


def evaluate_and_write_report(
    model: nn.Module,
    loader: DataLoader,
    itos: dict[int, str],
    blank_id: int,
    device: torch.device,
    decode_strategy: str,
    beam_size: int,
    decode_blank_penalty: float,
    predictions_path: Path,
    phoneme_report_path: Path,
    phoneme_top_k: int,
    phoneme_min_ref_count: int,
    max_batches: int | None = None,
    progress_desc: str | None = None,
) -> dict[str, float]:
    model.eval()
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    phoneme_report_path.parent.mkdir(parents=True, exist_ok=True)

    all_per: list[float] = []
    all_cer: list[float] = []
    analyzer = PhonemeErrorAnalyzer()
    total_ref_tokens = 0
    total_hyp_tokens = 0
    total_blank_mass = 0.0
    total_valid_frames = 0
    out_lt_target = 0
    sample_count = 0

    with predictions_path.open("w", encoding="utf-8") as fw, torch.no_grad():
        iterator = enumerate(loader, start=1)
        if progress_desc:
            total_batches = len(loader)
            if max_batches is not None:
                total_batches = min(total_batches, max_batches)
            iterator = tqdm(iterator, desc=progress_desc, total=total_batches, ncols=120)
        for batch_idx, batch in iterator:
            feats = batch["features"].to(device)
            feat_lens = batch["feature_lens"].to(device)
            target_lens = batch["target_lens"].to(device)
            logits, out_lens = model(feats, feat_lens)

            mask = valid_frame_mask(out_lens, logits.size(1))
            probs = torch.softmax(logits, dim=-1)
            blank_probs = probs[..., blank_id]
            total_blank_mass += float(blank_probs.masked_select(mask).sum().item())
            total_valid_frames += int(mask.sum().item())
            out_lt_target += int((out_lens < target_lens).sum().item())
            sample_count += int(feats.size(0))

            decoded_logits = apply_decode_blank_penalty(logits, blank_id=blank_id, penalty=decode_blank_penalty)
            pred_ids = decode_logits(decoded_logits, blank_id=blank_id, strategy=decode_strategy, beam_size=beam_size)
            pred_texts = [decode_ids(ids, itos) for ids in pred_ids]

            for utt, ref, hyp in zip(batch["utt_ids"], batch["texts"], pred_texts):
                sample_per = per(ref, hyp)
                sample_cer = cer(ref, hyp)
                all_per.append(sample_per)
                all_cer.append(sample_cer)
                analyzer.update(ref, hyp)
                ref_tokens = len(ref.replace(" ", ""))
                hyp_tokens = len(hyp.replace(" ", ""))
                total_ref_tokens += ref_tokens
                total_hyp_tokens += hyp_tokens
                fw.write(
                    json.dumps(
                        {
                            "utterance_id": utt,
                            "reference": ref,
                            "prediction": hyp,
                            "per": sample_per,
                            "cer": sample_cer,
                            "ref_tokens": ref_tokens,
                            "hyp_tokens": hyp_tokens,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if max_batches is not None and batch_idx >= max_batches:
                break

    metrics = {
        "per": float(sum(all_per) / max(1, len(all_per))),
        "cer": float(sum(all_cer) / max(1, len(all_cer))),
        "emission_ratio": float(total_hyp_tokens / max(1, total_ref_tokens)),
        "avg_blank_posterior": float(total_blank_mass / max(1, total_valid_frames)),
        "out_len_lt_target_frac": float(out_lt_target / max(1, sample_count)),
        "total_ref_tokens": float(total_ref_tokens),
        "total_hyp_tokens": float(total_hyp_tokens),
    }
    report = {
        "num_samples": len(all_cer),
        "avg_per": metrics["per"],
        "avg_cer": metrics["cer"],
        "decode_strategy": decode_strategy,
        "beam_size": beam_size,
        "decode_blank_penalty": decode_blank_penalty,
        "predictions": str(predictions_path),
        "diagnostics": metrics,
        "phoneme_analysis": analyzer.summary(top_k=phoneme_top_k, min_ref_count=phoneme_min_ref_count),
    }
    phoneme_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def read_history_jsonl(history_path: Path) -> list[dict[str, object]]:
    if not history_path.exists():
        return []
    rows: list[dict[str, object]] = []
    with history_path.open("r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"event": "plot_skipped", "reason": f"matplotlib unavailable: {exc}"}), flush=True)
        return None


def write_training_plots(history_path: Path, out_dir: Path, selection_metric: str) -> None:
    history = read_history_jsonl(history_path)
    if not history:
        return
    plt = _load_matplotlib()
    if plt is None:
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_cer = [float(row["val_cer"]) for row in history]
    val_per = [float(row["val_per"]) for row in history]
    emission_ratio = [float(row.get("val_emission_ratio", 0.0)) for row in history]
    blank_mass = [float(row.get("val_avg_blank_posterior", 0.0)) for row in history]
    selection_values = [float(row["selection_value"]) for row in history]
    best_so_far = []
    best = float("inf")
    for value in selection_values:
        best = min(best, value)
        best_so_far.append(best)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(epochs, train_loss, marker="o", label="Train CTC loss")
    axes[0, 0].set_title("CLDNN training loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, val_cer, marker="o", label="Val CER")
    axes[0, 1].plot(epochs, val_per, marker="o", label="Val PER")
    axes[0, 1].plot(epochs, best_so_far, linestyle="--", label=f"Best {selection_metric.upper()} so far")
    axes[0, 1].set_title("Validation error")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Error rate")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, emission_ratio, marker="o", color="tab:green")
    axes[1, 0].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1, 0].set_title("Predicted/reference token ratio")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Emission ratio")

    axes[1, 1].plot(epochs, blank_mass, marker="o", color="tab:red")
    axes[1, 1].set_title("Average blank posterior")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Blank posterior")

    fig.tight_layout()
    fig.savefig(plot_dir / "cldnn_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_phoneme_plots(phoneme_report_path: Path, out_dir: Path, label: str) -> None:
    if not phoneme_report_path.exists():
        return
    plt = _load_matplotlib()
    if plt is None:
        return

    report = json.loads(phoneme_report_path.read_text(encoding="utf-8"))
    analysis = report.get("phoneme_analysis", {})
    worst = list(analysis.get("worst_recall", []))[:20]
    substitutions = list(analysis.get("most_common_substitutions", []))[:20]
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if worst:
        phones = [str(row["phoneme"]) for row in worst]
        error_rates = [float(row.get("error_rate", 0.0)) for row in worst]
        deletions = [int(row.get("deletions", 0)) for row in worst]
        subs = [int(row.get("substitutions", 0)) for row in worst]
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].barh(phones, error_rates)
        axes[0].invert_yaxis()
        axes[0].set_title(f"Worst phonemes by error rate ({label})")
        axes[0].set_xlabel("Error rate")
        axes[0].set_ylabel("Phoneme")

        y = list(range(len(phones)))
        axes[1].barh(y, subs, label="Substitutions")
        axes[1].barh(y, deletions, left=subs, label="Deletions")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(phones)
        axes[1].invert_yaxis()
        axes[1].set_title(f"Error type split ({label})")
        axes[1].set_xlabel("Count")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"cldnn_phoneme_errors_{label}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if substitutions:
        labels = [f"{row['ref']}->{row['hyp']}" for row in substitutions]
        counts = [int(row["count"]) for row in substitutions]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels, counts)
        ax.invert_yaxis()
        ax.set_title(f"Most common substitutions ({label})")
        ax.set_xlabel("Count")
        fig.tight_layout()
        fig.savefig(plot_dir / f"cldnn_substitutions_{label}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def regenerate_existing_plots(out_dir: Path, selection_metric: str) -> None:
    history_path = out_dir / "history.jsonl"
    write_training_plots(history_path, out_dir, selection_metric)
    best_report = out_dir / "best_phoneme_report.json"
    if best_report.exists():
        write_phoneme_plots(best_report, out_dir, "best")
    epoch_reports = sorted(out_dir.glob("val_epoch_*_phoneme_report.json"))
    if epoch_reports:
        write_phoneme_plots(epoch_reports[-1], out_dir, "latest")
    print(
        json.dumps(
            {
                "event": "plots_regenerated",
                "out_dir": str(out_dir),
                "plots_dir": str(out_dir / "plots"),
                "history_found": history_path.exists(),
                "best_report_found": best_report.exists(),
                "latest_report": str(epoch_reports[-1]) if epoch_reports else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def dataloader_worker_kwargs(num_workers: int, prefetch_factor: int) -> dict[str, object]:
    if num_workers <= 0:
        return {}
    return {
        "persistent_workers": True,
        "prefetch_factor": max(1, int(prefetch_factor)),
    }


def make_config(args: argparse.Namespace, trainable_summary: dict[str, int]) -> dict[str, object]:
    train_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "data": {
            "train_manifest": str(args.train_manifest),
            "val_manifest": str(args.val_manifest),
            "test_manifest": str(args.test_manifest),
            "feature_type": "raw" if args.frontend_type == "wav2vec2_cnn" else "logmel",
            "vocab_path": str(args.vocab),
            "use_specaugment": args.use_specaugment,
            "specaugment_warmup_epochs": args.specaugment_warmup_epochs,
        },
        "model": {
            "architecture": "cldnn_deepspeech2_ctc",
            "frontend_type": args.frontend_type,
            "n_mels": args.n_mels,
            "conv_channels": args.conv_channels,
            "conv_time_strides": args.conv_time_strides,
            "conv_freq_strides": args.conv_freq_strides,
            "conv_dropout": args.conv_dropout,
            "wav2vec2_conv_dim": args.wav2vec2_conv_dim,
            "wav2vec2_conv_norm": args.wav2vec2_conv_norm,
            "wav2vec2_init": args.wav2vec2_init,
            "freeze_wav2vec2_cnn_epochs": args.freeze_wav2vec2_cnn_epochs,
            "rnn_type": args.rnn_type,
            "hidden_dim": args.hidden_dim,
            "rnn_layers": args.rnn_layers,
            "rnn_dropout": args.rnn_dropout,
            "proj_dim": args.proj_dim,
            "blank_bias_init": args.blank_bias_init,
            **trainable_summary,
        },
        "decode": {
            "strategy": args.decode_strategy,
            "beam_size": args.beam_size,
            "blank_penalty": args.decode_blank_penalty,
        },
        "train": train_args,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_num_threads))
    torch.set_num_threads(max(1, int(args.torch_num_threads)))
    torch.set_num_interop_threads(max(1, int(args.torch_num_inter_op_threads)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        regenerate_existing_plots(args.out_dir, args.selection_metric)
        return

    stoi, itos = load_vocab(args.vocab)
    blank_id = stoi["<blank>"]
    dataset_feature_type = "raw" if args.frontend_type == "wav2vec2_cnn" else "logmel"
    feature_specaugment_enabled = args.use_specaugment and args.frontend_type == "logmel_cnn"

    train_ds = PhonemeDataset(
        args.train_manifest,
        stoi=stoi,
        train=True,
        use_specaugment=False,
        feature_type=dataset_feature_type,
    )
    val_ds = PhonemeDataset(
        args.val_manifest,
        stoi=stoi,
        train=False,
        use_specaugment=False,
        feature_type=dataset_feature_type,
    )
    worker_kwargs = dataloader_worker_kwargs(args.num_workers, args.loader_prefetch_factor)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=DurationBucketBatchSampler(
            durations=[float(row.get("audio_duration_sec", 0.0)) for row in train_ds.rows],
            batch_size=args.batch_size,
            shuffle=True,
        ),
        num_workers=args.num_workers,
        collate_fn=ctc_collate,
        pin_memory=(device.type == "cuda"),
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ctc_collate,
        pin_memory=(device.type == "cuda"),
        **worker_kwargs,
    )
    test_loader = None
    if args.test_each_best:
        test_ds = PhonemeDataset(
            args.test_manifest,
            stoi=stoi,
            train=False,
            use_specaugment=False,
            feature_type=dataset_feature_type,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=ctc_collate,
            pin_memory=(device.type == "cuda"),
            **worker_kwargs,
        )

    model = CLDNNPhoneCTC(
        vocab_size=len(stoi),
        frontend_type=args.frontend_type,
        n_mels=args.n_mels,
        conv_channels=args.conv_channels,
        conv_time_strides=args.conv_time_strides,
        conv_freq_strides=args.conv_freq_strides,
        conv_dropout=args.conv_dropout,
        wav2vec2_conv_dim=args.wav2vec2_conv_dim,
        wav2vec2_conv_norm=args.wav2vec2_conv_norm,
        wav2vec2_init=args.wav2vec2_init,
        rnn_type=args.rnn_type,
        hidden_dim=args.hidden_dim,
        rnn_layers=args.rnn_layers,
        rnn_dropout=args.rnn_dropout,
        proj_dim=args.proj_dim,
        blank_id=blank_id,
        blank_bias_init=args.blank_bias_init,
    ).to(device)
    if args.frontend_type == "wav2vec2_cnn" and args.freeze_wav2vec2_cnn_epochs > 0:
        model.set_frontend_trainable(False)
    summary = trainable_parameter_summary(model)
    config = make_config(args, summary)
    (args.out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "device": str(device),
                "time_reduction": model.time_reduction,
                **summary,
                "num_workers": args.num_workers,
                "loader_prefetch_factor": args.loader_prefetch_factor if args.num_workers > 0 else None,
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "out_dir": str(args.out_dir),
                "wav2vec2_pretrained_source": getattr(model.frontend, "pretrained_source", None),
                "wav2vec2_pretrained_copied_tensors": getattr(model.frontend, "pretrained_copied_tensors", None),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    history_path = args.out_dir / "history.jsonl"
    best_metric = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.frontend_type == "wav2vec2_cnn":
            model.set_frontend_trainable(epoch > args.freeze_wav2vec2_cnn_epochs)
        if feature_specaugment_enabled and epoch > args.specaugment_warmup_epochs:
            train_ds.specaug = SpecAugment()
        else:
            train_ds.specaug = None

        running_loss = 0.0
        oom_skips = 0
        opt.zero_grad(set_to_none=True)
        prog = tqdm(train_loader, desc=f"cldnn epoch {epoch}", ncols=120)

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
        val_phoneme_report_path = args.out_dir / f"val_epoch_{epoch:02d}_phoneme_report.json"
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
            phoneme_report_path=val_phoneme_report_path,
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
            "frontend_trainable": any(param.requires_grad for param in model.frontend.parameters()),
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with history_path.open("a", encoding="utf-8") as fw:
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")

        write_training_plots(history_path, args.out_dir, args.selection_metric)
        write_phoneme_plots(val_phoneme_report_path, args.out_dir, "latest")

        ckpt = {
            "model_state": model.state_dict(),
            "config": config,
            **row,
        }
        torch.save(ckpt, args.out_dir / "last.pt")
        if selected_value < best_metric:
            best_metric = selected_value
            torch.save(ckpt, args.out_dir / "best.pt")
            (args.out_dir / "best_epoch.txt").write_text(str(epoch), encoding="utf-8")
            shutil.copyfile(val_phoneme_report_path, args.out_dir / "best_phoneme_report.json")
            shutil.copyfile(val_predictions_path, args.out_dir / "best_predictions.jsonl")
            write_phoneme_plots(args.out_dir / "best_phoneme_report.json", args.out_dir, "best")
            if test_loader is not None:
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
                    max_batches=None,
                    progress_desc=f"eval test best epoch {epoch}",
                )
                print(json.dumps({"event": "test_at_best", "epoch": epoch, **test_metrics}), flush=True)
            print(json.dumps({"event": "saved_best", "epoch": epoch, "best_metric": best_metric}), flush=True)

    print(json.dumps({"event": "training_complete", "best_metric": best_metric, "out_dir": str(args.out_dir)}), flush=True)


if __name__ == "__main__":
    main()
