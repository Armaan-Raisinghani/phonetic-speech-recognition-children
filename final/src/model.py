from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.hub import download_url_to_file


PANNS_CNN14_16K_URL = "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP=0.438.pth"
PANNS_CNN14_16K_FILENAME = "Cnn14_16k_mAP=0.438.pth"


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool_time: int, pool_freq: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(pool_time, pool_freq), stride=(pool_time, pool_freq)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Cnn14LikeBackbone(nn.Module):
    def __init__(self, num_blocks: int = 5) -> None:
        super().__init__()
        block_specs = [
            (1, 64, 1, 2),
            (64, 128, 1, 2),
            (128, 256, 2, 2),
            (256, 512, 2, 2),
            (512, 768, 1, 1),
        ]
        if num_blocks < 1 or num_blocks > len(block_specs):
            raise ValueError(f"num_blocks must be between 1 and {len(block_specs)}, got {num_blocks}")
        selected_specs = block_specs[:num_blocks]
        self.blocks = nn.ModuleList(
            [ConvBlock(in_ch, out_ch, pool_time, pool_freq) for in_ch, out_ch, pool_time, pool_freq in selected_specs]
        )
        self.out_dim = selected_specs[-1][1]
        self.time_reduction = 1
        for _, _, pool_time, _ in selected_specs:
            self.time_reduction *= pool_time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        # [B, C, T, F] -> [B, T, C]
        x = x.mean(dim=-1).transpose(1, 2).contiguous()
        return x


class PhonemeCTCModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        freeze_backbone: bool = True,
        use_transfer: bool = False,
        num_backbone_blocks: int = 5,
        head_type: str = "mlp",
        bilstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = Cnn14LikeBackbone(num_blocks=num_backbone_blocks)
        self.head_type = head_type.strip().lower()
        self.hidden_dim = int(hidden_dim)
        self.bilstm_layers = int(bilstm_layers)
        self.dropout = float(dropout)
        if self.head_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(self.backbone.out_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
            )
            classifier_in_dim = hidden_dim
        elif self.head_type == "linear":
            self.proj = nn.Identity()
            classifier_in_dim = self.backbone.out_dim
        elif self.head_type == "bilstm":
            self.proj = nn.LSTM(
                self.backbone.out_dim,
                hidden_dim,
                num_layers=self.bilstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=self.dropout if self.bilstm_layers > 1 else 0.0,
            )
            classifier_in_dim = hidden_dim * 2
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")
        self.classifier = nn.Linear(classifier_in_dim, vocab_size)
        self.transfer_loaded = False
        self.transfer_copied_tensors = 0
        self.transfer_checkpoint_path: str | None = None
        self.transfer_error: str | None = None

        if use_transfer:
            self.transfer_loaded = self.try_load_panns_transfer()

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def freeze_all_except_classifier(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def freeze_all_except_head_and_last_backbone_block(self) -> None:
        self.freeze_all_except_head_and_last_n_backbone_blocks(1)

    def freeze_all_except_head_and_last_n_backbone_blocks(self, num_blocks: int = 1) -> None:
        for p in self.parameters():
            p.requires_grad = False
        if len(self.backbone.blocks) == 0:
            raise ValueError("Cannot unfreeze backbone blocks because the backbone has no blocks")
        num_blocks = max(0, min(int(num_blocks), len(self.backbone.blocks)))
        if num_blocks > 0:
            for block in self.backbone.blocks[-num_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
        for p in self.proj.parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    @staticmethod
    def resolve_panns_checkpoint_path() -> Path:
        override = os.environ.get("PANNS_CHECKPOINT_PATH")
        if override:
            return Path(override).expanduser().resolve()
        return Path.home() / ".cache" / "phoneme_transfer" / PANNS_CNN14_16K_FILENAME

    def download_panns_checkpoint(self, checkpoint_path: Path) -> Path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists():
            return checkpoint_path
        download_url_to_file(PANNS_CNN14_16K_URL, str(checkpoint_path), progress=True)
        return checkpoint_path

    def copy_tensor_if_compatible(
        self,
        dst_state: dict[str, torch.Tensor],
        src_state: dict[str, torch.Tensor],
        dst_key: str,
        src_key: str,
    ) -> int:
        src_value = src_state.get(src_key)
        dst_value = dst_state.get(dst_key)
        if src_value is None or dst_value is None:
            return 0
        if dst_value.shape != src_value.shape:
            return 0
        dst_state[dst_key] = src_value
        return 1

    def try_load_panns_transfer(self) -> bool:
        # Load the official AudioSet-pretrained Cnn14_16k checkpoint and
        # transfer the compatible convolution + batchnorm layers into our
        # lighter CTC backbone.
        try:
            checkpoint_path = self.download_panns_checkpoint(self.resolve_panns_checkpoint_path())
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            src_state = checkpoint.get("model", checkpoint)
            dst_state = self.state_dict()
            copied = 0

            # The first four convolutional blocks line up exactly in shape with
            # our custom backbone. Later PANNs blocks are wider, so we skip them.
            src_to_dst = {
                "conv_block1.conv1.weight": "backbone.blocks.0.block.0.weight",
                "conv_block1.bn1.weight": "backbone.blocks.0.block.1.weight",
                "conv_block1.bn1.bias": "backbone.blocks.0.block.1.bias",
                "conv_block1.bn1.running_mean": "backbone.blocks.0.block.1.running_mean",
                "conv_block1.bn1.running_var": "backbone.blocks.0.block.1.running_var",
                "conv_block1.bn1.num_batches_tracked": "backbone.blocks.0.block.1.num_batches_tracked",
                "conv_block1.conv2.weight": "backbone.blocks.0.block.3.weight",
                "conv_block1.bn2.weight": "backbone.blocks.0.block.4.weight",
                "conv_block1.bn2.bias": "backbone.blocks.0.block.4.bias",
                "conv_block1.bn2.running_mean": "backbone.blocks.0.block.4.running_mean",
                "conv_block1.bn2.running_var": "backbone.blocks.0.block.4.running_var",
                "conv_block1.bn2.num_batches_tracked": "backbone.blocks.0.block.4.num_batches_tracked",
                "conv_block2.conv1.weight": "backbone.blocks.1.block.0.weight",
                "conv_block2.bn1.weight": "backbone.blocks.1.block.1.weight",
                "conv_block2.bn1.bias": "backbone.blocks.1.block.1.bias",
                "conv_block2.bn1.running_mean": "backbone.blocks.1.block.1.running_mean",
                "conv_block2.bn1.running_var": "backbone.blocks.1.block.1.running_var",
                "conv_block2.bn1.num_batches_tracked": "backbone.blocks.1.block.1.num_batches_tracked",
                "conv_block2.conv2.weight": "backbone.blocks.1.block.3.weight",
                "conv_block2.bn2.weight": "backbone.blocks.1.block.4.weight",
                "conv_block2.bn2.bias": "backbone.blocks.1.block.4.bias",
                "conv_block2.bn2.running_mean": "backbone.blocks.1.block.4.running_mean",
                "conv_block2.bn2.running_var": "backbone.blocks.1.block.4.running_var",
                "conv_block2.bn2.num_batches_tracked": "backbone.blocks.1.block.4.num_batches_tracked",
                "conv_block3.conv1.weight": "backbone.blocks.2.block.0.weight",
                "conv_block3.bn1.weight": "backbone.blocks.2.block.1.weight",
                "conv_block3.bn1.bias": "backbone.blocks.2.block.1.bias",
                "conv_block3.bn1.running_mean": "backbone.blocks.2.block.1.running_mean",
                "conv_block3.bn1.running_var": "backbone.blocks.2.block.1.running_var",
                "conv_block3.bn1.num_batches_tracked": "backbone.blocks.2.block.1.num_batches_tracked",
                "conv_block3.conv2.weight": "backbone.blocks.2.block.3.weight",
                "conv_block3.bn2.weight": "backbone.blocks.2.block.4.weight",
                "conv_block3.bn2.bias": "backbone.blocks.2.block.4.bias",
                "conv_block3.bn2.running_mean": "backbone.blocks.2.block.4.running_mean",
                "conv_block3.bn2.running_var": "backbone.blocks.2.block.4.running_var",
                "conv_block3.bn2.num_batches_tracked": "backbone.blocks.2.block.4.num_batches_tracked",
                "conv_block4.conv1.weight": "backbone.blocks.3.block.0.weight",
                "conv_block4.bn1.weight": "backbone.blocks.3.block.1.weight",
                "conv_block4.bn1.bias": "backbone.blocks.3.block.1.bias",
                "conv_block4.bn1.running_mean": "backbone.blocks.3.block.1.running_mean",
                "conv_block4.bn1.running_var": "backbone.blocks.3.block.1.running_var",
                "conv_block4.bn1.num_batches_tracked": "backbone.blocks.3.block.1.num_batches_tracked",
                "conv_block4.conv2.weight": "backbone.blocks.3.block.3.weight",
                "conv_block4.bn2.weight": "backbone.blocks.3.block.4.weight",
                "conv_block4.bn2.bias": "backbone.blocks.3.block.4.bias",
                "conv_block4.bn2.running_mean": "backbone.blocks.3.block.4.running_mean",
                "conv_block4.bn2.running_var": "backbone.blocks.3.block.4.running_var",
                "conv_block4.bn2.num_batches_tracked": "backbone.blocks.3.block.4.num_batches_tracked",
            }

            for src_key, dst_key in src_to_dst.items():
                copied += self.copy_tensor_if_compatible(dst_state, src_state, dst_key=dst_key, src_key=src_key)

            if copied == 0:
                return False

            self.load_state_dict(dst_state, strict=False)
            self.transfer_copied_tensors = copied
            self.transfer_checkpoint_path = str(checkpoint_path)
            return True
        except Exception as exc:
            self.transfer_error = f"{type(exc).__name__}: {exc}"
            return False

    def forward(self, x: torch.Tensor, input_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        if self.head_type == "bilstm":
            h, _ = self.proj(h)
        else:
            h = self.proj(h)
        logits = self.classifier(h)
        out_lens = torch.clamp(input_lens // self.backbone.time_reduction, min=1)
        return logits, out_lens


def build_model_from_config(
    vocab_size: int,
    cfg: dict,
    *,
    freeze_backbone: bool | None = None,
    use_transfer: bool | None = None,
) -> PhonemeCTCModel:
    model_cfg = cfg.get("model", {})
    return PhonemeCTCModel(
        vocab_size=vocab_size,
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True) if freeze_backbone is None else freeze_backbone),
        use_transfer=bool(model_cfg.get("use_transfer", False) if use_transfer is None else use_transfer),
        num_backbone_blocks=int(model_cfg.get("num_backbone_blocks", 5)),
        head_type=str(model_cfg.get("head_type", "mlp")),
        bilstm_layers=int(model_cfg.get("bilstm_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.3)),
    )
