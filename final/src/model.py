from __future__ import annotations

import torch
import torch.nn as nn


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
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ConvBlock(1, 64, 1, 2),
                ConvBlock(64, 128, 1, 2),
                ConvBlock(128, 256, 2, 2),
                ConvBlock(256, 512, 2, 2),
                ConvBlock(512, 768, 1, 1),
            ]
        )
        self.out_dim = 768
        self.time_reduction = 4

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
    ) -> None:
        super().__init__()
        self.backbone = Cnn14LikeBackbone()
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        self.transfer_loaded = False

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

    def try_load_panns_transfer(self) -> bool:
        # Best-effort loading: this keeps the pipeline runnable offline.
        try:
            hub_candidates = ["Cnn14", "cnn14"]
            ext_model = None
            for name in hub_candidates:
                try:
                    ext_model = torch.hub.load(
                        "qiuqiangkong/audioset_tagging_cnn",
                        name,
                        pretrained=True,
                    )
                    break
                except Exception:
                    continue

            if ext_model is None:
                return False

            src_state = ext_model.state_dict()
            dst_state = self.state_dict()
            copied = 0
            for k, v in src_state.items():
                if k in dst_state and dst_state[k].shape == v.shape:
                    dst_state[k] = v
                    copied += 1
            if copied == 0:
                return False
            self.load_state_dict(dst_state, strict=False)
            return True
        except Exception:
            return False

    def forward(self, x: torch.Tensor, input_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        h = self.proj(h)
        logits = self.classifier(h)
        out_lens = torch.clamp(input_lens // self.backbone.time_reduction, min=1)
        return logits, out_lens
