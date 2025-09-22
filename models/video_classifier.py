import torch
import torch.nn as nn
import os
from models.backbone import VitEncoder


class VideoClassifier(nn.Module):
    def __init__(
        self,
        encoder_config,
        num_classes,
        freeze_encoder=True,
        pretrained_encoder_path=None,
    ):
        super().__init__()
        self.encoder = VitEncoder(
            C_in=encoder_config["video_channels"],
            T_video=encoder_config["frames_per_clip"],
            H_video=encoder_config["resize_height"],
            W_video=encoder_config["resize_width"],
            patch_t=encoder_config["vit_patch_size_t"],
            patch_h=encoder_config["vit_patch_size_h"],
            patch_w=encoder_config["vit_patch_size_w"],
            embed_dim=encoder_config["vit_embed_dim"],
            depth=encoder_config["vit_depth"],
            num_heads=encoder_config["vit_num_heads"],
            mlp_ratio=encoder_config["vit_mlp_ratio"],
            dropout=encoder_config.get(
                "vit_dropout", 0.0
            ),
        )

        if pretrained_encoder_path and os.path.exists(pretrained_encoder_path):
            try:
                print(
                    f"Loading pre-trained ENCODER weights for VideoClassifier from: {pretrained_encoder_path}"
                )

                self.encoder.load_state_dict(
                    torch.load(pretrained_encoder_path, map_location="cpu")
                )
                print(
                    "Pre-trained ENCODER weights loaded successfully into VideoClassifier's encoder."
                )
            except Exception as e:
                print(
                    f"Error loading pre-trained ENCODER weights from {pretrained_encoder_path}: {e}. Encoder will use its initial weights."
                )
        elif pretrained_encoder_path:
            print(
                f"Warning: Pre-trained ENCODER weights file not found at {pretrained_encoder_path}. Encoder will use its initial weights."
            )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier_head = nn.Linear(encoder_config["vit_embed_dim"], num_classes)

    def forward(self, x, return_features=False):
        features = self.encoder(x)  # (B, num_patches, embed_dim)
        pooled_features = self.avgpool(features.transpose(1, 2)).squeeze(2)
        logits = self.classifier_head(pooled_features)
        if return_features:
            return logits, features
        return logits
