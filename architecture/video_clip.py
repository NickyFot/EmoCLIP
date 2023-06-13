import torch
from torch import nn
from torch.cuda.amp import custom_fwd

from .transformer import TemporalTransformer

import clip

device = "cuda" if torch.cuda.is_available() else "cpu"


class VClip(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            nhead: int = 8,
            num_layers: int = 4,
            dim_forward: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_forward = dim_forward

        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        for name, param in model.named_parameters():
            param.requires_grad = False
        self.backbone = model

        self.temporal = TemporalTransformer(
            input_dim=d_model,
            depth=num_layers,
            heads=nhead,
            mlp_dim=d_model,
            dim_head=dim_forward
        )
        self.logit_scale = nn.Parameter(self.backbone.logit_scale.clone().detach())
        self.logit_scale.requires_grad = True

    @custom_fwd
    def forward(self, x, text):
        image_features = self.encode_video(x)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def encode_video(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        v = self.backbone.encode_image(x).reshape(B, T, -1)
        v = self.temporal(v)

        v = v[:, 0]
        return v

    def encode_text(self, text):
        encoded_text = self.backbone.encode_text(text)
        return encoded_text
