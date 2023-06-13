import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class DownstreamTask(nn.Module):
    def __init__(self, clip_model, d_model: int = 512, n_classes: int = 7):
        super().__init__()
        self.backbone = clip_model.backbone.visual
        self.temporal = clip_model.temporal
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_classes)
        )

    def encode_video(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        v = self.backbone(x).reshape(B, T, -1)
        v = self.temporal(v)
        v = v[:, 0]
        return v

    def forward(self, x):
        v = self.encode_video(x)
        out = self.mlp(v)
        return out
