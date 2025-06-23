import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, ) :
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x, t):
        t = t.float().unsqueeze(-1) / 1000
        t_emb = self.time_embed(t)  # (B, 31)
        t_emb = t_emb[:, :, None, None]  # (B, 32, 1, 1)

        x1 = self.conv1(x) + t_emb
        x2 = self.conv2(x1)
        x = self.deconv1(x2) + x1
        x = self.deconv2(x)
        return x
