import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim) :
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(11, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Flatten()
        )
        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x, y):
        # x (batch_size, 1, 28, 28)
        # y (batch_size, 10)
        y = y.view(y.size(0), 10, 1, 1)
        y = y.expand(-1, -1, 28, 28)

        # x (batch_size, 11, 28, 28)
        x = torch.cat([x, y], dim=1)
        x = self.conv(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.from_latent = nn.Linear(latent_dim + 10, 64 * 7 * 7)

        self.deconv = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )

    def forward(self, z, y):
        # z (batch_size, latent_dim)
        # y (batch_size, 10)

        # z (batch_size, latent_dim+10)
        z = torch.cat([z, y], dim=1)
        x = self.from_latent(z)
        x = self.deconv(x)
        return x

class CVAE(nn.Module):
    def __init__(self, latent_dim, beta):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.beta = beta

    def reconstruction_error(self, x, x_hat):
        return F.mse_loss(x_hat, x, reduction='sum')

    def divergence(self, mu, logvar):
        return 0.5 * torch.sum(-logvar + torch.exp(logvar) + mu.pow(2) - 1)

    def VAE_loss(self, x, x_hat, mu, logvar):
        return self.reconstruction_error(x, x_hat) + self.beta * self.divergence(mu, logvar)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, y)
        return x_hat, mu, logvar

