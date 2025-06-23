import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def get_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


@torch.no_grad()
def sample(model, n_samples=5, img_size=28):
    # total diffusion timesteps
    T = 1000

    beta = get_beta_schedule(T)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    x_t = torch.randn(n_samples, 1, img_size, img_size)
    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, dtype=torch.long)
        epsilon_theta = model(x_t, t_batch)

        mu = 1 / alpha[t].sqrt() * (x_t - (1 - alpha[t]) / (1 - alpha_bar[t]).sqrt() * epsilon_theta)

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = beta[t].sqrt()
            x_t = mu + sigma_t * noise
        else:
            x_t = mu

    return x_t


