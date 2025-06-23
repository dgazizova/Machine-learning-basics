import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def get_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def q_sample(x_0, t, noise, alpha_bar):
    return alpha_bar[t].sqrt().view(-1, 1, 1, 1) * x_0 + (1 - alpha_bar[t]).sqrt().view(-1, 1, 1, 1) * noise


def train(model, optimizer, epochs, train_dataset, device):
    # total diffusion timesteps
    T = 1000

    beta = get_beta_schedule(T)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    for epoch in range(epochs):
        for x_0, _ in tqdm(train_dataset):
            x_0 = x_0.unsqueeze(1).to(device)
            t = torch.randint(0, T, (x_0.size(0),), device=device)
            noise = torch.randn_like(x_0)

            x_t = q_sample(x_0, t, noise, alpha_bar)
            noise_pred = model(x_t, t)

            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")



