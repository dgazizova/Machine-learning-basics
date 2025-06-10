import torch
from torch.utils.data import TensorDataset, DataLoader
from read_mnist import *
from vae import VAE
from tqdm import tqdm
import matplotlib.pyplot as plt


cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")


def train(model, optimizer, epochs, train_dataset):
    ss_epochs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, _ in tqdm(train_dataset):
            x = x.unsqueeze(1).to(device)
            x_hat, mu, logvar = model(x)
            loss = model.VAE_loss(x, x_hat, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        ss_epochs.append(avg_loss)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.2f}")

def evaluate(model, test_dataset):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, _ in test_dataset:
            x = x.unsqueeze(1).to(device)
            x_hat, mu, logvar = model(x)

            loss = model.VAE_loss(x, x_hat, mu, logvar)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_dataset)
    print(f"[Eval] Test Loss: {avg_loss:.2f}")

def plot_images(model, test_dataset):
    x, _ = next(iter(test_dataset))  # one batch
    x = x.to(device)
    x_hat, _, _ = model(x.unsqueeze(1))
    x_hat = x_hat.squeeze(1)

    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i].cpu(), cmap="gray")
        plt.axis("off")

        plt.subplot(2, 5, i + 6)
        plt.imshow(x_hat[i].detach().cpu(), cmap="gray")
        plt.axis("off")

    plt.suptitle("Top: Original, Bottom: Reconstructed")
    plt.show()

def sample_from_prior(model, n=5):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, model.encoder.mu.out_features).to(device)
        x_hat = model.decoder(z)

        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(x_hat[i, 0].cpu(), cmap="gray")
            plt.axis("off")
        plt.suptitle("Samples from prior")
        plt.show()


def main():
    # get train and test data
    train_images, train_labels, test_images, test_labels = get_images()
    # scaling from [0, 255], to [0, 1]
    train_images = train_images / 255
    test_images = test_images / 255

    X_tensor = torch.tensor(train_images, dtype=torch.float32)
    Y_tensor = torch.tensor(train_labels.T, dtype=torch.long)

    X_test_tensor = torch.tensor(test_images, dtype=torch.float32)
    Y_test_tensor = torch.tensor(test_labels.T, dtype=torch.long)

    train_dataset = DataLoader(TensorDataset(X_tensor, Y_tensor), batch_size=5, shuffle=True)
    test_dataset = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=1000, shuffle=False)

    model = VAE(latent_dim=20, beta=5.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 40

    train(model, optimizer, epochs, train_dataset)
    evaluate(model, test_dataset)
    plot_images(model, test_dataset)
    sample_from_prior(model)

    torch.save(model.state_dict(), f"vae_beta5_epoch{epochs}.pth")

if __name__ == '__main__':
    main()

