import torch
from model import CNN
from read_mnist import *
from torch.utils.data import TensorDataset, DataLoader
from train import train
from sample import sample
import matplotlib.pyplot as plt
import logging


cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")


def training():
    # get train and test data
    train_images, train_labels, test_images, test_labels = get_images()

    # scaling from [0, 255], to [0, 1]
    train_images = train_images / 255

    X_tensor = torch.tensor(train_images, dtype=torch.float32)
    Y_tensor = torch.tensor(train_labels.T, dtype=torch.long)
    train_dataset = DataLoader(TensorDataset(X_tensor, Y_tensor), batch_size=5, shuffle=True)

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 100

    train(model, optimizer, epochs, train_dataset, device)

    torch.save(model.state_dict(), f"ddpm_simple.pth")
    logging.info("Training is completed")


def sampling():
    model = CNN().to(device)
    model.load_state_dict(torch.load("ddpm_simple.pth", map_location="cpu"))
    model.eval()

    samples = sample(model, n_samples=5, img_size=28)

    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(samples[i, 0], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    training()
    sampling()

if __name__ == '__main__':
    main()

