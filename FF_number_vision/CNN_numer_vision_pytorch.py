import enum

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from read_mnist import *


class LossType(enum.Enum):
    NLL = "NLL"
    MSE = "MSE"


class ConvolutionaldNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # (28, 28), 1 -> 16 filters -> 28 + 2*p - (k-1) / s
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 28 / s = 14 -> (14, 14)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # (14, 14), 16 -> 32 filters
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # (14, 14) -> (7, 7)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            # flatten before, get 32*7*7
            nn.Linear(32*7*7, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.Sigmoid()
        )

    def forward(self, X):
        # add channel so shape becomes (batch_size, 1, 28, 28), n_channels = 1
        X = X.unsqueeze(1)
        X = self.conv_layers(X)

        # flattens image into a vector before going into fully connected layers
        X = X.reshape(X.size(0), -1)
        return self.fc_layers(X)


def train(model, loss_type: LossType, learning_rate, epochs, train_dataset, test_dataset):
    if loss_type == LossType.NLL:
        loss_fn = nn.BCELoss()
    elif loss_type == LossType.MSE:
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("Loss must be 'MSE' or 'NLL'")

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_epochs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, Y in train_dataset:  # for batches
            y_target = torch.nn.functional.one_hot(Y, num_classes=10).float()

            output = model(X)
            loss = loss_fn(output, y_target)
            # to clean gradient values
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_epochs.append(total_loss / len(train_dataset))

        print(f"Epoch {epoch}: Loss = {loss_epochs[-1]:.4f}")
        evaluate(model, test_dataset)

    return loss_epochs


def evaluate(model, test_dataset):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, Y in test_dataset:  # for batches
            output = model(X)
            preds = output.argmax(dim=1)
            correct += (preds == Y).sum().item()
            total += Y.size(0)
    print(f"Test Accuracy: {correct}/{total}")


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


loss_type = LossType.MSE
model = ConvolutionaldNet()
learning_rate = 0.1 if loss_type == LossType.MSE else 0.01
epochs = 20
loss = train(
    model, loss_type=loss_type, learning_rate=learning_rate, epochs=epochs,
    train_dataset=train_dataset, test_dataset=test_dataset
)

# Plotting loss
plt.plot(loss)
plt.title(f"Loss using {loss_type.value}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"Loss_using_{loss_type.value}_torch_CNN.png")
plt.show()


loss_type = LossType.NLL
model = ConvolutionaldNet()
learning_rate = 0.1 if loss_type == LossType.MSE else 0.1
epochs = 20
loss = train(
    model, loss_type=loss_type, learning_rate=learning_rate, epochs=epochs,
    train_dataset=train_dataset, test_dataset=test_dataset
)

# Plotting loss
plt.plot(loss)
plt.title(f"Loss using {loss_type.value}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"Loss_using_{loss_type.value}_torch_CNN.png")
plt.show()



