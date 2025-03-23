import enum

import numpy as np
from read_mnist import *
import matplotlib.pyplot as plt

class LossType(enum.Enum):
    NLL = "NLL"
    MSE = "MSE"


class Module:
    def sgd_step(self, nu, mini_batch_size): pass  # For modules w/o weights

class Linear(Module):
    def __init__(self, m, n):
        self.m, self.n = (m, n)  # (in size, out size)
        self.W0 = np.zeros([self.n, 1])  # (n x 1)
        self.W = np.random.randn(m, n)  # (m x n)

    def feedforward(self, A):
        self.A = A
        Z = np.dot(self.W.T, A) + self.W0
        return Z

    def backprop(self, dLdZ):
        dZdA = self.W
        dLdA = np.dot(dZdA, dLdZ)
        self.dLdW = np.dot(self.A, dLdZ.T)
        self.dLdW0 = np.sum(dLdZ, axis=1, keepdims=True)
        return dLdA

    def sgd_step(self, nu, mini_batch_size):
        self.W -= nu / mini_batch_size * self.dLdW
        self.W0 -= nu / mini_batch_size * self.dLdW0

class Sigmoid(Module):
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, Z):
        self.A = self.sigmoid(Z)
        return self.A

    def backprop(self, dLdA):
        dAdZ = self.A * (1 - self.A)
        dLdZ = dAdZ * dLdA
        return dLdZ

class NeuralNetwork:
    def __init__(self, modules, loss_type: LossType):
        self.modules = modules
        self.loss_type = loss_type

    def MSE_loss(self, Ypred, Y):
        return np.sum((Ypred - Y) ** 2)

    def MSE(self, Ypred, Y):
        dLdA = 2 * (Ypred - Y)
        return dLdA

    def NLL_loss(self, Ypred, Y):
        return np.sum(- (Y * np.log(Ypred) + (1 - Y) * np.log(1 - Ypred)))

    def NLL(self, Ypred, Y):  # negative log likelihood or cross entropy
        dLdA = (Ypred - Y) / (Ypred - Ypred**2)
        return dLdA

    def _shuffle_dataset(self):
        shuffle_indices = np.random.permutation(self.X.shape[1])
        X_shuffle = self.X[:, shuffle_indices]
        Y_shuffle = self.Y[:, shuffle_indices]
        self.X = X_shuffle
        self.Y = Y_shuffle

    def train(self, X, Y, nu, mini_batch_size, epochs, X_test, Y_test):  # Train
        if loss_type == LossType.NLL:
            loss_calc = self.NLL_loss
            loss_fn = self.NLL
        elif loss_type == LossType.MSE:
            loss_calc = self.MSE_loss
            loss_fn = self.MSE
        else:
            raise ValueError("Loss must be 'MSE' or 'NLL'")

        self.X = X
        self.Y = Y
        _, n = self.X.shape
        loss = []
        for i in range(epochs):
            self._shuffle_dataset()
            loss_epoch = 0
            for k in range(0, n, mini_batch_size):
                Ypred = self.feedforward(self.X[:, k:k+mini_batch_size])

                loss_epoch += loss_calc(Ypred, self.Y[:, k:k+mini_batch_size])
                delta = loss_fn(Ypred, self.Y[:, k:k+mini_batch_size])
                self.backward(delta)
                self.sgd_step(nu, mini_batch_size)
            loss.append(loss_epoch / n)
            if i % 10 == 0:
                print(f"Epoch: {i}")
                self.predict(X_test, Y_test)
        return loss

    def feedforward(self, X):
        for m in self.modules: X = m.feedforward(X)
        return X

    def backward(self, delta):  # Update dLdW and dLdW0
        # Note reversed list of modules
        for m in self.modules[::-1]: delta = m.backprop(delta)

    def sgd_step(self, nu, mini_batch_size):  # Gradient descent step
        for m in self.modules: m.sgd_step(nu, mini_batch_size)

    def predict(self, X_test, Y_test):
        Y_pred = self.feedforward(X_test)
        Ypred = np.argmax(Y_pred, axis=0)
        Ytest = np.argmax(Y_test, axis=0)
        n_test = np.sum(Ypred == Ytest)
        print(f"Test set accuracy: {n_test}/{len(Ytest)}")


train_images, train_labels, test_images, test_labels = get_images()
train_images = train_images.reshape(train_images.shape[0], -1) / 255  # scaling from [0, 255], to [0, 1]
test_images = test_images.reshape(test_images.shape[0], -1)

# one hot embedding
train_labels = np.eye(10, dtype=int)[train_labels]  # make vectors from numbers
test_labels = np.eye(10, dtype=int)[test_labels]

X = train_images.T
Y = train_labels.T

X_test = test_images.T
Y_test = test_labels.T

# FF use MSE as loss function
loss_type = LossType.MSE
net = NeuralNetwork([Linear(784, 30), Sigmoid(), Linear(30, 10), Sigmoid()], loss_type)
loss = net.train(X, Y, nu=2.0, mini_batch_size=5, epochs=201, X_test=X_test, Y_test=Y_test)

plt.plot(loss)
plt.title(f"Loss using {loss_type.value}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"Loss_using_{loss_type.value}.png")
plt.show()


# FF use NLL as loss function
loss_type = LossType.NLL
net = NeuralNetwork([Linear(784, 30), Sigmoid(), Linear(30, 10), Sigmoid()],loss_type)
loss = net.train(X, Y, nu=0.1, mini_batch_size=10, epochs=201, X_test=X_test, Y_test=Y_test)

plt.plot(loss)
plt.title(f"Loss using {loss_type.value}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"Loss_using_{loss_type.value}.png")
plt.show()




