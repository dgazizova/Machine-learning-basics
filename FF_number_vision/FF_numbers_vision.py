import numpy as np
from read_mnist import *


def sigmoid(z):
    return 1.0 / (1.0+np.exp(-z))

class NeuralNetwork:
    def __init__(self, layers):
        self.n_layers = len(layers)
        self.layers = layers
        self.W = [np.random.randn(layers[i], layers[i+1]) for i in range(self.n_layers - 1)]  # randn is Gaussian distribution
        self.W0 = [np.zeros((layer, 1)) for layer in layers[1:]]

    def feedforward(self, X):
        A = [X]  # A0 = X
        for i in range(self.n_layers - 1):
            Z = np.dot(self.W[i].T, A[i]) + self.W0[i]  # Find Z between layers
            A.append(sigmoid(Z))  # Use activation function to get new A
        self.A = A  # len(A) = 3

    def backpropagation(self, ac):
        dLdA = 2 * (self.A[-1] - ac)  # MSE loss
        dLdW = []
        dLdW0 = []
        for i in range(self.n_layers-1, 0, -1):
            dAdZ = self.A[i] * (1 - self.A[i])
            dLdZ = dLdA * dAdZ
            dLdW.append(np.dot(self.A[i-1], dLdZ.T))
            dLdW0.append(np.sum(dLdZ, axis=1, keepdims=True))
            dLdA = np.dot(self.W[i-1], dLdZ)
        self.dLdW = dLdW[::-1]
        self.dLdW0 = dLdW0[::-1]


    def sgd(self, nu, mini_batch_size):
        for i in range(self.n_layers-1):
            self.W[i] = self.W[i] - nu / mini_batch_size * self.dLdW[i]
            self.W0[i] = self.W0[i] - nu / mini_batch_size * self.dLdW0[i]

    def _shuffle_dataset(self):
        shuffle_indices = np.random.permutation(self.X.shape[1])
        X_shuffle = self.X[:, shuffle_indices]
        Y_shuffle = self.Y[:, shuffle_indices]
        self.X = X_shuffle
        self.Y = Y_shuffle

    def train(self, X, Y, nu, mini_batch_size, epochs, X_test, Y_test):
        self.X = X
        self.Y = Y
        _, n = self.X.shape
        for i in range(epochs):
            self._shuffle_dataset()
            for k in range(0, n, mini_batch_size):
                self.feedforward(self.X[:, k:k+mini_batch_size])
                self.backpropagation(self.Y[:, k:k+mini_batch_size])
                self.sgd(nu, mini_batch_size)
            if i % 10 == 0:
                print(f"Epoch: {i}")
                self.predict(X_test, Y_test)

    def predict(self, X_test, Y_test):
        self.feedforward(X_test)
        Ypred = np.argmax(self.A[-1], axis=0)
        Ytest = np.argmax(Y_test, axis=0)
        n_test = np.sum(Ypred == Ytest)
        print(f"Test set accuracy: {n_test} {len(Ytest)}")


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

net = NeuralNetwork([784, 30, 10])  # first layer is size of X second can be any size and third is size of Y
net.train(X, Y, nu=2.0, mini_batch_size=10, epochs=101, X_test=X_test, Y_test=Y_test)