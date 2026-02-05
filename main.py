import numpy as np
import pickle # для серіалізації об'єктів
import gzip # для стиснення файлів
import urllib.request # для завантаження файлів з інтернету
import os # для роботи з файловою системою
from PIL import Image # для обробки зображень
import matplotlib.pyplot as plt # для візуалізації
import ssl


# TODO: Проаналізувати код на помилки
ssl._create_default_https_context = ssl._create_unverified_context

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_loss(self, Y, Y_hat):
        m = Y.shape[0] # кількість прикладів у пакеті
        loss = -np.sum(Y * np.log(Y_hat + 1e-8)) / m
        return loss

    def backward(self, X, Y, Y_hat):
        m = Y.shape[-1] # кількість прикладів у пакеті
        dZ2 = Y_hat - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, Y,epochs=1000, batch_size=32):
        losses = []
        m = X.shape[1] # кількість прикладів у наборі даних
        for epoch in range(epochs):
            for i in range(0, m, batch_size):
                end =  min(i + batch_size, m)
                X_batch = X[:, i:end] # вибірка для поточного пакету
                Y_batch = Y[:, i:end] # відповідні мітки для поточного пакету
                Y_hat = self.forward(X_batch) # прямий прохід
                self.backward(X_batch, Y_batch, Y_hat) # зворотний прохід
            if epoch % 100 == 0:
                Y_hat_full = self.forward(X) # прямий прохід для всього набору даних
                loss = self.compute_loss(Y, Y_hat_full) # обчислення втрат
                losses.append(loss)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis=1)

    def accuracy(self, X, Y):
        Y_pred = self.predict(X)
        Y_true = np.argmax(Y, axis=1)
        return np.mean(Y_pred == Y_true) # обчислення точності


class DigitRecognizer:
    def __init__(self):
        self.model = None
        self.is_trained = False

    def load_mnist_data(self):
        url = 'https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz'
        filename = 'mnist.pkl.gz'
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)
        with gzip.open(filename, 'rb') as f:
            train_data, val_data, test_data = pickle.load(f, encoding='latin1')

        X_train = train_data[0].T
        Y_train = self.one_hot_encode(train_data[1]).T
        X_test = test_data[0].T
        if X_train.shape[0] == 784:
            X_train = X_train.T
            X_test = X_test.T
        Y_test = self.one_hot_encode(test_data[1]).T
        return (X_train, Y_train), (X_test, Y_test)

    def one_hot_encode(self, y, num_classes=10):
        return np.eye(num_classes)[y]

    def create_and_train_model(self, hidden_units=128, epochs=1000, learning_rate=0.1):
        (X_train, Y_train), (X_test, Y_test) = self.load_mnist_data()
        input_size = X_train.shape[0]
        output_size = Y_train.shape[0]
        self.model = NeuralNetwork(input_size, hidden_units, output_size, learning_rate)
        losses = self.model.train(X_train, Y_train, epochs=epochs)
        test_accuracy = self.model.accuracy(X_test, Y_test)
        self.is_trained = True
        return test_accuracy

    def predict_digit(self, image_path):
        if not self.is_trained:
            return None
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img_array = 1 - (np.array(img) / 255.0)
        img_flat = img_array.reshape(784, 1)
        prediction_probs = self.model.forward(img_flat)
        digit = np.argmax(prediction_probs)
        confidence = np.max(prediction_probs)
        return digit, confidence

def main():
    recognizer = DigitRecognizer()
    test_accuracy = recognizer.create_and_train_model(hidden_units=128, epochs=1000, learning_rate=0.1)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    digit, confidence = recognizer.predict_digit('5.png')
    print(f"Predicted Digit: {digit}, Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()