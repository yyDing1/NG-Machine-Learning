import numpy as np
from scipy.io import loadmat
from sklearn.metrics import classification_report


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, theta1, theta2):
    X = X.T
    X = np.insert(X, 0, np.ones(X.shape[1]), axis=0)
    output1 = sigmoid(theta1 @ X)
    output1 = np.insert(output1, 0, np.ones(output1.shape[1]), axis=0)
    output2 = sigmoid(theta2 @ output1)
    return np.argmax(output2, axis=0) + 1


def main():
    data = loadmat("ex3data1.mat")
    theta = loadmat("ex3weights.mat")
    theta1, theta2 = theta["Theta1"], theta["Theta2"]
    y_predict = forward(data['X'], theta1, theta2)
    print(classification_report(data['y'], y_predict))


if __name__ == "__main__":
    main()
