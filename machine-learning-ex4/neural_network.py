import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def predict(theta1, theta2, X):
    h = forward_prop(theta1, theta2, X)[4]
    return np.argmax(h, axis=1) + 1


def forward_prop(theta1, theta2, X):
    a1 = np.insert(X, 0, 1, axis=1)
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, 1, axis=1)
    z3 = a2 @ theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h


def compute_cost(theta_all, X, Y, punish):
    theta1 = theta_all[:25 * 401].reshape(25, 401)
    theta2 = theta_all[25 * 401:].reshape(10, 26)
    h = forward_prop(theta1, theta2, X)[4]
    loss = 1 / X.shape[0] * np.sum(-Y * np.log(h) - (1 - Y) * np.log(1 - h))
    reg = punish / (2 * X.shape[0]) * (np.sum(theta1[:, 1:] * theta1[:, 1:]) + np.sum(theta2[:, 1:] * theta2[:, 1:]))
    return loss + reg


# still cannot understand some detail
def gradient(theta_all, X, Y, punish):
    theta1 = theta_all[:25 * 401].reshape(25, 401)  # (25, 400)
    theta2 = theta_all[25 * 401:].reshape(10, 26)  # (10, 25)
    a1, z2, a2, z3, h = forward_prop(theta1, theta2, X)
    # a1: (5000, 401) z2: (5000, 25) a2: (5000, 26) z3, h: (5000, 10)
    d3 = h - Y  # (5000, 10)
    d2 = d3 @ theta2[:, 1:] * gradient_sigmoid(z2)  # (5000, 25)
    D2 = (d3.T @ a2) / X.shape[0]  # (10, 25)
    D1 = (d2.T @ a1) / X.shape[0]  # (25, 400)
    D1[:, 1:] += punish / X.shape[0] * theta1[:, 1:]
    D2[:, 1:] += punish / X.shape[0] * theta2[:, 1:]
    return np.append(D1.flatten(), D2.flatten())


def neural_network(init_theta, X, Y, punish):
    res = minimize(fun=compute_cost, x0=init_theta, args=(X, Y, punish), method="TNC", jac=gradient,
                   options={'maxiter': 300})
    return res.x


def show_hidden_layer(theta1):
    hidden_layer = theta1[:, 1:]
    fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(8, 8))
    for i in range(5):
        for j in range(5):
            ax[i, j].imshow(hidden_layer[i * 5 + j].reshape(20, 20).T, cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def main():
    data = loadmat("ex4data1.mat")
    init_theta = np.random.uniform(-0.12, 0.12, 10285)
    encoder = OneHotEncoder(sparse=False)
    X = data['X']
    Y = encoder.fit_transform(data['y'])
    ans_all = neural_network(init_theta, X, Y, 1)
    print(ans_all)
    theta1 = ans_all[:25 * 401].reshape(25, 401)
    theta2 = ans_all[25 * 401:].reshape(10, 26)
    print(classification_report(data['y'], predict(theta1, theta2, X)))
    show_hidden_layer(theta1)


if __name__ == "__main__":
    main()
