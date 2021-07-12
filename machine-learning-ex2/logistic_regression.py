import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_cost(theta, X, y):
    return -(y @ np.log(sigmoid(X @ theta)) + (1 - y) @ np.log(1 - sigmoid(X @ theta))) / X.shape[0]


def logistic_regression(X, y, alpha, upd):
    cost = np.zeros(upd + 1)
    theta = np.zeros(X.shape[1])
    cost[0] = compute_cost(theta, X, y)
    for i in range(upd):
        theta -= alpha / X.shape[0] * (X.T @ sigmoid(X @ theta) - X.T @ y)
        cost[i + 1] = compute_cost(theta, X, y)
    return theta, cost


def data_processing(data):
    for i in range(data.shape[1] - 1):
        data[:, i] = (data[:, i] - data[:, i].mean()) / data[:, ].std()


def gradient(theta, X, y):
    return (X.T @ sigmoid(X @ theta) - X.T @ y) / X.shape[0]


def quick_solve(X, y):
    res = opt.fmin_tnc(func=compute_cost, x0=np.zeros(X.shape[1]), fprime=gradient, args = (X, y))
    return res[0]


def main():
    data = np.loadtxt("ex2data1.txt", dtype=float, delimiter=',')
    # data_processing(data)
    data = np.insert(data, 0, values=np.ones(data.shape[0]), axis=1)
    X = data[:, :-1]
    y = data[:, -1]

    for i in range(X.shape[0]):
        if y[i] == 1:
            plt.scatter(X[i, 1], X[i, 2], c='b', marker='o')
        else:
            plt.scatter(X[i, 1], X[i, 2], c='r', marker='x')
    # theta, cost = logistic_regression(X, y, 0.1, 10000)
    theta = quick_solve(X, y)
    print(theta)
    x1 = np.arange(20, 100, 0.1)
    h1 = (-theta[0] - theta[1] * x1) / theta[2]
    plt.plot(x1, h1)
    plt.show()


if __name__ == "__main__":
    main()
