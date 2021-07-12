from matplotlib import pyplot as plt
import numpy as np


def compute_cost(X, y, theta):
    return 1 / (2 * X.shape[0]) * (X @ theta - y) @ (X @ theta - y)


def linear_regression(X, y, alpha, upd):
    cost = np.zeros(upd + 1)
    theta = np.zeros(X.shape[1])
    cost[0] = compute_cost(X, y, theta)
    for i in range(upd):
        theta -= alpha / X.shape[0] * (X.T @ X @ theta - X.T @ y)
        cost[i + 1] = compute_cost(X, y, theta)
    return theta, cost


def data_preprocessing(data):
    for i in range(1, data.shape[1]):
        data[:, i] = (data[:, i] - data[:, i].mean()) / data[:, i].std()


def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def main():
    data = np.loadtxt("ex1data2.txt", dtype=float, delimiter=',')
    data = np.insert(data, 0, values=np.ones((data.shape[0],)), axis=1)
    data_preprocessing(data)
    X, y = data[:, :-1], data[:, -1]
    theta, cost = linear_regression(X, y, 0.01, 2000)
    print(theta)
    print(normal_equation(X, y))
    v = np.arange(0, cost.shape[0], 1)
    plt.plot(v, cost)
    plt.xlabel("times of update")
    plt.ylabel("cost")
    plt.show()


if __name__ == "__main__":
    main()
