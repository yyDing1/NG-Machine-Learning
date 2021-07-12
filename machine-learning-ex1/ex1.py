import numpy as np
import matplotlib as plt


def calculate_cost(X, y, theta):
    temp = X * theta
    print(np.shape(temp), np.shape(y), np.shape(temp - y))
    return 1 / (2 * len(x)) * temp


def linear_regression(train_set, alpha, upd_times):
    n = len(train_set[0])
    m = len(train_set)
    x, y = np.split(train_set, (n - 1, ), axis=1)
    x = np.concatenate([np.ones([m, 1]), x], axis=1)
    theta = temp = np.zeros([n, ])
    cost = np.zeros(upd_times)
    cost[0] = calculate_cost(x, y, theta)
    for turn in range(1, upd_times + 1):
        for j in range(n):
            for i in range(m):
                temp[j, 0] = x[i] @ theta * x[i, j]
        theta = temp
        cost[turn] = calculate_cost(x, y, theta)
    return alpha, cost


if __name__ == "__main__":
    data = np.loadtxt("ex1data1.txt", dtype=float, delimiter=',')
    linear_regression(data, 0.1, 10)
    # print(ans)
