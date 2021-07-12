import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.metrics import classification_report


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(theta, X, y, punish):
    predict = sigmoid(X @ theta)
    loss = 1 / X.shape[0] * (-y @ np.log(predict) - (1 - y) @ np.log(1 - predict))
    reg = punish / (2 * X.shape[0]) * theta[1:] @ theta[1:]
    return loss + reg


def gradient(theta, X, y, punish):
    predict = sigmoid(X @ theta)
    loss_gradient = 1 / X.shape[0] * X.T @ (predict - y)
    reg_gradient = punish / X.shape[0] * theta
    reg_gradient[0] = 0
    return loss_gradient + reg_gradient


def logistic_regression(X, y, init_theta, punish):
    res = minimize(fun=compute_cost, x0=init_theta, args=(X, y, punish), method='TNC', jac=gradient)
    return res.x


def train(X, y, punish=1):
    num_labels = len(np.unique(y))
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    theta_all = np.zeros((num_labels, X.shape[1]))
    for i in range(num_labels):
        y_i = np.array([1 if i + 1 == y[j] else 0 for j in range(len(y))])
        theta_all[i] = logistic_regression(X, y_i, np.zeros(X.shape[1]), punish)
    return theta_all


def one_vs_all(X, theta_all):
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    temp = sigmoid(X @ theta_all.T)
    y_predict = np.argmax(temp, axis=1)
    y_predict += 1
    return y_predict


def main():
    data = loadmat("ex3data1.mat")
    ans = train(data['X'], data['y'])
    y_predict = one_vs_all(data['X'], ans)
    print(classification_report(data['y'], y_predict))


if __name__ == '__main__':
    main()
