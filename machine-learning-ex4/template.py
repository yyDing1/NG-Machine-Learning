import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def split(var_all, layer_size):
    var_list = [np.zeros((1, 1))] * len(layer_size)
    begin = 0
    for i in range(1, len(layer_size)):
        pre = layer_size[i - 1]
        now = layer_size[i]
        now_size = now * (pre + 1)
        var_list[i] = var_all[begin: begin + now_size].reshape(now, pre + 1)
        begin += now_size
    return var_list


def merge(var_list):
    var_all = var_list[1]
    for i in range(2, len(var_list)):
        var_all = np.append(var_all, var_list[i])
    return var_all


def forward_prop(theta, X):
    sz = len(theta) + 1
    a = [np.zeros((1, 1))] * sz
    z = [np.zeros((1, 1))] * sz
    a[1] = X
    for i in range(2, sz):
        a[i - 1] = np.insert(a[i - 1], 0, 1, axis=1)
        z[i] = a[i - 1] @ theta[i - 1].T
        a[i] = sigmoid(z[i])
    return a, z


def compute_cost(theta_all, X, Y, punish, layer_size):
    theta = split(theta_all, layer_size)
    h = forward_prop(theta, X)[0][-1]
    loss = 1 / X.shape[0] * np.sum(-Y * np.log(h) - (1 - Y) * np.log(1 - h))
    reg = 0
    for i in range(1, len(theta)):
        reg += np.sum(theta[i][:, 1:] * theta[i][:, 1:])
    reg *= punish / (2 * X.shape[0])
    return loss + reg


def gradient(theta_all, X, Y, punish, layer_size):
    theta = split(theta_all, layer_size)
    a, z = forward_prop(theta, X)
    sz = len(theta) + 1
    d = [np.zeros((1, 1))] * sz
    grad = [np.zeros((1, 1))] * (sz - 1)

    d[sz - 1] = a[sz - 1] - Y
    for i in range(sz - 2, 1, -1):
        d[i] = d[i + 1] @ theta[i][:, 1:] * gradient_sigmoid(z[i])
    for i in range(1, sz - 1):
        grad[i] = (d[i + 1].T @ a[i]) / X.shape[0]
        grad[i][:, 1:] += punish / X.shape[0] * theta[i][:, 1:]
    return merge(grad)


def solve(X, Y, punish, layer_size):
    theta_size = 0
    for i in range(1, len(layer_size)):
        theta_size += (layer_size[i - 1] + 1) * layer_size[i]
    init_theta = np.random.uniform(-0.12, 0.12, theta_size)
    res = minimize(fun=compute_cost, x0=init_theta, args=(X, Y, punish, layer_size), method="TNC", jac=gradient,
                   options={'maxiter': 500})
    return split(res.x, layer_size)


def predict(theta, X):
    h = forward_prop(theta, X)[0][-1]
    return np.argmax(h, axis=1) + 1


def train(X, y, layer_size):
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(y)
    return solve(X, Y, 1, layer_size)


def main():
    data = loadmat("ex4data1.mat")
    train_X, test_X, train_y, test_y = train_test_split(data['X'], data['y'], train_size=0.8, test_size=0.2)
    theta = train(train_X, train_y, [400, 50, 50, 10])
    print(classification_report(test_y, predict(theta, test_X)))


if __name__ == "__main__":
    main()
