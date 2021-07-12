import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(theta, X, y, punish):
    loss = -(y @ np.log(sigmoid(X @ theta)) + (1 - y) @ np.log(1 - sigmoid(X @ theta))) / X.shape[0]
    reg = punish / (2 * X.shape[0]) * theta @ theta
    return loss + reg


def gradient(theta, X, y, punish):
    return 1 / X.shape[0] * X.T @ (sigmoid(X @ theta) - y) + punish / X.shape[0] * theta


def logistic_regression(X, y, punish=1):
    res = opt.fmin_tnc(func=compute_cost, x0=np.zeros(X.shape[1]), fprime=gradient, args=(X, y, punish))
    return res[0]


def cal(theta, x1, x2, degree):
    ret = 0
    p = 0
    for i in range(degree + 1):
        for j in range(i + 1):
            ret += np.power(x1, i - j) * np.power(x2, j) * theta[p]
            p += 1
    return ret


def find_decision_boundary(theta, degree, eps):
    t1 = np.linspace(-1, 1.5, 1000)
    t2 = np.linspace(-1, 1.5, 1000)
    coordinate = [[x1, x2] for x1 in t1 for x2 in t2]
    x_cord, y_cord = zip(*coordinate)
    h_val = pd.DataFrame({"x1": x_cord, "x2": y_cord})
    h_val["prediction"] = cal(theta, h_val["x1"], h_val["x2"], degree)
    decision = h_val[np.abs(h_val["prediction"]) < eps]
    return decision.x1, decision.x2


def predict(theta, X):
    temp = sigmoid(X @ theta)
    return [1 if x >= 0.5 else 0 for x in temp]


def accuracy(theta, X, y):
    ans = predict(theta, X)
    return sum([1 if ans[i] == y[i] else 0 for i in range(len(ans))]) / len(ans)


def build_model(degree):
    data = pd.read_csv('ex2data2.txt', header=None, names=["Exam 1", "Exam 2", "Admitted"])
    positive = data[data["Admitted"].isin([1])]
    negative = data[data["Admitted"].isin([0])]
    plt.scatter(positive["Exam 1"], positive["Exam 2"], c='b', marker='o', label="Admitted")
    plt.scatter(negative["Exam 1"], negative["Exam 2"], c='r', marker='x', label="Not admitted")
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    data2 = data
    x1 = data2["Exam 1"]
    x2 = data2["Exam 2"]
    data2.drop("Exam 1", axis=1, inplace=True)
    data2.drop("Exam 2", axis=1, inplace=True)
    data.insert(1, "Ones", 1)
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            data2["F" + str(i - j) + str(j)] = np.power(x1, i - j) * np.power(x2, j)
    X = data2.values[:, 1:]
    y = data2.values[:, 0]
    theta = logistic_regression(X, y, 1)
    px, py = find_decision_boundary(theta, degree, 2e-3)
    plt.scatter(px, py, c='y', s=10, label='Prediction')
    plt.show()
    print("Accuracy in the training set: %.2f%%" % (accuracy(theta, X, y) * 100))


if __name__ == "__main__":
    build_model(6)
