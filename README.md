---
title: NG-machine learning课程前六周作业总结
date: 2021-06-02 19:58:42
mathjax: true
categories: neural network
tags:
- machine learning
- neural network
---

Upd on 2021/7/12：将本地文件重新整理了下，放到了[我的GitHub](https://github.com/yyDing1/NG-Machine-Learning)上，源代码在branch分支里面，main分支里面的是整理后的工程文件夹

<!--more-->

前六周主要内容包括：

1. 机器学习基础：线性回归，逻辑回归（应用到多分类）
2. 简单神经网络模型（前馈神经网络和反向传播算法）

主要是一些基础的实现，大都是分析损失函数，计算梯度和损失值这些过程。能用矩阵乘法不用循环（问就是有强迫症）

复制我的代码到pycharm里面会发现没有绿勾，有一些week warning，大都是矩阵$X$，向量$y$。emmm孩子已经尽力了，我只是觉得矩阵大写，向量小写表示应该很合理，所以这个week warning我就直接ignore掉了

工程文件里不想改warning了，大多是命名造成的week warning，没有绿勾，我妥协了

另外写了一个自定义层数与神经元数量的全连接层算法，可以作为一个$api$调用，类似于低配版的PyTorch全连接网络（在$master/machine-learning-ex4/template.py$里面），因为不属于作业内容所以没有放到$main$分支里

因为之前赶考试和复习，后面几个$exercise$都还没写（忍不住要吐槽$SCU$的$ai$班的培养计划了）

后面几个$exercise$准备用已有的框架写了，从$0$开始写代码太耗时间了。。。而且框架里的可视化做得很棒

计划$7.20$号前完成所有$exercise$然后去学pytorch复现论文

成果如下（均用Python实现）：

- 线性回归模型，梯度下降和最小二乘法两种方式实现
  - 梯度下降：固定学习率的方式进行梯度下降；调用$scipy.optimize$库中的$minimize$函数进行快速梯度下降
  - 最小二乘法：因为线性回归模型的损失函数为凸函数，固可以直接对其求导并令导数为$0$，来直接求得$argmin_{\theta} J(\theta)$，需要注意矩阵求逆的问题
- 逻辑回归模型，二分类问题，并将其扩展到多分类问题（$|y|$个分类器，一对多分类）
- 简单神经网络模型（前馈神经网络和反向传播算法），先构建两个隐层的情况，并加以改进，变为自定义层数以及自定义每层的神经元数量，增加了其泛用性

# 机器学习基础

## 线性回归

### 梯度下降方法

损失函数：$J(\theta)=\frac{1}{2m}\sum\limits_{i=1}^{m}({h}_{\theta}({x}^{(i)})-{y}^{(i)})^{2}$

梯度：$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum\limits_{i=1}^{m}({h}_{\theta}({x}^{(i)})-{y}^{(i)})x_j^{(i)}$，$\frac{\partial J(\theta)}{\partial \theta} = X^TX\theta - X^Ty$

因为固定学习率的方式特征每一维值极差相差较大，所以作了标准化

源代码：

```Python
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
    plt.show()


if __name__ == "__main__":
    main()
```

随迭代次数代价值的变化

![](阶段性总结1\ex1_1.png)

### 正规方程方法

$\frac{\partial J(\theta)}{\partial \theta} = X^TX\theta - X^Ty = 0$

$\theta = (X^TX)^{-1}X^Ty$

```python
def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y
```

## 逻辑回归

### 基础实现

逻辑回归函数：$h_\theta( x ) =\frac{1}{1+{e^{-\theta^T x}}}$

损失函数：$J(\theta)=\frac{1}{m}\sum\limits_{i=1}^{m}{(-{y}^{(i)}\log({h}_{\theta}({x}^{(i)}))-(1-{y}^{(i)})\log(1-{h}_{\theta}({x}^{(i)})))}$

梯度：$\frac{\partial J(\theta)}{\partial {\theta}_{j}}=\frac{1}{m}\sum\limits_{i=1}^{m}{({h}_{\theta }( {x}^{(i)})-{y}^{(i)})x_{j}^{(i)}}$

源代码：

```python
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
    data = np.loadtxt("ex2/ex2data1.txt", dtype=float, delimiter=',')
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
```

分类结果可视化：

![](阶段性总结1/ex2_1.png)

### 扩展到多项式版本

因为在观察第二个数据集的过程中，发现该数据集用线性分很难实现，所以考虑将模型扩展到多项式版本

即：$h_{\theta}{(x)} = \sum\limits_{i = 0}^{degree}\sum\limits_{j = 0}^{i}{(x_1)^{degree - j}(x_2)^j}$

多项式的最高次数$degree$需要人工设定

![](阶段性总结1/ex2_2.png)

损失函数：$J(\theta) = \frac{1}{m} \sum\limits_{i = 1}^{m}{-y^{(i)}logh_{\theta}{(x^{(i)})}} - (1 - y^{(i)})log(1 - h_{\theta}{(x^{(i)})}) + \frac{\alpha}{2m}\theta^2$

梯度：$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum\limits_{i = 1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})x_j^{(i)} + \frac{1}{m}\theta_j$，$\frac{\partial J(\theta)}{\partial \theta} = \frac{1}{m}X^T (h_{\theta}(x) - y) + \frac{\alpha}{m}\theta$

源代码：

```python
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
    data = pd.read_csv('ex2/ex2data2.txt', header=None, names=["Exam 1", "Exam 2", "Admitted"])
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
```

分类结果：

![](阶段性总结1/ex2_3.png)

### 关于正则项的一些尝试

$\lambda = 0$：无正则项（黄色点描的线为决策边界）

![](阶段性总结1/ex2_4.png)

$\lambda = 10$：正则项较大

![](阶段性总结1/ex2_5.png)

### 二分类扩展：多分类器

假设有$n$个类别，转换为$n$个二分类问题，最后取置信程度最大的那个类

```python
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
    data = loadmat("ex3/ex3data1.mat")
    ans = train(data['X'], data['y'])
    y_predict = one_vs_all(data['X'], ans)
    print(classification_report(data['y'], y_predict))


if __name__ == '__main__':
    main()
```

# 神经网络基础

## 简单前馈神经网络+反向传播

损失函数：$J(\theta) = \frac{1}{m}\sum\limits_{i = 1}^{m}\sum\limits_{k - 1}^{K}{-y_{k}^{(i)}log(h_{\theta}{(x^{(i)})})_k} - (1 - y_{k}^{(i)})log(1 - h_{\theta}{(x^{(i)})})_k + \frac{\alpha}{2m}\sum\limits_{l}\sum\limits_{i}\sum\limits_{j}{\theta_{i, j}^{(l)}}^2$

反向传播：

$sigmoid(z)=g(z)=\frac{1}{1+e^{-z}}$

$g'(z)=\frac{d}{dz}g(z)=g(z)(1-g(z))$

$\delta_i = \delta_{i + 1}\times\theta_i \times g'(z_{i})$

$\frac{\partial J(\theta)}{\partial \theta^{(l)}} = \delta_{i + 1}^T\times a_i + \frac{\alpha}{m}\theta^{(l)}$

### 一个两层的神经网络模型(含正则化项和反向传播梯度下降)

```python
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
    data = loadmat("ex4/ex4data1.mat")
    init_theta = np.random.uniform(-0.12, 0.12, 10285)
    encoder = OneHotEncoder(sparse=False)
    X = data['X']
    Y = encoder.fit_transform(data['y'])
    ans_all = neural_network(init_theta, X, Y, 1)
    print(ans_all)
    theta1 = ans_all[:25 * 401].reshape(25, 401)
    theta2 = ans_all[25 * 401:].reshape(10, 26)
    print(classification_report(data['y'], predict(theta1, theta2, X)))


if __name__ == "__main__":
    main()
```

### 自定义层数以及神经元数量

将上次的代码修改为自定义层数以及自定义每层的神经元数量，增加了其泛用性

因为$minimize$函数的第一个参数需要为一个一维向量，所以对所有$\theta$进行$flatten+append$处理，并且另外写了$merge, split$函数，用来将$\theta$封装成一维向量和将一维向量展开

通过尝试也验证了一个想法：理论上来讲，增加神经网络的层数与每层的神经元数量会使训练的结果更好

事实上，通过将$layersize = [400, 25, 10]$调整为$[400, 50, 50, 10]$后，模型准确率有所上升

选定$trainsize: testsize = 8:2$

源代码：

```python
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
    data = loadmat("ex4/ex4data1.mat")
    train_X, test_X, train_y, test_y = train_test_split(data['X'], data['y'], train_size=0.8, test_size=0.2)
    theta = train(train_X, train_y, [400, 50, 50, 10])
    print(classification_report(test_y, predict(theta, test_X)))


if __name__ == "__main__":
    main()
```

1. $layersize = [400, 25, 10]$

   ![](阶段性总结1/ex4_1.png)

2. $layersize = [400, 50, 50, 10]$

   ![](阶段性总结1/ex4_2.png)

