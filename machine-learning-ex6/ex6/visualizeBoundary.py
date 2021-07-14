import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets


def visualize_boundary(clf, X, x_min, x_max, y_min, y_max):
    x, y = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))
    Z = clf.predict(np.c_[x.ravel(), y.ravel()])
    Z = Z.reshape(x.shape)
    plt.contour(x, y, Z, colors='r')

