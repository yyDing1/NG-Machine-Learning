import numpy as np
from sklearn import svm


def select_best_parameters(X, y, X_val, y_val):
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    best_score = 0
    best_c = 0
    best_gamma = 0
    for c in C_values:
        for gamma in gamma_values:
            clf = svm.SVC(C=c, gamma=gamma)
            clf.fit(X, y)
            now_score = clf.score(X_val, y_val)
            if now_score > best_score:
                best_score = now_score
                best_c = c
                best_gamma = gamma
    return best_c, best_gamma, best_score
