# _*_ coding:utf-8 _*_
import numpy as np
import scipy

def L_i(x,y,W):
    delta = 1.0
    scores = W.dot(x)
    correct_class_score = scores[y]
    D = W.shape[0]
    loss_i = 0.0
    for j in list(range(D)):
        if j == y:
            continue
        else:
            loss_i += max(0, scores[j] - correct_class_score + delta)
    return loss_i
    pass


def L_i_vectorized(x,y,W):
    delta = 1.0
    scores = W.dot(x)
    margins = np.maximum(0, scores - scores[y] + delta)
    margins[y] = 0
    loss_i = sum(margins)
    return loss_i
    pass


def L(X, y, W):
    """
    :param X:
    :param y:
    :param W:
    :return:
    """
    pass