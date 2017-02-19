# _*_ coding:utf-8 _*_
import numpy as np
import math

def L_i(x,y,W):
    """
    非向量化的计算每一个样本数据的L值
    :param x: 列向量，cifar-10中的一个训练样本3073*1，其中第3073位为bias位，置为1
    :param y: 该样本对应的分类下标
    :param W: 权重矩阵  3073*10
    :return: 当前第i个损失函数得到的值
    """
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
    """
    向量化的计算每一个样本数据的L值
    :param x: 列向量，cifar-10中的一个训练样本3073*1，其中第3073位为bias位，置为1
    :param y: 该样本对应的分类下标0-9
    :param W: 权重矩阵  3073*10
    :return: 当前第i个损失函数得到的值
    """
    delta = 1.0
    scores = W.dot(x)
    margins = np.maximum(0, scores - scores[y] + delta)
    margins[y] = 0
    loss_i = sum(margins)
    return loss_i
    pass


def L(X, y, W):
    """
    向量化的计算数据的L值
    :param X: 列向量，cifar-10中的训练集3073*50000，其中第3073位为bias位，置为1
    :param y: 样本对应的分类下标0-9，50000*1
    :param W: 权重矩阵  3073*10
    :return: 当前第i个损失函数得到的值
    """
    delta = 1.0
    scores = W.T.dot(X) #10*50000
    print(scores.shape)
    D = X.shape[1] #测试集数据量
    K = W.shape[1] #分类的数目
    scores_correct = np.reshape(scores[(y.T, list(range(D)))],(1,D))  #1*50000
    wyx = np.ones((K,1)).dot(scores_correct)  # 10*50000
    margins = np.maximum(0,scores - wyx + delta)
    margins[(y.T, list(range(D)))] = 0.0
    return np.sum(margins)
    pass


W = np.array(range(1, 101))
W = np.reshape(W,(20, 5))
y = (np.random.random(20) * 5).astype(int)
y = np.reshape(y,(20,1))
X = np.array(range(100, 500)).reshape((20, 20))
print(X)
print(L(X,y,W))

