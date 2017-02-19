# _*_ coding:utf-8 _*_
import numpy as np
import time

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
    D = X.shape[1] #测试集数据量
    K = W.shape[1] #分类的数目
    scores_correct = np.reshape(scores[(y.T, list(range(D)))],(1,D))  #1*50000
    wyx = np.ones((K,1)).dot(scores_correct)  # 10*50000
    margins = np.maximum(0,scores - wyx + delta)
    margins[(y.T, list(range(D)))] = 0.0
    return np.sum(margins)
    pass

# st = time.time()
# W = np.random.random(10*3073).reshape(3073,10)
# y = (np.random.random(50000) * 10).astype(int)
# y = np.reshape(y,(50000,1))
# X = (np.random.random(50000*3073) * 255).reshape((3073, 50000))
# en = time.time()
# print("finish  ->", en - st)
# st = time.time()
# print(L(X,y,W))
# en = time.time()
# print(en - st)

f = np.array([1, -2, 0])
p = np.exp(f) / np.sum(np.exp(f))  #softmax

print(p)
print(np.sum(p))

# W = np.array([[0.001,-0.05,0.1,0.05,0.0],[0.7,0.2,0.05,0.16,0.2],[0.0,-0.45,-0.2,0.03,-0.3]]).T.reshape(5,3) #5*3
# X = np.array([-15,22,-44,56,1]).T.reshape(5,1) # 5*1
# st = time.time()
# y = np.array([2]) # 1*1
# en = time.time()
# print(L(X,y,W), "-->", en - st)
