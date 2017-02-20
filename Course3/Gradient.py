# _*_ coding:utf-8 _*_
import numpy as np
from scipy import io
import os
import time


CIFAR10_PATH = r"D:\PYProjects\cifar-10-batches-bin"
CIFAR10_CLASS = os.path.join(CIFAR10_PATH, "batches.meta.txt")
CIFAR10_TRAIN = os.path.join(CIFAR10_PATH, "train_data.mat")
CIFAR10_TEST = os.path.join(CIFAR10_PATH, "test_data.mat")
data = io.loadmat(CIFAR10_TRAIN)
X_train = data['train_data'].T # 3072*50000
Y_train = data['train_label'] #50000*1
bias_trick = np.ones((1,X_train.shape[1]))
X_train = np.append(X_train, bias_trick, axis=0) # 3073*50000
CLASS_NUM = 10
TRAIN_DATA_NUM = 50000
TEST_DATA_NUM = 10000
DATA_DIMENSION = 3072
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
    loss = np.sum(margins) / D + np.sum(np.square(W))
    return loss
    pass


def CIFAR10_loss_function(W):
    """
    :param W: 权重矩阵  3073*10
    :return: loss 对应W的损失函数值
    """
    return L(X_train,Y_train, W)
    # print(X_train.shape)
    # print(np.sum(X_train[3072]))

def eval_numerical_gradient(f, x):
    """
    :param f: 计算损失函数
    :param x: 待计算点（向量）
    :return:x的梯度
    """
    h = 0.00001
    fx = f(x)
    grad = np.zeros(x.shape)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h
        fpxh = f(x)
        x[ix] = old_value - h
        fdxh = f(x)
        x[ix] = old_value
        grad[ix] = (fpxh - fdxh) / (2*h)  # [f(x+h) - f(x-h)] / 2h
        it.iternext()
    return grad
    pass


if __name__ == "__main__":
    W = np.random.rand(DATA_DIMENSION+1, CLASS_NUM)*0.001
    st = time.time()
    df = eval_numerical_gradient(CIFAR10_loss_function, W)
    en = time.time()
    print("Running time is %f", en - st)
    print("df shape is", df.shape)

    loss_original = CIFAR10_loss_function(W)
    print("Original loss is %f", loss_original)
    for step in [-10, -9, -8, -7, -6, -5, -4, -3]:
        step_size = 10 ** step
        W_new = W - df * step_size
        loss_new = CIFAR10_loss_function(W_new)
        print("For step size is %f , loss is %f", step_size, loss_new)


    #test
    # a = np.array([[1,2],[4,5],[7,8]])
    # print(a)
    # b = np.array([3,6]).reshape(1,2)
    # print(b.shape)
    # print(np.append(a,b,axis=0))