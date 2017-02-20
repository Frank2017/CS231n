# _*_ coding:utf-8 _*_
import numpy as np
from scipy import io
import os
import time
import math


CIFAR10_PATH = r"D:\PYProjects\cifar-10-batches-bin"
CIFAR10_CLASS = os.path.join(CIFAR10_PATH, "batches.meta.txt")
CIFAR10_TRAIN = os.path.join(CIFAR10_PATH, "train_data.mat")
CIFAR10_TEST = os.path.join(CIFAR10_PATH, "test_data.mat")
data = io.loadmat(CIFAR10_TRAIN)
X_train = data['train_data'].T  # 3072*50000
Y_train = data['train_label']  # 50000*1
bias_trick = np.ones((1,X_train.shape[1]))
X_train = np.append(X_train, bias_trick, axis=0)  # 3073*50000
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
    scores = W.T.dot(X)  # 10*50000
    D = X.shape[1]  # 测试集数据量
    K = W.shape[1]  # 分类的数目
    scores_correct = np.reshape(scores[(y.T, list(range(D)))], (1, D))  # 1*50000
    wyx = np.ones((K, 1)).dot(scores_correct)  # 10*50000
    margins = np.maximum(0, scores - wyx + delta)
    margins[(y.T, list(range(D)))] = 0.0
    loss = np.sum(margins) / D + np.sum(np.square(W))
    return loss
    pass


def CIFAR10_loss_function(W):
    """
    :param W: 权重矩阵  3073*10
    :return: loss 对应W的损失函数值
    """
    return L(X_train, Y_train, W)
    # print(X_train.shape)
    # print(np.sum(X_train[3072]))


def eval_numerical_gradient(f, x):
    """
    :param f: 计算损失函数
    :param x: 待计算点（向量）
    :return:x的梯度
    """
    h = 0.00001
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
        print("Iteration index is ", ix, "Gradient is ", grad[ix])
    return grad
    pass


def F(x, y, z, step_size):
    q = x + y
    dfdq = z
    dfdz = q
    dqdx = 1
    dqdy = 1
    dfdx = dfdq * dqdx
    dfdy = dfdq * dqdy
    x += dfdx * step_size
    y += dfdy * step_size
    z += dfdz * step_size
    return x, y, z, (x+y)*z

if __name__ == "__main__":
    # W = np.random.rand(DATA_DIMENSION+1, CLASS_NUM)*0.001
    # st = time.time()
    # df = eval_numerical_gradient(CIFAR10_loss_function, W)
    # en = time.time()
    # print("Running time is %f", en - st)
    # print("df shape is", df.shape)
    #
    # loss_original = CIFAR10_loss_function(W)
    # print("Original loss is %f", loss_original)
    # for step in [-10, -9, -8, -7, -6, -5, -4, -3]:
    #     step_size = 10 ** step
    #     W_new = W - df * step_size
    #     loss_new = CIFAR10_loss_function(W_new)
    #     print("For step size is %f , loss is %f", step_size, loss_new)
    # # test1
    # f(x,y,z) = (x+y)*z
    # x = -2
    # y = 5
    # z = -4
    # step_size = 0.01
    # cnt = 20
    # while cnt:
    #     cnt -= 1
    #     (x, y, z, f) = F(x, y, z, step_size)
    #     print("%d====> x = %f  y = %f  z = %f  f = %f"%(cnt, x, y, z, f))
    # # test2
    # # f(x,y) = (x + sigmod(y)) / (sigmod(x) + (x + y)^2)
    # x = 3
    # y = -4
    # # forword-propagation
    # sigy = 1.0 / (1 + math.exp(-1 * y))
    # num = x + sigy
    # sigx = 1.0 / (1 + math.exp(-1 * x))
    # xpy = x + y
    # xpysqr = xpy ** 2
    # den = sigx + xpysqr
    # invden = 1.0 / den
    # f = num * invden
    # # back-propagation of f=num * invden
    # dnum = invden
    # dinvden = num
    # dden = -1.0 / (den ** 2) * dinvden
    # dsigx = 1.0 * dden
    # dxpysqr = 1.0 * dden
    # dxpy = 2 * xpy * dxpysqr
    # dx = 1.0 * dxpy
    # dy = 1.0 * dxpy
    # # 使用“+=”而不使用“=”的原因：符合多变量链式法则：如果一个变量在链路中存在于多个分支，当对其反向传递求导数时，
    # # 需要将所有汇入该单元的有关该变量的导数值求和。如果使用“=”则值只会被反复的赋值而不是累加。
    # # This follows the multi-variable chain rule in Calculus, which states that if a variable branches
    # # out to different parts of the circuit, then the gradients that flow back to it will add.
    # dx += ((1 - sigx) * sigx) * dsigx
    # dx += 1.0 * dnum
    # dsigy = 1.0 * dnum
    # dy += ((1 - sigy) * sigy) * dsigy
    # # test3 f = ((x * y) + max(z, w))*2
    x = 3.0
    y = -4.0
    z = 2.0
    w = -1.0
    # forword propagation
    xmy = x * y
    mzw = max(z, w)
    fsum = xmy + mzw
    f = fsum * 2
    # backword propagation
    dfsum = 2
    dxmy = 1.0 * dfsum
    dmzw = 1.0 * dfsum
    dz = dmzw * (1.0 if z >= w else 0.0)
    dw = dmzw * (1.0 if w >= z else 0.0)
    dx = y * dxmy
    dy = x * dxmy
    print(dx, dy, dz, dw)
