# _*_ coding:utf-8 _*_
import numpy as np
from numpy import *
import os
from scipy import io


width = 32
height = 32
channel = 3
# Linux Path
# data_dir = "/home/frank_ai/Desktop/"
# cifar_path = os.path.join(data_dir, "cifar-10-batches-bin")
# class_path = os.path.join(cifar_path, "batches.meta.txt")

# Windows Path
data_dir = "D:\PYProjects"
cifar_path = os.path.join(data_dir, "cifar-10-batches-bin")
class_path = os.path.join(cifar_path, "batches.meta.txt")


def readBinFile(filepath):
    """
    根据提供的filepath读取图片的二进制文件
    :param filepath:
    :return: picdata  10000*3073,第一列是class，剩余3072是图片数据，按R（1-1024）G（1025-2048）B（2049-3072）排列
    """
    binfile = open(file_path, mode='rb')
    filedata = binfile.read()
    filesize = binfile.tell()
    filedata2 = bytearray(filedata)
    Xlen = width * height * channel + 1
    cnt = 0
    temp = []
    picdata = []
    for i in range(0,filesize):
        if cnt == Xlen:
            cnt = 0
            picdata.append(temp)
            temp=[]
        if cnt < Xlen:
            temp.append(filedata2[i])
            cnt += 1
    picdata.append(temp)
    return picdata

def writeFile(outpath, data):
    """
    :param outpath: 输出文件的路径
    :param data: 写入文件的数据
    :return: null
    """
    try:
        fileout = open(outpath, 'w+')
        for i in list(range(len(data))):
            for j in list(range(len(data[i]))):
                if j < len(data[i]) - 1:
                    fileout.write(str(data[i][j])+' ')
                else:
                    fileout.write(str(data[i][j]))
            fileout.write('\n')
    finally:
        fileout.close()


def L1_Distance(arr1, arr2):
    """
    :notice: 传入的图片数据必须是去除标签后的纯照片数据,32*32*3
    :param arr1:图片数据1  类型均为numpy.narray
    :param arr2: 图片数据2
    :return: 返回两个图片的L1距离的绝对值
    """
    return np.sum(np.abs(arr1 - arr2),axis=1)
    pass


def L2_Distance(arr1, arr2):
    """
    :notice: 传入的图片数据必须是去除标签后的纯照片数据,32*32*3
    :param arr1:图片数据1 类型均为numpy.narray
    :param arr2: 图片数据2
    :return: 返回两个图片的L2距离的绝对值
    """
    return np.sqrt(np.sum(np.square(arr1 - arr2), axis= 1))
    pass


def predict_Y(train, test):
    """
    :notice: train 和 test均为读取的mat类型数据，内部将data和label分开存储
    :param train: 训练数据
    :param test: 测试数据
    :return: accuracy:float 返回预测结果的准确度
    :return:Ypridict: array 返回各项测试集中数据的预测值
    """
    train_data = train['train_data']
    train_label = train['train_label']
    test_data = test['test_data']
    test_label= test['test_label']
    test_num = test_data.shape[0]
    Ypredict = []
    for i in list(range(0,test_num)):
        distance = L1_Distance(test_data[i,:], train_data)
        # distance = L2_Distance(test_data[i, :], train_data)
        min_index = np.argmin(distance)
        Ypredict.append(train_label[min_index])
    accuracy = 0.0
    correct_pred = 0
    for j in list(range(0, test_num)):
        if Ypredict[i] == test_label[i]:
            correct_pred += 1
    accuracy = correct_pred / test_num * 100
    return (Ypredict, accuracy)
    pass
if __name__ == '__main__':

    # #将读入的二进制文件转换为数字型txt存入txt文件中
    # for i in list(range(1,6)):
    #     file_path = os.path.join(cifar_path, "data_batch_" + str(i) + ".bin")
    #     outpath = os.path.join(cifar_path, "data_batch_" + str(i) + ".txt")
    #     data = readBinFile(file_path)
    #     writeFile(outpath, data)
    #     data = []


    # #读入标签0-9对应的分类名称
    # lables = np.loadtxt(class_path, dtype="str").astype('str')
    # print(lables)

    # #将test文件转换为txt格式,10000*3073,其中第一列为标签列
    # file_path = os.path.join(cifar_path, "test_batch.bin")
    # outpath = os.path.join(cifar_path, "test_batch.txt")
    # data = readBinFile(file_path)
    # writeFile(outpath, data)


    # # 读取测试集txt文件,1*10000*3073
    # data_test = []
    # read_path = os.path.join(cifar_path, "test_batch.txt")
    # # data_test = np.loadtxt(read_path, dtype='int', delimiter=' ')
    # data_test.append(np.loadtxt(read_path,dtype='int', delimiter=' ').astype('int'))
    # data_test = np.array(data_test)
    # print(type(data_test))
    # print(shape(data_test))
    # print(shape(data_test[0][0:,1:]))
    # print(data_test[0][0:1, 2])
    # print(len(data_test))
    # print(len(data_test[0]))
    # print(len(data_test[0][0]))
    # print(data_test[0][0][2])
    # 将数据转化为mat格式，方便matlab读取,通过python读取出的数据类型为numpy.narray
    # mat_path = os.path.join(cifar_path, "test_data.mat")
    # io.savemat(mat_path, {'test_data': data_test[0][0:, 1:], 'test_label':data_test[0][0:, 0]})
    # test_data_mat = io.loadmat(mat_path)
    # print (shape(test_data_mat["test_data"]))
    # print (shape(test_data_mat["test_label"]))


    ## 读取txt文件，将其存入data中，data是5*10000*3073
    # data = []
    # for i in list(range(1,6)):
    #     read_path = os.path.join(cifar_path, "data_batch_" + str(i) + ".txt")
    #     data.append(np.loadtxt(read_path,dtype='int', delimiter=' ').astype('int'))
    # data = np.array(data)
    # mat_path = os.path.join(cifar_path, "train_data.mat")
    # io.savemat(mat_path, {"train_data":reshape(data[:,:,1:],(50000,3072)), "train_label":reshape(data[:,:,0],(50000,1))})
    # # mat_path = os.path.join(cifar_path, "train_data.mat")
    # data = io.loadmat(mat_path)
    # print(shape(data['train_data']))
    # print(shape(data['train_label']))


    # print(len(data))
    # print(len(data[0]))
    # print(len(data[0][0]))

    # arr1 = np.array([1,2,3,4])
    # arr2 = np.array([[[4,3,2,1],[4,2,1,3]],[[4,5,6,7],[5,6,4,7]],[[2,7,8,9],[8,9,2,7]]])
    # print(arr2.shape[0])
    # print(type(arr2))
    # print(arr2[:,0:1,2:3])
    #
    # print(type(arr2))

    train = io.loadmat(os.path.join(cifar_path,"train_data.mat"))
    test = io.loadmat(os.path.join(cifar_path, "test_data.mat"))
    print (shape(train['train_data']))
    print (shape(train['train_label']))
    print (shape(test['test_data']))
    print (shape(test['test_label']))
    # (pred, accuracy) = predict_Y(train, test)
    # print (shape(pred))
    # print (accuracy)

