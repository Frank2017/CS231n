# _*_ coding:utf-8 _*_
import numpy as np
import os
from scipy import io


width = 32
height = 32
channel = 3


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


data_dir = "/home/frank_ai/Desktop/"
cifar_path = os.path.join(data_dir, "cifar-10-batches-bin")
class_path = os.path.join(cifar_path, "batches.meta.txt")


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
    # data_test.append(np.loadtxt(read_path,dtype='int', delimiter=' ').astype('int'))
    # print(len(data_test))
    # print(len(data_test[0]))
    # print(len(data_test[0][0]))
    # print(data_test[0][0][2])
    # # 将数据转化为mat格式，方便matlab读取
    #mat_path = os.path.join(cifar_path, "test_data.mat")
    # io.savemat(mat_path, {'test': data_test})
    # test_data_mat = io.loadmat(mat_path)
    # print (len(test_data_mat["test"][0]))
    ## 读取txt文件，将其存入data中，data是5*10000*3073
    # data = []
    # for i in list(range(1,6)):
    #     read_path = os.path.join(cifar_path, "data_batch_" + str(i) + ".txt")
    #     data.append(np.loadtxt(read_path,dtype='int', delimiter=' ').astype('int'))
    # mat_path = os.path.join(cifar_path, "train_data.mat")
    # io.savemat(mat_path, {"train":data})
    # mat_path = os.path.join(cifar_path, "train_data.mat")
    # data = io.loadmat(mat_path)
    # print(data['train'][0][0][2])


    # print(len(data))
    # print(len(data[0]))
    # print(len(data[0][0]))
