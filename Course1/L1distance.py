import numpy as np
import pickle
import os


# def unpickle(file):
#     fileopen = open(file, 'rb')
#     dict = pickle.load(fileopen)
#     fileopen.close()
#     return dict


width = 32
height = 32
channel = 3


def readBinFile(filepath):
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
            temp.clear()
        if cnt < Xlen:
            temp.append(filedata2[i])
            cnt += 1
    picdata.append(temp)
    return picdata


data_dir = "/home/frank_ai/Desktop/"
cifar_path = os.path.join(data_dir, "cifar-10-batches-bin")
file_path = os.path.join(cifar_path, "data_batch_1.bin")
class_path = os.path.join(cifar_path, "batches.meta.txt")
if __name__ == '__main__':
    # dict = unpickle(file_path)
    print(file_path)
    data = readBinFile(file_path)
    lables = np.loadtxt(class_path, dtype="bytes").astype('str')
    print(len(data))
    print(data[9998][0])
    print(lables)

