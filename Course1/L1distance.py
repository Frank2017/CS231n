# _*_ coding:utf-8_*_
import  numpy as np
import pickle

def unpickle(file):
    fileopen = open(file, 'rb')
    dict = pickle.load(fileopen)
    fileopen.close()
    return dict


if __name__ == '__main__':
    dict = unpickle(".//cifar-10-batches-py//data_batch_1")
    len(dict)

