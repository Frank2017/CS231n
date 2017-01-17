import  numpy as np
import os

# def unpickle(file):
#     fileopen = open(file, 'rb')
#     dict = pickle.load(fileopen)
#     fileopen.close()
#     return dict

data_dir = "D:\PYProjects"
cifar_path = os.path.join(data_dir, "cifar-10-batches-py")
file_path = os.path.join(cifar_path, "data_batch_1")
if __name__ == '__main__':
    # dict = unpickle("D:\PYProjects\cifar-10-batches-py\data_batch_1")
    print(file_path)
    dict = np.load(file_path)
    print(len(dict))

