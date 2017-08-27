# coding: utf-8

# import sys, os
# sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
# from dataset.mnist import load_mnist 
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True)
# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

#--------------------------------------------------------------------------------
# try:
#     import urllib.request
# except ImportError:
#     raise ImportError('You should use Python 3.x')
# import os.path
# import gzip
# import pickle
# import os
# import numpy as np


# url_base = 'http://yann.lecun.com/exdb/mnist/'
# key_file = {
#     'train_img':'train-images-idx3-ubyte.gz',
#     'train_label':'train-labels-idx1-ubyte.gz',
#     'test_img':'t10k-images-idx3-ubyte.gz',
#     'test_label':'t10k-labels-idx1-ubyte.gz'
# }

# dataset_dir = os.path.dirname(os.path.abspath(__file__))
# print(dataset_dir)
# save_file = dataset_dir + "/mnist.pkl"
# print(save_file)

#--------------------------------------------------------------------------------
# import sys, os
# sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
# import numpy as np
# import pickle
# from dataset.mnist import load_mnist
# from common.functions import sigmoid, softmax


# def get_data():
#     (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
#     return x_test, t_test


# def init_network():
#     with open("sample_weight.pkl", 'rb') as f:
#         network = pickle.load(f)
#     return network


# def predict(network, x):
#     W1, W2, W3 = network['W1'], network['W2'], network['W3']
#     b1, b2, b3 = network['b1'], network['b2'], network['b3']

#     a1 = np.dot(x, W1) + b1
#     z1 = sigmoid(a1)
#     a2 = np.dot(z1, W2) + b2
#     z2 = sigmoid(a2)
#     a3 = np.dot(z2, W3) + b3
#     y = softmax(a3)

#     return y


# x, t = get_data()
# network = init_network()
# accuracy_cnt = 0
# print(len(x))
# print("X.shape" + str(x.shape))
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
#     if p == t[i]:
#         accuracy_cnt += 1

# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
#--------------------------------------------------------------------------------
import numpy as np
x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
print(x)
y = np.argmax(x, axis = 0)
print(y)
z = np.max(x, axis = 0)
print(z)
zz = np.max(x, axis = 1)
print(zz)