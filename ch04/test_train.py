# coding: utf-8
# import sys, os
# sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
# import numpy as np
# import matplotlib.pyplot as plt
# from dataset.mnist import load_mnist
# from two_layer_net import TwoLayerNet

# # データの読み込み
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# print(x_train.shape)
# print(t_train.shape)

# train_size = x_train.shape[0]
# print(train_size)
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch
# x_batch = x_train[batch_mask]so
# t_batch = t_train[batch_mask]
# print(x_batch.shape)
# print(t_batch.shape)
#--------------------------------------------------------------------------------
# import numpy as np
# def numerical_diff(f, x):
# 	h = 1e-4
# 	return (f(x + h) - f(x - h)) / (2 * h)
# def function_tmp1(x0):
# 	return x0 ** 2 + 4.0 ** 2.0

# print(function_tmp1(3)) #25
# print(numerical_diff(function_tmp1, 3.0))
#--------------------------------------------------------------------------------
# import numpy as np
# a = np.random.randn(2,3)
# print(a)
#--------------------------------------------------------------------------------
# import numpy as np

# def numerical_gradient(f, x):
#     h = 1e-4 # 0.0001
#     grad = np.zeros_like(x)
#     for idx in range(x.size): #x.size = 18*18 = 324
#         tmp_val = x[idx]
#         x[idx] = tmp_val + h
#         fxh1 = f(x) # f(x+h)
        
#         x[idx] = tmp_val - h 
#         fxh2 = f(x) # f(x-h)
#         grad[idx] = (fxh1 - fxh2) / (2*h)
        
#         x[idx] = tmp_val # 値を元に戻す
        
#     return grad
#--------------------------------------------------------------------------------
import numpy as np

def func (a, b):
	return a + b
print(func(3,5))

f = lambda a, b : a + b
print(f(3,5))
