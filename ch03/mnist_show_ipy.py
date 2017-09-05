# coding: utf-8

import sys, os
sys.path.append(os.pardir)  
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label = False)

pp = []
j = 0
for i in range(len(t_test)):
    if t_test[i] == 5:
        pp.append(i)
    else:
        pass
type(pp)
len(pp)

j = 0
for i in pp:
    j += 1
    if j > 70:
        break
    img = x_test[i]
    img = img.reshape(28, 28)
    plt.subplot(7,10,j)
    plt.axis("off")
    plt.imshow(img)
plt.show()