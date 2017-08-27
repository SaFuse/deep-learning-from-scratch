import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 5)
y = np.arange(-10, 10, 5)

X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2)

plt.contour(X, Y, Z)
plt.show()