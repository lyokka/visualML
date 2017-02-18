'''
import matplotlib.pyplot as plt
import numpy as np
X = np.matrix("1, 1; 1, 3; 1, 5; 1, 7; 1, 9; 1, 10")
Y = np.matrix([2, 1.5, 2, 3.5, 7, 7.2])
Y = Y.reshape([6, 1])


t0 = np.arange(-1000, 1000, 10)
t1 = np.arange(-1000, 1000, 10)

J = np.zeros([len(t0), len(t0)])

for i in range(len(t0)):
    for j in range(len(t1)):
        for k in range(len(X)):
            J[i, j] = J[i, j] + (t0[i] + t1[j]*X[k, 1] - Y[k])**2
        J[i, j] = J[i, j]/2
print(J)
         
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(t0, t1)
surf = ax.plot_surface(X, Y, J, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('train.csv')
X = data[['LotArea','LotFrontage']]
Y = data['SalePrice']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X['LotArea'], X['LotFrontage'], Y, c = 'blue', marker = 'o')
ax.set_xlabel('LotArea')
ax.set_ylabel('LotFrontage')
ax.set_zlabel('SalePrice')

#loss function
def J(Y_hat, Y):
    return np.dot((Y_hat - Y), np.transpose(Y_hat - Y))/2

#GD solution
X = data['LotArea']
N = len(X)
#initialization
Theta = np.array([1.588e+05, 2])
Y_hat = np.ones(N)
alpha = 6.5e-12
ones = pd.DataFrame(np.ones([N, 1]))
X = ones.join(X)

while J(Y_hat, np.asarray(Y)) > 4.2834599e+12:
    Y_hat = np.dot(X, Theta)
    Theta = Theta + alpha*np.dot((np.asarray(Y) - Y_hat), X)
    print(J(Y_hat, np.asarray(Y)))
