import numpy as np
import matplotlib.pyplot as plt

X = 2 * np.random.random((200, 2)) - 1# Return random floats in the half-open interval [200, 1.0).

def classifier1(X):
    # this takes an arry of input points and calculates y values that are 1 on one side of a sloping line
    return(np.sum( X * [0.8, 0.2], axis=1) > 0.3).astype(float)


y1 = classifier1(X)

plt.plot(X[y1 == 0.0, 0], X[y1 == 0.0,1], 'b.')
plt.plot(X[y1 == 1.0, 0], X[y1 == 1.0,1], 'r.')

plt.show()

def classifier2(X):
    return (np.sum( X * X, axis=1) < 0.66 ).astype(float)

y2 = classifier2( X )

plt.plot(X[y2==0.0,0],X[y2==0.0,1],'b.')
plt.plot(X[y2==1.0,0],X[y2==1.0,1],'r.')
plt.show()