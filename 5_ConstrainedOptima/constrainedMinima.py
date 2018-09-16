import numpy as np 
from matplotlib import pyplot as plt 

# 2D constraints in the form Wx, Wy, b
DivConstraint = [0.25, 1.0, -1]
DivFunc = []

x = np.linspace(-5.0, 5.0, 100)
y = np.linspace(-5.0, 5.0, 100)
X, Y = np.meshgrid(x,y)

F2 = X**2 + Y**2 - 4.0
F3 = X**2 + Y**2 - 9.0
F4 = X**2 + Y**2 - 16.0
F5 = X**2 + Y**2 - 25.0
plt.contour(X,Y,F2,[0])
plt.contour(X,Y,F3,[0])
plt.contour(X,Y,F4,[0])
plt.contour(X,Y,F5,[0])
plt.show()
