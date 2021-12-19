"""
Following section 3.3 of Geometric Algebra for Physicists
"""

import numpy as np
import matplotlib.pyplot as plt

from gapy.ga3d import *


# first pick two vectors to define the plane of our orbit
b = vec(1, 1, 0)
a = vec(0.25, 1, -0.5)

# get the unit bivector representing this plane
L = (a*b).project(2).unit
e = a.unit

# these are the two scalar parameters of the general solution
A = 1
B = 2

# calculate the analytical solution
s = np.linspace(0, 2*np.pi, 1001)
U = A*np.exp(s*L) + B*np.exp(-s*L)
x = U*U*e

# x makes two orbits for one in U, so to visualize this we slightly offset the orbit over time
p = I*L  
x = x + (np.linspace(0, 1, len(s))*p)

# unwrap the vectors and put them in raw numpy arrays for plotting
xx = np.array([xx.vector for xx in x]) 

# U is an ellipse in the 2d space of "scalar x unit bivector"
UU = np.array([[UU.scalar, (L*e*UU*e).scalar] for UU in U])

# plot it
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot(xx[:, 0], xx[:, 1], xx[:, 2], label="x(t)")
ax.plot([0, a.vector[0]], [0, a.vector[1]], [0, a.vector[2]], label="a")
ax.plot([0, b.vector[0]], [0, b.vector[1]], [0, b.vector[2]], label="b")
ax.scatter([0], [0], [0], marker=".", color="k")
ax.set_title("Spatial Trajectory")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.legend()

# hack to work around axes3d not supporting aspect "equal"
mx = max([np.max(xx[:, i]) for i in range(3)])
mn = min([np.min(xx[:, i]) for i in range(3)])
ax.scatter([mx, mn,  0,  0,  0,  0], 
           [ 0,  0, mx, mn,  0,  0],
           [ 0,  0,  0,  0, mx, mn],
           marker=".", alpha=0.0)

ax = fig.add_subplot(1, 2, 2)
ax.grid(True)
ax.set_axisbelow(True)
ax.plot(UU[:, 0], UU[:, 1])
ax.scatter([0], [0], marker=".", color="k")
ax.set_title("Spinor Trajectory")
ax.set_xlabel("Scalar")
ax.set_ylabel("Bivector")
mx = max([np.max(UU[:, i]) for i in range(2)])
mn = min([np.min(UU[:, i]) for i in range(2)])
ax.scatter([mx, mn,  0,  0], 
           [ 0,  0, mx, mn],
           marker=".", alpha=0.0)


plt.show()