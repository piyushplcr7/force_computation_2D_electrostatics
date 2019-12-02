import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#circular arc
fig, ax = plt.subplots()
plt.plot([ np.cos(theta) for theta in np.linspace(0,1.5*np.pi,50)],[np.sin(theta) for theta in np.linspace(0,1.5*np.pi,50)],color='k')

#lines
plt.plot([ 0 for x in np.linspace(0,1,10)],[y for y in np.linspace(-1,0,10)],color='k')
plt.plot([ x for x in np.linspace(0,1,10)],[0 for y in np.linspace(-1,0,10)],color='k')

#Fourier potential
points = [ [r*np.cos(theta),r*np.sin(theta)] for r in np.linspace(0,1,50) for theta in np.linspace(0,1.5*np.pi,50)  ]
x = np.array([ r*np.cos(theta) for r in np.linspace(0,1,50) for theta in np.linspace(0,1.5*np.pi,50)  ])
y = np.array([ r*np.sin(theta) for r in np.linspace(0,1,50) for theta in np.linspace(0,1.5*np.pi,50)  ])
potential = np.array([np.power(r,2./3.)*np.sin(2./3.*phi) for r in np.linspace(0,1,50) for phi in np.linspace(0,1.5*np.pi,50)])

M=50
N=50
C=ax.contourf(np.reshape(x,(M,N)),np.reshape(y,(M,N)),np.reshape(potential,(M,N)))
fig.colorbar(C)
plt.axis('equal')
#ax.set_xticks([0])
#ax.set_yticks([0])
plt.grid()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(np.reshape(x,(M,N)),np.reshape(y,(M,N)),np.reshape(potential,(M,N)),cmap=cm.coolwarm)
ax.set_zlim(0,2)
fig.colorbar(surf, shrink=0.5, aspect=5)

# plotting the gradient
#circular arc
fig, ax = plt.subplots()
plt.plot([ np.cos(theta) for theta in np.linspace(0,1.5*np.pi,50)],[np.sin(theta) for theta in np.linspace(0,1.5*np.pi,50)],color='k')

#lines
plt.plot([ 0 for x in np.linspace(0,1,10)],[y for y in np.linspace(-1,0,10)],color='k')
plt.plot([ x for x in np.linspace(0,1,10)],[0 for y in np.linspace(-1,0,10)],color='k')

#Fourier potential
rmin = 0.05
points = [ [r*np.cos(theta),r*np.sin(theta)] for r in np.linspace(rmin,1,50) for theta in np.linspace(0,1.5*np.pi,50)  ]
x = np.array([ r*np.cos(theta) for r in np.linspace(rmin,1,50) for theta in np.linspace(0,1.5*np.pi,50)  ])
y = np.array([ r*np.sin(theta) for r in np.linspace(rmin,1,50) for theta in np.linspace(0,1.5*np.pi,50)  ])
potential = np.array([4./9.*np.power(r,-2./3.) for r in np.linspace(rmin,1,50) for phi in np.linspace(0,1.5*np.pi,50)])

M=50
N=50
C=ax.contourf(np.reshape(x,(M,N)),np.reshape(y,(M,N)),np.reshape(potential,(M,N)))
fig.colorbar(C)
plt.axis('equal')
#ax.set_xticks([0])
#ax.set_yticks([0])
plt.grid()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(np.reshape(x,(M,N)),np.reshape(y,(M,N)),np.reshape(potential,(M,N)),cmap=cm.coolwarm, edgecolor='none')

fig.colorbar(surf, shrink=0.5, aspect=5)


plt.show()
