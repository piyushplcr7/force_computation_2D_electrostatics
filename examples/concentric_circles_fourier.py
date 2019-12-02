import numpy as np
import matplotlib.pyplot as plt
import math


#inner circle
fig, ax = plt.subplots()
plt.plot([ 1.5*np.cos(theta) for theta in np.linspace(0,2*np.pi,50)],[1.5*np.sin(theta) for theta in np.linspace(0,2*np.pi,50)],color='k')

#outer circle
plt.plot([ 3*np.cos(theta) for theta in np.linspace(0,2*np.pi,50)],[3*np.sin(theta) for theta in np.linspace(0,2*np.pi,50)],color='k')

#Fourier potential
points = [ [r*np.cos(theta),r*np.sin(theta)] for r in np.linspace(1.5,3,50) for theta in np.linspace(0,2*np.pi,50)  ]
x = np.array([ r*np.cos(theta) for r in np.linspace(1.5,3,50) for theta in np.linspace(0,2*np.pi,50)  ])
y = np.array([ r*np.sin(theta) for r in np.linspace(1.5,3,50) for theta in np.linspace(0,2*np.pi,50)  ])
potential = np.array([2.3774437510817346 + (1.2/r**2 - 0.014814814814814815*r**2)*np.cos(2*phi) - 2.164042561333445*np.log(r) + (2./r - 0.2222222222222222*r)*np.sin(phi) for r in np.linspace(1.5,3,50) for phi in np.linspace(0,2*np.pi,50)])

M=50
N=50
C=ax.contourf(np.reshape(x,(M,N)),np.reshape(y,(M,N)),np.reshape(potential,(M,N)))
fig.colorbar(C)
plt.axis('equal')
#ax.set_xticks([0])
#ax.set_yticks([0])
plt.grid()

#-- Generate Data -----------------------------------------
# Using linspace so that the endpoint of 360 is included...
#azimuths = np.radians(np.linspace(0, 360, 50))
#zeniths = np.arange(1.5, 3, 50)

azimuths = np.linspace(0, 2*np.pi, 50)
zeniths = np.linspace(1.5, 3, 50)

r, theta = np.meshgrid(zeniths, azimuths)
print("theta shape: ",theta.shape)
print("r shape: ",r.shape)
values = 2.3774437510817346 + (1.2/r**2 - 0.014814814814814815*r**2)*np.cos(2*theta) - 2.164042561333445*np.log(r) + (2./r - 0.2222222222222222*r)*np.sin(theta)

#-- Plot... ------------------------------------------------
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
ax.contourf(theta, r, values)

plt.axis('equal')
plt.show()
