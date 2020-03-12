import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

qa = np.loadtxt('quad_approx_circle.txt')
lra = np.loadtxt('lowrank_approx_circle.txt')
vel = np.loadtxt('eigvecs.txt')
vfa = np.loadtxt('velapprox.txt')

start = 4
maxpanels = 50

numpanels = np.empty((0,0),int)
K = np.empty((0,0),int)
error = np.empty((0,0),int)
verror = np.empty((0,0),int)

for i in range(start, maxpanels):
    #test = np.full(i-1,i)
    test = np.full(i,i)
    #print test.shape
    numpanels = np.append(numpanels,test)
    #K = np.append(K,np.arange(1,i))
    K = np.append(K,np.arange(1,i+1))

for panel in range(start,maxpanels):
    for k in range(1,panel+1):
        #error = np.append(error,[ abs(lra[panel,k] - qa[panel]) ])
        error = np.append(error,[ abs(lra[panel,k] - qa[panel])/abs(qa[panel]) ])
        verror = np.append(verror,[vfa[panel,k] ])

print numpanels
print K
print error

fig = plt.figure()
#ax = plt.axes(projection='3d')

#ax.plot_surface(numpanels,K,error,cmap='viridis', edgecolor='none')
#ax.set_title('Surface plot')

ax = fig.gca(projection='3d')
ax.plot_trisurf(numpanels,K,error, linewidth=0.2, antialiased=True)
ax.set_xlabel('numpanels')
ax.set_ylabel('K')
ax.set_zlabel('error')
ax.set_title('Low rank approximation error')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(numpanels,K,verror, linewidth=0.2, antialiased=True)
ax.set_xlabel('numpanels')
ax.set_ylabel('K')
ax.set_zlabel('verror')
ax.set_title('velocity field approximation error')

#plt.figure()
#plt.plot(range(start,maxpanels),qa[start:])

fig, axs = plt.subplots(2,5, figsize=(15, 15), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.1)

axs = axs.ravel()
numpanels = vel.shape[0]
theta = np.linspace(0,2*np.pi,numpanels+1)[0:numpanels]
x = np.cos(theta)
y = np.sin(theta)

for i in range(10):
    field = vel[:,i]
    u = np.multiply(x,field)
    v = np.multiply(y,field)
    #axs[i].contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
    #axs[i].set_title(str(250+i))
    #plt.quiver(x,y,u,v)
    axs[i].quiver(x,y,u,v)
    #plt.plot([np.cos(theta) for theta in np.linspace(0,2*np.pi,100)],[np.sin(theta) for theta in np.linspace(0,2*np.pi,100)])
    axs[i].plot([np.cos(theta) for theta in np.linspace(0,2*np.pi,100)],[np.sin(theta) for theta in np.linspace(0,2*np.pi,100)])
    axs[i].axis('equal')
    axs[i].set_title(str(i))
    axs[i].axis('off')
    #plt.axis('equal')

plt.show()
