# -*- coding: utf-8 -*-

# ---------------------- Import packages -----------------------

import numpy as np
import math
from numba import jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# constants - used to calculate endresults in SI. 
# Computation is in natural units, i.e. m=1 eps=1 sigma=1.
kb = 1.38 * 10**(-23) #J/K
eps_per_kb = 120 #K
sigma = 0.34*10**(-9) #m
m = 6.633853*10**(-26) #kg - mass of argon atom
tau = math.sqrt(m*sigma*sigma/eps_per_kb/kb) #s - typical unit of time.

#### ----------------------------------- Parameters ------------
# Initialize parameters
def init_params(rhopar,Tpar,npar,tfinpar=2000,dtpar=0.004):
    Npar = 4*npar*npar*npar # Total number of particles
    Lpar = (Npar/rhopar)**(1/3) # Length of the computational domain
    return init_pos(npar,Lpar),init_vel(Npar,Tpar),rhopar,Tpar,npar,tfinpar,dtpar,Npar,Lpar


# ---------------------------------- Function definitions ---------------
# Compute initial position, a FFC structure
def init_pos(n,L):
    v = np.linspace(0,1,n,False)
    w = np.linspace(1/(2*n),(2*n-1)/(2*n),n,True)
    
    X1,Y1,Z1 = np.meshgrid(v,v,v) #corners
    X2,Y2,Z2 = np.meshgrid(w,w,v) #face centers per axis
    X3,Y3,Z3 = np.meshgrid(v,w,w)
    X4,Y4,Z4 = np.meshgrid(w,v,w)
    x = np.hstack((X1.reshape(n**3),X2.reshape(n**3),X3.reshape(n**3),X4.reshape(n**3)))
    y = np.hstack((Y1.reshape(n**3),Y2.reshape(n**3),Y3.reshape(n**3),Y4.reshape(n**3)))
    z = np.hstack((Z1.reshape(n**3),Z2.reshape(n**3),Z3.reshape(n**3),Z4.reshape(n**3)))
    
    pos = np.array([L*x,L*y,L*z])    
    
    return pos
    
# Compute initial velocity based on a temperature. Note that since we take the
# velocity from a random distribution, the actual temperature may differ from
# the desired initial temperature.
def init_vel(N,T):
    v = np.random.normal(0,math.sqrt(T),(3,N))
    v[0,:] -= v[0,:].mean()
    v[1,:] -= v[1,:].mean()
    v[2,:] -= v[2,:].mean()
    return v


# Force between particle i and j
# also returns the distance between particles and potential energy
@jit
def forceLJ(xi,yi,zi,xj,yj,zj, L):
    dx = xi-(xj + np.around((xi-xj)/L)*L)
    dy = yi-(yj + np.around((yi-yj)/L)*L)
    dz = zi-(zj + np.around((zi-zj)/L)*L)
    r2 = dx*dx + dy*dy + dz*dz
    pref = 48/(r2**7) - 24/(r2**4)
    Fx = pref*dx
    Fy = pref*dy
    Fz = pref*dz
    V = 4/(r2**6) - 4/(r2**3)
    rV = -48/(r2**6) + 24/(r2**3) #r_ij * dV/dr (r_ij)
    return Fx, Fy, Fz, V, rV, r2
    
# Total force on each particle in a vector, and total potential energy of all particles.
@jit
def forceTotal(pos,N,L):
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    Fz = np.zeros(N)
    V = 0.0
    rV = 0.0
    for i in range(N):
        for j in range(i):
            dFx,dFy,dFz,dV,drV, r2 = forceLJ(pos[0,i],pos[1,i],pos[2,i],pos[0,j],pos[1,j],pos[2,j],L)
            Fx[i] += dFx
            Fy[i] += dFy
            Fz[i] += dFz
            # action = - reaction
            Fx[j] -= dFx
            Fy[j] -= dFy
            Fz[j] -= dFz
            
            V += dV
            rV += drV
    F = np.array([Fx,Fy,Fz])
    return F,V,rV

@jit
def forceTotalAndN(pos, N, L, nBins):
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    Fz = np.zeros(N)
    nr = np.zeros(nBins)
    binSize  = np.sqrt(3) * L / nBins

    V = 0.0
    rV = 0.0
    for i in range(N):
        for j in range(i):
            dFx,dFy,dFz,dV,drV, r2 = forceLJDis(pos[0,i],pos[1,i],pos[2,i],pos[0,j],pos[1,j],pos[2,j],L)
            Fx[i] += dFx
            Fy[i] += dFy
            Fz[i] += dFz

            # action = - reaction
            Fx[j] -= dFx
            Fy[j] -= dFy
            Fz[j] -= dFz
            binIndex = round(math.sqrt(r2) / binSize)
            nr[binIndex] += 1 #add count of distance
            V += dV
            rV += drV

    F = np.array([Fx,Fy,Fz])
    return F ,V , rV , nr
    
def evolveTimeConserveE(pos,vel,numtime,N,L,dt):
    #Initialize kinetic and potential energy vectors
    K = np.zeros(numtime)
    V = np.zeros(numtime)
    rV = np.zeros(numtime)
    F = forceTotal(pos,N,L)[0]
    for i in range(numtime):
        vel += 0.5*dt*F
        pos += dt*vel
        pos %= L

        F,V[i],rV[i] = forceTotal(pos,N,L)
        vel += 0.5*dt*F   
        
        K[i] = 0.5*np.sum(vel[0,:]**2+vel[1,:]**2+vel[2,:]**2)
        
        
    return pos,vel,K,V,rV
    
def evolveTimeFixedT(pos,vel,numtime,N,L,dt,Tfixed):
    #Initialize potential energy vectors
    K = np.zeros(numtime)
    V = np.zeros(numtime)
    rV = np.zeros(numtime)
    F = forceTotal(pos,N,L)[0]
    for i in range(numtime):
        vel += 0.5*dt*F
        pos += dt*vel
        pos %= L

        F,V[i],rV[i] = forceTotal(pos,N,L)
        vel += 0.5*dt*F

        scale = np.sqrt(3*Tfixed/np.mean(vel[0,:]**2+vel[1,:]**2+vel[2,:]**2))
        vel *= scale
        
        K[i] = 0.5*np.sum(vel[0,:]**2+vel[1,:]**2+vel[2,:]**2)
        
    return pos,vel,K,V,rV

#evolve time and calculate the correlation function
def evolveTimeFixedTAndCalcG(pos, vel, numtime, N, L, dt, Tfixed, nBins):
    #Initialize potential energy vectors
    K = np.zeros(numtime)
    V = np.zeros(numtime)
    rV = np.zeros(numtime)

    nr = np.zeros((numtime, nBins))

    F = forceTotal(pos,N,L)[0]

    for i in range(numtime):
        vel += 0.5*dt*F
        pos += dt*vel
        pos %= L

        F, V[i], rV[i], nri = forceTotalAndN(pos,N,L, nBins)
        nr[i, :] = nri
        vel += 0.5*dt*F

        scale = np.sqrt(3*Tfixed/np.mean(vel[0,:]**2+vel[1,:]**2+vel[2,:]**2))
        vel *= scale

        K[i] = 0.5*np.sum(vel[0,:]**2+vel[1,:]**2+vel[2,:]**2)

    return pos,vel,K,V,rV, nr

def correlation(N, L, nBins, nr, offset = 2000): #offset is the waiting time until equilibrium is reached
     #calculate g
    rVec = np.linspace(0, np.sqrt(3) * L, nBins)
    dr = rVec[1] - rVec[0]
    rVec += 0.5 * dr
    #skip first 2000 steps.
    g = 2 * (L * L * L) / (N * (N-1)) * np.mean(nr[offset:len(nr),:], axis = 0) / (4 * 3.14 * rVec ** 2 * dr)
    return rVec, g



    
def plotEnergy(timesteps,K,V):
    plt.figure(1)
    plt.plot(range(timesteps),K,'b',range(timesteps),V,'g',range(timesteps),K+V,'r')
    plt.legend(('Kinetic', 'Potential', 'Total'))
    plt.draw()
    
def plotParticles(pos,L):
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[0,:],pos[1,:],pos[2,:])
    plt.xlim(0,L)
    plt.ylim(0,L)
    ax.set_zlim(0,L)
    plt.draw()

def plotPressure(timesteps,P):
    plt.figure(3)
    plt.plot(range(timesteps),P)
    plt.legend(('Pressure'))
    plt.draw()
    
def plotPversusRho(rhoij,Tij,Pij):
    plt.figure(4)
    plt.plot(rhoij[:,0],Pij[:,0]/Tij[:,0],rhoij[:,1],Pij[:,1]/Tij[:,1],rhoij[:,2],Pij[:,2]/Tij[:,2])
    plt.legend(('T = ' + str(Tij[0][0]),'T = ' + str(Tij[0][1]),'T = ' + str(Tij[0][2])),loc=2) 
    plt.draw()
 
    


