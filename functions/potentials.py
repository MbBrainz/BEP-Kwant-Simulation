__author__ = "Maurits Bos"

import numpy as np
from numpy.core.fromnumeric import shape

# Defining the potential: 2 HARMONIC POTENTIALs AREN'T ADDABLE BECAUSE IT GIVES A HOLE IN THE MIDDLE
def harm_potential(x0, y0, V0 = 1e-6, L=80, W = 40, tune2=1, tune4=0): 
    # FOR VALUES tune2=0.05, tune4=0.001 YOU GET A SAME SCALE BUT STEEPER FUNCTION!
    pot=np.zeros((L,W))
    for x in range(L):
        for y in range(W):
            
            pot[x, y] =tune2*V0*((x-x0)**2 + (y-y0)**2) + tune4*V0*((x-x0)**2 + (y-y0)**2)**2 
    return pot

    

def gauss_potential(x0, y0, V0 = 1e-6, L=80, W = 40, sigma=20):
    """Create a Gaussian potential
    Args:
        x0 (float): x-coordinate of center of the Gaus
        y0 (float): y-coordinate of the center of the Gauss
        V0 (float, optional): potential at the top of the Gaussian. Defaults to 1e-6.
        L (int, optional): length of the potential landscape you want to plot. Defaults to 80.
        W (int, optional): width of the potential landscape you want to plot. Defaults to 40.
        sigma (float, optional): Full width at Half of the maximum value. Defaults to 20.

    Returns:
        [2D-ndarray]: [potential lanscape with a gaussian profile at the given coordinates]
    """
    pot=np.zeros((L,W))
    for x in range(L):
        for y in range(W):
            pot[x, y] = -V0*np.exp(-(((x-x0)**2)+((y-y0)**2))/(2*(sigma**2)))
    return pot 


def double_potential(V0 = 1e-6, L=80, W = 40): 
    pot=np.zeros((L,W))
    #variables for potential in x direction
    a = 5e-5; b = 1e-2; c = 0.5
    x_hole = np.sqrt(2*b/a) + 40
    print("location of right x-hole: ")
    print(x_hole)
    xtune = 0.5
    ytune = 0.05
    for x in range(L):
        for y in range(W):
            pot[x, y] = V0 * ((a*(xtune*(x-L/2))**4 - b*(xtune*(x-L/2)) **2) - c + (ytune*(y-W/2))**2) 
            if (pot[x, y] > 0):
                pot[x, y] = 0
    return pot + V0

def disk_potential(D, L=40, W=40, q=10, Y=0, full=False):
    # Asumed as point charge
    e0 = 1e-9/36/np.pi
    pot = np.zeros((L,W))
    e = 1.60218e-19
    Q = q * e#eV
    pot_full = np.zeros((D+2*L,W))
    for i in range(D+2*L):
        for j in range(W):
            r = np.sqrt((i)**2 + (j-0.5*W + Y)**2)
            pot_full[i,j] = -Q/(2*np.pi*e0*r)*1e9 #nm 

    pot = pot_full[D:D+L,:]

    if (full==True):
        return pot, pot_full
    else: return pot

def barrier_potential(x0 , V0, vertical=False, L=40, W=40, sigma= 20):
    pot = np.zeros((L,W))

    for x in range(L):
        for y in range(W):
            pot[x,y] = V0*np.exp(-(((x-x0)**2))/(2*(sigma**2)))
    if(vertical):
        pot = pot.transpose()

    return pot

def correct_potential(pot,r0,V0,x0,y0):
    new_pot = np.zeros(shape(pot))
    for i in range(len(pot[:,0])):
        for j in range(len(pot[0,:])):
            r = np.sqrt(((i-x0)**2 + (j-y0)**2))
            if (r >= r0):
                new_pot[i,j] = V0
            else:
                new_pot[i,j] = pot[i,j]
    return new_pot
