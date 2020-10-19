__author__ = "Maurits Bos"
__credits__ = ["Chien-An Wang", "Menno Veldhorst", "Caroline Smulders"]


import matplotlib.pyplot as plt
import kwant
import numpy as np


def spectrum_energy(p, ev, ax=None):
    """Plots the eigenenergies in a plot together with a intersection of the given potential

    Args:
        p (SimpleNamespace): simple namespace with at least a numpy array with p.pot[x,y] = V(x,y)
        ev (array): Array with eigenvalues. this array is outputted by Kwant when calculating for a certain system
        ax (axes, optional): axes on which the plot has to be drawn. Defaults to None.

    Returns:
        [axes]: [description]
    """
    if (ax == None):
        ax = plt.gca()
    y_mid = round(len(p.pot[0,:])/2)
    ax.plot(p.pot[:,y_mid])
    xarr = np.arange(len(p.pot[:,y_mid]))
    for i in range(len(ev)):
        ax.plot((0,len(p.pot[:,y_mid])),(ev[i],ev[i]),'r',linewidth=1)

    # ax.set_ylim(2,10)
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('Energy (meV)')
    return ax

def addition_energy(ev, ax=None, label=None, marker='.',linestyle='-', return_add_energy=False):

    coul = np.zeros(9)
    totale = np.ones(10)
    for n in range(0 ,9):
        for n in range(2):
            coul[n]=6.17*(10**-3)
        for n in range(2 ,7):
            coul[n]=3.5*(10** -3)
        for n in range(7 ,9):
            coul[n] = 2.7*(10** -3)

    coul[0] = coul[0]
    for i in range(8):
        coul[i+1] = ev[i+2] - ev[i+1] + coul[i+1]

    if (return_add_energy):
        return coul

    if (ax == None):
        ax = plt.gca()

    ax.scatter ( range(1 ,10), coul*10**3, marker=marker)
    ax.plot ( range(1 ,10), coul*10**3, label=label, linestyle=linestyle)
    ax.set_xticks([2,4,6,8])
    ax.set_xticks([1,3,5,7,9], minor=True)
    ax.set_ylabel ('Addition energy ( meV )')
    ax.set_xlabel ('Number of holes on the dot ')

    return ax


def plot_all(syst, p,  ev, title=None, y=None):
    
    fig, ax = plt.subplots(2,2)
    
    fig.suptitle(title, fontsize=16)

    fig.set_figheight(8)
    fig.set_figwidth(9)

    kwant.plot(syst, ax=ax[0, 0], fig_size=(5,5))
    ax[0, 0].set_title("(a) kwant lattice points")
    cf = ax[0, 1].contourf(p.pot.transpose())
    ax[0, 1].set_title("(b) potential landscape")
    # ax[0,1].set_ticks(None)
    fig.colorbar(cf, ax=[ax[1,1], ax[0,1]])
    addition_energy(ev, ax=ax[1, 0])
    ax[1, 0].set_title("(c) Addition Energy")

    spectrum_energy(p, ev=ev, ax=ax[1, 1])
    if(y != None):
        ax[1, 1].axhline(y=y, c='grey', linestyle='--')

    ax[1, 1].set_title("(d) Energy spectrum")
    ax[1,1].set_ylabel(None)
    plt.tight_layout()
    fig.show()