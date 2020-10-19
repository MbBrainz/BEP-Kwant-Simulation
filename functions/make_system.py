__author__ = "Maurits Bos"
__credits__ = ["Chien-An Wang", "Menno Veldhorst", "Caroline Smulders"]

import kwant 
from types import SimpleNamespace
import numpy as np

pauli = SimpleNamespace(s0=np.array([[1., 0.], [0., 1.]]),
                        sx=np.array([[0., 1.], [1., 0.]]),
                        sy=np.array([[0., -1j], [1j, 0.]]),
                        sz=np.array([[1., 0.], [0., -1.]]))


# ----------- Prerequesites of make system --------


def onsite(site, p): 
    n = site.tag
    x = int(n[0])
    y = int(n[1])
    return (4 * p.t +  p.pot[x,y] ) * pauli.s0 + p.Ez[0] * pauli.sx + p.Ez[1] * pauli.sy + p.Ez[2] * pauli.sz      

def hopx(site1, site2, p):
    return -p.t * pauli.s0 + .5j * p.alpha * pauli.sy

def hopy(site1, site2, p):
    return -p.t * pauli.s0 - .5j * p.alpha * pauli.sx



# ----------- Make systems ----------------
def make_system_circular(r, a=1, norbs=2):
    
    lat= kwant.lattice.square(a,norbs=norbs)

    def circle(pos): #treats individual sites
        (x0, y0) = (r, r)
        (x, y) = pos
        rsq = (x-x0) ** 2 + (y-y0) ** 2
        return rsq < r ** 2

    syst = kwant.Builder()
    syst[lat.shape(circle, (r, r))] = onsite

    # Add hoppings to system
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopx 
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopy
    return syst

def make_system_square(L, W, a=1, norbs=2):

    lat= kwant.lattice.square(a,norbs=norbs)

    syst = kwant.Builder()
    syst[(lat(x, y) for x in range(L) for y in range(W))] = onsite

    # Add hoppings to system
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopx 
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopy
    return syst

