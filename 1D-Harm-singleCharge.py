__author__ = "Maurits Bos"
__credits__ = ["Chien-An Wang", "Menno Veldhorst", "Caroline Smulders"]
#%%
import  numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import kwant
from types import SimpleNamespace

#these files need to be in the functions folder
import functions.plots as plots
import functions.potentials as pt
import functions.make_system as dot
from functions.data import *

# %% Calculating the hopping E and lattice constant 
a   = 2.5   *1e-9  # m; latiche spacing
m_e = 9.10938356*1e-31 #kg
hbar= 1.0545718 *1e-34 
e   = 1.60218   *1e-19 #Joule
alpha_R = 0.001 #eV * nm

t     = (hbar**2)/(2*0.05*m_e*a**2) *(e**-1)
alpha = alpha_R / (a*1e9)
Ez = np.array([0e-6, 0e-6, 0e-6])

#%% ------- Defining a circular system in Kwant
r=40 

syst = dot.make_system_circular(r)
syst = syst.finalized()
#%% -------------------------- Run Simulation for Multiple Potentials -------------

V0 = 10*1e-6 
q = [0, 0.1, 0.07, 0.05, 0.04, 0.03]
d_sens = [80, 80, 40, 20, 10, 0]


ev1 ,evec1, p1 = [],[],[]
for i in range(len(d_sens)):
    #defines potential, Comment barrier if you dont want that
    pot =  1 + pt.harm_potential(40, r, V0=V0,L=2*r,W=2*r)              # Potential of dot
    pot = pot + pt.disk_potential(d_sens[i], q=q[i],L=2*r,W=2*r)        # " of sensor
    # pot = pot + pt.barrier_potential(0,V0=0.10, L=2*r, W=2*r, sigma=5)  # " of barrier

    Ez = np.array([0e-6, 0e-6, 0e-6])
    p=SimpleNamespace( t=t, alpha=alpha, Ez= Ez, pot=pot)

    ham = syst.hamiltonian_submatrix(params=dict(p=p), sparse=True)
    ev0, evec0 = sp.sparse.linalg.eigsh(ham.tocsc(), k=30, sigma=0, tol=1e-6) 
    idx = np.argsort(ev0)

    evec1.append(evec0[:,idx])
    ev1.append(ev0[idx])
    p1.append(p)

print('calculation succesful')

# %% ---------------------------- Plot Contour and Spectrum, All in One ------------------------------------

fig, ax = plt.subplots(2,len(p1),sharey='row',gridspec_kw={'height_ratios':[2,5]})
fig.set_figwidth(24)
fig.set_figheight(16)
levels = np.arange(0.997,1.015,0.002)
fig.suptitle("Potential change due to point-charge at varied distance", fontsize=30)

for i in range(len(p1)):
    ax[0, i].contourf(p1[i].pot.transpose(),levels, cmap='viridis')
    plots.spectrum_energy(p1[i], ev1[i], ax=ax[1,i])
    ax[1, i].axhline(y=1.01, c='grey', linestyle='--')
    ax[1, i].set_ylabel(None)
    ax[1, i].set_xlabel('x (nm)',fontsize=15)
    ax[1, i].set_ylim(0.997,1.012)

    s1='d_sens: ' + str(d_sens[i])
    s2='q: ' + str(q[i])
    ax[1, i].text(0.9,0.05,s1, horizontalalignment='right', verticalalignment='center', transform = ax[1,i].transAxes,fontsize=14)
    ax[1, i].text(0.9,0.075,s2, horizontalalignment='right', verticalalignment='center', transform = ax[1,i].transAxes,fontsize=14)

ax[1,0].set_ylabel('energy (eV)',fontsize=15)
plt.tight_layout()
filename = ''
# plt.savefig(filename)




#%% ------------------- Plot All Addition energies in one plot ------------------------

fig, ax = plt.subplots(1,1)
s='Addition energy of a dot perturbed by a charge at (0,40)'
fig.suptitle(s)
for pot_id in range(len(q)):
    label = 'q=' + str(q[pot_id])
    ax = plots.addition_energy(ev1[pot_id], label=label)
    s1='d_sens: ' + str(d_sens[pot_id])
    s2='q: ' + str(q[pot_id])
# # Uncomment to include experimental data
# ax.errorbar(np.arange(1, len(charging_E_P1) + 1), charging_E_P1, yerr=std_E_P1,fmt='r-',label='Data dot 1')
# ax.errorbar(np.arange(1, len(charging_E_P2) + 1), charging_E_P2, yerr=std_E_P2,fmt='b-',label='Data dot 2')
filename = '1D-Harm-1C-dsens=0-AdittionEnergy.pdf'

ax.legend()
plt.tight_layout()
# plt.savefig(filename)


#%% --------- Plot Addition Energy of Selected Results Together with Dot1 or Dot2 ------
fig, ax = plt.subplots(1,1)
s='Addition energy of a dot perturbed by a near charge'
fig.suptitle(s)
selection = [0,1]

for pot_id in selection:
    label = 'd=' + str(d_sens[pot_id]) +'; q=' + str(q[pot_id])
    ax = plots.addition_energy(ev1[pot_id],label=label)
    s1='d_sens: ' + str(d_sens[pot_id])
    s2='q: ' + str(q[pot_id])
plt.tight_layout()
# filename = '1D-Harm-1C-dsens'+str(d_sens[pot_id]) +'-q'+str(q[pot_id]) +'-addEnergy.pdf'
filename = '1D-Harm-1C-dsens_variated-AdittionEnergy-WithDots.pdf'
ax.errorbar(np.arange(1, len(charging_E_P1) + 1), charging_E_P1, yerr=std_E_P1,fmt='r-',label='Data dot 1')
ax.errorbar(np.arange(1, len(charging_E_P2) + 1), charging_E_P2, yerr=std_E_P2,fmt='b-',label='Data dot 2')
ax.legend()
# plt.savefig(filename)

# %% --------------------- PLot Wavefunctions -------------------------

for pot_id in range(1):
    fig = plt.figure()
    fig.set_figwidth(7)
    fig.set_figheight(8)
    for i in range(12):
        ax = fig.add_subplot(4, 3, 1 + i, xticks=[],yticks=[], xlabel="index: " + str(i))
        state_id = i
        density = np.abs(evec1[pot_id][0::2, state_id])**2  # spin up component
        kwant.plotter.map(syst, density, colorbar=True,ax=ax)
    fig.suptitle('Probability densities; d_sens = '+ str(d_sens[pot_id]) +'; q = ' + str(q[pot_id]))
    plt.tight_layout()
    simulation_code = '1D-Harm-1C'
    filename = simulation_code + 'dsens='+str(d_sens[pot_id]) +'-q='+str(q[pot_id]) +'-PD.pdf'
    # plt.savefig(filename)     #uncomment to save


