##Mueller calculator test

import numpy as np
import matplotlib.pyplot as plt

Celerity = 3e8 #Speed of light in medium


sapphire_transmission = np.loadtxt('dummy_sapphire.txt')

def transmission(nu):

    return [ np.interp(nu, sapphire_transmission[:,0], sapphire_transmission[:,1]), 
    np.interp(nu, sapphire_transmission[:,0], sapphire_transmission[:,2]) ]

def phi(nu, n_s, n_f, d):

    return 2.0*np.pi*d*(n_s-n_f)/Celerity *nu

def mueller_single_plate(nu, n_s, n_f, d):#DONT USE as model for eps!=0

    mueller_mat = np.zeros((4,nu.size))
    a, b = transmission(nu)
    phase = phi(nu, n_s,n_f,d)
    mueller_mat[0,:] = (a**2+b**2)/2.0 #00 and 11 elements
    mueller_mat[1,:] = (a**2-b**2)/2.0 #01 and 10 elements
    mueller_mat[2,:] = a*b*np.cos(phase) #22 and 33 elements
    mueller_mat[3,:] = a*b*np.sin(phase) #23 and minus 32 elements

    return mueller_mat

#Input parameters
eps_1 = 0+0j
eps_2 = 0+0j

n_s = 3.36 #From 1006.3874
n_f =3.04 #From 1006.3874
d = 3.05e-3

nu = np.arange(0, 3e11, 1e9)

mueller_response = mueller_single_plate(nu, n_s, n_f, d)

fig, ax = plt.subplots(2, 2, sharey='col', sharex='row')
ax[0,0].plot(nu/1e9, mueller_response[0,:], 'r--')
ax[0,0].set(xlabel="Frequency [GHz]", title='T')

ax[0,1].plot(nu/1e9, mueller_response[1,:], 'r--')
ax[0,1].set(xlabel="Frequency [GHz]", title=r'$\rho$')

ax[1,0].plot(nu/1e9, mueller_response[2,:], 'r--')
ax[1,0].set(xlabel="Frequency [GHz]", title='c')

ax[1,1].plot(nu/1e9, mueller_response[3,:], 'r--')
ax[1,1].set(xlabel="Frequency [GHz]", title='s')


plt.subplots_adjust(left=0.1, bottom=0.08, right=0.9, top=0.92, wspace=0.2, hspace=0.35)
plt.show()