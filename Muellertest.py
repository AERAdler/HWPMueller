##Mueller calculator test

import numpy as np
import matplotlib.pyplot as plt

eps_1 = 0+0j
eps_2 = 0+0j

Celerity = 3e8 #Speed of light in medium

sapphire_transmission = np.loadtxt('dummy_sapphire.txt')

def transmission(nu):

    return [ np.interp(nu, sapphire_transmission[:,0], sapphire_transmission[:,1]), 
    np.interp(nu, sapphire_transmission[:,0], sapphire_transmission[:,2]) ]

def phi(nu, n_s, n_f, d):

    return 2.0*np.pi*d*(n_s-n_f)/Celerity *nu

def mueller_single_plate(nu, n_s, n_f, d):#DONT USE as model for eps!=0

    mueller_mat = np.zeros(4,nu.size)
    a, b = transmission(nu)
    phase = phi(nu, n_s,n_f,d)
    mueller_mat[0,:] = (a**2+b**2)/2.0 #00 and 11 elements
    mueller_mat[1,:] = (a**2-b**2)/2.0 #01 and 10 elements
    mueller_mat[2,:] = a*b*np.cos(phase) #22 and 33 elements
    mueller_mat[3,:] = a*b*np.sin(phase) #23 and minus 32 elements

    return mueller_mat