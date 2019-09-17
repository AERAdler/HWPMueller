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

    return 2.0*np.pi*d*(n_s-n_f)*nu

