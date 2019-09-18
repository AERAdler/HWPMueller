##Mueller calculator test, based on 1006.3359

import numpy as np
import matplotlib.pyplot as plt

Celerity = 3e8 #Speed of light in medium


sapphire_properties = np.loadtxt('dummy_sapphire.txt', dtype=complex)

def properties(nu):

    return [ np.interp(nu, np.real(sapphire_properties[:,0]), np.real(sapphire_properties[:,1])), 
    np.interp(nu, np.real(sapphire_properties[:,0]), np.real(sapphire_properties[:,2])), 
    np.interp(nu, np.real(sapphire_properties[:,0]), sapphire_properties[:,3]),
    np.interp(nu, np.real(sapphire_properties[:,0]), sapphire_properties[:,4]),
     ]

def phi(nu, n_s, n_f, d):

    return 2.0*np.pi*d*(n_s-n_f)/Celerity *nu

def mueller_single_plate(nu, n_s, n_f, d):#DONT USE as model for eps!=0

    a, b, eps_1, eps_2 = properties(nu)
    phase = phi(nu, n_s,n_f,d)

    mueller_mat = np.zeros((4,nu.size))
    mueller_mat[0,:] = 0.5*(a**2+b**2) #00 and 11 elements
    mueller_mat[1,:] = 0.5*(a**2-b**2)#01 and 10 elements
    mueller_mat[2,:] = a*b*np.cos(phase) #22 and 33 elements
    mueller_mat[3,:] = a*b*np.sin(phase) #23 and minus 32 elements

    return mueller_mat

def mueller_multiple_layer(nu, n_s, n_f, d):# Equation 9

    a, b, eps_1, eps_2 = properties(nu)
    phase = phi(nu, n_s,n_f,d)


    mueller_mat = np.array([
                    [
                    0.5*(a**2 + b**2 + np.absolute(eps_1)**2 + np.absolute(eps_2)**2), 
                    0.5*(a**2 - b**2 - np.absolute(eps_1)**2 + np.absolute(eps_2)**2),  
                    +a*np.real(eps_1) + b*(np.real(eps_2)*np.cos(phase) + np.imag(eps_2)*np.sin(phase)),
                    -a*np.imag(eps_1) - b*(np.real(eps_2)*np.sin(phase) - np.imag(eps_2)*np.cos(phase))
                    ], 
                    [
                    0.5*(a**2 - b**2 + np.absolute(eps_1)**2 - np.absolute(eps_2)**2),
                    0.5*(a**2 + b**2 - np.absolute(eps_1)**2 - np.absolute(eps_2)**2),
                    +a*np.real(eps_1) - b*(np.real(eps_2)*np.cos(phase) + np.imag(eps_2)*np.sin(phase)),
                    -a*np.real(eps_1) + b*(np.real(eps_2)*np.sin(phase) - np.imag(eps_2)*np.cos(phase))
                    ], 
                    [
                    a*np.real(eps_2) + b*(np.real(eps_1)*np.cos(phase) + np.imag(eps_1)*np.sin(phase)),
                    a*np.real(eps_2) - b*(np.real(eps_1)*np.cos(phase) + np.imag(eps_1)*np.sin(phase)),
                    np.real(eps_1)*np.real(eps_2) + np.imag(eps_1)*np.imag(eps_2) + a*b*np.cos(phase),
                    np.real(eps_1)*np.imag(eps_2) - np.imag(eps_1)*np.real(eps_2) - a*b*np.sin(phase) 
                    ], 
                    [
                    a*np.imag(eps_2) + b*(np.real(eps_1)*np.sin(phase) - np.imag(eps_1)*np.cos(phase)),
                    a*np.imag(eps_2) - b*(np.real(eps_1)*np.sin(phase) - np.imag(eps_1)*np.cos(phase)),
                    +np.real(eps_1)*np.imag(eps_2) - np.imag(eps_1)*np.real(eps_2) + a*b*np.sin(phase),
                    -np.real(eps_1)*np.real(eps_2) - np.imag(eps_1)*np.imag(eps_2) + a*b*np.cos(phase)
                    ]
                ])

    return mueller_mat

def jones_to_mueller(J):# Equation 3
    pauli =  (np.array([[1,0],[0,1]]), np.array([[1,0],[0,-1]]), np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]))
    mueller = np.zeros((4,4), dtype=complex)
    for i in range(4):
        for j in range(4):
            mueller[i,j] = 0.5*np.trace(np.matmul(np.matmul(pauli[i], J),np.matmul(pauli[j], np.conjugate(np.transpose(J)))) )
    return mueller

def inst_rot_matrix(psi):#Tinbergen, Astronoimical Polarimetry
    a = np.ones(psi.shape)
    b = np.zeros(psi.shape)
    c = np.cos(2*psi)
    d = np.sin(2*psi)
    return np.array([[a,b,b,b],[b,c,d,b],[b,-d,c,b],[b,b,b,a]])

def instrument_total_matrix(nu, n_s, n_f, d, eps_1, eps_2, eta, delta, chi, theta, psi):
    #nu is the frequency of the light
    #n_s, n_f, d, eps_1 and eps_2 are the properties of the hwp
    #eta and delta are the two Jones parameters for the detector
    #chi, theta, and psi are rotation of detector, hwp and boresight

    mesh_chi, mesh_theta, mesh_psi, mesh_nu = np.meshgrid(chi, theta, psi, nu)#4D arrays of all possible positions and frequencies
    #This might cause memory size issues ? 
    return (inst_rot_matrix(mesh_chi)*inst_rot_matrix(-mesh_theta)
    *mueller_multiple_layer(mesh_nu, n_s, n_f, d, eps_1, eps_2)
    *inst_rot_matrix(-mesh_theta)*inst_rot_matrix(mesh_psi))

#Input parameters

n_s = 3.36 #From 1006.3874
n_f =3.04 #From 1006.3874
d = 3.05e-3

nu = np.arange(0, 3e11, 5e9)
psi = np.arange(0, 2*np.pi, np.pi/10)
chi = np.pi/6
theta = np.arange(0, np.pi, np.pi/10)
#total_intru = instrument_total_matrix(nu, n_s, n_f, d, eps_1, eps_2, 0.7, 0.3, chi, psi, psi)
#print total_intru.shape
mueller_response = mueller_single_plate(nu, n_s, n_f, d)
mueller_multiple = mueller_multiple_layer(nu, n_s, n_f, d)
#print mueller_multiple.shape
fig, ax = plt.subplots(2, 2, sharey='col', sharex='row')

ax[0,0].plot(nu/1e9, mueller_response[0,:], 'r--')
ax[0,0].plot(nu/1e9, mueller_multiple[0,0,:], 'b')
ax[0,0].set(xlabel="Frequency [GHz]", title='T')

ax[0,1].plot(nu/1e9, mueller_response[1,:], 'r--')
ax[0,1].plot(nu/1e9, mueller_multiple[1,0,:], 'b')
ax[0,1].set(xlabel="Frequency [GHz]", title=r'$\rho$')

ax[1,0].plot(nu/1e9, mueller_response[2,:], 'r--')
ax[1,0].plot(nu/1e9, mueller_multiple[2,2,:], 'b')
ax[1,0].set(xlabel="Frequency [GHz]", title='c')

ax[1,1].plot(nu/1e9, mueller_response[3,:], 'r--')
ax[1,1].plot(nu/1e9, mueller_multiple[3,2,:], 'b')
ax[1,1].set(xlabel="Frequency [GHz]", title='s')


plt.subplots_adjust(left=0.1, bottom=0.08, right=0.9, top=0.92, wspace=0.2, hspace=0.35)
plt.show()