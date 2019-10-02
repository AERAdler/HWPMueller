import numpy as np

def hwp_mueller_matrix(use_preset_thickness_and_index='150GHz', cooled=True, n_s=False,n_f=False, n_ar=False,n_gap=False, d_hwp=False,d_ar_near=False,d_ar_far=False, d_gap_near=False, d_gap_far=False, freq=np.linspace(135,165,31)):
    # SAB
    #
    # I've been reading "Microwave Engineering" by Pozar to sort of get my bearings for KIDs stuff
    # Both because Spider's python code could use it, and also as a good exercise,
    # this code calculates the HWP mueller matrix vs frequency
    # but using equations just from that textbook and my thesis.

    # first determine if we are using presets for the indices of refraction and the thicknesses
    # input "False" here, and none of this preset stuff will trigger, and the values entered
    # as function arguments will be used instead

    #AEA
    #
    #There are six slots in the n, d, and Z arrays, for, in order:
    #The AR 
    #The gap between AR and Sapphire
    #Sapphire's two indices, slow then fast
    #The back gap
    #the back AR
    #Usually a bunch of them are similar, but it's easier arithmeticaly for me that way


    n = np.ones(6)
    d = np.zeros(6)
    Z = np.zeros(6)
    if use_preset_thickness_and_index:
        # then see if we are cold or not
        if cooled:

            # set the preset thicknesses and materials for each band
            if use_preset_thickness_and_index=='150GHz':
                n=np.array([1.935, 1.51, 3.336, 3.019, 1.51, 1.935])
                d=np.array([0.254, 0.006, 3.160, 3.160, 0.006, 0.254]) # mm thickness

            if use_preset_thickness_and_index=='95GHz':
                n=np.array([1.951, 1.000, 3.336, 3.019, 1.000, 1.951])
                d=np.array([0.427, 0.000, 4.930, 4.930, 0.000, 0.427]) # mm thickness

        if not cooled:
 
            # set the preset thicknesses and materials for each band
            if use_preset_thickness_and_index=='150GHz':
                n=np.array([1.935, 1.51, 3.3736, 3.0385, 1.51, 1.935])
                d=np.array([0.254, 0.006, 3.160, 3.160, 0.006, 0.254]) # mm thickness

            if use_preset_thickness_and_index=='95GHz':
                n=np.array([1.951, 1.000, 3.3736, 3.0385, 1.000, 1.951])
                d=np.array([0.427, 0.000, 4.930, 4.930, 0.000, 0.427]) # mm thickness

    else:
        n = np.array([n_ar,n_gap,n_s, n_f, n_gap, n_ar])
        d = np.array([d_ar_near, d_gap_near, d_hwp, d_hwp, d_gap_far, d_ar_far])

    # convert frequency from GHz to Hz
    f_Hz = freq*1.0e9
    # convert the thicknesses to meters
    d *= 1e-3
    # convert the indices to impedances
    c_light = 299792458.0 #m/s, exact speed of light
    Z0 = 119.9169832*np.pi #ohms, exact impedance of free space, from wikipedia
    Z = Z0/n

    T = np.zeros(len(freq))
    rho=np.zeros(len(freq))
    c = np.zeros(len(freq))
    s = np.zeros(len(freq))
    # and also the reflection mueller matrix elements too
    # equation 9.6 in my thesis
    R = np.zeros(len(freq))
    tau=np.zeros(len(freq))
    h = np.zeros(len(freq))
    q = np.zeros(len(freq))

    f_mesh, Z_mesh = np.meshgrid(f_Hz, Z)

    bl = np.zeros((6,)+freq.shape) #in Pozar bl = omega/v_p

    for i in range(6):#not the most elegant
        bl[i,:] = 2*np.pi*f_Hz/c_light*(n[i]*d[i])

    ABCD = np.array([[np.cos(bl), 1j*Z_mesh*np.sin(bl)], [(1j/Z_mesh)*np.sin(bl), np.cos(bl)]])
    #Go from matrix of frequency,material array to frequency,material array of matrices
    ABCD = np.moveaxis(ABCD, (0,1),(-2,-1))
    total_slow_ABCD = ABCD[0,:,:,:]@ABCD[1,:,:,:]@ABCD[2,:,:,:]@ABCD[4,:,:,:]@ABCD[5,:,:,:]
    total_fast_ABCD = ABCD[0,:,:,:]@ABCD[1,:,:,:]@ABCD[3,:,:,:]@ABCD[4,:,:,:]@ABCD[5,:,:,:]

    # for slow-axis co-polar thru transmission, we want s21 from the slow axis scattering matrix
    t_slow = 2.0 / (total_slow_ABCD[:,0,0] + total_slow_ABCD[:,0,1]/Z0 + total_slow_ABCD[:,1,0]*Z0 + total_slow_ABCD[:,1,1])
    # for co-polar reflection, we want s11
    r_slow = ((total_slow_ABCD[:,0,0] + total_slow_ABCD[:,0,1]/Z0 - total_slow_ABCD[:,1,0]*Z0 - total_slow_ABCD[:,1,1])
        *2.0/t_slow)

    # for fast-axis co-polar thru transmission, we want s21 from the fast axis scattering matrix
    t_fast = 2.0 / (total_fast_ABCD[:,0,0] + total_fast_ABCD[:,0,1]/Z0 + total_fast_ABCD[:,1,0]*Z0 + total_fast_ABCD[:,1,1])
    # for co-polar reflection, we want s11
    r_fast = ((total_fast_ABCD[:,0,0] + total_fast_ABCD[:,0,1]/Z0 - total_fast_ABCD[:,1,0]*Z0 - total_fast_ABCD[:,1,1])
        *2.0/t_fast)

    #Jones matrices for transmission and reflection
    J_ret = np.array([[t_slow,np.zeros(t_slow.size)],[np.zeros(t_fast.size),t_fast]])
    J_ret = np.moveaxis(J_ret, (0,1), (-2,-1))#Same trick as for ABCD
    J_ref = np.array([[r_slow,np.zeros(r_slow.size)],[np.zeros(r_fast.size),r_fast]])
    J_ref = np.moveaxis(J_ref, (0,1), (-2,-1))#Same trick as for ABCD
    #Pauli matrices to go from Jones to Mueller
    sigma1 = np.array([[1.0,0.0],[0.0, 1.0]])
    sigma2 = np.array([[1.0,0.0],[0.0,-1.0]])
    sigma3 = np.array([[0.0,1.0],[1.0, 0.0]])
    sigma4 = np.array([[0.0,-1j],[ 1j, 0.0]])

    T = 0.5*np.trace( np.real(sigma1@J_ret@sigma1@np.conj(np.transpose(J_ret, (0,2,1))) ), axis1=1, axis2=2 )
    rho=0.5*np.trace( np.real(sigma1@J_ret@sigma2@np.conj(np.transpose(J_ret, (0,2,1))) ), axis1=1, axis2=2 )
    c = 0.5*np.trace( np.real(sigma3@J_ret@sigma3@np.conj(np.transpose(J_ret, (0,2,1))) ), axis1=1, axis2=2 )
    s = 0.5*np.trace( np.real(sigma4@J_ret@sigma3@np.conj(np.transpose(J_ret, (0,2,1))) ), axis1=1, axis2=2 )

    R = 0.5*np.trace( np.real(sigma1@J_ref@sigma1@np.conj(np.transpose(J_ret, (0,2,1))) ), axis1=1, axis2=2 )
    tau=0.5*np.trace( np.real(sigma2@J_ref@sigma2@np.conj(np.transpose(J_ref, (0,2,1))) ), axis1=1, axis2=2 )
    h = 0.5*np.trace( np.real(sigma3@J_ref@sigma3@np.conj(np.transpose(J_ref, (0,2,1))) ), axis1=1, axis2=2 )
    q = 0.5*np.trace( np.real(sigma4@J_ref@sigma3@np.conj(np.transpose(J_ref, (0,2,1))) ), axis1=1, axis2=2 )

    return nu, T, rho, c, s, R, tau, h, q

#### call
f = np.arange(0, 1000, 1)

table = np.zeros((f.size,5))
table[:,0] = f*1e9
T, rho, c, s, R, tau, h, q = hwp_mueller_matrix(use_preset_thickness_and_index='150GHz', cooled=True, 
    n_s=False,n_f=False, n_ar=False,n_gap=False,
    d_hwp=False,d_ar_near=False,d_ar_far=False, d_gap_near=False, d_gap_far=False, freq=f)

table[:,1] = np.sqrt(T+rho)
table[:,2] = np.sqrt(T-rho)

np.savetxt("dummy_sapphire.txt", table)

