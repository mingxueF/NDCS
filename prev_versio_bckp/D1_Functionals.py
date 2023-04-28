"""NDCS functionalities for F&T calculations 
    author: Elias Polak, version: 15.07.2022
"""

# import NumPy
import numpy as np

#functional sellection

def compute_kinetic_potential(rhoa_devs, rhob_devs, functional):

    if functional == "NDCS":
        v_t=compute_kinetic_ndcs_potential(rhoa_devs, rhob_devs)

    if functional == 'LDA' or functional == 'TF':
        v_t=compute_kinetic_lda_potential(rhoa_devs, rhob_devs)

    
    return v_t

#Thomas-Fermi kinetic energy and potential
def compute_kinetic_tf(rho):
    cf = (3./10.)*(np.power(3.0*np.pi**2, 2./3.))
    et = cf*(np.power(rho, 5./3.))
    vt = cf*5./3.*(np.power(rho, 2./3.))
    return et, vt
     # return compute_corr_pyscf(rho, xc_code='LDA_K_TF')

#Correlation energy functional
def compute_corr_pyscf(rho, xc_code=',VWN'):
    from pyscf.dft import libxc
    exc, vxc, fxc, kxc = libxc.eval_xc(xc_code, rho)
    return exc, vxc[0]


#The NDCS kinetic limit potential
def compute_kinetic_limit_experiment(rho_devs):
    """
    Parameters
    ----------
    rho_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    """
	#Limit potential with the gamma=1 factor:
    zero_mask = np.where(rho_devs[0] > 1e-10)[0] #A zero mask is added to exclude areas where rho=0
    wpot = np.zeros(rho_devs[0].shape)
    grad_rho = rho_devs[1:4].T
    wpot[zero_mask] += (np.einsum('ij,ij->i', grad_rho, grad_rho)[zero_mask])
    wpot[zero_mask] /= pow(rho_devs[0][zero_mask], 2)
    wpot[zero_mask] += - (rho_devs[4][zero_mask])/rho_devs[0][zero_mask]
    return wpot


#Switching function NDCS
def ndcs_switch_factor(rhoB):
    """     
    This formula is constructed artificially after reasoning with the NDCS potential.
    It motivates from the condition in the Lastra et. al. 2008 paper. 
    Details can be found in the theoretical notes. 

    Input: 

    rhoB : np.array((1, N))
        Array with the density derivatives,
        density = rhoB


    Output: Real-valued switching constant between 0 and 1. """

    #Setting a zero mask for avoiding to small densities in rhoB (for wpot):
    zero_maskB=np.where(rhoB>1e-10)

    #Preallocate sfactor
    sfactor = np.zeros(rhoB.shape)

    #Formula for f^{NDSC}(rho_B)=(1-exp(-rho_B))
    sfactor[zero_maskB] = (1-np.exp(-(rhoB[zero_maskB])))

    return sfactor


#The non-additive kinetic NDCS potential
def compute_kinetic_ndcs_potential(rho0_devs, rho1_devs):
    """
    Parameters
    ----------
    rho0_devs, rho1_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    """
    rho_tot = rho0_devs + rho1_devs                         #rho_tot=rho_A+rho_B

    etf_tot, vtf_tot = compute_kinetic_tf(rho_tot[0])       #Evaluating the TF potential at rho_A+rho_B
    etf_0, vtf_0 = compute_kinetic_tf(rho0_devs[0])         #Evaluating the TF potential at rho_A
    
    sfactor = ndcs_switch_factor(rho1_devs[0])                #NDSD2 switching function
    
    wpot = compute_kinetic_limit_experiment(rho1_devs)   #Limit potential (gamma=1)

    v_t = vtf_tot - vtf_0 + sfactor * 1/8 * wpot            #NDSD potential


    return v_t

#The LDA potential
def compute_kinetic_lda_potential(rho0_devs, rho1_devs):
    """
    Parameters
    ----------
    rho0_devs, rho1_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    """
    rho_tot = rho0_devs + rho1_devs                         #rho_tot=rho_A+rho_B

    etf_tot, vtf_tot = compute_kinetic_tf(rho_tot[0])       #Evaluating the TF potential at rho_A+rho_B
    etf_0, vtf_0 = compute_kinetic_tf(rho0_devs[0])         #Evaluating the TF potential at rho_A
    
    v_t = vtf_tot - vtf_0            #TF potential

    return v_t

#The (old) NDSD switching function
def ndsd1_switch_factor(rhoB_devs, plambda):
    """Compute the NDSD1 switch functional.

    This formula follows eq. 21 from Garcia-Lastra 2008.

    Parameters
    ---------
    rhoB_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    plambda :  float
        Smoothing parameter.
    """

    #(by Elias)
    #====

    rhob = rhoB_devs[0]
    sb_min = 0.3 #(Standard: 0.3)
    sb_max = 0.9 #(Standard: 0.9, Opt: 1.5)
    rhob_min = 0.7 #(Standard: 0.7, Opt: 0.1)

    #Setting a zero mask for avoiding to small densities:
    zero_mask=np.where(rhoB_devs[0]>1e-10)[0]

    #Computation of reduced density gradient:
    Gradnorm = np.sqrt(rhoB_devs[1][zero_mask]**2+rhoB_devs[2][zero_mask]**2+rhoB_devs[3][zero_mask]**2)
    sbdenom = 2.0*(3.0*np.pi**2.0)**(1./3.)
    rhodenom = rhob[zero_mask]**(4./3.)
    sb=Gradnorm/(sbdenom*rhodenom)

    #Computing switching function

    Fone=np.exp(plambda*(sb_min-sb))

    Ftwo=np.exp(plambda*(sb_max-sb))

    Fthr=np.exp(plambda*(rhob_min-rhob[zero_mask]))

    sfactor = np.zeros(rhoB_devs[0].shape)

    #Final formula:
    sfactor[zero_mask]=1.0/(Fone+1)*(1.0-1.0/(Ftwo+1.0))*(1.0/(Fthr+1))
 
    return sfactor


#The NDCS kinetic energy symmetry correction
def ndcs_energy_correc(rhoA_devs, rhoB_devs):
    """
    The total NDCS Symmetry correction comes from the first order functional derivative of the asymmetry part of the energy functional. 

    Parameters
    ---------
    rho_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    """
    zero_maskA = np.where(rhoA_devs[0] > 1e-10)[0] #A zero mask is added to exclude areas where rhoA=0

    #Preallocate
    part1 = np.zeros(rhoA_devs[0].shape)
    part2 = np.zeros(rhoA_devs[0].shape)
    part3 = np.zeros(rhoA_devs[0].shape)
    
    #Norm of the gradient of rhoA    
    GradnormA = np.sqrt(rhoA_devs[1]**2+rhoA_devs[2]**2+rhoA_devs[3]**2)
    
    #The 3 parts of the integrand: Functional derivative of T^NDCS[rhoB,rhoA] w.r.t. rhoA
    part1[zero_maskA] = -rhoB_devs[4][zero_maskA]*ndcs_switch_factor(rhoA_devs[0][zero_maskA])
    part2[zero_maskA] = rhoB_devs[0][zero_maskA]*(1-ndcs_switch_factor(rhoA_devs[0][zero_maskA]))*GradnormA[zero_maskA]**2*(rhoA_devs[0][zero_maskA]**(-1))
    part3[zero_maskA] = rhoB_devs[0][zero_maskA]*(1-ndcs_switch_factor(rhoA_devs[0][zero_maskA]))*GradnormA[zero_maskA]**2

    tot = part1 + part2 + part3

    return tot, part1, part2, part3