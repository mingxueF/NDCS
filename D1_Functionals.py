"""NDCS functionalities for F&T calculations 
    author: Elias Polak, version: 15.07.2022
"""

# import NumPy
import numpy as np
from NDCS2.func_deriv import diff_operator 
from sympy import *
#functional sellection

def compute_kinetic_potential(rhoa_devs, rhob_devs, functional):

    if functional == "NDCS":
        v_t=compute_kinetic_ndcs_potential(rhoa_devs, rhob_devs)
    if functional == "NDCS2":
        v_t=compute_kinetic_ndcs2_potential(rhoa_devs, rhob_devs)
    if functional == "NDCS2-lc94":
        v_t=compute_kinetic_ndcs2lc_potential(rhoa_devs, rhob_devs)
    if functional == "NDCS2-gea2":
        v_t=compute_kinetic_ndcs2gea2_potential(rhoa_devs, rhob_devs)
    if functional == 'LDA' or functional == 'TF':
        v_t=compute_kinetic_lda_potential(rhoa_devs, rhob_devs)
    if functional =="GEA2":
        v_t=compute_kinetic_gea2_potential(rhoa_devs,rhob_devs)  
    if functional =="LC94":
        rho_tot = rhoa_devs + rhob_devs                         #rho_tot=rho_A+rho_B
        vt_tot = compute_lc_potential(rho_tot)       #Evaluating the lc potential at rho_A+rho_B
        vt_0 = compute_lc_potential(rhoa_devs)       #Evaluating the lc potential at rho_A
        v_t = vt_tot - vt_0            # potential
    return v_t

#Thomas-Fermi kinetic energy and potential
def compute_kinetic_tf(rho):
    """
    TF energy density
    """
    zero_mask = np.where(rho > 1e-10)[0] #A zero mask is added to exclude areas where rho=0
    cf = (3./10.)*(np.power(3.0*np.pi**2, 2./3.))
    et = np.zeros(rho.shape)
    vt = np.zeros(rho.shape)
    et[zero_mask] = cf*(np.power(rho[zero_mask], 5./3.))
    vt[zero_mask] = cf*5./3.*(np.power(rho[zero_mask], 2./3.))
    return et, vt
     # return compute_corr_pyscf(rho, xc_code='LDA_K_TF')

def compute_kinetic_lc(rho_devs):
    """
    energy density of GGA kinetic functional LC94
    fs- the enhancement factor
    """
    zero_mask = np.where(rho_devs[0] > 1e-10)[0] #A zero mask is added to exclude areas where rho=0
    print('zero_mask',zero_mask)
    cs = 2*(np.power(3*np.pi**2,1./3.)) 
    grad_rho = rho_devs[1:4].T
    norm_grad = np.sqrt(np.einsum('ij,ij->i', grad_rho, grad_rho)[zero_mask]) # |\nabla\rho|
    s = norm_grad/(cs*np.power(rho_devs[0][zero_mask],4/3)) 
    fs = np.zeros(rho_devs[0].shape)
    print("density matrix shape:",rho_devs[0].shape)
    a = 0.093907
    b = 76.32
    c = 0.26608
    d = 0.0809615
    e = 0.000057767
    etf,vtf = compute_kinetic_tf(rho_devs[0])
    print("dimension of tf", etf.shape)
    fs[zero_mask] = (1 + a*s*np.arcsinh(b*s)+(c-d*np.exp(-100*s**2))*s**2)/(1 + a*s*np.arcsinh(b*s) + e*s**4) 
    print("dimension of fs", fs.shape)
    #e_lc = np.einsum("i,i->i",etf,fs)
    e_lc = etf*fs
    return e_lc


def compute_lc_potential(rho_devs):
    cs = 2*(np.power(3*np.pi**2,1./3.)) 
    cf = (3./10.)*(np.power(3.0*np.pi**2, 2./3.))
    a = 0.093907
    b = 76.32
    c = 0.26608
    d = 0.0809615
    e = 0.000057767
    zero_mask = np.where(rho_devs[0] > 1e-10)[0] #A zero mask is added to exclude areas where rho=0
    print("shape of rho:",rho_devs[0].shape)
    grad_rho = rho_devs[1:4].T
    norm_grad = np.sqrt(np.einsum('ij,ij->i', grad_rho, grad_rho)) # |\nabla\rho|
    s = np.zeros(rho_devs[0].shape)
    s[zero_mask] = norm_grad[zero_mask]/(cs*np.power(rho_devs[0][zero_mask],4/3)) 
    x = s * cs  
    fs = np.zeros(rho_devs[0].shape)
    fs_1 = np.zeros(rho_devs[0].shape)
    fs_2 = np.zeros(rho_devs[0].shape)
    vt = np.zeros(rho_devs[0].shape)
    # get the derivatives of enhancement factor f(s)
    t = symbols('t')
    # define the function f(s)
    f = (1 + a*t*asinh(b*t)+(c-d*exp(-100*t**2))*t**2)/(1 + a*t*asinh(b*t) + e*t**4)
    df = diff(f, t) # first derivative
    ddf = diff(df, t) # second derivative
    # to take multiple entries
    f = lambdify(t, f)
    df = lambdify(t, df)
    ddf = lambdify(t,ddf)
    fs = f(s)
    fs_1 = df(s)
    fs_2 = ddf(s)
    vt[zero_mask] = 5/3*cf*np.power(rho_devs[0][zero_mask],2./3.)*(fs[zero_mask]-s[zero_mask]*fs_1[zero_mask] + 4/5*s[zero_mask]**2*fs_2[zero_mask])
    #vt = 5/3*cf*np.power(rho_devs[0],2./3.)*(fs-s*fs_1 + 4/5*s**2*fs_2)
    D = diff_operator(rho_devs)
    vt[zero_mask] += cf*D[zero_mask]*(np.power(rho_devs[0][zero_mask],1./3.)/np.power(norm_grad,3)[zero_mask])*(fs_1[zero_mask]/cs - s[zero_mask]*fs_2[zero_mask]/(cs))
    vt[zero_mask] -= (cf/(cs**2)*(fs_1[zero_mask]/s[zero_mask])*(rho_devs[4][zero_mask]/rho_devs[0][zero_mask]))
    return vt
    

#Correlation energy functional
def compute_corr_pyscf(rho, xc_code=',VWN'):
    from pyscf.dft import libxc
    exc, vxc, fxc, kxc = libxc.eval_xc(xc_code, rho)
    return exc, vxc[0]

def compute_kinetic_weizsacker_potential(rho_devs):
    """Compute the Weizsacker Potential.

    Parameters
    ----------
    rho_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    """
    #Non-experimental functional derivative with the original gamma=2 factor:
    zero_mask = np.where(rho_devs[0] > 1e-10)[0] #A zero mask is added to exclude areas where rho=0
    wpot = np.zeros(rho_devs[0].shape)
    grad_rho = rho_devs[1:4].T
    wpot[zero_mask] += (np.einsum('ij,ij->i', grad_rho, grad_rho)[zero_mask])
    wpot[zero_mask] /= pow(rho_devs[0][zero_mask], 2)
    wpot[zero_mask] += - 2*(rho_devs[4][zero_mask])/rho_devs[0][zero_mask]
    return wpot

def compute_kinetic_weizsacker(rho_devs):
    "Compute the TFW energy"""
    
    """Parameters
    ----------
    rho_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    """
    #Nonadditive GEA (the gradient expansion of the kinetic energy truncated to the second order) potential
    zero_mask = np.where(rho_devs[0] > 1e-10)[0] #A zero mask is added to exclude areas where rho=0
    eW = np.zeros(rho_devs[0].shape)
    grad_rho = rho_devs[1:4].T
    #eW = (np.einsum('ij,ij->i', grad_rho, grad_rho)[zero_mask]) / (rho_devs[0][zero_mask])
    eW[zero_mask] = (np.einsum('ij,ij->i', grad_rho, grad_rho)[zero_mask]) / (rho_devs[0][zero_mask])
    return eW

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
    sfactor[zero_maskB] = (1.0-np.exp(-(rhoB[zero_maskB])))

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

def compute_kinetic_ndcs2_potential(rho0_devs, rho1_devs):
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
    vt_tot = compute_lc_potential(rho_tot)       #Evaluating the lc potential at rho_A+rho_B
    vt_0 = compute_lc_potential(rho0_devs)       #Evaluating the lc potential at rho_A
    v_lc = vt_tot - vt_0            # potential
    vt_tf = compute_kinetic_lda_potential(rho0_devs,rho1_devs) # have the gea2 instead of lda vt_nad 
    sfactor = ndcs_switch_factor(rho1_devs[0])                #NDSD2 switching function
    wpot = compute_kinetic_limit_experiment(rho1_devs)   #Limit potential (gamma=1)
    v_t = vt_tf + (1.0-sfactor)*(v_lc - vt_tf) + sfactor * 1/8 * wpot
    #v_t = v_lc + sfactor * 1/8 * wpot
    return v_t

def compute_kinetic_ndcs2lc_potential(rho0_devs, rho1_devs):
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
    vt_tot = compute_lc_potential(rho_tot)       #Evaluating the lc potential at rho_A+rho_B
    vt_0 = compute_lc_potential(rho0_devs)       #Evaluating the lc potential at rho_A
    v_lc = vt_tot - vt_0            # potential
    vt_tf = compute_kinetic_lda_potential(rho0_devs,rho1_devs) # have the gea2 instead of lda vt_nad 
    sfactor = ndcs_switch_factor(rho1_devs[0])                #NDSD2 switching function
    wpot = compute_kinetic_limit_experiment(rho1_devs)   #Limit potential (gamma=1)
    #v_t = vt_tf + (1.0-sfactor)*(v_lc - vt_tf) + sfactor * 1/8 * wpot
    v_t = v_lc + sfactor * 1/8 * wpot
    return v_t

def compute_kinetic_ndcs2gea2_potential(rho0_devs, rho1_devs):
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
    vt_tf = compute_kinetic_lda_potential(rho0_devs,rho1_devs) # have the gea2 instead of lda vt_nad 
    vt_vw = compute_kinetic_weizsacker_potential(rho_tot) # have the gea2 instead of lda vt_nad 
    vt_vw0 = compute_kinetic_weizsacker_potential(rho0_devs) # have the gea2 instead of lda vt_nad 
    sfactor = ndcs_switch_factor(rho1_devs[0])                #NDSD2 switching function
    wpot = compute_kinetic_limit_experiment(rho1_devs)   #Limit potential (gamma=1)
    v_t = vt_tf + (1.0-sfactor)*1/72*(vt_vw - vt_vw0) + sfactor * 1/8 * wpot
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

def compute_kinetic_gea2_potential(rhoa_devs,rhob_devs):
    rho_tot_devs = rhoa_devs + rhob_devs
    v_tf = compute_kinetic_lda_potential(rhoa_devs,rhob_devs)
    vW_tot = compute_kinetic_weizsacker_potential(rho_tot_devs)
    vW_0 = compute_kinetic_weizsacker_potential(rhoa_devs)
    v_t = v_tf + 1/72 * (vW_tot - vW_0)
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

    tot = part1 + part2 - part3

    return tot, part1, part2, part3
