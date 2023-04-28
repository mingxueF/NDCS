""" Utilities for the Observable Evaluation Script
---------------------------------------------------

Author: Tanguy Englert & Elias Polak
Version: 08.12.2022 """

import numpy as np

from pyscf import dft
from pyscf.dft import gen_grid, rks
from taco.embedding.pyscf_emb_pot import get_charges_and_coords
from taco.embedding.pyscf_emb_pot import make_potential_matrix
from taco.embedding.pyscf_wrap_single import get_density_from_dm
from taco.embedding.pyscf_wrap_single import compute_nuclear_repulsion
from Modules.D1_Functionals import ndcs_energy_correc, compute_kinetic_tf, compute_kinetic_limit_experiment, ndcs_switch_factor

#Computing the distance norm
############################################
def distance_norm(molA, molB, molAB, dmA, dmB, A, B, grid_ref_coords, grid_ref_weights):
    """ Distance norm computation: L1-norm difference between sum of FnT densities and Ref densitiy
        
        Output: Distance Norm, L1-norm of sum of FnT densities, L1-norm of Ref density
        Author: Elias Polak ; Version: 08.12.2022"""

    #Evaluate densities on the reference grid
    rhoa=get_density_from_dm(molA, dmA, grid_ref_coords,
                                deriv=3, xctype='meta-GGA')
    rhob=get_density_from_dm(molB, dmB, grid_ref_coords,
                                deriv=3, xctype='meta-GGA')
    rho_tot = rhoa[0] + rhob[0]

    #Load the reference density matrix
    ref_path='./'+A+'_'+B+'_ref_DM.npy'
    ref_dm=np.load(ref_path)

    #Evaluate reference density on the grid
    rho_ref=get_density_from_dm(molAB, ref_dm, grid_ref_coords,
                                    deriv=3, xctype='meta-GGA')

    #Integrating densities on the grid
    L1_ref=np.dot(grid_ref_weights,rho_ref[0])
    L1_tot=np.dot(grid_ref_weights,rho_tot)
    L2_norm=np.dot(grid_ref_weights,np.absolute(rho_ref[0]-rho_tot))
    return L2_norm, L1_ref, L1_tot

#Computing the total dipole moment in debye
############################################
def compute_total_dipole(molA, molB, dmA, dmB, xc_code):
    """ Total Dipole Moment in debye from the FnT densities
        
        Output: Total Dipole Moment
        Author: Tanguy Englert ; Version: 08.12.2022 """

    scfres = dft.RKS(molA)
    scfres.xc = xc_code
    scfres.verbose=0
    scfres.conv_tol = 1e-12
    scfres.kernel()
    dma_iso = scfres.make_rdm1()
    EA_iso=scfres.e_tot
    
    scfres1 = dft.RKS(molB)
    scfres1.xc = xc_code
    scfres1.verbose=0
    scfres1.conv_tol = 1e-12
    scfres1.kernel()
    dmb_iso = scfres1.make_rdm1()
    EB_iso=scfres1.e_tot

    tot_dip_iso = scfres.dip_moment(molA, dm=dma_iso) + scfres1.dip_moment(molB, dm=dmb_iso)
    tot_dip_iso_norm = np.linalg.norm(tot_dip_iso)
    tot_dip = scfres.dip_moment(molA, dm=dmA) + scfres1.dip_moment(molB, dm=dmB)
    tot_dip_norm=np.linalg.norm(tot_dip)
    return tot_dip_norm, EA_iso, EB_iso, tot_dip_iso_norm

#Functions to compute the Coloumb and Attraction potential, and Electrostatic terms:
########################################################################################
def compute_coulomb_potential(mol0, mol1, dm1):
    """Compute the electron-electron repulsion potential.

    Returns
    -------
    v_coulomb : np.ndarray(NAO,NAO)
        Coulomb repulsion potential.

    """
    mol1234 = mol1 + mol1 + mol0 + mol0
    shls_slice = (0, mol1.nbas,
                  mol1.nbas, mol1.nbas+mol1.nbas,
                  mol1.nbas+mol1.nbas, mol1.nbas+mol1.nbas+mol0.nbas,
                  mol1.nbas+mol1.nbas+mol0.nbas, mol1234.nbas)
    eris = mol1234.intor('int2e', shls_slice=shls_slice)
    v_coulomb = np.einsum('ab,abcd->cd', dm1, eris)
    return v_coulomb

def compute_attraction_potential(mol0, mol1):
    """Compute the nuclei-electron attraction potentials.

    Returns
    -------
    v0nuc1 : np.ndarray(NAO,NAO)
        Attraction potential between electron density of fragment0
        and nuclei of fragment1.
    v1nuc0 : np.ndarray(NAO,NAO)
        Attraction potential between electron density of fragment0
        and nuclei of fragment1.

    """
    # Nuclear-electron attraction integrals
    mol0_charges, mol0_coords = get_charges_and_coords(mol0)
    mol1_charges, mol1_coords = get_charges_and_coords(mol1)
    v0_nuc1 = 0
    for i, q in enumerate(mol1_charges):
        mol0.set_rinv_origin(mol1_coords[i])
        v0_nuc1 += mol0.intor('int1e_rinv') * -q
    v1_nuc0 = 0
    for i, q in enumerate(mol0_charges):
        mol1.set_rinv_origin(mol0_coords[i])
        v1_nuc0 += mol1.intor('int1e_rinv') * -q
    return v0_nuc1, v1_nuc0

def compute_electrostatic_terms(molA, molB, dma, dmb):
    v_coul = compute_coulomb_potential(molA, molB, dmb)
    ecoulomb = np.einsum('ab,ba', v_coul, dma)

    v_nuc0, v_nuc1 = compute_attraction_potential(molA, molB)

    a_charges, a_coords = get_charges_and_coords(molA)
    b_charges, b_coords = get_charges_and_coords(molB)
    enuc = compute_nuclear_repulsion(a_charges, a_coords, b_charges, b_coords)
    
    # Nuclear-electron attraction integrals
    evbnuca = np.einsum('ab,ba', v_nuc1, dmb)
    evanucb = np.einsum('ab,ba', v_nuc0, dma)

    E_es_tot=ecoulomb+enuc+evbnuca+evanucb

    return ecoulomb, enuc, evbnuca, evanucb, E_es_tot



#########################################################################
#Interaction energy evaluation
########################################################################


def compute_fragment_energies(molA, molB, dma, dmb, xc_code):
    #fragment A
    dftresa = dft.RKS(molA)
    dftresa.xc = xc_code
    nuca = dftresa.energy_nuc()
    core_ea, e2a = rks.energy_elec(dftresa, dma)
    Ea = core_ea + nuca

    #fragment B
    dftresb = dft.RKS(molB)
    dftresb.xc = xc_code
    nucb = dftresb.energy_nuc()
    core_eb, e2b = rks.energy_elec(dftresb, dmb)
    Eb = core_eb + nucb

    return Ea, Eb

def compute_energy_results(molA, molB, molAB, dmA, dmB, grid_ref_coords, grid_ref_weights, xc_code, A, B, functional):
    

    #Load densities
    rhoa_dev = get_density_from_dm(molA, dmA, grid_ref_coords, deriv=3, xctype='meta-GGA')
    rhob_dev = get_density_from_dm(molB, dmB, grid_ref_coords, deriv=3, xctype='meta-GGA')

    rho_tot = rhoa_dev[0] + rhob_dev[0]

    #Get total dipole moment and isolated energies
    total_dipole, Ea_iso, Eb_iso, total_iso_dipole=compute_total_dipole(molA, molB, dmA, dmB, xc_code)
    
    #################################################################################################################
    #Non-additive Kinetic energies (LDA and/or NDCS)
    #################################################################################################################

    #LDA
    etf_tot, vtftot = compute_kinetic_tf(rho_tot)
    etfa, vtfa = compute_kinetic_tf(rhoa_dev[0])
    etfb, vtfb = compute_kinetic_tf(rhob_dev[0])
    
    e_nad_TF = np.dot(grid_ref_weights, etf_tot - etfa - etfb)

    e_LDA = e_nad_TF
    Tsnad_corr = 0
    #NDCS
    if functional == "NDCS":
        #sym1
        ndcspot = compute_kinetic_limit_experiment(rhob_dev) 
        sfactor = ndcs_switch_factor(rhob_dev[0]) 
        e_limit1 = np.dot(grid_ref_weights, rhoa_dev[0]*ndcspot*1/8*sfactor) 

        e_NDCS_sym1 = e_limit1

        #sym2
        ndcspot = compute_kinetic_limit_experiment(rhoa_dev) 
        sfactor = ndcs_switch_factor(rhoa_dev[0]) 
        e_limit2 = np.dot(grid_ref_weights, rhob_dev[0]*ndcspot*1/8*sfactor) 

        e_NDCS_sym2 = e_limit2

        Tsnad_sym1=e_nad_TF+e_NDCS_sym1
        Tsnad_sym2=e_nad_TF+e_NDCS_sym2

        #First order Symmetry correction
        #sym1
        correc_func, part1, part2, part3 = ndcs_energy_correc(rhoa_dev, rhob_dev)
        correc_s1= np.dot(grid_ref_weights, correc_func*1/8)
        part1_int_s1= np.dot(grid_ref_weights, part1*1/8)
        part2_int_s1= np.dot(grid_ref_weights, part2*1/8)
        part3_int_s1= np.dot(grid_ref_weights, part3*1/8)

        #sym2
        correc_func, part1, part2, part3 = ndcs_energy_correc(rhob_dev, rhoa_dev)
        correc_s2= np.dot(grid_ref_weights, correc_func*1/8)
        part1_int_s2= np.dot(grid_ref_weights, part1*1/8)
        part2_int_s2= np.dot(grid_ref_weights, part2*1/8)
        part3_int_s2= np.dot(grid_ref_weights, part3*1/8)

        #final first order correction
        corr1=correc_s1-e_NDCS_sym1
        corr2=correc_s2-e_NDCS_sym2

        #Symmetry correction constant
        C_corr=(e_NDCS_sym2-e_NDCS_sym1)/(corr1-corr2)

        #First order perturbation symmetry energy
        T_ndcs_corr = (1-C_corr)*e_NDCS_sym1+C_corr*correc_s1

        Tsnad_corr=e_nad_TF+T_ndcs_corr


    #################################################################################################################
    #Exchange correlation terms
    #################################################################################################################
    
    grids = gen_grid.Grids(molAB)
    grids.level = 4 #Density of the grid
    grids.build()

    exc_nad, v_nad_xc = make_potential_matrix(molA, molB, molAB, dmA, dmB, dmA + dmB, grids, xc_code)

    #################################################################################################################
    #coulomb electrostatic terms
    #################################################################################################################

    ecoulomb, enuc, evbnuca, evanucb, E_es_tot=compute_electrostatic_terms(molA, molB, dmA, dmB)

    #################################################################################################################
    #Energy of separate fragments
    #################################################################################################################

    Ea, Eb=compute_fragment_energies(molA, molB, dmA, dmB, xc_code)

    #################################################################################################################
    #E FDET ENERGY
    #################################################################################################################

    E_FDET = Ea+Eb+e_LDA+exc_nad+E_es_tot
    E_DIFF = (Ea+Eb)-Ea_iso-Eb_iso

    if functional == "NDCS":
        E_FDET = Ea+Eb+Tsnad_corr+exc_nad+E_es_tot

    E_int = E_FDET-Ea_iso-Eb_iso

    #Saving everything in a library
    FDET_results={
        'EA_iso':Ea_iso,
        'EB_iso':Eb_iso,
        'EA':Ea,
        'EB':Eb,
        'Et_TF':e_LDA,
        'Ets_nad':Tsnad_corr,
        'Exc_nad':exc_nad,
        'J_AB':ecoulomb,
        'Va(rhob)':evanucb,
        'Vb(rhoa)':evanucb,
        'Vrep_AB':enuc,
        'Enad_tot':(e_LDA+exc_nad),
        'Ees_tot':(ecoulomb+evanucb+evbnuca+enuc),
        'EB+EA-Eiso':E_DIFF,
        'E_FDET':E_FDET,
        'Eint_FDET':E_int,
    }

    FDET_res_for_csv={
        'fragment energies':'-',
        'EA':Ea,
        'EB':Eb,
        'isolated fragment energies':'-',
        'EA_iso':Ea_iso,
        'EB_iso':Eb_iso,
        'diff':'-',
        'EB+EA-Eiso':E_DIFF,
        'XC':'-',
        'Exc_nad':exc_nad,
        'electrostatic terms':'-',
        'J_AB':ecoulomb,
        'Va(rhob)':evanucb,
        'Vb(rhoa)':evanucb,
        'Vrep_AB':enuc,
        'Ees_tot':(ecoulomb+evanucb+evbnuca+enuc),
        'non-additive kinetic energy':'-',
        'Et_TF':e_LDA,
        'Ets_nad':Tsnad_corr,
        'final results':'-',
        'E_FDET':E_FDET,
        'Eint_FDET':E_int,   
    }

    return FDET_results, FDET_res_for_csv

