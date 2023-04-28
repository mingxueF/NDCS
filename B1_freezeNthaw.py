""" Author: Tanguy Englert
    Last modified: 07.12.2022 """
import numpy as np
from pyscf import dft
from pyscf.dft import gen_grid
from pyscf.dft.numint import eval_ao, eval_mat
from taco.embedding.pyscf_emb_pot import get_charges_and_coords, make_potential_matrix_change
from taco.embedding.pyscf_wrap_single import get_density_from_dm, get_coulomb_repulsion
from NDCS2.D1_Functionals import compute_kinetic_potential
import os.path
import sys
import json



def compute_embedding_mat(mola, dma, molb, dmb, grids, functional,xc_code):

    # Evaluate electron densities
    functionals = ["NDCS","NDCS2","NDCS2-lc94","NDCS2-gea2","LDA","TF","GEA2","LC94"]
    rhoa_devs = get_density_from_dm(mola, dma, grids.coords, deriv=3, xctype='meta-GGA')
    rhob_devs = get_density_from_dm(molb, dmb, grids.coords, deriv=3, xctype='meta-GGA')

    # Coulomb repulsion potential
    v_coul = get_coulomb_repulsion(molb, dmb, grids.coords)

    # Nuclear-electron attraction potential
    molb_charges, molb_coords = get_charges_and_coords(molb)
    vb_nuca = np.zeros(rhoa_devs[0].shape)
    for j, point in enumerate(grids.coords):
        for i in range(len(molb_charges)):
            vb_nuca[j] += - molb_charges[i]/np.linalg.norm(point-molb_coords[i])

    # DFT nad potential
    if functional in functionals:
        rho_tot = rhoa_devs[0] + rhob_devs[0]
        vts_nad = compute_kinetic_potential(rhoa_devs, rhob_devs, functional)
        vemb_tot = v_coul + vb_nuca + vts_nad
        # transfrom in a AO basis-matrix form
        ao_molA = eval_ao(mola, grids.coords, deriv=0)
        ao_molB = eval_ao(molb, grids.coords, deriv=0)
        vemb_mat = eval_mat(mola, ao_molA, grids.weights, rhoa_devs[0], vemb_tot, xctype='LDA')
    else:
        vemb_tot = v_coul + vb_nuca 
        # transfrom in a AO basis-matrix form
        ao_molA = eval_ao(mola, grids.coords, deriv=0)
        ao_molB = eval_ao(molb, grids.coords, deriv=0)
        vemb_mat = eval_mat(mola, ao_molA, grids.weights, rhoa_devs[0], vemb_tot, xctype='LDA')
        print("using kenetic functional:",functional)
        ets_nad,vts_nad = make_potential_matrix_change(mola, rhoa_devs,rhob_devs, grids, functional)
        vemb_mat += vts_nad
    # add the part of v_xc_nad
    exc_nad, v_nad_xc = make_potential_matrix_change(mola, rhoa_devs,rhob_devs, grids, xc_code)
    vemb_mat += v_nad_xc
    print("make embedding potential in a Fock-like matrix")
    return vemb_mat

def compute_ft_densities(molA, molB, molAB, xc_code, functional, A, B, path, format, thresh, gridlvl):

    mo_energies = {"homo_A":[],"lumo_A":[],"homo_B":[],"lumo_B":[]}
    #############################################################
    # Get reference densities
    #############################################################
    # TIP: For HF you only need: scfres1 = scf.RHF(mol)
    a_nelec = molA.nelectron
    print("number of electron:",a_nelec)
    b_nelec = molB.nelectron
    print("number of electron:",b_nelec)
    scfres = dft.RKS(molA)
    scfres.xc = xc_code
    scfres.conv_tol = 1e-12
    scfres.kernel()
    dma = scfres.make_rdm1()
    ##
    scfres1 = dft.RKS(molB)
    scfres1.xc = xc_code
    scfres1.conv_tol = 1e-12
    scfres1.kernel()
    dmb = scfres1.make_rdm1()

    # Construct grid for integration
    #==================================
    # This could be any grid, but we use the Becke 
    gridsa = gen_grid.Grids(molA)
    gridsa.level = gridlvl
    gridsa.build()
    gridsb = gen_grid.Grids(molB)
    gridsb.level = gridlvl
    gridsb.build()

    #############################################################
    # Make embedding potential
    ############################################################
    vemb_mat = compute_embedding_mat(molA, dma, molB, dmb, gridsa,functional,xc_code)
    ao_molA = eval_ao(molA, gridsa.coords, deriv=0)
    ao_molB = eval_ao(molB, gridsb.coords, deriv=0)
    print("prepare the embedding potential matrix")

    #############################################################
    # Run freeze and thaw
    #############################################################
    converged = False
    count = 0
    threshold = thresh
    # Loop until embedded density and energy converges
    while True:
        print("SCF cycle: ", count)
        if count % 2 == 0:
            mol0 = molA
            mol1 = molB
            ao_mol0 = ao_molA
            ao_mol1 = ao_molB
            fock_ref = scfres.get_hcore()
            fock = fock_ref.copy()
            grids0 = gridsa
            grids1 = gridsb
            if count == 0:
                dm0 = dma.copy()
                dm1 = dmb.copy()
            else:
                dm0 = dm1.copy()
                dm1 = dm_final.copy()
        else:
            mol0 = molB
            mol1 = molA
            ao_mol0 = ao_molB
            ao_mol1 = ao_molA
            fock_ref = scfres1.get_hcore()
            fock = fock_ref.copy()
            dm0 = dm1.copy()
            dm1 = dm_final.copy()
            grids0 = gridsb
            grids1 = gridsa
        scfemb = dft.RKS(mol0)
        scfemb.xc = xc_code
        fock += vemb_mat
        scfemb.get_hcore = lambda *args: fock
        scfemb.conv_tol = 1e-11
        # Solve
        scfemb.kernel()
        dm_final = scfemb.make_rdm1()
        print("dipole moment:", np.linalg.norm(scfemb.dip_moment(mol0, dm_final)))
        energy_final = scfemb.e_tot
        if count == 0:
            dm_old = dm_final.copy()
            energy_old = energy_final
        elif count % 2 == 0:
            denergy = abs(energy_final - energy_old)
            ddensity = np.linalg.norm(dm_final - dm_old)
            if denergy <= threshold:
                converged = True
                break
            else:
                dm_old = dm_final.copy()
                energy_old = energy_final

        int_vemb = np.einsum('ab,ba', vemb_mat, dm_final)
        print("Expectation energy of vemb:  %.8f a.u." % int_vemb)
        if count % 2 == 0:

            mo_energy = scfemb.mo_energy
            homo = mo_energy[(int(a_nelec)//2-1)]
            lumo = mo_energy[int(a_nelec)//2]
            mo_energies["homo_A"].append(homo)
            mo_energies["lumo_A"].append(lumo)
            print("homo energy:",homo)
            print("lumo energy:",lumo)
        if count % 2 != 0:
            mo_energy = scfemb.mo_energy
            homo = mo_energy[(int(b_nelec)//2-1)]
            lumo = mo_energy[int(b_nelec)//2]
            mo_energies["homo_B"].append(homo)
            mo_energies["lumo_B"].append(lumo)
            print("homo energy:",homo)
            print("lumo energy:",lumo)
        if count >= 30:
            print("Maximal cycles reached")
            print("Break the caculations now----------")
            break
            #sys.exit()
        else:
            del vemb_mat
            # Re-evaluate the embedding potential
            # But now for the other molecule
            vemb_mat = compute_embedding_mat(mol1, dm1, mol0, dm_final, grids1,functional,xc_code)
            count += 1
            del scfemb
            del fock_ref, fock
    if converged:
        if count % 2 == 0:
            print("Energy is converged in "+str(count) +" cycles--------")
            dma_final = dm_final
            dmb_final = dm1
        else:
            dma_final = dm1
            dmb_final = dm_final

        if format == 'txt':
            np.savetxt(''+path+'/'+A+'_'+B+'_ft_dma.txt', dma_final, delimiter='\n')
            np.savetxt(''+path+'/'+A+'_'+B+'_ft_dmb.txt', dmb_final, delimiter='\n')
        else:
            print("Save the density matrix-------")
            np.save(''+path+'/'+A+'_'+B+'_ft_dma.npy', dma_final)
            np.save(''+path+'/'+A+'_'+B+'_ft_dmb.npy', dmb_final)

    if not converged:
        print("Energy is not converged in 30 cycles----------")
    with open(functional+'_mo_energy.json','w') as f:
        json.dump(mo_energies,f)
    return count
