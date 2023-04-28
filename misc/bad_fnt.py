""" Author: Tanguy Englert
    Last modified: 22.09.2022 """
import os.path
import os
import numpy as np
from pyscf import dft
from pyscf.dft import gen_grid
from pyscf.dft.numint import eval_ao, eval_mat
from taco.embedding.pyscf_emb_pot import get_charges_and_coords, make_potential_matrix
from taco.embedding.pyscf_wrap_single import get_density_from_dm, get_coulomb_repulsion
from modules.functionals import compute_kinetic_potential
from modules.cube.cube import make_cubic_grid
from modules.cube.cube_utils import mol_list, bohr, atls, make_cube, cube_op
def compute_embedding_potential(mola, dma, molb, dmb, points, functional):
    
    # Evaluate electron densities
    rhoa_devs = get_density_from_dm(mola, dma, points, deriv=3, xctype='meta-GGA')
    rhob_devs = get_density_from_dm(molb, dmb, points, deriv=3, xctype='meta-GGA')
    
    # Coulomb repulsion potential
    v_coul = get_coulomb_repulsion(molb, dmb, points)
    
    # Nuclear-electron attraction potential
    molb_charges, molb_coords = get_charges_and_coords(molb)
    vb_nuca = np.zeros(rhoa_devs[0].shape)
    for j, point in enumerate(points):
        for i in range(len(molb_charges)):
            vb_nuca[j] += - molb_charges[i]/np.linalg.norm(point-molb_coords[i]) 
    
    # DFT nad potential
    rho_tot = rhoa_devs[0] + rhob_devs[0]
    vts_nad = compute_kinetic_potential(rhoa_devs, rhob_devs, functional)
   
    vemb_tot = v_coul + vb_nuca + vts_nad
    return vemb_tot


def compute_ft_densities(molA, molB, molAB, xc_code, functional):
    #############################################################
    # Get reference densities
    #############################################################
    # TIP: For HF you only need: scfres1 = scf.RHF(mol)
    # Na+
    scfres = dft.RKS(molA)
    scfres.xc = xc_code
    scfres.conv_tol = 1e-12
    scfres.kernel()
    dma = scfres.make_rdm1()
    # H2O
    scfres1 = dft.RKS(molB)
    scfres1.xc = xc_code
    scfres1.conv_tol = 1e-12
    scfres1.kernel()
    dmb = scfres1.make_rdm1()

    # Construct grid for integration
    #==================================
    # This could be any grid, but we use the Becke 
    gridsa = gen_grid.Grids(molA)
    gridsa.level = 4
    gridsa.build()
    gridsb = gen_grid.Grids(molB)
    gridsb.level = 4
    gridsb.build()


    #############################################################
    # Make embedding potential 
    ############################################################
    vemb_tot = compute_embedding_potential(molA, dma, molB, dmb, gridsa.coords, functional)
    ao_molA = eval_ao(molA, gridsa.coords, deriv=0)
    ao_molB = eval_ao(molB, gridsb.coords, deriv=0)
    rhoa = get_density_from_dm(molA, dma, gridsa.coords)
    vemb_mat = eval_mat(molA, ao_molA, gridsa.weights, rhoa, vemb_tot, xctype='LDA')
    exc_nad, v_nad_xc = make_potential_matrix(molA, molB, molAB, dma, dmb, dma + dmb, gridsa, xc_code)
    vemb_mattot = vemb_mat + v_nad_xc


    #############################################################
    # Run freeze and thaw
    #############################################################
    converged = False
    count = 0
    threshold = 1e-9
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
        fock += vemb_mattot
        scfemb.get_hcore = lambda *args: fock
        scfemb.conv_tol = 1e-11
        # Solve
        scfemb.kernel()
        dm_final = scfemb.make_rdm1()
        energy_final = scfemb.e_tot
        if count == 0:
            dm_old = dm_final.copy()
            energy_old = energy_final
        elif count % 2 == 0:
            denergy = abs(energy_final - energy_old)
            ddensity = np.linalg.norm(dm_final - dm_old)
            if denergy <= threshold:
                break
            else:
                dm_old = dm_final.copy()
                energy_old = energy_final
        int_vemb = np.einsum('ab,ba', vemb_mattot, dm_final)
        print("Expectation energy of vemb:  %.8f a.u." % int_vemb)
        if count >= 30:
            print("Maximal cycles reached")
            break
        else:
            del vemb_mattot
            # Re-evaluate the embedding potential
            # But now for the other molecule
            vemb_tot = compute_embedding_potential(mol1, dm1, mol0, dm_final, grids1.coords, functional)
            rho1 = get_density_from_dm(mol1, dm1, grids1.coords)
            vemb_mat = eval_mat(mol1, ao_mol1, grids1.weights, rho1, vemb_tot, xctype='LDA')
            exc_nad, v_nad_xc = make_potential_matrix(mol1, mol0, molAB, dm1, dm0, dm1 + dm0, grids1, xc_code)
            vemb_mattot = vemb_mat + v_nad_xc
            count += 1
            del scfemb
            del fock_ref, fock

    if count % 2 == 0:
        dma_final = dm_final
        dmb_final = dm1
    else:
        dma_final = dm1
        dmb_final = dm_final

    return dma_final, dmb_final

def compute_ft_frames(molA, molB, molAB, xc_code, functional, atoms, crds, A, B):
    #############################################################
    # Get reference densities
    #############################################################
    # TIP: For HF you only need: scfres1 = scf.RHF(mol)
    # Na+
    dircheck = os.path.isdir('./ftframes/'+A+'_'+B+'/')
    if dircheck == False:
        os.makedirs('./ftframes/'+A+'_'+B+'/')

    # #Origin of the grid at o=(-5,-5,-5)
    o=np.array([-10,-10,-10])
    # #Shape of the grid:
    ncube=130 #Number of points in each direction
    cubegrid=((ncube,ncube,ncube))
    acc=20 #Parameter for the intervall length of the grid (o+[acc,acc,acc])
    # #The stepsize in each direction from the origin o is given by acc and ncube-1
    scube=np.array([[acc/(ncube-1), 0, 0],[0, acc/(ncube-1),0],[0, 0, acc/(ncube-1)]])
    cube = make_cubic_grid(cubegrid, scube, o)

    scfres = dft.RKS(molA)
    scfres.xc = xc_code
    scfres.conv_tol = 1e-12
    scfres.kernel()

    scfres1 = dft.RKS(molB)
    scfres1.xc = xc_code
    scfres1.conv_tol = 1e-12
    scfres1.kernel()

    dma=np.load('./data/PBE,PBE/6-311G**/LDA/'+A+'_'+B+'/'+A+'_'+B+'_ft_dma.npy')
    dmb=np.load('./data/PBE,PBE/6-311G**/LDA/'+A+'_'+B+'/'+A+'_'+B+'_ft_dmb.npy')


    # Construct grid for integration
    #==================================
    # This could be any grid, but we use the Becke 
    gridsa = gen_grid.Grids(molA)
    gridsa.level = 4
    gridsa.build()
    gridsb = gen_grid.Grids(molB)
    gridsb.level = 4
    gridsb.build()


    #############################################################
    # Make embedding potential 
    ############################################################
    vemb_tot = compute_embedding_potential(molA, dma, molB, dmb, gridsa.coords, functional)
    ao_molA = eval_ao(molA, gridsa.coords, deriv=0)
    ao_molB = eval_ao(molB, gridsb.coords, deriv=0)
    rhoa = get_density_from_dm(molA, dma, gridsa.coords)
    vemb_mat = eval_mat(molA, ao_molA, gridsa.weights, rhoa, vemb_tot, xctype='LDA')
    exc_nad, v_nad_xc = make_potential_matrix(molA, molB, molAB, dma, dmb, dma + dmb, gridsa, xc_code)
    vemb_mattot = vemb_mat + v_nad_xc


    #############################################################
    # Run freeze and thaw
    #############################################################
    converged = False
    count = 0
    threshold = 1e-9
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
        fock += vemb_mattot
        scfemb.get_hcore = lambda *args: fock
        scfemb.conv_tol = 1e-11
        # Solve
        scfemb.kernel()
        dm_final = scfemb.make_rdm1()
        energy_final = scfemb.e_tot
        if count == 0:
            dm_old = dm_final.copy()
            energy_old = energy_final
        elif count % 2 == 0:
            denergy = abs(energy_final - energy_old)
            ddensity = np.linalg.norm(dm_final - dm_old)
            if denergy <= threshold:
                break
            else:
                dm_old = dm_final.copy()
                energy_old = energy_final
        int_vemb = np.einsum('ab,ba', vemb_mattot, dm_final)
        print("Expectation energy of vemb:  %.8f a.u." % int_vemb)
        if count >= 30:
            print("Maximal cycles reached")
            break
        else:
            del vemb_mattot
            # Re-evaluate the embedding potential
            rho0cube= get_density_from_dm(mol0, dm0, cube, deriv=3, xctype='meta-GGA')
            

            

            if count % 2 == 0:
                a=A
            
            else:
                a=B
            

            strcount=str(count)

            make_cube(rho0cube[0], './ftframes/'+A+'_'+B+'/'+a+'_rho'+strcount+'_'+functional+'.cube', atoms, crds, o, scube, cubegrid)
            

            # But now for the other molecule
            vemb_tot = compute_embedding_potential(mol1, dm1, mol0, dm_final, grids1.coords, functional)
            rho1 = get_density_from_dm(mol1, dm1, grids1.coords)
            vemb_mat = eval_mat(mol1, ao_mol1, grids1.weights, rho1, vemb_tot, xctype='LDA')
            exc_nad, v_nad_xc = make_potential_matrix(mol1, mol0, molAB, dm1, dm0, dm1 + dm0, grids1, xc_code)
            vemb_mattot = vemb_mat + v_nad_xc
            count += 1
            del scfemb
            del fock_ref, fock

    if count % 2 == 0:
        dma_final = dm_final
        dmb_final = dm1
    else:
        dma_final = dm1
        dmb_final = dm_final