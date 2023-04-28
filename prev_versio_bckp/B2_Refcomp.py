""" Author: Tanguy Englert
    Last modified: 22.09.2022 """

import json
from pyscf import dft
from pyscf.dft import gen_grid
import numpy as np
from taco.embedding.pyscf_wrap_single import get_density_from_dm


def compute_ref(molA, molB, molAB, xc_code, A, B, path, format, gridlvl):
    #Full DFT calculations:
    scfres = dft.RKS(molAB)
    scfres.xc = xc_code
    #scfres.verbose=0
    scfres.conv_tol = 1e-12
    scfres.kernel()
    ref_dm = scfres.make_rdm1()
    ref_dip=np.linalg.norm(scfres.dip_moment())
    Etot_ref=scfres.e_tot

    scfres_A = dft.RKS(molA)
    scfres_A.xc = xc_code
    #scfres_A.verbose=0
    scfres_A.conv_tol = 1e-12
    scfres_A.kernel()
    dma = scfres_A.make_rdm1()
    EA_ref=scfres_A.e_tot
    # isocca=np.size(scfres_A.mo_occ[scfres_A.mo_occ != 0]) #Number of occupied orbitals
    # homoa=scfres_A.mo_energy[isocca-1] #Highest occupied orbital
    # lumoa=scfres_A.mo_energy[isocca] #Lowest unoccupied orbital


    scfres_B = dft.RKS(molB)
    scfres_B.xc = xc_code
    #scfres_B.verbose=0
    scfres_B.conv_tol = 1e-12
    scfres_B.kernel()
    dmb = scfres_B.make_rdm1()
    EB_ref=scfres_B.e_tot
    # isoccb=np.size(scfres_B.mo_occ[scfres_B.mo_occ != 0]) #Number of occupied orbitals
    # homob=scfres_B.mo_energy[isoccb-1] #Highest occupied orbital
    # lumob=scfres_B.mo_energy[isoccb] #Lowest unoccupied orbital

    Eint_ref=Etot_ref-EA_ref-EB_ref

    #======
    #Saving density matrix and full density with the griven grid
    #======

    #Generating a Becke-grid for the integration
    grids = gen_grid.Grids(molAB)
    grids.level = gridlvl #Density of the grid
    grids.build()

    grid_ref_coords=grids.coords
    grid_ref_weights=grids.weights

    #Evaluate reference density on the grid
    rho_ref=get_density_from_dm(molAB, ref_dm, grids.coords,
                                    deriv=3, xctype='meta-GGA')

    rhoa_ref=get_density_from_dm(molA, dma, grid_ref_coords,
                                    deriv=3, xctype='meta-GGA')
    rhob_ref=get_density_from_dm(molB, dmb, grid_ref_coords,
                                    deriv=3, xctype='meta-GGA')

    rho_tot2 = rhoa_ref[0] + rhob_ref[0]
    L2_diff=np.dot(grid_ref_weights,np.absolute(rho_ref[0]-rho_tot2))**0.5
    L2_diffsq=L2_diff**2

    if format == 'txt':
        np.savetxt(''+path+'/'+A+'_'+B+'_ref_DM.txt', ref_dm, delimiter='\n')
        np.savetxt(''+path+'/'+A+'_'+B+'_ref_grid.txt', grid_ref_coords, delimiter='\n')
        np.savetxt(''+path+'/'+A+'_'+B+'_ref_gridweights.txt', grid_ref_weights, delimiter='\n')
    else:    
        np.save(''+path+'/'+A+'_'+B+'_ref_DM.npy', ref_dm)
        np.save(''+path+'/'+A+'_'+B+'_ref_grid.npy', grid_ref_coords)
        np.save(''+path+'/'+A+'_'+B+'_ref_gridweights.npy', grid_ref_weights)


    #storing the reference values in a dictionary
    refs={
        'reference values':'-',
        'Etot_ref':Etot_ref,
        'EA_iso_ref':EA_ref,
        'EB_iso_ref':EB_ref,
        'Eint_ref': Eint_ref,
        'ref_dipole': ref_dip,
        'ref_dist': L2_diffsq
    }
    #Save the reference values
    json.dump(refs, open(''+path+'/'+A+'_'+B+'_refdata.json','w'))

    return ref_dm, grid_ref_coords, grid_ref_weights, refs
