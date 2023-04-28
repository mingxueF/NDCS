import numpy as np
import re
from modules.cube.cube import write_cube
from pathlib import Path
from modules.functionals import compute_corr_pyscf, compute_kinetic_potential, compute_kinetic_limit_experiment, ndcs_switch_factor
from taco.embedding.pyscf_wrap_single import get_density_from_dm, get_coulomb_repulsion
from taco.embedding.pyscf_emb_pot import get_charges_and_coords
import modules.cube.cube_tools as ccube




def mol_list(mol):
    ls_coords =[]
    ls_atoms =[]
    a=0
    spmol=mol.split()
    for i in spmol:
        if bool(re.search(r'\d', str(i)))==True:
            ls_coords.append(i)   
        
        if a%4==0:
            ls_atoms.append(i)
            
        a=a+1
    return ls_coords, ls_atoms

def bohr(ls_coords):
    listemp=[]
    coords = np.zeros((1,3))
    coords = np.delete(coords, 0, axis=0)
    b=1
    for i in ls_coords:
        j=float(i)
        j=j*1.88973
        listemp.append(j)

        if b%3==0:
            rw=np.array(listemp)
            coords = np.append(coords,[rw],axis= 0)
            listemp=[]

        b=b+1
    return coords

def atls(ls_atoms):
    atoms=[]
    for i in ls_atoms:
        j=(i)
        atoms.append(j)
    
    #atoms = (', '.join(ato))
    return atoms

def make_cube(data, filename, atoms, coords, o, scube, cubegrid):
    strAB='''graphical data from '''+str(atoms[:])+' \n\n'
    write_cube(np.array(atoms,dtype='<U2'), coords, o, scube,
           cubegrid, data, filename, strAB,bool(True))
           #cubegrid, data, ''+cubepath+'/'+A+'_'+B+'_'+filename, strAB,bool(True))

def cube_op(op, fileA, fileB, fileout):
    #Read relative file path


    #fileA=''+cubepath+'/'+A+'_'+B+'_rhoA_NDCS.cube'
    #fileB=''+cubepath+'/'+A+'_'+B+'_rhoB_NDCS.cube'

    #Check if the file exists:
    fileA_path=Path(fileA)
    fileB_path=Path(fileB)
    if fileA_path.is_file() & fileB_path.is_file():
    #Adding both cube files in order to obtain total density
        if op is 'add':
            ccube.add_cubes([fileA,fileB], fileout)  
        elif op is 'diff':
            ccube.diff_cubes([fileA,fileB], fileout)

def compute_embedding_potential_old(mola, dma, molb, dmb, points, xc_code, functional):
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
    ##XC term
    exc_tot, vxc_tot = compute_corr_pyscf(rho_tot, xc_code)
    exc_a, vxc_a = compute_corr_pyscf(rhoa_devs[0], xc_code)
    exc_b, vxc_b = compute_corr_pyscf(rhob_devs[0], xc_code)
    vxc_nad = vxc_tot - vxc_a

    
    vts_nad = compute_kinetic_potential(rhoa_devs, rhob_devs, functional)
    
    vemb_tot = v_coul + vb_nuca + vxc_nad + vts_nad
    return vemb_tot

def compute_lim_pot(rho):
    sfactor = ndcs_switch_factor(rho[0])                #NDSD2 switching function
    
    wpot = compute_kinetic_limit_experiment(rho) 

    v_lim = sfactor * 1/8 * wpot
    
    return v_lim