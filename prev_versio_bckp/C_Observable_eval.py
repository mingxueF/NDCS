""" Observable Evaluation Script from optimized FnT density Matrices
----------------------------------------------------------------------
Input: DM_A, DM_B, DM_Ref

Output: 
-----------------------------------------------------------------------
Author: Elias Polak ; Version: 08.12.2022"""

import json
import sys
import imp
import numpy as np
from pathlib import Path
from pyscf import gto
from B2_Refcomp import compute_ref
from D2_Utilities import distance_norm, compute_total_dipole, compute_energy_results


#Load input geometry and variables
filename = sys.argv[1]
data = imp.load_source('input', filename)
from input import *

#Build molecule from the input
molA = gto.M(atom=geoa, basis=basis, charge=chargeA)
molB = gto.M(atom=geob, basis=basis, charge=chargeB)
molAB = gto.M(atom=geoab, basis=basis, charge=(chargeA+chargeB))

#Specification of the path
mainpath='.'
refpath='.'
ftpath='.'


#Check if Reference is available
refdata_path=refpath+'/'+A+'_'+B+'_refdata.json'
ref_path = Path(refdata_path)
if ref_path.is_file() == False:
    print('Reference data is missing')

# reconstructing the data as a dictionary
refs = json.load(open(refdata_path))

#Loading the FnT densities
dmA=np.load(''+mainpath+'/'+A+'_'+B+'_ft_dma_'+functional+'.npy')
dmB=np.load(''+mainpath+'/'+A+'_'+B+'_ft_dmb_'+functional+'.npy')

#Loading the Grid
grid_ref_coords=np.load(''+refpath+'/'+A+'_'+B+'_ref_grid.npy')
grid_ref_weights=np.load(''+refpath+'/'+A+'_'+B+'_ref_gridweights.npy')


#################################
#Evaluate the distance norm
#################################

L2_norm, L1_ref, L1_tot = distance_norm(molA, molB, molAB, dmA, dmB, A, B, grid_ref_coords, grid_ref_weights)


#################################
#Evaluate the total dipole moment
#################################

tot_dip_norm, EA_iso, EB_iso, tot_dip_iso_norm = compute_total_dipole(molA, molB, dmA, dmB, xc_code)

###################################
#Evaluate the interaction energies
###################################

FDET_results, FDET_res_for_csv = compute_energy_results(molA, molB, molAB, dmA, dmB, grid_ref_coords, grid_ref_weights, xc_code, A, B, functional)


###################################
#Printing the output in a text file
###################################

original_stdout = sys.stdout # Save the original output destination
line = "-"*72
with open ('Output.txt', 'w') as file:
    sys.stdout = file #Change the standard output to the text file above
    print('FDET DFT F&T Results as Observables')
    print(line)
    print('Calculation parameters from the Input')
    print(line)
    print('Chemical System: '+A+'-'+B)
    print('Ts-nad functional: '+functional)
    print('Basis set: '+basis_set)
    print('E_xc-nad functional: '+xc_code)
    print('Method-settings: '+geo_settings)
    print(line)
    print('Results') 
    print(line)   
    print('Integration observables')
    print('Distance_norm_FnT:\t %.10f a.u.' % L2_norm)
    print('Distance_norm_iso:\t %.10f a.u.' % refs['ref_dist'])
    print('L1-norm of FnT densities:\t %.10f a.u.' % L1_tot)
    print('L1-norm of the reference density: \t %.10f a.u.' % L1_ref)
    print(line)
    print('Total dipole moment in debye')
    print('Dipole_moment_FnT:\t %.10f a.u.' % tot_dip_norm)
    print('Dipole_moment_Iso:\t %.10f a.u.' % tot_dip_iso_norm)
    print('Dipole_moment_Ref:\t %.10f a.u.' % refs['ref_dipole'])
    print(line)
    print('Interaction Energies')
    print(line)
    print('E_int_ref:\t %.10f a.u.' % refs['Eint_ref'])
    print('E_int_FnT:\t %.10f a.u.' % FDET_results['Eint_FDET'])
    print(line)
    print('Isolated fragment energies')
    print('E_iso_A:\t %.10f a.u.' % refs['EA_iso_ref'])
    print('E_iso_B:\t %.10f a.u.' % refs['EB_iso_ref']) 
    print(line)
    print('Fragment energies after FnT')
    print('Ea:\t %.10f a.u.' % FDET_results['EA'])
    print('Eb:\t %.10f a.u.' % FDET_results['EB'])
    print(line)
    print('Total FDET energy')
    print('E_FDET:\t %.10f a.u.' % FDET_results['E_FDET'])
    print(line)
    print('More energy observables:')
    print(line)
    print('Exchange correlation')
    print('Exc_nad[rhoa,rhob]:\t %.10f a.u.' % FDET_results['Exc_nad'])
    print('Electrostatic terms')
    print('J_AB:\t %.10f a.u.' % FDET_results['J_AB'])
    print('Nuclear attraction')
    print('Va[rhob]:\t %.10f a.u.' % FDET_results['Va(rhob)'])
    print('Vb[rhoa]:\t %.10f a.u.' % FDET_results['Vb(rhoa)'])
    print('V_AB:\t %.10f a.u.' % FDET_results['Vrep_AB'])
    print('Electrostatic potential')
    print('E_es_tot:\t %.10f a.u.' % FDET_results['Ees_tot'])
    print(line)
    print('Non-additive Kinetic energy')
    print(line)
    print('Ts_nad(TF):\t %.10f a.u.' % FDET_results['Et_TF'])
    print('Ts_nad(TF+NDCS):\t %.10f a.u.' %FDET_results['Ets_nad'])
    print(line)
    print('Observable calculation successfull')
    print(line)
    print('######################################')
    print('Have a nice day!')

    sys.stdout = original_stdout #reset the standard output to the terminal
