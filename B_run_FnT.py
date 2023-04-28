""" Author: Tanguy Englert
    Last modified: 07.12.2022 """

import sys
import imp
from pyscf import gto
from B2_refcomp import compute_ref
from B1_freezeNthaw import compute_ft_densities
from datetime import timedelta
import time

start=time.perf_counter()

#defaut settings that will be overwritten if specified in teh input
format='npy'
threshold=1e-9
gridlvl=4
ref = True
ft = True

#load input geometry and variables
filename = sys.argv[1]
data = imp.load_source('input', filename)
from input import *

#build molecule from the input
molA = gto.M(atom=geoa, basis=basis, charge=chargeA)
molB = gto.M(atom=geob, basis=basis, charge=chargeB)
molAB = gto.M(atom=geoab, basis=basis, charge=(chargeA+chargeB))

mainpath='.'
refpath='.'
ftpath='.'

if ref == True:
    compute_ref(molA, molB, molAB, xc_code, A, B, refpath, format, gridlvl)

if ft == True:
    compute_ft_densities(molA, molB, molAB, xc_code, functional, A, B, ftpath, format, threshold, gridlvl)

end=time.perf_counter()
tm=(end - start)
tm=timedelta(seconds=tm)
print()
print('elapsed time:')
print(tm)
print()