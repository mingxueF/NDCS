"""FDET Freeze and Thaw Scripts 


B: Submiting the Calculation
-> Running the calculation using an input file and specifying the use of Module B1 and/or B2

Input: Given in the input file

Output: Optimized Density Matrices [DM_A] and [DM_B]


B.1 Freeze and Thaw Module
-> Runs a Freeze and Thaw calculation on the input parameters from A.

Output: DM_A and DM_B


B.2 Reference Computation
-> Runs the Reference Calculations
Output: DM_Ref, Grid_Ref, Gridweights_Ref, refs (dictonary with all reference values)


C Observable Evaluation
-> Creates an output file with all the observables from the F&T calculations using the specified functionals in the input

Input: DM_A, DM_B and DM_Ref (from part B)

Output: Distance Norm, Dipole Moments, Interaction Energies, Orbital levels, etc. 

D.1_Functionals
-> Defined functionals used for the F&T calculation 

D.2_Utility
-> Defined functionals used for the observable evaluation

"""