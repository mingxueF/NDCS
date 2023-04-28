import numpy as np


def diff_operator(rho_devs):
    """
    see EQ.19 in paper T.A Wesolowski, JCP,2003.
    differential operator in calculating functional derivative
    D = sum_i sum_j p_i\rho p_i p_j\rho p_j\rho
    """
    print("rho_devs shape",rho_devs.shape)
    X, Y, Z = 1, 2, 3  # first partial derivative
    XX, XY, XZ, YY, YZ, ZZ = 6, 7, 8, 9, 10, 11 # second partial derivative
    D = np.zeros(rho_devs[0].shape)
    D += rho_devs[X]*rho_devs[XX]*rho_devs[X]+rho_devs[Y]*rho_devs[YY]*rho_devs[Y]+rho_devs[Z]*rho_devs[ZZ]*rho_devs[Z]
    D += 2*(rho_devs[X]*rho_devs[XY]*rho_devs[Y] + rho_devs[X]*rho_devs[XZ]*rho_devs[Z]+ rho_devs[Y]*rho_devs[YZ]*rho_devs[Z])
    return D


#def fs_1_deri(x):
#    fs_1 = ((2*x*(0.26608 - 0.0809615* np.exp(-100*x**2)) + (7.16698*x)/np.sqrt(5824.74*x**2 + 1) + 16.1923* np.exp(-100*x**2)*x**3 + 0.093907*np.arcsinh(76.32*x))/(0.000057767*x**4 + 0.093907*x*np.arcsinh(76.32*x) + 1) - (0.000231068*x**3 + (7.16698*x)/np.sqrt(5824.74*x**2 + 1) + 0.093907*np.arcsinh(76.32*x))*(x**2*(0.26608 - 0.0809615* np.exp((-100*x**2)) + 0.093907*x*np.arcsinh(76.32*x) + 1))/(0.000057767*x**4 + 0.093907* x* np.arcsinh(76.32 *x) + 1)**2)
#    return fs_1

def fs_2_deri(x):
    fs_2 = (2*(0.26608 - 0.0809615* np.exp(-100* x**2)) + 64.7692*np.exp(-100 *x**2)* x**2 - 0.0809615* (40000* np.exp(-100* x**2)* x**2 - 200* np.exp(-100* x**2))* x**2 - (41745.8* x**2)/np.power((5824.74* x**2 + 1),3/2) + 14.334/np.sqrt(5824.74* x**2 + 1))/(0.000057767* x**4 + 0.093907* x* np.arcsinh(76.32* x) + 1) + ((2* (0.000231068* x**3 + (7.16698* x)/np.sqrt(5824.74* x**2 + 1) + 0.093907* np.arcsinh(76.32* x))**2)/(0.000057767* x**4 + 0.093907* x* np.arcsinh(76.32* x) + 1)**3 - (-(41745.8* x**2)/np.power((5824.74* x**2 + 1),3/2) + 0.000693204* x**2 + 14.334/np.sqrt(5824.74* x**2 + 1))/(0.000057767* x**4 + 0.093907* x* np.arcsinh(76.32* x) + 1)**2)* (x**2* (0.26608 - 0.0809615* np.exp(-100* x**2)) + 0.093907* x* np.arcsinh(76.32* x) + 1) - (2* (0.000231068* x**3 + (7.16698* x)/np.sqrt(5824.74 *x**2 + 1) + 0.093907* np.arcsinh(76.32* x))* (2* x* (0.26608 - 0.0809615* np.exp(-100* x**2)) + (7.16698* x)/np.sqrt(5824.74* x**2 + 1) + 16.1923* np.exp(-100* x**2)* x**3 + 0.093907* np.arcsinh(76.32* x)))/(0.000057767* x**4 + 0.093907* x* np.arcsinh(76.32* x) + 1)**2
    return fs_2
