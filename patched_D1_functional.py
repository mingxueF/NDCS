--- D1_Functionals.py	2022-12-14 23:25:14.000000000 +0100
+++ ../NDCS/functionals.py	2022-12-16 17:15:03.000000000 +0100
@@ -1,5 +1,6 @@
 """NDCS functionalities for F&T calculations 
-    author: Elias Polak, version: 15.07.2022
+    author: Elias Polak, version: 11.06.2021
+    modified by Tanguy Englert, last change 22.09.2022
 """
 
 # import NumPy
@@ -15,7 +16,8 @@
     if functional == 'LDA' or functional == 'TF':
         v_t=compute_kinetic_lda_potential(rhoa_devs, rhob_devs)
 
-    
+    if functional =="GEA2":
+        v_t=compute_kinetic_gea2_potentail(rhoa_devs,rhob_devs)  
     return v_t
 
 #Thomas-Fermi kinetic energy and potential
@@ -32,8 +34,47 @@
     exc, vxc, fxc, kxc = libxc.eval_xc(xc_code, rho)
     return exc, vxc[0]
 
+def compute_kinetic_weizsacker_potential(rho_devs):
+    """Compute the Weizsacker Potential.
+
+    Parameters
+    ----------
+    rho_devs : np.array((6, N))
+        Array with the density derivatives,
+        density = rho_devs[0]
+        grad = rho_devs[1:3] (x, y, z) derivatives
+        laplacian = rho_devs[4]
+    """
+    #Non-experimental functional derivative with the original gamma=2 factor:
+    zero_mask = np.where(rho_devs[0] > 1e-10)[0] #A zero mask is added to exclude areas where rho=0
+    wpot = np.zeros(rho_devs[0].shape)
+    grad_rho = rho_devs[1:4].T
+    wpot[zero_mask] += (np.einsum('ij,ij->i', grad_rho, grad_rho)[zero_mask])
+    wpot[zero_mask] /= pow(rho_devs[0][zero_mask], 2)
+    wpot[zero_mask] += - 2*(rho_devs[4][zero_mask])/rho_devs[0][zero_mask]
+    return wpot
+
+def compute_kinetic_weizsacker(rho_devs):
+    "Compute the TFW Potential"""
+    
+    """Parameters
+    ----------
+    rho_devs : np.array((6, N))
+        Array with the density derivatives,
+        density = rho_devs[0]
+        grad = rho_devs[1:3] (x, y, z) derivatives
+        laplacian = rho_devs[4]
+    """
+    #Nonadditive GEA (the gradient expansion of the kinetic energy truncated to the second order) potential
+    zero_mask = np.where(rho_devs[0] > 1e-10)[0] #A zero mask is added to exclude areas where rho=0
 
-#The NDCS kinetic limit potential
+    eW = np.zeros(rho_devs[0].shape)
+    grad_rho = rho_devs[1:4].T
+
+    eW = (np.einsum('ij,ij->i', grad_rho, grad_rho)[zero_mask]) / (rho_devs[0][zero_mask])
+    return eW
+
+#The v_t^limit(gamma=1) experimental potential
 def compute_kinetic_limit_experiment(rho_devs):
     """
     Parameters
@@ -82,7 +123,7 @@
     return sfactor
 
 
-#The non-additive kinetic NDCS potential
+#The NDCS potential
 def compute_kinetic_ndcs_potential(rho0_devs, rho1_devs):
     """
     Parameters
@@ -102,12 +143,11 @@
     
     wpot = compute_kinetic_limit_experiment(rho1_devs)   #Limit potential (gamma=1)
 
-    v_t = vtf_tot - vtf_0 + sfactor * 1/8 * wpot            #NDSD potential
-
+    v_t = vtf_tot - vtf_0 + sfactor * 1/8 * wpot            #NDCS potential
 
     return v_t
 
-#The LDA potential
+#The NDCS potential
 def compute_kinetic_lda_potential(rho0_devs, rho1_devs):
     """
     Parameters
@@ -127,7 +167,14 @@
 
     return v_t
 
-#The (old) NDSD switching function
+def compute_kinetic_gea2_potential(rhoa_devs,rhob_devs):
+    rho_tot_devs = rhoa_devs + rhob_devs
+    v_tf = compute_kinetic_lda_potential(rhoa_devs,rhob_devs)
+    vW_tot = compute_kinetic_weizsacker_potential(rho_tot_devs)
+    vW_0 = compute_kinetic_weizsacker_potential(rhoa_devs)
+    v_t = v_tf + 1/72 * (vW_tot - vW_0)
+    return v_t
+
 def ndsd1_switch_factor(rhoB_devs, plambda):
     """Compute the NDSD1 switch functional.
 
@@ -176,11 +223,9 @@
  
     return sfactor
 
-
-#The NDCS kinetic energy symmetry correction
 def ndcs_energy_correc(rhoA_devs, rhoB_devs):
     """
-    The total NDCS Symmetry correction comes from the first order functional derivative of the asymmetry part of the energy functional. 
+    The NDCS Symmetry correction comes from the first order functional derivative of the asymmetry part of the functional
 
     Parameters
     ---------
@@ -208,3 +253,4 @@
     tot = part1 + part2 - part3
 
     return tot, part1, part2, part3
+
