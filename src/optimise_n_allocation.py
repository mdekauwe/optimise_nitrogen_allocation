#!/usr/bin/env python

"""
Optimise the distribution of nitrogen within the leaf such that photosynthesis 
is maximised for a given PAR and CO2 concentration. 

Reference:
=========
* Medlyn (1996) The Optimal Allocation of Nitrogen within the C3 Photsynthetic 
  System at Elevated CO2. Australian Journal of Plant Physiology, 23, 593-603.

"""
__author__ = "Martin De Kauwe & Belinda Medlyn"
__version__ = "1.0 (20.01.2014)"
__email__ = "mdekauwe@gmail.com; belinda.medlyn@mq.edu.au"

import sys
import numpy as np
import scipy.optimize as opt
from math import fabs

class OptimiseNitrogenAllocation(object):
    """
    Optimisation approach is used to find the optimal N distribution given that
    the total photosynthetic nitrogen content (N_p) is specified. There are four
    N pools, but this procedure is only solving for two of these pools as N_p is
    specified and N_s is solved within the photosynthesis model. 
             
    """
    def __init__(self, model=None, tol=1E-14):
        """
        Parameters
        ----------
        model : object
            photosynthesis model function
        tol : float
            tolerance that summed N pools must agree with N_p           
        """
        
        # if we want to minimise a function we need to return f(x), if we
        # want to maximise a function we return -f(x)
        self.sign = -1.0
        self.model = model
        self.tol = tol
    
    def main(self, fit_params=None, par=None, Rd=None, 
                   Ci=None, N_p=None, Tleaf=None, Q10=None, Eaj=None, Eav=None, 
                   deltaSj=None, deltaSv=None, r25=None, Hdv=200000.0, 
                   Hdj=200000.0, Tref=None):
        """ figure out the optimal allocation of nitrogen """
        
        result = opt.fmin_cobyla(self.objective, x0=fit_params, disp=0,
                                 cons=[self.constraint_Np, self.constraint_Nc,
                                       self.constraint_Ne, self.constraint_Nr], 
                                 args=(par, Rd, Ci, N_p, Tleaf, Q10, 
                                       Eaj, Eav, deltaSj, deltaSv, r25, 
                                       Hdv, Hdj, Tref), consargs=(N_p,))
        
        return result
        
   
    def objective(self, x, *args, **kws):
        """ function that we are maximising, this is just a wrapper around the
        call to the photosynthesis model. 
        
        Parameters
        ----------
        x : array
            containing two N pool values (Nc & Ne) that the optimisation 
            routine is varying to maximise the mean An value.
        *args : tuple
            series of args required by the photosynthesis model  
        
        """

        # argument unpacking...
        (par, Rd, Ci, N_p, Tleaf, Q10, 
         Eaj, Eav, deltaSj, deltaSv, 
         r25, Hdv, Hdj, Tref) = args
        
        # call photosynthesis model with new N values
        (An, Ac, Aj, 
         Jmax25, 
         Vcmax25) = self.model.calc_photosynthesis(N_pools=x, par=par, Rd=Rd, 
                                                      Ci=Ci, N_p=N_p, 
                                                      Tleaf=Tleaf, Q10=Q10, 
                                                      Eaj=Eaj, Eav=Eav, 
                                                      deltaSj=deltaSj, 
                                                      deltaSv=deltaSv, r25=r25, 
                                                      Hdv=Hdv, Hdj=Hdj, 
                                                      Tref=Tref)
        # maximising the function
        return self.sign * np.mean(An)
    
    def constraint_Np(self, x, *args):
        """ N_s + N_e + N_r + N_c < N_p 
        
        optimised N pools must be equal to total N (N_p)
        
        Parameters
        ----------
        x : array
            containing two N pool values (Nc & Ne) that the optimisation 
            routine is varying to maximise the mean An value.
        *args : tuple
            series of args required by the photosynthesis model  
        """
        N_p, = args
        
        # unpack calculated values from the photosynthesis model
        (N_c, N_e, N_r, N_s) = self.model.N_pool_store
        
        return self.tol - fabs(N_p - N_e - N_c - N_r - N_s)
    
    def constraint_Nc(self, x, *args):
        """ N_c >= 0.0 
        
        returns a positive number if within bound and 0.0 it is exactly on the 
        edge of the bound
        
        Parameters
        ----------
        x : array
            containing two N pool values (Nc & Ne) that the optimisation 
            routine is varying to maximise the mean An value.
        *args : tuple
            series of args required by the photosynthesis model  
        """
        return x[0]
        
    def constraint_Ne(self, x, *args):
        """ N_e >= 0.0 
        
        returns a positive number if within bound and 0.0 it is exactly on the 
        edge of the bound
        
        Parameters
        ----------
        x : array
            containing two N pool values (Nc & Ne) that the optimisation 
            routine is varying to maximise the mean An value.
        *args : tuple
            series of args required by the photosynthesis model  
        """
        return x[1]
    
    def constraint_Nr(self, x, *args):
        """ N_r >= 0.0 
        
        returns a positive number if within bound and 0.0 it is exactly on the 
        edge of the bound
        
        Parameters
        ----------
        x : array
            containing two N pool values (Nc & Ne) that the optimisation 
            routine is varying to maximise the mean An value.
        *args : tuple
            series of args required by the photosynthesis model  
        """
        (N_c, N_e, N_r, N_s) = self.model.N_pool_store
        return N_r
    

    
   
if __name__ == "__main__":
    
    # Example - testing the impact of increasing N_p
    from evans_photosynthesis_model import PhotosynthesisModel
    import matplotlib.pyplot as plt
    plt.rcParams['legend.fontsize'] = 9
    
    
    
    MMOL_2_MOL = 1000.0
    par = 800.0
    Ci = 385. * 0.7
    Rd = 0.5
    Tleaf = 25.
    deg2kelvin = 273.15
    Tleaf += deg2kelvin
    Eaj = 30000.0
    Eav = 60000.0
    deltaSj = 650.0
    deltaSv = 650.0
    Hdv = 200000.0
    Hdj = 200000.0
    r25 = None
    Q10 = None
    N_c = 0.015
    N_e = 0.015
    #===== Medlyn paper ====#
    alpha = 0.425   # (-)
    g_m = 0.4       # mol m-2 s-1 bar-1 ie mol m-2 s-1 at 1 atm
    K_cat = 24.0    # mmol CO2 mol-1 Rubisco-1
    K_s = 1.25E-04  # mol N s umol-1
    #=======================#
   
    
    P = PhotosynthesisModel(peaked_Jmax=True, peaked_Vcmax=True, 
                                 alpha=alpha, g_m=g_m, K_cat=K_cat, K_s=K_s)
    O = OptimiseNitrogenAllocation(model=P)
    
    
    
    N_c_store = []
    N_e_store = []
    N_r_store = []
    N_s_store = []
    
    An_store = []
    for N_p in np.linspace(0.05, 0.15, 10):
        x0 = np.array([N_c, N_e])
        # ===== OPTIMIZE parameter set ===== # 
        result = O.main(fit_params=x0, par=par, Rd=Rd, Ci=Ci, 
                           N_p=N_p, Tleaf=Tleaf, Tref=25.0, Eaj=Eaj, Eav=Eav, 
                           deltaSj=deltaSj, deltaSv=deltaSv, r25=r25, Q10=Q10, 
                           Hdv=Hdv, Hdj=Hdj)
    
    
        
        (N_c, N_e, N_r, N_s) = P.N_pool_store
        fitted_x0 = np.array([N_c, N_e])
        
        (An, Anc, Anj, 
         Jmax25, 
         Vcmax25) = P.calc_photosynthesis(N_pools=fitted_x0, par=par, Ci=Ci, 
                                               Rd=Rd, N_p=N_p, Tleaf=Tleaf, 
                                               Tref=25.0, Eaj=Eaj, Eav=Eav, 
                                               deltaSj=deltaSj, deltaSv=deltaSv, 
                                               r25=r25, Q10=Q10, Hdv=Hdv, 
                                               Hdj=Hdj)
        An_store.append(np.mean(An))
        N_c_store.append(N_c)
        N_e_store.append(N_e)
        N_r_store.append(N_r)
        N_s_store.append(N_s)
    
    
    N_p = np.linspace(0.05, 0.15, 10)
    #plt.plot(N_p, An_store, lw=2, ls="-", c="red", label="An")
    plt.plot(N_p, N_c_store, lw=2, ls="-", c="red", label="N_c")
    plt.plot(N_p, N_e_store, lw=2, ls="-", c="orange", label="N_e")
    plt.plot(N_p, N_r_store, lw=2, ls="-", c="blue", label="N_r")
    plt.plot(N_p, N_s_store, lw=2, ls="-", c="green", label="N_s")
    
    plt.legend(loc="best", numpoints=1)
    plt.xlabel("N_p")
    plt.ylabel("allocated prop")
    
    plt.show()
 