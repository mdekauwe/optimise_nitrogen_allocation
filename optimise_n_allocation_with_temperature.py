#!/usr/bin/env python

"""
Figure out how J/V ratio changes with temperature
"""
__author__ = "Belinda Medlyn"
__version__ = "1.0 (20.01.2014)"
__email__ = "belinda.medlyn@mq.edu.au"

import matplotlib.pyplot as plt
import numpy as np

import os
import sys

sys.path.append("src")

from optimise_n_allocation import OptimiseNitrogenAllocation
from evans_photosynthesis_model import PhotosynthesisModel


def main(Ca=None):
    
    par = 800.0
    Ci = Ca * 0.7
    Rd = None
    Tleaf = 25.
    deg2kelvin = 273.15
    Tleaf += deg2kelvin
    Eaj = 27900.0
    Eav = 69900.0
    deltaSj = 619.0
    deltaSv = 634.0
    Hdv = 200000.0
    Hdj = 200000.0
    r25 = 2.0
    Q10 = 2.0
    N_c = 0.015
    N_e = 0.015
    N_p = 0.10
    #===== Medlyn paper ====#
    alpha = 0.425   # (-)
    g_m = 0.4       # mol m-2 s-1 bar-1 ie mol m-2 s-1 at 1 atm
    K_cat = 24.0    # mmol CO2 mol-1 Rubisco-1
    K_s = 1.25E-04  # mol N s umol-1
    #=======================#
   
    
    P = PhotosynthesisModel(peaked_Jmax=True, peaked_Vcmax=True, 
                                 alpha=alpha, g_m=g_m, K_cat=K_cat, K_s=K_s)
    O = OptimiseNitrogenAllocation(model=P)
    
    
    
    Jm_store = np.zeros(0)
    Vm_store = np.zeros(0)
    leaf_temp = np.linspace(5, 30, 10)
    Npools_store = np.zeros((len(leaf_temp),4))
    for i, Tleaf in enumerate(leaf_temp):
        
        initial_guess = np.array([N_c, N_e])
        # ===== OPTIMIZE parameter set ===== # 
        result = O.main(fit_params=initial_guess, par=par, Rd=Rd, Ci=Ci, 
                        N_p=N_p, Tleaf=Tleaf+deg2kelvin, Tref=25.0, Eaj=Eaj, 
                        Eav=Eav, deltaSj=deltaSj, deltaSv=deltaSv, r25=r25, 
                        Q10=Q10, Hdv=Hdv, Hdj=Hdj)
    
    
        # retrieve all fitted N pools
        (N_c, N_e, N_r, N_s) = P.N_pool_store
        Npools_store[i] = N_c, N_e, N_r, N_s
        
        fitted_x0 = np.array([N_c, N_e])
        
        (An, Anc, Anj, 
         Jmax25, 
         Vcmax25) = P.calc_photosynthesis(N_pools=fitted_x0, par=par, Ci=Ci, 
                                               Rd=Rd, N_p=N_p, 
                                               Tleaf=Tleaf+deg2kelvin, 
                                               Tref=25.0, Eaj=Eaj, Eav=Eav, 
                                               deltaSj=deltaSj, deltaSv=deltaSv, 
                                               r25=r25, Q10=Q10, Hdv=Hdv, 
                                               Hdj=Hdj)
        
        Jm_store = np.append(Jm_store, Jmax25)
        Vm_store = np.append(Vm_store, Vcmax25)
    jv_ratio = Jm_store / Vm_store

    return jv_ratio, leaf_temp, Npools_store
    
def make_plot(leaf_temp, jv_ratio_amb, jv_ratio_ele):    
    # Make plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    simpleaxis(ax)
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    ax.plot(leaf_temp, jv_ratio_amb, lw=2, ls="-", c="blue", label="Amb")
    ax.plot(leaf_temp, jv_ratio_ele, lw=2, ls="-", c="red", label="Ele")
    ax.set_ylabel("Optimal Jm25:Vm25 ratio")
    ax.set_xlabel("Tleaf")
    ax.legend(numpoints=1, loc="best")
    fig.savefig(os.path.join("plots", "opt_JV_ratio.eps"), bbox_inches='tight')
    
    plt.show()

def make_plot2(leaf_temp, Npools_store):    
    # Make plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    simpleaxis(ax)
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    ax.plot(leaf_temp, Npools_store_amb[:,0], ls=" ", marker="o", c="royalblue", 
            label="N$_c$")
    ax.plot(leaf_temp, Npools_store_amb[:,1], ls=" ", marker="o", c="red", 
            label="N$_e$")
    ax.plot(leaf_temp, Npools_store_amb[:,2], ls=" ", marker="o", c="green", 
            label="N$_r$")
    ax.plot(leaf_temp, Npools_store_amb[:,3], ls=" ", marker="o", c="yellow", 
            label="N$_s$")
    ax.set_ylabel("Allocation proportion")
    ax.set_xlabel("Tleaf")
    ax.legend(numpoints=1, loc="best")
    ax.set_xlim(0, 35)
    fig.savefig(os.path.join("plots", "alloc_frac_vs_tleaf.eps"), bbox_inches='tight')
    fig.savefig(os.path.join("plots", "alloc_frac_vs_tleaf.png"), dpi=150, 
                bbox_inches='tight')
    plt.show()

def simpleaxis(ax):
    """ Remove the top line and right line on the plot face """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
if __name__ == "__main__":
    
    
    (jv_ratio_amb, leaf_temp, Npools_store_amb) = main(Ca=390.0)
    (jv_ratio_ele, leaf_temp, Npools_store_ele) = main(Ca=600.0)
    make_plot(leaf_temp, jv_ratio_amb, jv_ratio_ele)
    
    make_plot2(leaf_temp, Npools_store_amb)
    