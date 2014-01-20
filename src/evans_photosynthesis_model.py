#!/usr/bin/env python

"""
Photosynthesis model based on Evans but with temperature dependancies to be
used within Belinda's optimal N allocation scheme.

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
import matplotlib.pyplot as plt
import warnings

class PhotosynthesisModel(object):
    """
    Photosynthesis model based on Evans (1989)
    
    References:
    -----------
    * Evans, 1989
    + temperature dependancies
    """
    def __init__(self, peaked_Jmax=False, peaked_Vcmax=False, Oi=205.0, 
                 gamstar25=42.75, Kc25=404.9, Ko25=278.4, Ec=79430.0,
                 Eo=36380.0, Egamma=37830.0, theta_J=0.7, alpha=0.425, 
                 quantum_yield=0.3, absorptance=0.8, g_m=0.4, 
                 K_cat=24.0, K_s=1.25E-04, Jmax25 = 100, Vcmax25 = 60):
    
        """
        Parameters
        ----------
        alpha : float
            quantum yield of electron transport (mol mol-1) 
       
        g_m : float
            Conductance to CO2 transfer between intercellular spaces and sites 
            of carboxylation (mol m-2 s-1 bar-1) 
        K_cat : float
            specific activitt of Rubisco (mol CO2 mol-1 Rubisco s-1)
        K_s : float
            constant of proportionality between N_s and Jmax (g N m2 s umol-1)
        Oi : float
            intercellular concentration of O2 [mmol mol-1]
        gamstar25 : float
            co2 compensation point (i.e. the value of Ci at which net CO2 
            uptake is zero due to photorespiration) in the absence of 
            dark resp at 25 degC [umol mol-1] or 298 K
        Kc25 : float
            Michaelis-Menten coefficents for carboxylation by Rubisco at 
            25degC [umol mol-1] or 298 K
        Ko25: float
            Michaelis-Menten coefficents for oxygenation by Rubisco at 
            25degC [umol mol-1]. Note value in Bernacchie 2001 is in mmol!!
            or 298 K
        Ec : float
            Activation energy for carboxylation [J mol-1]
        Eo : float
            Activation energy for oxygenation [J mol-1]
        Egamma : float
            Activation energy at CO2 compensation point [J mol-1]
        RGAS : float
            Universal gas constant [J mol-1 K-1]
        theta_J : float
            Curvature of the light response 
        alpha : float
            Leaf quantum yield (initial slope of the A-light response curve)
            [mol mol-1]
        peaked_Jmax : logical
            Use the peaked Arrhenius function (if true)
        peaked_Vcmax : logical
            Use the peaked Arrhenius function (if true)
        """
        
        self.g_m = g_m
        self.K_cat = K_cat
        self.K_s = K_s
        self.peaked_Jmax = peaked_Jmax
        self.peaked_Vcmax = peaked_Vcmax
        self.deg2kelvin = 273.15
        self.RGAS = 8.314 
        self.Oi = Oi    
        self.gamstar25 = gamstar25
        self.Kc25 = Kc25    
        self.Ko25 = Ko25
        self.Ec = Ec   
        self.Eo = Eo  
        self.Egamma = Egamma
        self.theta_J = theta_J    
        self.alpha = alpha
        self.Jmax25 = Jmax25
        self.Vcmax25 = Vcmax25
        
        
    def calc_photosynthesis(self, N_pools=None, par=None, Rd=None, 
                            Ci=None, N_p=None, Tleaf=None, Q10=None, 
                            Eaj=None, Eav=None, deltaSj=None, deltaSv=None, 
                            r25=None, Hdv=200000.0, Hdj=200000.0, Tref=25.0):
        
        """
        Parameters
        ----------
        N_pools : list of floats
            list containing the 2 free N pools:
            N_c - amount of N in chlorophyll (mol m-2)
            N_e - amount of N in electron transport components (mol m-2)
            
            === calculated internally from the above and N_p
            N_s - amount of N in soluble protein other than Rubisco (mol m-2)
            N_r - amount of N in Rubisco (mol m-2)        
            
        par : float
            incident PAR (umol m-2 s-1)
        Rd : float
            rate of dark respiration (umol m2 s-1)
        Ci : float
            intercellular CO2 concentration [umol mol-1]
        N_p : float
            total amount of N in photosynthesis (mol m-2)
        N_s : float
            amount of N in soluble protein (mol m-2)
        N_r : float
            amount of N in Rubisco (mol m-2)
        Tleaf : float
            leaf temp [deg K]
        Tref : float
            reference temp, default is 25. [deg K]
        Q10 : float
            ratio of respiration at a given temperature divided by respiration 
            at a temperature 10 degrees lower
        Eaj : float
            activation energy for the parameter [kJ mol-1]
        Eav : float
            activation energy for the parameter [kJ mol-1]
        deltaSj : float
            entropy factor [J mol-1 K-1)
        deltaSv : float
            entropy factor [J mol-1 K-1)
        HdV : float
            Deactivation energy for Vcmax [J mol-1]
        Hdj : float
            Deactivation energy for Jmax [J mol-1]
        r25 : float
            Estimate of respiration rate at the reference temperature 25 deg C
             or 298 K [deg K]
            
        Returns:
        --------
        An : float
            Net leaf assimilation rate [umol m-2 s-1] 
          
        """
        
        # Michaelis-Menten constant for O2/CO2, Arrhenius temp dependancy
        Kc = self.arrh(self.Kc25, self.Ec, Tleaf)
        Ko = self.arrh(self.Ko25, self.Eo, Tleaf)
        Km = Kc * (1.0 + self.Oi / Ko)
        #Km = 544.0
        
        
        # Effect of temp on CO2 compensation point 
        gamma_star = self.arrh(self.gamstar25, self.Egamma, Tleaf)
        #gamma_star = 38.6
        
        
        # Calculations at 25 degrees C or the measurement temperature
        if r25 is not None:
            Rd = self.resp(Tleaf, Q10, r25, Tref)
               
        # unpack...we need to pass as a list for the optimisation step
        (N_c, N_e) = N_pools
        
        # Leaf absorptance depends on the chlorophyll protein complexes (N_c)
        absorptance = self.calc_absorptance(N_c)

        # Max rate of electron transport (umol m-2 s-1) 
        Jmax25 = self.calc_jmax(N_e, N_c)
        
        # Calculate Ns & Nr
        N_s = self.calc_n_alloc_soluble_protein(Jmax25).mean() 
        N_r = N_p - N_c - N_e - N_s        
        
        # Max rate of carboxylation velocity (umol m-2 s-1) 
        Vcmax25 = self.calc_vcmax(N_r)
        

        # Effect of temperature on Vcmax and Jamx
        if self.peaked_Vcmax:
            Vcmax = self.peaked_arrh(Vcmax25, Eav, Tleaf, deltaSv, Hdv)
        else:
            Vcmax = self.arrh(Vcmax25, Eav, Tleaf)
        
        #print "Vcmax T", Vcmax25, Eav, Tleaf, deltaSv, Hdv, Vcmax        
        if self.peaked_Jmax:
            Jmax = self.peaked_arrh(Jmax25, Eaj, Tleaf, deltaSj, Hdj)
        else:
            Jmax = self.arrh(Jmax25, Eaj, Tleaf)
        #print "Jmax T", Jmax25, Eaj, Tleaf, deltaSj, Hdj, Jmax

        self.N_pool_store = np.array([N_c, N_e, N_r, N_s])

        # rate of electron transport, a saturating function of absorbed PAR
        J = self.quadratic(a=self.theta_J, 
                           b=-(self.alpha * absorptance * par + Jmax), 
                           c=self.alpha * absorptance * par * Jmax)
       
        # Photosynthesis when Rubisco is limiting
        a = 1. / self.g_m
        b = (Rd - Vcmax) / self.g_m - Ci - Km
        c = Vcmax * (Ci - gamma_star) - Rd * (Ci + Km)
        Acn = self.quadratic(a=a, b=b, c=c)
        #print "Vcmax", Vcmax, Rd, Ci, gamma_star, Km, a, b, c, Acn
        
        # Photosynthesis when electron transport is limiting
        VJ = (J / 4.0)
        a = 1.0 / self.g_m
        b = (Rd - VJ) / self.g_m - Ci - (2.0 * gamma_star)
        c = VJ * (Ci - gamma_star) - Rd * (Ci + 2.0 * gamma_star)
        Ajn = self.quadratic(a=a, b=b, c=c)
        #print "Jmax", VJ, Rd, Ci, gamma_star, a, b, c, Ajn
         
        # By default we assume a everything under Ci<150 is Ac limited
        An = np.where(Ci < 150.0, Acn, np.minimum(Acn, Ajn))
        
        # net assimilation rates.
        #An = A - Rd
        #Acn = Ac - Rd
        #Ajn = Aj - Rd
        # print A, Rd, An, Jmax25, Vcmax25
        return An, Acn, Ajn, Jmax25, Vcmax25
    
    def quadratic(self, a=None, b=None, c=None):
        """ minimilist quadratic solution as root for J solution should always
        be positive, so I have excluded other quadratic solution steps. I am 
        only returning the smallest of the two roots 
        
        Parameters:
        ----------
        a : float
            co-efficient 
        b : float
            co-efficient
        c : float
            co-efficient
        
        Returns:
        -------
        val : float
            positive root
        """
        
        d = b**2 - 4.0 * a * c # discriminant
        d = np.where(np.logical_or(d<=0, np.any(np.isnan(d))), -999.9, d)
        root1 = np.where(d>0.0, (-b - np.sqrt(d)) / (2.0 * a), d)
        #root2 = np.where(d>0.0, (-b + np.sqrt(d)) / (2.0 * a), d)
        
        return root1
        
    
    def calc_jmax(self, N_e, N_c, a_j=15870.0, b_j=2775.0):
        """ Evans (1989( found a linear reln between the rate of electron
        transport and the total amount of nitrogen in the thylakoids per unit
        chlorophyll 
        
        Parameters:
        ----------
        N_e : float
            amount of N in electron transport components (mol m-2)
        N_c : float
            amount of N in chlorophyll (mol m-2)
        a_j : float
            umol (mol N)-1 s-1
        b_j : float
            umol (mol N)-1 s-1
            
        Returns:
        --------
        Jmax : float
            Max rate of electron transport (umol m-2 s-1)
        """
        Jmax = (a_j * N_e) + (b_j * N_c)
        
        return Jmax
    
    def calc_vcmax(self, N_r):
        """ 
        Calculate Rubisco activity 
        
        Parameters:
        ----------
        N_r : float
            amount of N in Rubisco (mol m-2)
        
        Returns:
        --------
        Vcmax : float
            Max rate of carboxylation velocity (umol m-2 s-1) 
        """
        conv = 7000.0 / 44.0 # mol N to mol Rubiso
        
        return self.K_cat * conv * N_r
        
    
    def calc_absorptance(self, N_c):
        """ Calculate absorptance in the leaf based on N in the chlorophyll
        protein complexes
        
        Parameters:
        ----------
        N_c : float
            amount of N in chlorophyll (mol m-2)
            
        Returns:
        --------
        absorptance : float
            Leaf absorptance (-)
        """
        return (25.0 * N_c) / (25.0 * N_c + 0.076) # units cancel
    
    def calc_n_alloc_soluble_protein(self, Jmax):
        """ Calculate N allocated to soluble protein
        
        Parameters:
        --------
        Jmax : float
            Max rate of electron transport (umol m-2 s-1)
        
        Returns:
        --------
        N_s : float
            amount of N in soluble protein other than Rubisco (mol m-2)
        """
        return self.K_s * Jmax 
    
    def arrh(self, k25, Ea, Tk):
        """ Temperature dependence of kinetic parameters is described by an
        Arrhenius function.    

        Parameters:
        ----------
        k25 : float
            rate parameter value at 25 degC or 298 K
        Ea : float
            activation energy for the parameter [kJ mol-1]
        Tk : float
            leaf temperature [deg K]

        Returns:
        -------
        kt : float
            temperature dependence on parameter 
        
        References:
        -----------
        * Medlyn et al. 2002, PCE, 25, 1167-1179.   
        """
        return k25 * np.exp((Ea * (Tk - 298.15)) / (298.15 * self.RGAS * Tk)) 
    
    def peaked_arrh(self, k25, Ea, Tk, deltaS, Hd):
        """ Temperature dependancy approximated by peaked Arrhenius eqn, 
        accounting for the rate of inhibition at higher temperatures. 

        Parameters:
        ----------
        k25 : float
            rate parameter value at 25 degC or 298 K
        Ea : float
            activation energy for the parameter [kJ mol-1]
        Tk : float
            leaf temperature [deg K]
        deltaS : float
            entropy factor [J mol-1 K-1)
        Hd : float
            describes rate of decrease about the optimum temp [KJ mol-1]
        
        Returns:
        -------
        kt : float
            temperature dependence on parameter 
        
        References:
        -----------
        * Medlyn et al. 2002, PCE, 25, 1167-1179. 
        
        """
        arg1 = self.arrh(k25, Ea, Tk)
        arg2 = 1.0 + np.exp((298.15 * deltaS - Hd) / (298.15 * self.RGAS))
        arg3 = 1.0 + np.exp((Tk * deltaS - Hd) / (Tk * self.RGAS))
                
        return arg1 * arg2 / arg3
    
    
    def resp(self, Tleaf, Q10, r25, Tref=None):
        """ Calculate leaf respiration accounting for temperature dependence.
        
        Parameters:
        ----------
        r25 : float
            Estimate of respiration rate at the reference temperature 25 deg C
            or or 298 K
        Tref : float
            reference temperature
        Q10 : float
            ratio of respiration at a given temperature divided by respiration 
            at a temperature 10 degrees lower
        
        Returns:
        -------
        Rt : float
            leaf respiration
        
        References:
        -----------
        Tjoelker et al (2001) GCB, 7, 223-230.
        """
        return r25 * Q10**(((Tleaf - self.deg2kelvin) - Tref) / 10.0)
    


if __name__ == "__main__":
    
    Ci = np.arange(10.0, 1500.0, 20.0)
    par = 800.0
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

    #===== Medlyn paper ====#
    alpha = 0.425   # (-)
    g_m = 0.4       # mol m-2 s-1 bar-1 ie mol m-2 s-1 at 1 atm
    K_cat = 24.0    # mmol CO2 mol-1 Rubisco-1
    K_s = 1.25E-04  # mol N s umol-1
    #=======================#
    
    P = PhotosynthesisModel(peaked_Jmax=True, peaked_Vcmax=True, alpha=alpha,
                            g_m=g_m, K_cat=K_cat, K_s=K_s)
    
    # Testing, just equally divide up the total N available
    N_p = 0.09 # mol m-2 
    N_c = 0.015
    N_e = 0.015
    N_pools = np.array([N_c, N_e])
   
    
    (An, Anc, Anj, Jmax25, Vcmax25) = P.calc_photosynthesis(N_pools=N_pools, par=par, Rd=Rd, 
                                          Ci=Ci, N_p=N_p, Tleaf=Tleaf, Q10=Q10, 
                                          Eaj=Eaj, Eav=Eav, deltaSj=deltaSj, 
                                          deltaSv=deltaSv, r25=r25, Hdv=Hdv, 
                                          Hdj=Hdj, Tref=25.0)
    
    plt.title("PAR = 800 $\mu$mol $m^{-2}$ $s^{-1}$")
    plt.plot(Ci, Anc, "b-", label="$A_c$", lw=5)
    plt.plot(Ci, Anj, "g-", label="$A_j$", lw=5)
    plt.plot(Ci, An, "ro", label="$A_n$", lw=1)
    plt.ylabel("Assimilation ($\mu$mol $m^{-2}$ $s^{-1}$)")
    plt.xlabel("Ci ($\mu$mol mol$^{-1}$)")
    plt.legend(numpoints=1, loc="best")
    plt.show()
    
        