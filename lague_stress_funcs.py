# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:00:16 2023

@author: gjg882
"""

import numpy as np
import math
from matplotlib import pyplot as plt

import xarray as xr

#%%

class Stress_Funcs:
    
    def __init__(self, W, D, S, theta, rho):
        self.W = W
        self.D = D
        self.theta = theta
        self.S = S
        self.g = 9.81 #acceleration of gravity, m/s 
        self.rho = rho
        

    def calc_Fw(self):
        Fw = 1.78*((self.W/self.D)*math.sin(self.theta) - (2*math.cos(self.theta)) + 1.5)**-1.4
        return Fw
    
    def calc_Tbank(self, Fw):
        
        Tbank = self.rho * self.g * self.D * self.S * (Fw/2) * ((self.W/self.D)*math.sin(self.theta) - math.cos(self.theta))
        
        return Tbank
    
    def calc_Tbed(self, Fw):
        
        coeff = ((self.rho*self.g)/2 * self.D * self.S * (1-Fw))
        
        Tbed = coeff * (1 + (self.W*math.tan(self.theta))/ (self.W * math.tan(self.theta) - 2*self.D))
        
        return Tbed
    
    def calc_psi_bed (self, Fw):
       psi_bed = 0.5 * self.rho * self.g * (1-Fw) * ((self.W*math.tan(self.theta) / (self.W*math.tan(self.theta) - 2*self.D)))
       return psi_bed

    def calc_psi_bank(self, Fw):
        psi_bank = self.rho * self.g * (Fw/2) * ((self.W/self.D) * math.sin(self.theta) - math.cos(self.theta))
        return psi_bank
                                            
        

#%%
    
    
    