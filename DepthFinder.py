# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:51:46 2023

@author: gjg882
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from landlab import RasterModelGrid


#Need Q - comes from flow accumulator



def DepthFinder(mg, theta_rad, manning_n, Qwerrorfraction, error_tolfraction):
    
    Qwerrorfraction = 10  # just a dummy initial value to start iterating for depth, reset before each iteration
    manning_itnum = 0
    htmepall = []
    
    while Qwerrorfraction > error_tolfraction:
        
        
        Qw = mg.at_node['surface_water__discharge']
        ws = mg.at_node['channel__sed_width']
        S = mg.at_node['topographic__steepest_slope']
        h = mg.at_node['flow__depth']
        
        #calculate estimated discharge, Qwg (Qw guess)
        Qwg = (1 / manning_n) * ((h * (ws + h / math.tan(thetarad))) ** (5 / 3)) * (
                    (ws + 2 * h / math.sin(thetarad)) ** (-2 / 3)) * S ** 0.5  # Qwg for Qw guess, ie calculation
        Qwerrorfraction = np.abs((Qw - Qwg) / Qw)
        
        
        
        
        #Calculate the error
        Qwerrorfraction = np.abs((Qw - Qwg) / Qw)
        manning_itnum += 1
        
        
        
    
    #Needed landlab fields
    #w_rock
    #w_sed
    #w_watersurface
    #sediment depth on bed
    #U_initguess = 2.75 #should this be field or a float?
    