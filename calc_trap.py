# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:33:14 2024

@author: gjg882
"""

import numpy as np
import math
import cmath
from matplotlib import pyplot as plt




def quadratic_formula(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return "Complex Roots"
    root1 = (-b + math.sqrt(discriminant)) / (2*a)
    root2 = (-b - math.sqrt(discriminant)) / (2*a)
    return root1, root2
 
    

def calc_H_quadratic(total_sed_vol, dx, wbr, theta_rad):
    
    #Cross-sectional area of sediment
    A = total_sed_vol / dx
    
    #Quadratic equation coefficients for symmetrical trapezoid
    #Derived here https://www.desmos.com/calculator/01u2jhx03h
    a = 1
    b = wbr * math.tan(theta_rad)
    c = -A * math.tan(theta_rad)
    
    roots = quadratic_formula(a, b, c)
    
    return roots
    
