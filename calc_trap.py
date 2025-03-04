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
    

def calc_H_trapezoid(mg, thetarad):
    

    dx = mg.dx
    total_sed_vol = mg.at_node['sediment__volume']
    wr = mg.at_node['channel_bedrock__width']
    #Cross-sectional area of sediment
    A = total_sed_vol / dx
    
    #Quadratic equation coefficients for symmetrical trapezoid
    #Derived here https://www.desmos.com/calculator/01u2jhx03h
      
    a = np.ones(mg.number_of_nodes) #coefficient a in quadratic equation
    b = wr * math.tan(thetarad)
    c = -A * math.tan(thetarad)
    
    
    
    # Calculate discriminant
    discriminant = b**2 - 4*a*c
    
    
    # Initialize output arrays with NaN values
    root1 = np.full_like(discriminant, np.nan, dtype=float)
    root2 = np.full_like(discriminant, np.nan, dtype=float)
    
    # Only calculate roots for positive discriminants
    valid = (discriminant > 0)
    
    # Handle positive discriminants - real roots
    root1[valid] = (-b[valid] + np.sqrt(discriminant[valid])) / (2*a[valid])
    root2[valid] = (-b[valid] - np.sqrt(discriminant[valid])) / (2*a[valid])
    
    return root1, root2
    
        


def quadratic_formula_array(a, b, c):
    """
    Solve quadratic equations of the form ax^2 + bx + c = 0 for multiple inputs.
    
    Parameters:
    a, b, c: numpy arrays or scalars - coefficients of the quadratic equation
    
    Returns:
    tuple: (root1, root2) where each is an array of solutions
           np.nan for zero or negative discriminants (non-physical solutions)
    """
    # Convert inputs to numpy arrays if they aren't already
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    
    # Calculate discriminant
    discriminant = b**2 - 4*a*c
    
    # Initialize output arrays with NaN values
    root1 = np.full_like(discriminant, np.nan, dtype=float)
    root2 = np.full_like(discriminant, np.nan, dtype=float)
    
    # Only calculate roots for positive discriminants
    valid = (discriminant > 0)
    
    # Handle positive discriminants - real roots
    root1[valid] = (-b[valid] + np.sqrt(discriminant[valid])) / (2*a[valid])
    root2[valid] = (-b[valid] - np.sqrt(discriminant[valid])) / (2*a[valid])
    
    return root1, root2