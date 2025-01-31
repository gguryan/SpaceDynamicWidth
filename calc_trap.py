# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:33:14 2024

@author: gjg882
"""

import numpy as np
import math
import cmath
from matplotlib import pyplot as plt


def calc_H_trapezoid(total_sed_vol, dx, wbr, bank_angle):
    
    tan_alpha = math.tan(180 - bank_angle) #TODO check degree vs radians
    
    a = 1
    
    b = wbr * tan_alpha
    
    A = total_sed_vol/dx
    
    c = -A*tan_alpha
    
    # calculate the discriminant
    d = (b**2) - (4*a*c)

    # find two solutions
    sol1 = (-b-cmath.sqrt(d))/(2*a)
    sol2 = (-b+cmath.sqrt(d))/(2*a)
    
    return sol1, sol2
    
    
    
    