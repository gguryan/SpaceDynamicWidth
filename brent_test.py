# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:27:43 2023

@author: gjg882
"""

from scipy.optimize import brentq

# Define the function to find the root of
def yfun(y, Q_target, b, m, S, C, n):
    return yfun(y, b, m, S, C, n) - Q_target

# Provided input variables
Q_target = 225.0
n = 0.016
m = 2
b = 10.0
S = 0.0006
C = 1.486

# Initial guess for the root
y_guess = 1.0

# Use brentq to find the root
result = brentq(yfun, 0, 10, args=(Q_target, b, m, S, C, n))

print(f'The calculated y for Q = {Q_target} is approximately: {result}')
