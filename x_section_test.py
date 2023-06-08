# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:21:43 2023

@author: gjg882
"""

import numpy as np
import math
from matplotlib import pyplot as plt

import xarray as xr

from lague_stress_funcs import Stress_Funcs

#%%

Qin = 100 #TODO units? different number?

theta = 60 #From Lague

manning_n = .05 #from Lague

dx = 100 #m


#%%

#Need to use manning to iteratively calculate flow depth
#Try scipy root_scalar(method='newton')

