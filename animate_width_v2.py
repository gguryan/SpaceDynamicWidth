# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:40:32 2024

@author: gjg882
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

#%%

ds1 = xr.open_dataset('C:\\Users\\gjg882\\Desktop\\Projects\\SpaceDynamicWidth\\NormalFaultSims\\Faulted_no_sed.nc')

#%%
ds2 = xr.open_dataset('C:\\Users\\gjg882\\Desktop\\Projects\\SpaceDynamicWidth\\NormalFaultSims\\Faulted_w_sed.nc')

#%%

ds1.topographic__steepest_slope.sel(time=800000).plot()


#%%

fig, ax = plt.subplots(2, 1, figsize=(6,12))

