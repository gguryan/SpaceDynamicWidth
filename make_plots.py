# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:37:18 2023

@author: gjg882
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

#from illustrate_cell import draw_channel


from landlab import RasterModelGrid
from landlab import load_params
from landlab.plot import imshow_grid
from landlab.components import (FlowAccumulator, 
                                DepressionFinderAndRouter,
                                FastscapeEroder,
                                ChannelProfiler,
                                SteepnessFinder,
                                ChiFinder)


#%%

#ds = xr.open_dataset('10x10_1500kyr_V4.nc')
ds = xr.open_dataset('10x10_1myr_V5_fsc500.nc')

plot_time = 1e6

#load the input parameters that are saved as metadata on the ds
ds_attrs = ds.attrs

#%%

#look at plot of mean elevation over time to determine if topo is at steady state
topo = ds.topographic__elevation


mean_topo = topo.mean(dim=["x", "y"])

plt.figure()
mean_topo.plot()

#%%

br_width = ds.channel_bedrock__width

mean_br_width = br_width.mean(dim=["x", "y"])

plt.figure()
mean_br_width.plot()

#%%

outlet_br_width = br_width.sel(x=100, y=100)

plt.figure()
outlet_br_width.plot()

#%%

xy200_width = br_width.sel(x=200, y=200)

plt.figure()
xy200_width.plot()

#%%

xy200_topo = topo.sel(x=200, y=200)

plt.figure()
xy200_topo.plot()

#%%

xy700_topo = topo.sel(x=700, y=700)

plt.figure()
xy700_topo.plot()


#%%

slope = ds.topographic__steepest_slope

xy200_slope = slope.sel(x=200, y=200)

plt.figure()
xy200_slope.plot()

#%%
 #years


topo_800 = ds.topographic__elevation.sel(time=plot_time)
br_w_800 = ds.channel_bedrock__width.sel(time=plot_time)
sed_w_800 = ds.channel_sed__width.sel(time=plot_time)
Qw_800 = ds.surface_water__discharge.sel(time=plot_time)

Q_47 = Qw_800 ** 0.47 

Q_ratio = br_w_800 / Q_47


#topo_800.plot()

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

topo_800.plot(ax=axs[0, 0], cmap='pink')
sed_w_800.plot(ax=axs[0, 1])
Qw_800.plot(ax=axs[1, 0], cmap='Blues')
Q_ratio.plot(ax=axs[1, 1])

axs[0, 0].set_title('Topography')
axs[0, 1].set_title('Channel Width')
axs[1, 0].set_title('Discharge')
axs[1, 1].set_title('Sed Width / Q^0.47')

plt.tight_layout()

#%%

Q_ratio_a = Q_ratio.values

#%%

def calc_relief(topo_da):
    max_elev = topo_da.max()
    min_elev = topo_da.min()
    
    relief = max_elev - min_elev
    
    return relief.values

#%%

plt.figure()
topo_outlet = ds.topographic__elevation.sel(x=100, y=100)
topo_outlet.plot()

#%%

mean_topo = ds.topographic__elevation.mean(dim='time')

#%%

ds_coords = [100, 100] 
us_coords = [200, 600]
mid_coords = [300, 400]

#%%

topo1d = ds.topographic__elevation.sel(x=us_coords[0], y=us_coords[1])

topo1d_100 = ds.topographic__elevation.sel(x=100, y=100)

topo1d_25 = ds.topographic__elevation.sel(x=mid_coords[0], y=mid_coords[1])

plt.figure()
topo1d.plot(color='red', label='upstream')
topo1d_25.plot(color='magenta', label='mid-channel')
topo1d_100.plot(color='black', label='outlet')
plt.legend(loc='best')

#plt.xlim(0, 1200000)
plt.title('')

#%%

width1d = ds.channel_bedrock__width.sel(x=700, y=700)

width1d_100 = ds.channel_bedrock__width.sel(x=100, y=100)

width1d_25 = ds.channel_bedrock__width.sel(x=300, y=500)

plt.figure()
width1d.plot(color='red', label='upstream')
width1d_25.plot(color='magenta', label='mid-channel')
width1d_100.plot(color='black', label='outlet')
plt.legend(loc='best')

#plt.xlim(0, 1200000)
plt.title('')

