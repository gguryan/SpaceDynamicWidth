# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:53:09 2024

@author: gjg882
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import xarray as xr

from matplotlib.animation import PillowWriter

import plotly.graph_objects as go

#from illustrate_cell import draw_channel

from calc_prf_width import calc_prf_width

from matplotlib.ticker import ScalarFormatter

from landlab.io.native_landlab import load_grid, save_grid

from matplotlib.animation import FFMpegWriter
#%%

ds1 = xr.open_dataset('NormalFaultSims\\Faulted_no_sed.nc')

ds2 = xr.open_dataset('NormalFaultSims\\Faulted_w_sed.nc')


#%%

plot_time = 1100000

nf_x =[ 0, 3500]
nf_y = [2000,100]



fig, axs = plt.subplots(2, 1, figsize=(6,11), dpi=300)

ds1.topographic__steepest_slope.sel(time=plot_time).plot(cmap='jet', ax = axs[0], vmax=3)
axs[0].set_title('Slope')
axs[0].plot(nf_x, nf_y, color='red')

ds1.channel_bedrock__width.sel(time=plot_time).plot(cmap='pink', ax = axs[1], vmax=30)
axs[1].set_title('Channel Width')
axs[1].plot(nf_x, nf_y, color='red')

#%%


nf_x =[ 0, 3500]
nf_y = [2000,100]

fig1, axs = plt.subplots(2, 1, figsize=(6,11), dpi=300)

ds1.topographic__steepest_slope.sel(time=plot_time).plot(cmap='jet', ax = axs[0], vmax=3)
axs[0].set_title('Slope')
axs[0].plot(nf_x, nf_y, color='red')

ds1.channel_bedrock__width.sel(time=plot_time).plot(cmap='pink', ax = axs[1], vmax=30)
axs[1].set_title('Channel Width')
axs[1].plot(nf_x, nf_y, color='red')

#%%
max_t = ds1.channel_bedrock__width.time.max().values
step_size = 10000

plot_times = np.arange(0, max_t+1000, step_size)


#%%


fig, axs = plt.subplots(2, 1, figsize=(6,11))


writer = PillowWriter(fps=10)

with writer.saving(fig, 'width_slope_mov.gif', 100):

    for plot_time in plot_times:
        
        ds1.topographic__steepest_slope.sel(time=plot_time).plot(cmap='jet', ax = axs[0], vmax=3, add_colorbar=False)
        axs[0].set_title('Slope')
        axs[0].plot(nf_x, nf_y, color='red')
    
        ds1.channel_bedrock__width.sel(time=plot_time).plot(cmap='pink', ax = axs[1], vmax=30, add_colorbar = False)
        axs[1].set_title('Channel Width')
        axs[1].plot(nf_x, nf_y, color='red')
        
        writer.grab_frame()
        
#%%

