# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:37:18 2023

@author: gjg882
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import xarray as xr


import plotly.graph_objects as go

#from illustrate_cell import draw_channel

from calc_prf_width import calc_prf_width

from matplotlib.ticker import ScalarFormatter

from landlab.io.native_landlab import load_grid, save_grid

from landlab import RasterModelGrid
from landlab import load_params
from landlab.plot import imshow_grid, drainage_plot
from landlab.components import (FlowAccumulator, 
                                DepressionFinderAndRouter,
                                FastscapeEroder,
                                ChannelProfiler,
                                SteepnessFinder,
                                NormalFault)


#%%

ds1 = xr.open_dataset('Faulted_no_sed_2.nc')
ds2 = xr.open_dataset('Faulted_w_sed_2.nc')


#%%

mg_1 = load_grid('NormalFaultSims/nx50_normal_fault_FromSS_NoSed_MG.nc.grid')

#%%

mg_2 = load_grid('NormalFaultSims/nx50_normal_fault_FromSS_MG.nc.grid')
#%%

nf1 = NormalFault(mg_1, fault_trace = {'x1': 0,
                                      'y1': 2000, 
                                      'y2': 100, 
                                      'x2': 3500}, include_boundaries=True)


#%%
imshow_grid(mg_1, nf1.faulted_nodes.astype(int), cmap="viridis")

#%%

ds_attrs = ds1.attrs

#%%

topo1 = ds1.topographic__elevation

mean_topo1 = topo1.mean(dim=["x", "y"])

plt.figure()
plt.title('Mean Topographic Elevation')
mean_topo1.plot()


#%%

sec_per_yr =  60 * 60 * 24 * 365

plot_time = 300000+800000

dx = ds_attrs['dx']

sf_min_DA = dx**2

mg1, df1 = calc_prf_width(ds1, plot_time, sf_min_DA)

mg2, df2 = calc_prf_width(ds2, plot_time, sf_min_DA)

#%%

time_arr = ds1.time.to_numpy()

#%%
#look at plot of mean elevation over time to determine if topo is at steady state
topo1 = ds1.topographic__elevation

mean_topo1 = topo1.mean(dim=["x", "y"])

plt.figure()
plt.title('Mean Topographic Elevation')
mean_topo1.plot()

#%%

br_width1 = ds1.channel_bedrock__width
br_width2 = ds1.channel_bedrock__width

mean_br_width1 = br_width1.mean(dim=["x", "y"])

plt.figure()
mean_br_width1.plot()
plt.title('Mean Channel Width')



#%%

prf1 = ChannelProfiler(mg_1, minimum_channel_threshold=0, main_channel_only=True)

prf1.run_one_step()


#%%

prf_nodes = prf1.nodes[0]

prf_x = []
prf_y = []

for n in prf_nodes:
    
    x_val = mg_1.x_of_node[n]
    prf_x.append(x_val)
    
    
    y_val = mg_1.y_of_node[n]
    prf_y.append(y_val)



#plt.scatter(ds_coords[0], ds_coords[1], c='goldenrod')
#plt.scatter(mid_coords[0]+100, 2400, c='teal')
#plt.scatter(us_coords[0]-300,3700, c='indigo')



#ds_coords = [100, 100] 
#us_coords = [2400, 4700]
#mid_coords = [2200, 3700]


#ds_coords = [100, 100] 

#mid_coords = [2600, 2400]

#us_coords = [4300, 3700]


ds_coords = [100, 100] 

mid_coords = [2600-800, 2400-800]

us_coords = [4300, 3700]

#ds_coords = [100, 100] 
#us_coords = [900, 900]
#mid_coords = [500, 500]


#%%Plot topography through time at selected nodes

topo1d_1 = ds1.topographic__elevation.sel(x=us_coords[0], y=us_coords[1])

topo1d_outlet_1 = ds1.topographic__elevation.sel(x=dx, y=dx)

topo1d_25_1 = ds1.topographic__elevation.sel(x=mid_coords[0], y=mid_coords[1])

topo1d_2 = ds2.topographic__elevation.sel(x=us_coords[0], y=us_coords[1])

topo1d_outlet_2 = ds2.topographic__elevation.sel(x=dx, y=dx)

topo1d_25_2 = ds2.topographic__elevation.sel(x=mid_coords[0], y=mid_coords[1])

#%%

plt.figure(figsize=(10, 4), dpi=300)

xfmt = ScalarFormatter()
xfmt.set_powerlimits((0,3))

topo1d_1.plot(color='indigo', label='upstream')
topo1d_25_1.plot(color='teal', label='mid-channel')
topo1d_outlet_1.plot(color='goldenrod', label='outlet')

topo1d_2.plot(color='indigo', linestyle='--')
topo1d_25_2.plot(color='teal', linestyle='--')
topo1d_outlet_2.plot(color='goldenrod', linestyle='--')



plt.legend(loc='best')


plt.gca().xaxis.set_major_formatter(xfmt)

plt.ylabel('Elevation (m)')
plt.xlabel('Time (kyr)')
plt.xlim(0, plot_time)
plt.title('Topography through Time')

plt.rcParams['font.size'] = 12

xfmt = ScalarFormatter()
xfmt.set_powerlimits((0,3))
#xfmt.set_useOffset(100000)

plt.axvline(x=800000, color='darkgrey', linestyle=':')



#plt.savefig('HighUplift_topo.tiff')


#%%Plot width through time at selected nodes




width1d_1 = ds1.channel_bedrock__width.sel(x=us_coords[0], y=us_coords[1])

width1d_100_1 = ds1.channel_bedrock__width.sel(x=ds_coords[0], y=ds_coords[1])

width1d_25_1 = ds1.channel_bedrock__width.sel(x=mid_coords[0], y=mid_coords[1])


width1d_2 = ds2.channel_bedrock__width.sel(x=us_coords[0], y=us_coords[1])

width1d_100_2 = ds2.channel_bedrock__width.sel(x=ds_coords[0], y=ds_coords[1])

width1d_25_2 = ds2.channel_bedrock__width.sel(x=mid_coords[0], y=mid_coords[1])


plt.figure(figsize=(10, 4), dpi=300)



xfmt = ScalarFormatter()
xfmt.set_powerlimits((0,3))
#xfmt.set_useOffset(100000)

plt.axvline(x=800000, color='darkgrey', linestyle=':')


width1d_1.plot(color='indigo', label='upstream')
width1d_25_1.plot(color='teal', label='mid-channel')
width1d_100_1.plot(color='goldenrod', label='outlet')

width1d_2.plot(color='indigo', linestyle='--')
width1d_25_2.plot(color='teal', linestyle='--')
width1d_100_2.plot(color='goldenrod', linestyle='--')




lh_out = mlines.Line2D([], [], color='goldenrod', label='outlet')
lh_mid = mlines.Line2D([], [], color='teal', label='mid-channel')
lh_us = mlines.Line2D([], [], color='indigo', label='upstream')


sed_line = mlines.Line2D([], [], color='k', linestyle='--', label='w/ sediment cover')

ax = plt.gca()

plt.gca().xaxis.set_major_formatter(xfmt)



#ax.legend(handles=[lh_out, lh_mid, lh_us, sed_line])


plt.ylabel('Width (m)')
plt.xlabel('Time (kyr)')
plt.xlim(0, plot_time)
plt.title('Channel Width')

plt.rcParams['font.size'] = 12

#plt.savefig('HighUplift_width.tiff')

#%%

slope_us_1 = ds1.topographic__steepest_slope.sel(x=us_coords[0], y=us_coords[1])

slope_ds_1 = ds1.topographic__steepest_slope.sel(x=ds_coords[0], y=ds_coords[1])

slope_out_1 = ds1.topographic__steepest_slope.sel(x=mid_coords[0], y=mid_coords[1])


slope_us_2 = ds2.topographic__steepest_slope.sel(x=us_coords[0], y=us_coords[1])

slope_ds_2 = ds2.topographic__steepest_slope.sel(x=ds_coords[0], y=ds_coords[1])

slope_out_2 = ds2.topographic__steepest_slope.sel(x=mid_coords[0], y=mid_coords[1])


plt.figure(figsize=(10, 4), dpi=300)



xfmt = ScalarFormatter()
xfmt.set_powerlimits((0,3))
#xfmt.set_useOffset(100000)

plt.axvline(x=800000, color='darkgrey', linestyle=':')


slope_us_1.plot(color='indigo', label='upstream')
slope_ds_1.plot(color='teal', label='mid-channel')
slope_out_1.plot(color='goldenrod', label='outlet')

slope_us_2.plot(color='indigo', linestyle='--')
slope_ds_2.plot(color='teal', linestyle='--')
slope_out_2.plot(color='goldenrod', linestyle='--')




lh_out = mlines.Line2D([], [], color='goldenrod', label='outlet')
lh_mid = mlines.Line2D([], [], color='teal', label='mid-channel')
lh_us = mlines.Line2D([], [], color='indigo', label='upstream')


sed_line = mlines.Line2D([], [], color='k', linestyle='--', label='w/ sediment cover')

ax = plt.gca()

plt.gca().xaxis.set_major_formatter(xfmt)



#ax.legend(handles=[lh_out, lh_mid, lh_us, sed_line])


plt.ylabel('Slope')
plt.xlabel('Time (kyr)')
plt.xlim(0, plot_time)
plt.ylim(0,.2)
#plt.title('Slope Through Time')
plt.title('Channel Slope')

plt.rcParams['font.size'] = 12

#
#plt.savefig('HighUplift_width.tiff')



#%%
topo_800_1 = ds1.topographic__elevation.sel(time=plot_time)
br_w_800_1 = ds1.channel_bedrock__width.sel(time=plot_time)
sed_w_800_1 = ds1.channel_sed__width.sel(time=plot_time)
Qw_800_1  = ds1.surface_water__discharge.sel(time=plot_time) 
flow_depth_1 =  ds1.flow__depth.sel(time=plot_time)



#%%

mg_1.at_node['channel_bedrock__width'][0] = 0
imshow_grid(mg_1, 'channel_bedrock__width')

plt.title('Channel Width without Sediment Cover Effects')
#%%

fig1 = plt.figure()



mg_2.at_node['channel_bedrock__width'][0] = 0
imshow_grid(mg_2, 'channel_bedrock__width', colorbar_label='Channel Width[m]')
#plt.scatter(ds_coords[0], ds_coords[1], c='goldenrod')
#plt.scatter(mid_coords[0], mid_coords[1], c='teal')
#plt.scatter(us_coords[0], us_coords[1], c='indigo')




plt.scatter(ds_coords[0], ds_coords[1], c='goldenrod', marker='s', s=12)
plt.scatter(mid_coords[0], mid_coords[1], c='teal', marker='s', s=12)
plt.scatter(us_coords[0], us_coords[1], c='indigo', marker='s', s=12)



plt.title('Model with Sediment Cover')

#fig1.add_trace(go.Scatter(x=[ds_coords[0], mid_coords[0], us_coords[0]], 
                        # y=[ds_coords[1], mid_coords[1], us_coords[1]]))

#%%

mid_ind = int(len(prf_x)/2)


mid_x = prf_x[mid_ind]
mid_y = prf_y[mid_ind]

#%%

#mg_1.keys('node')
nf_x = [0, 3500]
nf_y= [2000, 100]

fig, axs = plt.subplots(1, 3, figsize=(25, 6), dpi=300)

ds2.topographic__steepest_slope.sel(time=plot_time).plot(cmap='jet', ax = axs[0])
axs[0].set_title('Slope')
axs[0].plot(nf_x, nf_y, color='red')

ds2.soil__depth.sel(time=plot_time).plot(cmap='viridis', ax = axs[2])
axs[2].set_title('Sediment Thickness')
axs[2].plot(nf_x, nf_y, color='red')


ds2.channel_bedrock__width.sel(time=plot_time).plot(cmap='pink', ax = axs[1], vmax=30)
axs[1].set_title('Channel Width')
axs[1].plot(nf_x, nf_y, color='red')

#%%

fig, axs = plt.subplots(1, 3, figsize=(24, 6), dpi=300)

ds1.topographic__steepest_slope.sel(time=plot_time).plot(cmap='jet', ax = axs[0], vmax=3)
axs[0].set_title('Slope')
axs[0].plot(nf_x, nf_y, color='red')

ds1.topographic__elevation.sel(time=plot_time).plot(cmap='pink', ax = axs[2])
axs[2].set_title('Topographic Elevation')
axs[2].plot(nf_x, nf_y, color='red')


ds1.channel_bedrock__width.sel(time=plot_time).plot(cmap='pink', ax = axs[1], vmax=30)
axs[1].set_title('Channel Width')
axs[1].plot(nf_x, nf_y, color='red')

#%%


fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

plt.sca(ax=axs[0])
axs[0].set_title('Topography - No Sediment Cover')


imshow_grid(mg_1, 'topographic__elevation', vmax=750, allow_colorbar=False)
plt.plot(nf_x, nf_y, color='red')


plt.sca(ax=axs[1])
axs[1].set_title('Topography With Sediment Cover Effects')

imshow_grid(mg_2, 'topographic__elevation', vmax=750)
plt.plot(nf_x, nf_y, color='red')


#%%



fig, axs = plt.subplots(2, 1, figsize=(6,11), dpi=300)

ds1.topographic__steepest_slope.sel(time=plot_time).plot(cmap='jet', ax = axs[0], vmax=3)
axs[0].set_title('Slope')
axs[0].plot(nf_x, nf_y, color='red')

ds1.channel_bedrock__width.sel(time=plot_time).plot(cmap='pink', ax = axs[1], vmax=30)
axs[1].set_title('Channel Width')
axs[1].plot(nf_x, nf_y, color='red')



#%%

#ds1.to_netcdf('Faulted_no_sed_2.nc')
#ds2.to_netcdf('Faulted_w_sed_2.nc')

#%%
# =============================================================================
# Q_47 = Qw_800 ** 0.5
# 
# Q_ratio = br_w_800 / Q_47
# 
# 
# #topo_800.plot()
# 
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
# 
# topo_800.plot(ax=axs[0, 0], cmap='pink')
# sed_w_800.plot(ax=axs[0, 1])
# Qw_800.plot(ax=axs[1, 0], cmap='Blues')
# Q_ratio.plot(ax=axs[1, 1], cmap='coolwarm')
# 
# 
# #Q_ratio.plot(ax=axs[1, 1], cmap='coolwarm')
# 
# axs[0, 0].set_title('Topography')
# axs[0, 1].set_title('Channel Width')
# axs[1, 0].set_title('Discharge')
# axs[1, 1].set_title('Channel Width / $Q^0.5$')
# 
# 
# #(m$^3$/s)
# 
# plt.suptitle(str(plot_time) + ' years')
# plt.tight_layout()
# =============================================================================


#%%

# =============================================================================
# plt.figure()
# topo_outlet = ds.topographic__elevation.sel(x=100, y=100)
# topo_outlet.plot()

#%%


#%%

# =============================================================================
# xy200_width = br_width.sel(x=200, y=200)
# 
# plt.figure()
# xy200_width.plot()
# 
# #%%
# 
# xy200_topo = topo.sel(x=200, y=200)
# 
# plt.figure()
# xy200_topo.plot()
# 
# #%%
# 
# xy700_topo = topo.sel(x=700, y=700)
# 
# plt.figure()
# xy700_topo.plot()
# 
# 
# #%%
# 
# slope = ds.topographic__steepest_slope
# 
# xy200_slope = slope.sel(x=200, y=200)
# 
# plt.figure()
# xy200_slope.plot()
# 
# 
# =============================================================================
