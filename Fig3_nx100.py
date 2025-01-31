# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:37:18 2023

@author: gjg882
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import xarray as xr

#from illustrate_cell import draw_channel

from calc_prf_width import calc_prf_width

from matplotlib.ticker import ScalarFormatter



from landlab import RasterModelGrid
from landlab import load_params
from landlab.plot import imshow_grid
from landlab.components import (FlowAccumulator, 
                                DepressionFinderAndRouter,
                                FastscapeEroder,
                                ChannelProfiler,
                                SteepnessFinder)


#%%

#ds1 = xr.open_dataset('ModelOutput/SDW_20x20_e-14_2_highU_6.nc')

#ds1 = xr.open_dataset('ModelOutput/SDW_20x20_e-14_2_highU_NoSed.nc')
ds1 = xr.open_dataset('ModelOutput/SDW_100x100_test3.nc')
ds2 = xr.open_dataset('ModelOutput/SDW_100x100_test3_sed3_729.nc')

#ds2 = 

#load the input parameters that are saved as metadata on the ds
ds_attrs = ds1.attrs

#%%

sec_per_yr =  60 * 60 * 24 * 365

plot_time = 729000
plot_time_myr = plot_time/1000000

dx = ds_attrs['dx']

sf_min_DA = dx**2

mg1, df1 = calc_prf_width(ds1, plot_time, sf_min_DA)
mg2, df2 = calc_prf_width(ds1, plot_time, sf_min_DA)

df_trim = df1.drop(index=0)
df2_trim = df2.drop(index=0)

#mg2, df2 = calc_prf_width(ds2, plot_time, sf_min_DA)

#%%

time_arr = ds1.time.to_numpy()





#%%Plot width vs drainage area scaling

plt.figure(dpi = 300)
plt.plot(df_trim['drainage_area'], df_trim['br_width'], label='no sediment')
plt.plot(df2_trim['drainage_area'], df2_trim['br_width'], label='with sediment')

#plt.plot(df_trim['DA_exp'], df_trim['br_width'], color='blue')


plt.xlabel("Drainage Area ($m^{2}$)")


plt.ylabel('Channel Width (m)')

plt.yscale('log')
plt.xscale('log')



#%%plot width vs slope

plt.figure(dpi = 300)

plt.plot(df_trim['br_width'], df_trim['channel_slope'])
plt.plot(df2_trim['br_width'], df2_trim['channel_slope'], label='with sediment')

plt.ylabel('Channel Slope')
plt.xlabel('Channel Width')

plt.yscale('log')
plt.xscale('log')

plt.legend(loc='best')


#%%plot main channel profile

#TODO update calc prf function to include channel bedrock width in data frame

plt.figure()
plt.plot(df1['channel_dist'], df1['channel_elev'], label='no sediment')
plt.plot(df2['channel_dist'], df2['channel_elev'], label='with sediment')
#plt.plot(df1['channel_dist'], df1['br_width'], label='bedrock width')
plt.xlabel('Distance Upstream (m)')

plt.title('Main Channel Profile')
plt.ylabel('Elevation (m)')

#%%

plt.figure()
plt.plot(df1['channel_dist'], df1['br_width'], label='no sediment')
plt.plot(df2['channel_dist'], df2['br_width'], label='with sediment')
#plt.plot(df1['channel_dist'], df1['br_width'], label='bedrock width')
plt.xlabel('Distance Upstream (m)')

plt.title('Width Along Profile')
plt.ylabel('Width (m)')

#%%plot topography

plt.figure()
imshow_grid(mg1, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')
plt.title('Topography at ' + str(plot_time_myr) + ' myr')

#%%plot width in map view

#zero out width at outlet

mg1.at_node['channel_bedrock__width'][0] = 0

plt.figure()
imshow_grid(mg1, 'channel_bedrock__width', colorbar_label='Channel Width (m)', cmap='viridis')
plt.title('Width at ' + str(plot_time_myr) + ' myr')




#%%#look at plot of mean elevation over time to determine if topo is at steady state
topo1 = ds1.topographic__elevation

mean_topo1 = topo1.mean(dim=["x", "y"])

plt.figure()
plt.title('Mean Topographic Elevation')
mean_topo1.plot()

#%%


ds1.channel_bedrock__width.loc[dict(time=plot_time, x=0,y=0)] = np.nan

#ds2.channel_bedrock__width.loc[dict(time=plot_time, x=0,y=0)] = np.nan

#%%

br_width1 = ds1.channel_bedrock__width
#br_width2 = ds2.channel_bedrock__width

mean_br_width1 = br_width1.mean(dim=["x", "y"])

plt.figure()
mean_br_width1.plot()
plt.title('Mean Channel Width')



#%%

ds_coords = [100, 100] 
us_coords = [1600, 1600]
mid_coords = [900, 900]


#ds_coords = [100, 100] 
#us_coords = [900, 900]
#mid_coords = [500, 500]


#%%Plot topography through time at selected nodes

topo1d_1 = ds1.topographic__elevation.sel(x=us_coords[0], y=us_coords[1])

topo1d_outlet_1 = ds1.topographic__elevation.sel(x=dx, y=dx)

topo1d_25_1 = ds1.topographic__elevation.sel(x=mid_coords[0], y=mid_coords[1])

#topo1d_2 = ds2.topographic__elevation.sel(x=us_coords[0], y=us_coords[1])

#topo1d_outlet_2 = ds2.topographic__elevation.sel(x=dx, y=dx)

#topo1d_25_2 = ds2.topographic__elevation.sel(x=mid_coords[0], y=mid_coords[1])

#%%

plt.figure(dpi=300)

xfmt = ScalarFormatter()
xfmt.set_powerlimits((0,3))

topo1d_1.plot(color='indigo', label='upstream')
topo1d_25_1.plot(color='teal', label='mid-channel')
topo1d_outlet_1.plot(color='goldenrod', label='outlet')

# =============================================================================
# topo1d_2.plot(color='indigo', linestyle='--')
# topo1d_25_2.plot(color='teal', linestyle='--')
# topo1d_outlet_2.plot(color='goldenrod', linestyle='--', label='Constant Q')
# =============================================================================



plt.legend(loc='best')


plt.gca().xaxis.set_major_formatter(xfmt)

plt.ylabel('Elevation (m)')
plt.xlabel('Time (kyr)')
plt.xlim(0, plot_time)
plt.title('Topography through Time')

plt.rcParams['font.size'] = 12





#plt.savefig('HighQ_topo.tiff')


#%%Plot width through time at selected nodes




width1d_1 = ds1.channel_bedrock__width.sel(x=us_coords[0], y=us_coords[1])

width1d_100_1 = ds1.channel_bedrock__width.sel(x=ds_coords[0], y=ds_coords[1])

width1d_25_1 = ds1.channel_bedrock__width.sel(x=mid_coords[0], y=mid_coords[1])


#width1d_2 = ds2.channel_bedrock__width.sel(x=us_coords[0], y=us_coords[1])

#width1d_100_2 = ds2.channel_bedrock__width.sel(x=ds_coords[0], y=ds_coords[1])

#width1d_25_2 = ds2.channel_bedrock__width.sel(x=mid_coords[0], y=mid_coords[1])


plt.figure(dpi=300)



xfmt = ScalarFormatter()
xfmt.set_powerlimits((0,3))
#xfmt.set_useOffset(100000)

plt.axvline(x=600000, color='darkgrey', linestyle=':')


width1d_1.plot(color='indigo', label='upstream')
width1d_25_1.plot(color='teal', label='mid-channel')
width1d_100_1.plot(color='goldenrod', label='outlet')

# =============================================================================
# width1d_2.plot(color='indigo', linestyle='--')
# width1d_25_2.plot(color='teal', linestyle='--')
# width1d_100_2.plot(color='goldenrod', linestyle='--')
# =============================================================================




lh_out = mlines.Line2D([], [], color='goldenrod', label='outlet')
lh_mid = mlines.Line2D([], [], color='teal', label='mid-channel')
lh_us = mlines.Line2D([], [], color='indigo', label='upstream')


# =============================================================================
# sed_line = mlines.Line2D([], [], color='k', linestyle='--', label='w/ sediment cover')
# =============================================================================

ax = plt.gca()

plt.gca().xaxis.set_major_formatter(xfmt)



ax.legend(handles=[lh_out, lh_mid, lh_us])


plt.ylabel('Width (m)')
plt.xlabel('Time (kyr)')
plt.xlim(0, plot_time)
plt.title('Channel Width')

plt.rcParams['font.size'] = 12

#plt.savefig('HighQ_width.tiff')

#%%


topo_800_1 = ds1.topographic__elevation.sel(time=plot_time)
br_w_800_1 = ds1.channel_bedrock__width.sel(time=plot_time)
sed_w_800_1 = ds1.channel_sed__width.sel(time=plot_time)
Qw_800_1  = ds1.surface_water__discharge.sel(time=plot_time) 
flow_depth_1 =  ds1.flow__depth.sel(time=plot_time)


#%%

Q_47 = Qw_800 ** 0.5

Q_ratio = br_w_800 / Q_47


#topo_800.plot()

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

topo_800.plot(ax=axs[0, 0], cmap='pink')
sed_w_800.plot(ax=axs[0, 1])
Qw_800.plot(ax=axs[1, 0], cmap='Blues')
Q_ratio.plot(ax=axs[1, 1], cmap='coolwarm')


#Q_ratio.plot(ax=axs[1, 1], cmap='coolwarm')

axs[0, 0].set_title('Topography')
axs[0, 1].set_title('Channel Width')
axs[1, 0].set_title('Discharge')
axs[1, 1].set_title('Channel Width / $Q^0.5$')


#(m$^3$/s)

plt.suptitle(str(plot_time) + ' years')
plt.tight_layout()


#%%


plt.figure(dpi=300, figsize=(5,4))
Q_ratio.plot(cmap='coolwarm')

plt.title('Channel Width / $Q^{1/2}$')
plt.rcParams['font.size'] = 12

#%%

plt.figure(dpi=300, figsize=(5,4))
br_w_800_1.plot(cmap='viridis')
plt.title('Channel Bedrock Width')
plt.rcParams['font.size'] = 12



#%%

plt.figure(dpi=300, figsize=(5,4))
topo_800_1.plot(cmap='pink', cbar_kwargs={'label': 'Elevation (m)'})
plt.title('Topography at 1.2 myr')

plt.rcParams['font.size'] = 12


#Q_ratio_a = Q_ratio.values



#%%

def calc_relief(topo_da):
    max_elev = topo_da.max()
    min_elev = topo_da.min()
    
    relief = max_elev - min_elev
    
    return relief.values

#%%

# =============================================================================
# plt.figure()
# topo_outlet = ds.topographic__elevation.sel(x=100, y=100)
# topo_outlet.plot()
# =============================================================================

#%%

mean_topo = ds.topographic__elevation.mean(dim='time')

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

#%%




#%%



#%%

plt.figure()
plt.plot(df1['channel_dist'], df1['channel_Er'])
plt.xlabel('Distance Upstream (m)')
plt.ylabel('Erosion Rate')

#%%

plt.figure()
imshow_grid(mg1, 'topographic__steepest_slope', at='node')
plt.title('Slope')

#%%



#%%

df_trim['DA_exp'] = df_trim['drainage_area'] ** (1/2)

#%%


plt.figure(dpi = 300)
plt.plot(df_trim['drainage_area'], df_trim['br_width'])

#plt.plot(df_trim['DA_exp'], df_trim['br_width'], color='blue')


plt.xlabel("Drainage Area ($m^{2}$)")


plt.ylabel('Channel Width (m)')

plt.yscale('log')
plt.xscale('log')

#%%


plt.figure(dpi = 300)

plt.plot(df_trim['br_width'], df_trim['channel_slope'])

plt.ylabel('Channel Slope')
plt.xlabel('Channel Width')

#%%


df1.drop(index=0)

plt.figure(dpi = 300)
plt.plot(df1['channel_dist'], df1['br_width'])


plt.xlabel('Distance Upstream (m)')
plt.ylabel('Channel Width (m)')

#plt.yscale('log')
#plt.xscale('log')


#%%

topo_init = ds.topographic__elevation.sel(time=0)

plt.figure()
plt.title('Initial Topography')
topo_init.plot()

#%%



#%%

topo_diff = topo_800 - topo_init

plt.figure()
topo_diff.plot()
plt.title('Topo Final - Topo Init')

#%%


plot_time_2 = plot_time - 1000
mg2, df2 = calc_prf_width(ds, plot_time_2, sf_min_DA)

#%%

prf_diff = df['channel_elev'] - df2['channel_elev'] 

plt.figure()
plt.title('Diff Between topo final and 1 kyr earlier')
plt.plot(df['channel_dist'], prf_diff)

#%%

flow_depth.plot()

#%%  

outlet_width = np.ones(801,)

#%%


# =============================================================================
# print(br_width.sel(time=plot_time, x=100,y=100).values)
# print(flow_depth.sel(x=100,y=100).values)
# print(ds.soil__depth.sel(time=plot_time, x=100,y=100).values)
# print(ds_attrs['Kr'])
# print(ds_attrs['Kbank'])
# print(ds_attrs['Ks'])
# print(ds_attrs['V_mperyr'])
# 
# =============================================================================



outlet_width = br_width.sel(time=plot_time, x=100,y=100).values
outlet_depth = flow_depth.sel(x=100,y=100).values
outlet_soil = ds.soil__depth.sel(time=plot_time, x=100,y=100).values
Kr = ds_attrs['Kr']
Kbank = ds_attrs['Kbank']
Ks = ds_attrs['Ks']
V = ds_attrs['V_mperyr']


print(outlet_width, outlet_depth, outlet_soil, Kr, Kbank, Ks, V)


#%%

ds.channel_bedrock__width.sel(time=plot_time).plot()