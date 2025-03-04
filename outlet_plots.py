# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 07:53:13 2025

@author: grace
"""


import numpy as np
import math
from matplotlib import pyplot as plt



import scipy.optimize


import xarray as xr



from landlab import load_params

from landlab import RasterModelGrid

from landlab.plot import imshow_grid

from landlab.components import (FlowAccumulator, 
                                DepressionFinderAndRouter,
                                FastscapeEroder,
                                Lithology,
                                LithoLayers,
                                ChannelProfiler,
                                ChiFinder,
                                Space, 
                                PriorityFloodFlowRouter, 
                                SpaceLargeScaleEroder)
import os


#%%

from plot_funcs import (xr_to_mg, 
                        plot_channel_prf, 
                        calc_channel_dims, 
                        plot_xy_timeseries_multi)

#%%


#ds_file = 'C:/Users/grace/Desktop/Projects/output/newQ_200kyr_lowKbank2.nc'

#ds_file = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_lowKbank2.nc'


#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_lowKbank2.nc'
#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_lowKbank.nc'
#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sameK.nc'

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed3.nc'
#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_kbank2x.nc'

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_kbank2x_nx100.nc'
#ds_file='C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_test_kr2_150_nx50_2.nc' #didn't work

#ds_file = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed3.nc'


#ds_file = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed_nx100_1e-11.nc'

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed_nx100_5e-12.nc' #Failed
#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed_nx100_5e-12.nc' #Failed

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed_nx100_1e-11.nc' #THIS ONE WORKS


#ds_file='C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_test_kr15_nx50_ctrl.nc'

#ds_file = "C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_kbank2x_nx100.nc"


#ds_file = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_ratio_05.nc'

#ds_file = "C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_kbank2x.nc" #20x20, 100kyr




#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_ratio_05_sed.nc'
#ds2 = xr.open_dataset(ds_file) 

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_ratio_05_sed_v3.nc'
#ds1 = xr.open_dataset(ds_file)

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_test_kr15_nx50_ctrl_4.nc'
#ds1 = xr.open_dataset(ds_file)

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_ratio_05.nc'
#ds1 = xr.open_dataset(ds_file) 


#ds_file = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_lowKbank.nc' #n "default", kbank = 1.5e-12, Kbr = 1e-12
#ds_file = "C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/space_fixed_width_sed3.nc"
#ds_file = "C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/space_fixed_width_sed3.nc"
#ds_file = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_ratio_05_sed_v3.nc'

ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/dynamic_layers_ctrl.nc'
model_name = 'SDW'

ds1 = xr.open_dataset(ds_file)

sec_per_yr = 3.154e+7


max_time = ds1.time.max().item()


plot_time = max_time


#Calculate additional channel dimensions
var_list1 = list(ds1.variables)
ds1 = calc_channel_dims(ds1)



# var_list2 = list(ds1.variables)
# ds2 = calc_channel_dims(ds2)
data_vars_dimensionless = ['topographic__steepest_slope', 'width_depth__ratio' ]
data_vars_m = ['channel_bedrock__width', 'flow__depth',  'topographic__elevation', 'soil__depth' ] #data variables w/ units of L
data_vars_rates = ['bank_erosion__rate', 'bedrock_erosion__rate'] #data variables w/ units of L/T

ds1['bank_erosion__rate'] *= sec_per_yr

ds1['bedrock_erosion__rate'] *= sec_per_yr
#data_vars_m = ['topographic__elevation', 'soil__depth' ] #data variables w/ units of L
#data_vars_rates = ['bedrock_erosion__rate'] #data variables w/ units of L/T
    

#%%




fig, xy_coords, mg, ref_node_ind = plot_channel_prf(ds1, max_time, 'SDW w/ Sediment')

#%%


fig, axes = plot_xy_timeseries_multi(ds1, data_vars_rates, xy_coords, vline=300000)




#for v0 and v1
# axes[0].set_ylim([0, 5])
# axes[1].set_ylim([0, 0.75])
# axes[2].set_ylim([0, 9])
# axes[3].set_ylim([0, 80])
#plot_xy_timeseries_multi(ds2, data_vars_m, xy_coords2)

# axes[0].set_ylim([0, 6])
# axes[1].set_ylim([0, 0.75])
# axes[2].set_ylim([0, 16])
# axes[3].set_ylim([0, 80])


#%%


fig, axes = plot_xy_timeseries_multi(ds1, data_vars_m, xy_coords, vline=300000)



fig, axes = plot_xy_timeseries_multi(ds1, data_vars_dimensionless, xy_coords, vline=300000)

#%%


ds2 = xr.open_dataset('C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/space_fixed_width_sed_layers_50x50.nc')
title2 = 'SPACE w/ sediment'
fig, xy_coords2, mg2, ref_node_ind = plot_channel_prf(ds2, max_time, title2)


ds3 = xr.open_dataset('C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/space_fixed_width_layers_50x50.nc')
title_3 = 'SPACE, no sediment'
fig, xy_coords3, mg3, ref_node_ind = plot_channel_prf(ds3, max_time, title2)


#%%
data_vars_rates_SPACE = ['bedrock_erosion__rate', 'sediment_erosion__rate']
data_vars_dim_SPACE = ['topographic__steepest_slope']
data_vars_m_SPACE = ['topographic__elevation', 'soil__depth']

#%%

#fig, axes = plot_xy_timeseries_multi(ds2, data_vars_rates_SPACE, xy_coords, vline=300000)
fig, axes = plot_xy_timeseries_multi(ds2, data_vars_m_SPACE, xy_coords, vline=300000)
#fig, axes = plot_xy_timeseries_multi(ds2, data_vars_dim_SPACE, xy_coords, vline=300000)




#%%

fig, axes = plot_xy_timeseries_multi(ds3, data_vars_rates_SPACE, xy_coords, vline=300000)
fig, axes = plot_xy_timeseries_multi(ds3, data_vars_m_SPACE, xy_coords, vline=300000)
fig, axes = plot_xy_timeseries_multi(ds3, data_vars_dim_SPACE, xy_coords, vline=300000)


#%%

ds4 = xr.open_dataset('C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_lowKbank.nc') #n "default", kbank = 1.5e-12, Kbr = 1e-12)



#%%
plt.figure(dpi=300)

ds_list = [ds1, ds2, ds3, ds4]
xy_mid = [2500, 1200]

labels = ['SDW w/sed', 'SPACE w/ sed', 'SPACE no sed', 'SDW no sed (small grid)']


# mid_topo_1 = ds1['topographic__elevation'].sel(x=xy_mid[0], y=xy_mid[1], method='nearest')
# mid_topo_2 = ds2['topographic__elevation'].sel(x=xy_mid[0], y=xy_mid[1], method='nearest')
# mid_topo_3 = ds3['topographic__elevation'].sel(x=xy_mid[0], y=xy_mid[1], method='nearest')


for ds, label in zip(ds_list, labels):
    var_data = ds['topographic__steepest_slope'].sel(x=xy_mid[0], y=xy_mid[1], method='nearest')
    var_data.plot(label=label)

plt.legend()
plt.title('Slope @mid-channel node')

#%%

ds4['bedrock_erosion__rate'] *= sec_per_yr


plt.figure()

for ds, label in zip(ds_list, labels):
    # Calculate the mean elevation across the entire grid at each timestep
    var_data = ds['bedrock_erosion__rate'].mean(dim=['x', 'y'])
    var_data.plot(label=label)

plt.legend(loc = 'lower right')
plt.ylabel('Mean Bedrock Erosion Rate')

#%%

last_z = var_data.sel(time=200000).item()
print(last_z)


#%%

z_mean_2 = ds2['topographic__elevation'].mean(dim=['x', 'y'])
last_z2 = z_mean_2.sel(time=200000).item()
print(last_z2)


#%%

# mg = xr_to_mg(ds1, plot_time=max_time)

# #ref_node_ind = (ds1.attrs['nx'] * 2) + 1

# slope_ref = mg.at_node['topographic__steepest_slope'][ref_node_ind]
# DA_ref = mg.at_node['drainage_area'][ref_node_ind]
# bed_er_ref = mg.at_node['bedrock_erosion__rate'][ref_node_ind] #erosion rate in m/yr
# bank_er_ref = mg.at_node['bank_erosion__rate'][ref_node_ind]  #erosion rate in m/yr

# print('slope ref,', slope_ref)
# print('DA ref,', DA_ref)
# print('bed_er_ref,', bed_er_ref)
# print('bank_er_ref,', bank_er_ref)

# Am = DA_ref ** 0.5
# Sn = slope_ref ** 1

# K_ref = bed_er_ref / (Am * Sn)


# plt.figure()
# imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')
# plt.title('Initial Topo')
# plt.show()

#%%Convert erosion rate units if necessary

# sec_per_yr = 3.154e+7

# mean_er = ds['bedrock_erosion__rate'].mean()

# order_of_magnitude = math.floor(math.log10(abs(mean_er)))

# Check if order of magnitude is around -11
# if order_of_magnitude == -11:
#     ds['bedrock_erosion__rate'] *= sec_per_yr
#     ds['bank_erosion__rate'] *= sec_per_yr



#%%Run fastscape to get to same elevation

# mg = xr_to_mg(ds1, plot_time=0)

# #All boundaries are closed except outlet node
# mg.set_closed_boundaries_at_grid_edges(bottom_is_closed=True,
#                                        left_is_closed=True,
#                                        right_is_closed=True,
#                                        top_is_closed=True)

# #Setting Node 0 (bottom left corner) as outlet node 
# mg.set_watershed_boundary_condition_outlet_id(0, 
#                                               mg.at_node['topographic__elevation'],
#                                               -9999.)


# fa1 = PriorityFloodFlowRouter(mg)
# fa1.run_one_step()
# fsc_uplift = .001 #m/yr
# fsc = FastscapeEroder(mg, K_sp=(K_ref/1.14))
# fsc_time = 0
# fsc_dt = 100

# fsc_nts = int(500000/fsc_dt)

# for i in range (fsc_nts):
    
#     dz_ad = np.zeros(mg.size('node'))
#     dz_ad[mg.core_nodes] = fsc_uplift * fsc_dt
#     mg.at_node['topographic__elevation'] += dz_ad
    
#     fa1.run_one_step()
    
#     fsc.run_one_step(dt=fsc_dt)
#     fsc_time += fsc_dt
    
#     if fsc_time % 10000 == 0:
#         mean_z = np.mean(mg.at_node['topographic__elevation'])
#         print(fsc_time, ', mean z= ', mean_z)




    
#%%
plt.figure()
imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')
plt.title('Final Topo')
plt.show()

slope_ref2 = mg.at_node['topographic__steepest_slope'][ref_node_ind]
DA_ref2 = mg.at_node['drainage_area'][ref_node_ind]


print('slope ref,', slope_ref2)
print('DA ref,', DA_ref2)


#%%

prf = ChannelProfiler(
    mg,
    main_channel_only=True,
)

prf.run_one_step()

ordered_dict = prf.data_structure

#get the first watershed from prf ordered dict
#watershed is a tuple with two elements: outlet ID and a nested dict 
watershed = next(iter(ordered_dict.items()))

#watershed data is the nested dict
outlet_ID, watershed_data = watershed


#nested_dict = watershed_data
#channel_key = next(iter(nested_dict))
channel_dict = watershed_data
channel_key = next(iter(channel_dict))


#get node ids of all nodes in the main channel
ids_array = channel_dict[channel_key]['ids']


#Get outlet, midpoint, and upstream end of main channel
first_value = ids_array[1]
middle_value = ids_array[len(ids_array) // 2]
last_value = ids_array[-3]

x_coords = mg.x_of_node[[first_value, middle_value, last_value]]
y_coords = mg.y_of_node[[first_value, middle_value, last_value]]

xy_coords = list(zip(x_coords, y_coords))

fig = plt.figure()
#imshow_grid(mg1, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')   
prf.plot_profiles_in_map_view()
#plt.title(f'Main Channel, {model_name}, Time={plot_time/1000} kyr') 
plt.plot(x_coords[0], y_coords[0], 'o', color='tab:blue', markersize=3)  
plt.plot(x_coords[1], y_coords[1], 'o', color='orange', markersize=3)  
plt.plot(x_coords[2], y_coords[2], 'o', color='green', markersize=3)  

plt.show()

#%%
mg2 = xr_to_mg(ds1, plot_time=max_time)
z_avg_2 = np.mean(mg2.at_node['topographic__elevation'])
print('SDW avg z=', z_avg_2)
