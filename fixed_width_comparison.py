# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:21:43 2023

@author: gjg882
"""

import timeit

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

#%%inputs and outputs

#inputs = load_params('dynamic_w_inputs_10x10_gjg.txt')
inputs = load_params('dynamic_w_inputs_layers.txt')
#inputs = load_params("C:/Users/grace/Desktop/Projects/SpaceDynamicWidth/dynamic_w_inputs_10x10_gjg.txt")
#inputs = load_params('C:/Users/gjg882/Desktop/Code/SpaceDynamicWidth/dynamic_w_inputs.txt')

#path to save netcdf file to 

#ds_file_out = 'C:/Users/gjg882/Desktop/Projects/SDW_Output/ModelOutput/Qcalc_test_threshold2.nc'
#ds_file_out = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_kbank2x_nx100.nc'

#ds_file_out = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/space_fixed_width_sed.nc'

#ds_file_out = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_Ke-13.nc'''

#ds_file_out = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed_nx100_5e-12.nc'

#ds_file_out = 'C:/Users/gjg882/Desktop/Projects/SDW_Output/ModelOutput/SDW_20x20_e-14_2_highQ_sed.nc'
#ds_file_out = 'C:/Users/grace/Desktop/Projects/output/threshold_temp.nc'


ds_file_out = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/space_fixed_width_layers_50x50.nc'


#TODO - try model run with thresholds from original lague model

#%%Input variables



space_runtime = 1000000 #just for testing purposes


dx = 100

nx=50
ny=50


# nx=inputs['nx'] 
# ny=inputs['ny']


space_dt = inputs['dt'] #years


H_init = inputs["H_init"] #Initial sediment depth on bed, in m



#Try making things less erodible
#K_br =  K_br / 2
#K_bank = K_bank / 2

n_sp = 1 
m_sp = 0.5 
H_star = inputs["Hstar"]

#sp_crit_br = inputs["omegacr"]
#sp_crit_sed = inputs["omegacs"]
#omegacbank = inputs["omegacbank"]

V_mperyr = inputs["V_mperyr"]
phi = inputs["porosity"]
Ff = inputs["Ff"]


sp_crit_br = 1.8 * .03 #avg velocity of regular model run * .03 (critical shields)
sp_crit_sed = sp_crit_br
omegacbank = sp_crit_br*1.2

#replace inputs w/ updated values so they get saved to XR dataset 
inputs["omegacr"] = sp_crit_br 
inputs["omegacs"] = sp_crit_sed 
inputs["omegacbank"] = omegacbank 


Upliftrate_mperyr = inputs['space_uplift']

#thresholds
sp_crit_br = inputs["omegacr"]
sp_crit_sed = inputs["omegacs"]
omegacbank = inputs["omegacbank"]

# sp_crit_br = .03 * .12  #tau x estimated velocity (.12 m3/sec for 20x20x100 grid)
# sp_crit_sed = sp_crit_br 
# omegacbank = sp_crit_br * 1.15 #From phillips et al 2022



#Make a landlab model grid
mg = RasterModelGrid((nx, ny), dx)


#create model grid fields

H_soil = mg.add_zeros('node', 'soil__depth') #Create a grid field for soil depth

mg.at_node['soil__depth'][:] = H_init


bed_er = mg.add_zeros('bedrock_erosion__rate', at='node')
sed_er = mg.add_zeros('sediment_erosion__rate', at='node')


#%% Create initial topography 

#Random initial roughness
np.random.seed(seed = 200) #constant seed for constant random roughness


z = np.zeros(mg.size('node'))
mg.add_field('topographic__elevation',z, at='node')


#add random roughness to topography
random_field = 0.01 * np.random.randn(mg.size('node'))
z += random_field - random_field.min()



#All boundaries are closed except outlet node
mg.set_closed_boundaries_at_grid_edges(bottom_is_closed=True,
                                       left_is_closed=True,
                                       right_is_closed=True,
                                       top_is_closed=True)

#Setting Node 0 (bottom left corner) as outlet node 
mg.set_watershed_boundary_condition_outlet_id(0, 
                                              mg.at_node['topographic__elevation'],
                                              -9999.)


imshow_grid(mg, mg.status_at_node, color_for_closed='blue')

#%%

plt.figure()
imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')
plt.title('Initial Topo')
plt.show()


#%%

# K_br = 5.6622360440729185e-05 #calibrated by trial and error - for fsc, produces same steady state avg. z as default run
# K_br *= 1.195 #multiplier for use w/ sediment

K_br = 4.04850E-05 #For no sediment


K_sed = K_br * 1.5


K_br2 = K_br/2

#Set erodibility for each rock type
lith_attrs = {'K_sp': {1: K_br, 2: K_br2, 3: (K_br/10000)}}



#lith_attrs_ctrl = {'K_sp': {1: K_br, 2: K_br, 3: (K_br/1000)}} #first try layers with same K to make sure everything is working

layer_depths = [500, 2000] #Make bottom layer super thick ]

layer_ids = [1,2] #3 only for when litholayers needs to deposit - if it needs this something is wrong


lith = LithoLayers(mg, 
                    layer_depths, 
                    layer_ids,
                    attrs=lith_attrs,
                    layer_type='MaterialLayers',
                    rock_id=3)

lith.run_one_step()











#%%Use FastscapeEroder component to develop initial drainage network


fa1 = FlowAccumulator(mg,  flow_director='D8')
fa1.run_one_step()

fsc_uplift = .001 #m/yr
fsc = FastscapeEroder(mg, K_sp=1e-4)
fsc_dt = 100
fsc_time = 0

fsc_nts = 2000

for i in range (fsc_nts):
    
    dz_ad = np.zeros(mg.size('node'))
    dz_ad[mg.core_nodes] = fsc_uplift * fsc_dt
    mg.at_node['topographic__elevation'] += dz_ad
    
    lith.dz_advection=dz_ad
    
    lith.run_one_step()
    
    if 3 in mg.at_node['rock_type__id']:
        raise Exception("Litholayers is depositing material (rock type 3 detected)")
    
    fa1.run_one_step()
    
    fsc.run_one_step(dt=fsc_dt)
    fsc_time += fsc_dt
    

plt.figure()
imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')
plt.title('Topo after FSC')
plt.show()




mg.at_node["bedrock__elevation"] = mg.at_node["topographic__elevation"][:] - mg.at_node["soil__depth"][:]



#Coefficients calculated from regression line for bedrock rivers on fig 2E of Buckley et al 2024
#Q_calc = 0.25 * (DA[nx+1]**0.6)
#Runoff rate = Q/drainage area, DA * runoff = Q
#Runoff rate is in m/second
#runoff_calc = Q_calc / DA[nx+1]

runoff_mperyr = 3




fa = PriorityFloodFlowRouter(mg, runoff_rate=runoff_mperyr) 
fa.run_one_step()

lith_thick_fsc = lith.thickness[:]



#%%

t = np.arange(0, space_runtime+space_dt, space_dt)            
nts = len(t)


#how often to write model output to xarray dataset
save_interval = 1000 #years

#array of times where model output will be saved
out_times = np.arange(0, space_runtime+save_interval, save_interval)
out_count = len(out_times)



#%% Create Xarray dataset to save model output

ds = xr.Dataset(
    data_vars={
        
        'topographic__elevation':  (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': 'meters',  # dictionary with data attributes
                'long_name': 'Topographic Elevation'
            }),
            
        
        'bedrock_erosion__rate' : (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': 'meters/yr',
                'long_name': 'Bedrock Erosion Rate',
                
            }),    
        
        
        'sediment_erosion__rate' : (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': 'meters/yr',
                'long_name': 'Sediment Erosion Rate',
                
            }), 
        
        
        'soil__depth' : (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': 'meters',
                'long_name': 'sediment thickness',
                
            }),   
        
        'rock_type__id' : (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': '-',
                'long_name': 'Rock Type ID',
                
            }),     
        

        
        'topographic__steepest_slope' : (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': '-',
                'long_name': 'Topographic Steepest Slope',
                
            })  
    },
        
    coords={
        'x': (
            ('x'),  # tuple of dimensions
            mg.x_of_node.reshape(
                mg.shape)[0, :],  # 1-d array of coordinate data
            {
                'units': 'meters'
            }),  # dictionary with data attributes
        'y': (('y'), mg.y_of_node.reshape(mg.shape)[:, 1], {
            'units': 'meters'
        }),
        'time': (('time'), out_times, {
            'units': 'years',
            'standard_name': 'time'
        })
    },
    attrs=dict(inputs))
   
#list of model grid fields to save to output dataset    
out_fields = ['topographic__elevation',
              'bedrock_erosion__rate',
              'sediment_erosion__rate',
              'soil__depth',
              'rock_type__id',
              'topographic__steepest_slope']
             

 

ds.attrs.update({"runoff_mperyr": runoff_mperyr})

        
#%%


ds.attrs.update({"K_br": K_br})

space = SpaceLargeScaleEroder(mg,
            K_sed =K_sed,
            K_br = K_br,
            F_f = Ff,
            phi = phi,
            H_star = H_star,
            v_s = V_mperyr,
            m_sp = m_sp,
            n_sp = n_sp,
            sp_crit_sed = 0,
            sp_crit_br = 0)


#%% Main model loop - uniform rock type


# elapsed_time = 0
# total_U = 0
# starttime = timeit.default_timer()

# space_uplift = Upliftrate_mperyr


# #for i in range(10000):
# for i in range(nts):
# #for i in range(1):


    
#     #flow routing    
#     fa.run_one_step()
#     #Use getter to update bedrock erosion rate from space 

#     #erode with space
#     space.run_one_step(dt=space_dt)
    
#     bed_er[mg.core_nodes] = space._Er[mg.core_nodes]
    
    
#     #uplift the landscape 
#     dz_ad = np.zeros(mg.size('node'))
#     dz_ad[mg.core_nodes] = space_uplift * space_dt
#     mg.at_node['bedrock__elevation'] += dz_ad
    
#     mg.at_node['topographic__elevation'][:] = mg.at_node['bedrock__elevation'] + mg.at_node['soil__depth']
  
    
#     #save output, check how long code has been running
#     if elapsed_time %save_interval== 0:
        
#         time_diff = timeit.default_timer() - starttime
#         time_diff_minutes = np.round(time_diff/60, decimals=2)
        
#         ds_ind = int((elapsed_time/save_interval))
    
#         for of in out_fields:
#             ds[of][ds_ind, :, :] = mg['node'][of].reshape(mg.shape)
            
#         print(elapsed_time, time_diff_minutes )
#         print('mean elev', np.mean(z[mg.core_nodes]))
        
#         #write output to netcdf file
#         #ds.to_netcdf(ds_file_out)
    
#     # if elapsed_time == 150000:
        
#         # runoff_mperyr *= 2
#         # fa = PriorityFloodFlowRouter(mg, runoff_rate=runoff_mperyr) 
        
    
#     #update elapsed time
#     elapsed_time += space_dt
    
    

#%%Main model loop - LITHOLAYERS

elapsed_time = 0
total_U = 0
starttime = timeit.default_timer()

space_uplift = Upliftrate_mperyr


#save initial condition as time 0
for of in out_fields:
    ds[of][0, :, :] = mg['node'][of].reshape(mg.shape)



#for i in range(10000):
for i in range(nts):
#for i in range(1):


    
    #flow routing    
    fa.run_one_step()
    #Use getter to update bedrock erosion rate from space 



    space.K_br = mg.at_node['K_sp'][:] 
    
    #erode with space
    space.run_one_step(dt=space_dt)
    
    bed_er[mg.core_nodes] = space._Er[mg.core_nodes]
    sed_er[mg.core_nodes] = space._Es[mg.core_nodes]
    
    
    #uplift the landscape 
    dz_ad = np.zeros(mg.size('node'))
    dz_ad[mg.core_nodes] = space_uplift * space_dt
    mg.at_node['bedrock__elevation'] += dz_ad
    
    
    #temporarily set topo = to br elev so litholayers doesn't "see" sediment
    mg.at_node['topographic__elevation'][:] = mg.at_node['bedrock__elevation']
    lith.dz_advection=dz_ad
    lith.run_one_step()
    
    if 3 in mg.at_node['rock_type__id']:
        raise Exception("Litholayers is depositing material (rock type 3 detected)")
        
        
    
    mg.at_node['topographic__elevation'][:] = mg.at_node['bedrock__elevation'] + mg.at_node['soil__depth']
    

    
    #save output, check how long code has been running
    if elapsed_time %save_interval== 0:
        
        time_diff = timeit.default_timer() - starttime
        time_diff_minutes = np.round(time_diff/60, decimals=2)
        
        ds_ind = int((elapsed_time/save_interval))
    
        for of in out_fields:
            ds[of][ds_ind, :, :] = mg['node'][of].reshape(mg.shape)
            
        print(elapsed_time, time_diff_minutes )
        print('mean elev', np.mean(z[mg.core_nodes]))
        
        #write output to netcdf file
        ds.to_netcdf(ds_file_out)
    
        
    
    #update elapsed time
    elapsed_time += space_dt







#%%
plt.figure()
imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')   
plt.title('Final Topo') 

    
plt.show()
#%%

mg.at_node['soil__depth'][0] = 0
plt.figure()
imshow_grid(mg, 'soil__depth', colorbar_label='Sed Thickness (m)')   
plt.title('Final Sediment Thickness') 
plt.show()

#%%

plt.figure()
imshow_grid(mg,'topographic__steepest_slope')   
plt.title('Slope') 
plt.show()

 

#%%Plot zero slope nodes in red

# zero_slope = np.where(mg.at_node['topographic__steepest_slope'] == 0)[0]
# flat_x = mg.x_of_node[zero_slope]
# flat_y = mg.y_of_node[zero_slope]


# from matplotlib.colors import ListedColormap

# # Create a custom colormap based on 'viridis'
# viridis = plt.cm.get_cmap('viridis', 256)
# new_colors = viridis(np.linspace(0, 1, 256))

# # Set the color for zero values to red
# new_colors[0] = [1, 0, 0, 1]  # RGBA for red (fully opaque)

# # Create a new colormap
# custom_cmap = ListedColormap(new_colors)


# plt.figure()
# imshow_grid(mg, 'topographic__steepest_slope', colorbar_label='Slope', cmap=custom_cmap)  
# #plt.plot(flat_x, flat_y, 'rs') 
# plt.title('Final Slope') 
# plt.show()


#%%


topo_mean = ds['topographic__elevation'].mean(dim=["x", "y"])
plt.plot(topo_mean["time"], topo_mean)
plt.title('Mean Elevation over Time')
plt.show()



