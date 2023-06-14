# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:56:49 2023

@author: gjg882
"""

import numpy as np
from matplotlib import pyplot as plt
import xarray as xr

import time

from landlab import RasterModelGrid

from landlab.plot import imshow_grid
from landlab import load_params
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

#%%


#Inputs to create RasterModelGrid
inputs = load_params('fixed_width_inputs_ctrl.txt')

model_name = inputs['model_name']

dx=inputs['dx']
nx=inputs['nx']
ny=inputs['ny']
H_init=inputs['H_init']


K1 = inputs['K_br']
K3 = 5e-5 #TODO CHANGE THIS
K2 = K1*K3

m_sp = inputs['m_sp']
n_sp = inputs['n_sp']


mg = RasterModelGrid((nx, ny), dx)

np.random.seed(seed = 200) #constant seed for constant random roughness
z = mg.add_zeros('topographic__elevation', at='node')
random_field = 0.01 * np.random.randn(mg.size('node'))
z += random_field - random_field.min()


mg.add_zeros('node', 'soil__depth') #Create a grid field for soil depth
mg.at_node['soil__depth'][:] = H_init

w = np.zeros((nx, ny))
w = mg.add_field("channel__width", w, at="node")

K = np.zeros((nx, ny))
K = mg.add_field("K", K, at="node")


mg.set_watershed_boundary_condition_outlet_id(0, 
                                              mg.at_node['topographic__elevation'],
                                              -9999.)

#Check boundary conditions
#imshow_grid(mg, mg.status_at_node, color_for_closed='blue')


#%%

#Inputs to instantiate SPACE
K_sed = inputs['K_sed']
F_f = inputs['F_f']
phi = inputs['phi']
H_star = inputs['H_star']
v_s = inputs['v_s']
sp_crit_sed = inputs['sp_crit_sed']
sp_crit_br = inputs['sp_crit_br']

space = SpaceLargeScaleEroder(mg,
           K_sed =K_sed,
           K_br = K1,
           F_f = F_f,
           phi = phi,
           H_star = H_star,
           v_s = v_s,
           m_sp = m_sp,
           n_sp = m_sp,
           sp_crit_sed = sp_crit_sed,
           sp_crit_br = sp_crit_br)


#space runtime parameters
space_dt = inputs['space_dt'] #years
space_uplift = inputs['space_uplift']
space_runtime = inputs['space_runtime']

#Array of timesteps
t = np.arange(0, space_runtime+space_dt, space_dt)
nts = len(t)

#How often to save model output to xr dataset
save_interval = 1000
out_times = np.arange(0, space_runtime+save_interval, save_interval)
out_count = len(out_times)


#%%

ds = xr.Dataset(
    data_vars={
        
        'topographic__elevation': (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': 'meters',  # dictionary with data attributes
                'long_name': 'Topographic Elevation'
            }),

        'soil__depth':
        (('time', 'y', 'x'), np.empty((out_count, mg.shape[0], mg.shape[1])), {
            'units': 'meters',
            'long_name': 'Sediment Depth'
        
        }),
        
        'channel__width':
        (('time', 'y', 'x'), np.empty((out_count, mg.shape[0], mg.shape[1])), {
            'units': '"m"',
            'long_name': 'Channel Width'
            
        }),
        
        'surface_water__discharge':
        (('time', 'y', 'x'), np.empty((out_count, mg.shape[0], mg.shape[1])), {
            'units': 'm**3/s',
            'long_name': 'Bedrock Erosion'
            
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


out_fields = ['topographic__elevation', 
              'soil__depth', 
              'channel__width',
              'surface_water__discharge']


#Save initial condition to xarray dataset
for of in out_fields:
    ds[of][0, :, :] = mg['node'][of].reshape(mg.shape)


#%%

#Run the model

start_time = time.time()

 #output file path
ds_file = 'C:/Users/gjg882/Desktop/Projects/SpaceDynamicWidth/ModelOutput/FixedWidthTest_ctrl.nc'

#instantiate flow router
fr = PriorityFloodFlowRouter(mg, flow_metric='D8', suppress_out = True)


elapsed_time = 0

for i in range(nts):
    
    #New priority flow router component
    fr.run_one_step()
    
    #erode with space
    _ = space.run_one_step(dt=space_dt)
    
    #Layers are advected upwards due to uplift
    dz_ad = np.zeros(mg.size('node'))
    dz_ad[mg.core_nodes] = space_uplift * space_dt
    mg.at_node['bedrock__elevation'] += dz_ad
    
    #Recalculate topographic elevation to account for rock uplift
    mg.at_node['topographic__elevation'][:] = \
        mg.at_node['bedrock__elevation'][:] + mg.at_node['soil__depth'][:]

    #Update space K values
    space.K_br = mg.at_node['K_sp']
    
    if elapsed_time %save_interval== 0:
        
        ds_ind = int((elapsed_time/save_interval))
    
        for of in out_fields:
            ds[of][ds_ind, :, :] = mg['node'][of].reshape(mg.shape)
        
        print(elapsed_time, ds_ind)
        ds.to_netcdf(ds_file)
    
    elapsed_time += space_dt

end_time = time.time()
loop_time = round((end_time - start_time) / 60)
print('Loop time =', loop_time)


