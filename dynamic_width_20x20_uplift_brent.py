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

from lague_stress_funcs import Stress_Funcs

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

#%%
inputs = load_params('dynamic_w_inputs_20x20_brent4.txt')
#inputs = load_params('C:/Users/gjg882/Desktop/Code/SpaceDynamicWidth/dynamic_w_inputs.txt')


#%%

ds_file_out = 'SDW_20x20_1myr_dt50_fill_medK.nc'

#%%

#Model time in years
space_runtime = 1000000 #todo change var names to model_runtime


#grid dimensions

dx = inputs['dx']
nx=inputs['nx']
ny=inputs['ny']

space_dt = inputs['dt']

theta_deg = inputs['theta_deg']
thetarad = math.radians(theta_deg) #convert to radians

manning_n = inputs['mannings_n'] #from Lague
rho_w = inputs['rho_w'] #Density of water, kg/m3
rho_sed = inputs["rho_sed"] # Density of sediment, kg/m^3

H_init = inputs["H_init"] #Initial sediment depth on bed, in m

wr_init = inputs["wr_init"] # Bedrock bed width, initial, in m.
wr_min = inputs['wr_min']  # Minimum bedrock bed width, code just stops if it gets this small.

U_initguess = inputs['U_initguess']  # Initial guess of avg velocity, in m/s, just to start the iteration process.

h_initguess = 1.0 #TODO probably want to change this later

K_br = inputs["Kr"]
K_sed = inputs["Ks"]
K_bank = inputs["Kbank"]
n_sp = inputs["n_sp"]
m_sp = inputs["m_sp"] #NOTE THAT THIS IS CURRENTLY SET TO ONE, gets subsumed into new K calculation
H_star = inputs["Hstar"]
sp_crit_br = inputs["omegacr"]
sp_crit_sed = inputs["omegacs"]
omegacbank = inputs["omegacbank"]
v_s = inputs["V_mperyr"]
phi = inputs["porosity"]
Ff = inputs["Ff"]

space_uplift = inputs['space_uplift']

V_mperyr = v_s


# Other variables, generally won't change:
rhow = 1000  # Water density, kg/m^3
rhos = 2500  # Sediment density, kg/m^3
g = 9.81  # Gravitational acceleration, m/s^2




#%%

#Make a landlab model grid

mg = RasterModelGrid((nx, ny), dx)


mg.add_zeros('node', 'soil__depth') #Create a grid field for soil depth
mg.at_node['soil__depth'][:] = H_init

wr = np.ones((nx, ny)) * wr_init
wr = mg.add_field("channel_bedrock__width", wr, at="node")

#ws = np.zeros((nx, ny)) 
mg.add_zeros("channel_sed__width", at="node")
ws = mg.at_node['channel_sed__width']

mg.add_zeros('psi_bed', at='node')
mg.add_zeros('psi_bank', at='node')

mg.add_zeros('bank_erosion__rate', at='node') #TODO
mg.add_zeros('bedrock_erosion__rate', at='node')
mg.add_zeros('water_surface__width', at='node')
mg.add_zeros('depth_averaged__width', at='node')

mg.add_zeros('shear_stress__partitioning', at='node')

mg.add_zeros('normalized__discharge', at='node')

K_sp_arr = np.ones((nx, ny)) * K_br
mg.add_field("K_sp", K_sp_arr, at="node")

K_sed_arr = np.ones((nx, ny)) * K_sed
mg.add_field("K_sed", K_sed_arr, at="node")

K_bank_arr = np.ones((nx, ny)) * K_bank
mg.add_field("K_bank", K_bank_arr, at="node")


h = np.ones((nx, ny)) * h_initguess
mg.add_field('flow__depth', h, at = 'node') 

#%% Create initial topography 

#Random initial roughness
np.random.seed(seed = 200) #constant seed for constant random roughness

z = mg.node_x * .015 + mg.node_y * .015
mg.add_field('topographic__elevation',z, at='node')
z[mg.boundary_nodes] = 0

#shallow_func = lambda x, y: ((0.001 * x) + (0.003 * y))
#shallow_func(mg.at_node['topographic__elevation'][mg.x_of_node], mg.at_node['topographic__elevation'][mg.y_of_node]) 
 

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


#%%

fr = PriorityFloodFlowRouter(mg, flow_metric='D8', runoff_rate=0.002, depression_handler='fill', suppress_out = True)
fr.run_one_step()

Qw = mg.at_node['surface_water__discharge']
S = mg.at_node['topographic__steepest_slope']

Qw_outlet_exp = Qw[0]**0.5
print('initial Q**0.5=', Qw_outlet_exp)

#print('initial slopes:', S)
#%%


plt.figure()
imshow_grid(mg, S, colorbar_label='Slope')
plt.title('Initial Slope')


#%%Use FastscapeEroder component to develop initial drainage network


fsc = FastscapeEroder(mg)
fsc_dt = 100

fa = FlowAccumulator(mg)

for i in range (25):
    fa.run_one_step()
    fsc.run_one_step(fsc_dt)


plt.figure()
imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')
plt.title('Topo after FSC')


#%%
space = SpaceLargeScaleEroder(mg,
           K_sed =K_sed,
           K_br = K_br,
           F_f = Ff,
           phi = phi,
           H_star = H_star,
           v_s = v_s,
           m_sp = m_sp,
           n_sp = n_sp,
           sp_crit_sed = sp_crit_sed,
           sp_crit_br = sp_crit_br)





t = np.arange(0, space_runtime+space_dt, space_dt)
nts = len(t)

save_interval = 1000

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
            
        'channel_sed__width':
            (('time', 'y', 'x'), np.empty((out_count, mg.shape[0], mg.shape[1])), {
                'units': 'm',
                'long_name': 'Channel Sediment Width'
            }),
        
        'bedrock_erosion__rate' : (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': 'meters/yr',
                'long_name': 'Bedrock Erosion Rate',
                
            }),    
        
        'bank_erosion__rate' : (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': 'meters/yr',
                'long_name': 'Bank Erosion Rate',
            
            }),
                
        'flow__depth' : (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': 'meters',
                'long_name': 'Flow Depth Manning Eqn',
                
            }),   
        
        'surface_water__discharge' : (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': 'm**3/s',
                'long_name': 'Discharge',
                
            }), 
        
        'topographic__steepest_slope' : (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': '-',
                'long_name': 'Topographic Steepest Slope',
                
            }),     
        
        'channel_bedrock__width' : (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': 'm',
                'long_name': 'Channel Bedrock Width',
                
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
              'channel_sed__width',
              'bank_erosion__rate',
              'bedrock_erosion__rate',
              'flow__depth',
              'surface_water__discharge',
              'topographic__steepest_slope',
              'channel_bedrock__width']
              
              
              
              


#%%Define a funciton to calculate estimated discharge for a given flow depth, h, then compare estimated discharge to actual

#calculate new width of sediment based on soil depth, bank ankle, and bedrock width
def calc_ws(mg, thetarad):
    mg.at_node['channel_sed__width'][:] = mg.at_node['channel_bedrock__width'][:] 
    + 2 * mg.at_node['soil__depth'][:] / np.tan(thetarad)

#%%
plt.figure()
imshow_grid(mg, 'channel_sed__width')
plt.title('sed width before calculating')


#%%

#calculate initial sediment width
calc_ws(mg, thetarad)

print('initial Ws/Q**0.5=',  (ws[0] / Qw_outlet_exp)) 

plt.figure()
imshow_grid(mg, 'channel_sed__width')
plt.title('sed width after calculating')


#%%


#function to guess discharge based on flow depth h, manning, width of sediment in channel, previous discharge 
#model uses root finding function to find value of h that gives this function the smallest Q_error
def Qwg(h, manning_n, ws, thetarad, S, Qw_target):
    
    Qwg = (1 / manning_n) * ((h * (ws + h / math.tan(thetarad))) ** (5 / 3)) * (
                (ws + 2 * h / math.sin(thetarad)) ** (-2 / 3)) * S ** 0.5

    #Q_error = np.abs(Qw_target - Qwg) #For Newton Solver
    Q_error = (Qwg - Qw_target) #For Brent solver 
    
    return Q_error



#%%


#TODO - TRY USING BRENT INSTEAD OF NEWTON SOLVER
#SEE https://waterprogramming.wordpress.com/2016/08/18/root-finding-in-matlab-r-python-and-c/'
#SEE ALSO https://www.engr.scu.edu/~emaurer/hydr-watres-book/flow-in-open-channels.html


def calc_ws(mg, thetarad):
    mg.at_node['channel_sed__width'][:] = mg.at_node['channel_bedrock__width'][:] + 2 * mg.at_node['soil__depth'][:] / np.tan(thetarad)



#%%

#Save model initial condition to xarray output
for of in out_fields:
    ds[of][0, :, :] = mg['node'][of].reshape(mg.shape)


#%%

#Main model loop

elapsed_time = 0
starttime = timeit.default_timer()

#Upper and lower bounds 
lower_bound = 0
#upper_bound = dx-1
upper_bound = 10000 #for troubleshooting purposes ONLY 

for i in range(nts):
    
    
    #flow routing
    fr.run_one_step()
        
    #iterate through nodes, upstream to downstream to calculate flow depth
    #not actually sure if order is important here
    nodes_ordered = mg.at_node["flow__upstream_node_order"]


    for j in range(len(nodes_ordered) - 1, -1, -1):

        func_args = (manning_n, ws[j], thetarad, S[j], Qw[j])
        
        
        
        #if j == 5 or j == 6:
            #print(j, 'ws=', func_args[1], 'S=', func_args[3], 'Qw=', func_args[4] )
        
        if mg.status_at_node[j] == 0: #operate on core nodes only
            
            #print('i=', i, 'j=', j, 'S=', S[j], 'Qw=', Qw[j], 'ws=', ws[j])
            
            if S[j] == 0:
                print('S=0 at', i, j)
                #pass
            
            else:

                mg.at_node['flow__depth'][j] = scipy.optimize.brentq(Qwg, a=lower_bound, b=upper_bound, args=func_args, disp=True, maxiter=100)
                #print('ws=', func_args[1], 'S=', func_args[3], 'Qw=', func_args[4], 'j=', j)
                
                #print('flow depth =', mg.at_node['flow__depth'][j] )
                #print('soil depth =', mg.at_node['soil__depth'][j] )
        
    #calculate width at water surface
    mg.at_node['water_surface__width'][:] = mg.at_node['channel_sed__width'][:] + 2 * mg.at_node['flow__depth'][:] / np.tan(thetarad)
    
    
    #Calculate depth-averaged width
    mg.at_node['depth_averaged__width'][:] = (mg.at_node['water_surface__width'][:] + mg.at_node['channel_sed__width'][:]) / 2
    
    #Caluclate normalized discharge
    mg.at_node['normalized__discharge'][:] = mg.at_node['surface_water__discharge'][:] / mg.at_node['depth_averaged__width'][:]

    #Calculate Fw,shear stress partitioning between bed and banks
    mg.at_node['shear_stress__partitioning'][:] = 1.78 * (mg.at_node['water_surface__width'][:] / mg.at_node['flow__depth'][:] * np.sin(thetarad) - 2 * np.cos(thetarad) + 1.5) ** (-1.4)
     
    
    #Next calculate width coefficients, Psi bed and Psi bank
    #psibed = rhow * g * (1 - Fw) / 2 * (1 + wws * np.tan(thetarad) / (wws * np.tan(thetarad) - 2 * h))
    mg.at_node['psi_bed'][:] = rhow * g * (1 - mg.at_node['shear_stress__partitioning'][:]) / 2 * (1 + mg.at_node['water_surface__width'][:] * np.tan(thetarad) / (mg.at_node['water_surface__width'][:] * np.tan(thetarad) - 2 * mg.at_node['flow__depth'][:]))
    
    #psibank = rhow * g * Fw / 2 * (wws / h * np.sin(thetarad) - np.cos(thetarad))
    mg.at_node['psi_bank'][:] = rhow * g * mg.at_node['shear_stress__partitioning'][:] / 2 * (mg.at_node['water_surface__width'][:] / mg.at_node['flow__depth'][:] * np.sin(thetarad) - np.cos(thetarad))
    
    
    #Multiply erodibilities by width coefficient
    space.K_br = mg.at_node['K_sp'][:] * mg.at_node['psi_bed'][:]
    space.K_sed = mg.at_node['K_sed'][:] * mg.at_node['psi_bed'][:]
    
    #erode with space
    _ = space.run_one_step(dt=space_dt)
    
    #Calculate bank erosion
    mg.at_node['bank_erosion__rate'][:] = mg.at_node['K_bank'][:] * mg.at_node['psi_bank'][:] * mg.at_node['surface_water__discharge'][:] * (mg.at_node['topographic__steepest_slope'][:]**n_sp)- omegacbank
    
    #Bedrock erosion rate from space
    #mg.at_node['bedrock_erosion__rate'][:] = space._Er.reshape(mg.shape[0], mg.shape[1])
    mg.at_node['bedrock_erosion__rate'][:] = space._Er
    
    #Calculate change in channel bedrock width at each node
    dwrdt = (mg.at_node['bank_erosion__rate'][:] / math.sin(thetarad) - mg.at_node['bedrock_erosion__rate'][:]/math.tan(thetarad)) * 2 
    mg.at_node['channel_bedrock__width'][:] += dwrdt * space_dt
    
    if  mg.at_node['channel_bedrock__width'].any() < wr_min:
        raise Exception("Channel is too narrow")
    
    if  mg.at_node['channel_bedrock__width'].any() > dx:
        raise Exception("Channel width is greater than one grid cell")
    
    #Update channel sediment width, ws
    mg.at_node['channel_sed__width'][:] = mg.at_node['channel_bedrock__width'][:] + 2 * mg.at_node['soil__depth'][:] / np.tan(thetarad)
    
    
    dz_ad = np.zeros(mg.size('node'))
    dz_ad[mg.core_nodes] = space_uplift * space_dt
    mg.at_node['topographic__elevation'] += dz_ad
    
    

    if elapsed_time %save_interval== 0:
        
        time_diff = timeit.default_timer() - starttime
        time_diff_minutes = np.round(time_diff/60, decimals=2)
        
        ds_ind = int((elapsed_time/save_interval))
    
        for of in out_fields:
            ds[of][ds_ind, :, :] = mg['node'][of].reshape(mg.shape)
        
        print(elapsed_time, ds_ind, time_diff_minutes)
        ds.to_netcdf(ds_file_out)
        

    elapsed_time += space_dt

    

    
    
#%%

plt.figure()
imshow_grid(mg, 'channel_bedrock__width', colorbar_label='Channel Width(m)')    

#%%

plt.figure()
imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')    
    
#%%

# =============================================================================
# print(wr[5])
# print(mg.at_node['channel_bedrock__width'][5])
# print(mg.at_node['topographic__elevation'][5])
# print(mg.at_node['flow__depth'])
# =============================================================================

sed_flux_in = mg.at_node['sediment__influx']

#%%

test_depth = ds.isel()