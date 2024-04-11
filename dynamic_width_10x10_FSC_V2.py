# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:21:43 2023

@author: gjg882
"""

import numpy as np
import math
from matplotlib import pyplot as plt

import timeit

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
inputs = load_params('dynamic_w_inputs_10x10_V5a.txt')
#inputs = load_params('C:/Users/gjg882/Desktop/Code/SpaceDynamicWidth/dynamic_w_inputs.txt')


#%%

save_output = False #Do you want to save the model output to netcdf file?
ds_file_out = '10x10_500kyr_V5_fsc200_newq_2024.nc'


#%%

#Model time in years
space_runtime = 10000 #todo change var names to model_runtime

#grid dimensions
dx = inputs['dx']
#nx=inputs['nx']
#ny=inputs['ny']

nx=20
ny=20



space_dt = 20
space_dt_sec = space_dt * 365 * 24 * 60 * 60

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

timestep_s = 31536000  # Timestep duration, in seconds. Note that SPACE model uses 1 year.

Upliftrate_mmperyr = 1  # Uplift rate in mm/year, converted below to m/s

# Other variables, generally won't change:
rhow = 1000  # Water density, kg/m^3
rhos = 2500  # Sediment density, kg/m^3
g = 9.81  # Gravitational acceleration, m/s^2


# Calculate, convert:
# Calculate ws_init, initial sediment bed width:
V_sec = V_mperyr * 1 / (60 * 60 * 24 * 365)  # Convert effective settling velocity V to meters per second.

Upliftrate_mpers = Upliftrate_mmperyr / (1000 * 60 * 60 * 24 * 365)
#R_init = distdownstrtozeroelev_m * S_init - H_init  # Initial elevation (m above sea level)
timestep_yrs = timestep_s / (60 * 60 * 24 * 365)

yr_to_s = 60 * 60 * 24 * 365

#%%

#Make a landlab model grid

mg = RasterModelGrid((nx, ny), dx)


H = np.zeros(mg.size('node'))
H = mg.add_field('soil__depth', H, at='node') #Create a grid field for soil depth
H[mg.core_nodes] = H_init

wr = np.ones(mg.size('node'))
wr = mg.add_field("channel_bedrock__width", wr, at="node")
wr[mg.core_nodes] = wr[mg.core_nodes] * wr_init


ws = np.ones(mg.size('node'))
ws = mg.add_field("channel_sed__width", ws, at="node")


E_bank = np.zeros(mg.size('node'))
mg.add_field('bank_erosion__rate', E_bank, at='node', Clobber=True) #TODO


mg.add_zeros('bedrock_erosion__rate', at='node')

#wws = np.zeros((nx, ny)) #width at water surface, in meters
wws = np.zeros(mg.size('node'))
mg.add_field('water_surface__width', wws, at='node')

#wwa = np.zeros((nx, ny)) #meters
wwa = np.zeros(mg.size('node'))
mg.add_field('depth_averaged__width', wwa, at='node')

Fw = np.zeros(mg.size('node'))
mg.add_field('shear_stress__partitioning', Fw, at='node')

q = np.zeros(mg.size('node')) #needs to be in m**2/s
mg.add_field('normalized__discharge', q, at='node')


K_br_arr = np.ones(mg.size('node')) * K_br
mg.add_field("K_br", K_br_arr, at="node")

K_sed_arr = np.ones(mg.size('node')) * K_sed
mg.add_field("K_sed", K_sed_arr, at="node")

K_bank_arr = np.ones(mg.size('node')) * K_bank
mg.add_field("K_bank", K_bank_arr, at="node")

#h = np.ones((nx, ny)) * h_initguess
h = np.zeros(mg.size('node'))
mg.add_field('flow__depth', h, at = 'node') 

U = np.zeros(mg.size('node'))
mg.add_field('flow__velocity', U, at = 'node')

psi_bank =  np.zeros(mg.size('node'))
mg.add_field('psi_bank', psi_bank, at = 'node', clobber=True)

psi_bed =  np.zeros(mg.size('node'))
mg.add_field('psi_bed', psi_bed, at = 'node', Clobber=True)

rh = np.zeros(mg.size('node'))
mg.add_field('hydraulic__radius', rh, at='node')

#%% Create initial topography 

#Random initial roughness
np.random.seed(seed = 200) #constant seed for constant random roughness


#z = mg.node_x * slope_init + mg.node_y * slope_init
z = np.ones(mg.size('node'))
mg.add_field('topographic__elevation',z, at='node')


#shallow_func = lambda x, y: ((0.001 * x) + (0.003 * y))
#shallow_func(mg.at_node['topographic__elevation'][mg.x_of_node], mg.at_node['topographic__elevation'][mg.y_of_node]) 
 
#add some roughness 
random_field = 0.01 * np.random.randn(mg.size('node'))
z += random_field - random_field.min()

z[mg.boundary_nodes] = 0

#All boundaries are closed except outlet node
mg.set_closed_boundaries_at_grid_edges(bottom_is_closed=True,
                                       left_is_closed=True,
                                       right_is_closed=True,
                                       top_is_closed=True)

#Setting Node 0 (bottom left corner) as outlet node 
mg.set_watershed_boundary_condition_outlet_id(0, 
                                              mg.at_node['topographic__elevation'],
                                              -9999.)

#z1 = np.ones(mg.size('node'))
#z1[mg.boundary_nodes] = 0
#mg.at_node['topographic__elevation'][:]  = mg.at_node['topographic__elevation'][:]  * z1

imshow_grid(mg, mg.status_at_node, color_for_closed='blue')




#%%Use FastscapeEroder component to develop initial drainage network

fa = FlowAccumulator(mg,  flow_director='D8')
fa.run_one_step()

fsc_uplift = .001 #m/yr
fsc = FastscapeEroder(mg, K_sp=1e-4)
fsc_dt = 100
fsc_time = 0

fsc_nts = 300

dz_ad_fsc = np.zeros(mg.size('node'))
dz_ad_fsc[mg.core_nodes] = fsc_uplift * fsc_dt

for i in range (fsc_nts):
    
    z += dz_ad_fsc
    
    fa.run_one_step()
    
    fsc.run_one_step(dt=fsc_dt)
    fsc_time += fsc_dt
    

    
#%%    

#plt.close()
    
plt.figure()
imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')
plt.title('Topo after FSC')



#%%

fr = PriorityFloodFlowRouter(mg, flow_metric='D8', suppress_out = True)
fr.run_one_step()

Qw = mg.at_node['surface_water__discharge']
Qw_sec = Qw/yr_to_s
S = mg.at_node['topographic__steepest_slope']

Qw_outlet_exp = Qw[0]**0.5
Qw_outlet_exp_sec = Qw_sec[0] ** 0.5

print('initial Q**0.5=', Qw_outlet_exp)
print('initial Q_sec**0.5=', Qw_outlet_exp_sec)


#print('initial slopes:', S)
#%%

plt.figure()
imshow_grid(mg, S, colorbar_label='Slope')
plt.title('Initial Slope')


#%%

#Instantiate SPACE component
#Remember to make sure variables w/ time units are converted to seconds
space = SpaceLargeScaleEroder(mg,
           K_sed = K_sed,
           K_br = K_br,
           F_f = Ff,
           phi = phi,
           H_star = H_star,
           v_s = V_sec,
           m_sp = m_sp,
           n_sp = n_sp,
           sp_crit_sed = sp_crit_sed,
           sp_crit_br = sp_crit_br, 
           discharge_field=q)


t = np.arange(0, space_runtime+space_dt, space_dt)
nts = len(t)

save_interval = 20

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
    ws[mg.core_nodes] = wr[mg.core_nodes] + 2 * H[mg.core_nodes] /np.tan(thetarad)
    
    #mg.at_node['channel_sed__width'][:] = mg.at_node['channel_bedrock__width'][:] 
    #+ 2 * mg.at_node['soil__depth'][:] / np.tan(thetarad)
    
    if np.any(ws <= 0) == True:
        print('too narrow!')
        input('press any key to continue')

#calculate new width at water surface
def calc_wws(mg): 
    wws[mg.core_nodes] = ws[mg.core_nodes] + 2 * h[mg.core_nodes] / np.tan(thetarad)
    
    if wws.any() <= 0:
        print('too narrow!')
        input('press any key to continue')

#calculate new depth-averaged width    
def calc_wwa(mg):
    wwa[mg.core_nodes] = (wws[mg.core_nodes] + ws[mg.core_nodes]) / 2
    

def calc_rh(mg):
    rh[mg.core_nodes] = h[mg.core_nodes] * (ws[mg.core_nodes] + 
                                            h[mg.core_nodes] / 
                                            np.tan(thetarad)) / (ws[mg.core_nodes] + 2  * h[mg.core_nodes] / math.sin(thetarad)) 
    
def calc_U(mg):
    
    rh_power = rh[mg.core_nodes] ** (2/3)
    
    S_power = S[mg.core_nodes] ** 0.5
    
    U[mg.core_nodes] = (1 / manning_n) * rh_power * S_power
    
    
def calc_psi_bed(mg):
    psi_bed[mg.core_nodes] = rhow * g * (1 - Fw[mg.core_nodes]) / 2 * (1 + wws[mg.core_nodes] * np.tan(thetarad) / 
         (wws[mg.core_nodes] * np.tan(thetarad) - 2 * h[mg.core_nodes]))  
    

def calc_psi_bank(mg):
    psi_bank[mg.core_nodes]  = rhow * g * Fw[mg.core_nodes] /  2 * (wws[mg.core_nodes] / h[mg.core_nodes] * np.sin(thetarad) - np.cos(thetarad))
    
    


#%%
plt.figure()
imshow_grid(mg, 'channel_sed__width')
plt.title('sed width before calculating')


#%%

#calculate initial sediment width
calc_ws(mg, thetarad)

plt.close()

print('initial Ws/Q_sec**0.5=',  (ws[0] / Qw_outlet_exp_sec)) 

#%%

plt.figure()
imshow_grid(mg, 'channel_sed__width')
plt.title('sed width after calculating')

#%%

calc_wws(mg)
calc_wwa(mg)


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

#Save model initial condition to xarray output
for of in out_fields:
    ds[of][0, :, :] = mg['node'][of].reshape(mg.shape)
    


#%%

#array of uplift
dz_ad = np.zeros(mg.size('node'))
dz_ad[mg.core_nodes] = Upliftrate_mpers * space_dt_sec


elapsed_time_yrs = 0
elapsed_time_sec = 0
starttime = timeit.default_timer()

#Upper and lower bounds for h in Qwg
lower_bound = 0
upper_bound = 100

#%%
dwrdt = np.zeros(mg.size('node'))


z[mg.core_nodes] = 100 #FOR TESTING ONLY


#%%NEW main model loop

for i in range(1):
    
    fr.run_one_step()
    
    Qw_sec = Qw/yr_to_s
    
    nodes_ordered = mg.at_node["flow__upstream_node_order"]
    
    #calculate flow depth at each node
    for j in range(len(nodes_ordered) - 1, -1, -1):
        func_args = (manning_n, ws[j], thetarad, S[j], Qw_sec[j])
        
        if mg.status_at_node[j] == 0: #operate on core nodes only
        
            if S[j] == 0:
                print('S=0 at', i, j)

            else:
               h[j] = scipy.optimize.brentq(Qwg, a=lower_bound, b=upper_bound, args=func_args, disp=True, maxiter=100)
               print(h)
              
    #calculate hydraulic radius using function defined above
    calc_rh(mg)

    #calculate velocity
    calc_U(mg)
    
    q[mg.core_nodes] = Qw_sec[mg.core_nodes] / wwa[mg.core_nodes]
    
    Fw[mg.core_nodes] = 1.78 * (wws[mg.core_nodes] / h[mg.core_nodes] * np.sin(thetarad) - 2 * np.cos(thetarad) + 1.5) ** (-1.4)
    
    calc_psi_bed(mg)
    
    calc_psi_bank(mg)
    
    #update K values in space eqns using psi bed coefficient
    space.K_br = mg.at_node['K_br'][:] * psi_bed
    space.K_sed = mg.at_node['K_sed'][:] * psi_bed
    
    
    #Calculate bank erosion rate
    E_bank[mg.core_nodes] = K_bank_arr[mg.core_nodes] * psi_bank[mg.core_nodes] * q[mg.core_nodes] * (S[mg.core_nodes] ** n_sp) - omegacbank
    
    #erode channel bed
    _ = space.run_one_step(dt=space_dt_sec)
    
    #Get bedrock and sediment erosion rates from space
    E_rock = space.Er
    E_sed = space.Es
    
    #Calculate total bank erosion and update width
    dwrdt[mg.core_nodes] = 2 * (E_bank[mg.core_nodes] / np.sin(thetarad) - E_rock[mg.core_nodes] / np.tan(thetarad))
    wr[mg.core_nodes] += ( dwrdt[mg.core_nodes] * space_dt_sec ) # UPDATED bedrock width
    
    #Update widths based on flow 
    calc_ws(mg, thetarad) #sediment width
    calc_wws(mg)   #water surface
    calc_wwa(mg) #depth averaged
    
    #Uplift
    z += dz_ad
    
    if elapsed_time_yrs %save_interval== 0:
        
        ds_ind = int((elapsed_time_yrs/save_interval))
        
        time_diff = timeit.default_timer() - starttime
        time_diff_minutes = np.round(time_diff/60, decimals=2)
        
        if save_output == True:
    
            for of in out_fields:
                ds[of][ds_ind, :, :] = mg['node'][of].reshape(mg.shape)
                ds.to_netcdf(ds_file_out)
        
        #print(elapsed_time, ds_ind)
        print(elapsed_time_yrs, ds_ind, time_diff_minutes)
        
    elapsed_time_yrs += space_dt
    elapsed_time_sec += space_dt_sec
    
    elapsed_time_test = elapsed_time_sec / yr_to_s
    print(elapsed_time_yrs, elapsed_time_test)
    

#%%

z0 = ds.topographic__elevation.sel(time=0).values

z0=z0.reshape(mg.size('node'))

z_diff = z - z0

#%%

plt.figure()
imshow_grid(mg, 'normalized__discharge', colorbar_label='Unit Discharge(m2/s)')
plt.title('Final q')

plt.figure()
imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')    
plt.title('FInal Topo')


#%%Main model loop
# =============================================================================
# =============================================================================
# =============================================================================
# 
# for i in range(nts):
#     
#     #flow routing
#     fr.run_one_step()
#     
#     
#     #iterate through nodes, upstream to downstream to calculate flow depth
#     #not actually sure if order is important here
#     nodes_ordered = mg.at_node["flow__upstream_node_order"]
# 
#     for j in range(len(nodes_ordered) - 1, -1, -1):
# 
#         #func_args = (manning_n, ws[j], thetarad, S[j], Qw[j])
#         func_args = (manning_n, ws[j], thetarad, S[j], Qw[j])
#         #func_args = (manning_n, ws[j], thetarad, S[j], 100)
#         
#         
#         if mg.status_at_node[j] == 0: #operate on core nodes only
#         
#             if S[j] == 0:
#                 print('S=0 at', i, j)
#                 #input('press enter to continue')
#         
#             #print('i=', i, 'j=', j, 'S=', S[j], 'Qw=', Qw[j], 'ws=', ws[j])
#             
#             else:
#                 mg.at_node['flow__depth'][j] = scipy.optimize.brentq(Qwg, a=lower_bound, b=upper_bound, args=func_args, disp=True, maxiter=100)
#                 #print('ws=', func_args[1], 'S=', func_args[3], 'Qw=', func_args[4], 'j=', j)
#     
#             
#     
#         
#     #calculate width at water surface
#     mg.at_node['water_surface__width'][:] = mg.at_node['channel_sed__width'][:] + 2 * mg.at_node['flow__depth'][:] / np.tan(thetarad)
#     
#     #Calculate depth-averaged width
#     mg.at_node['depth_averaged__width'][:] = (mg.at_node['water_surface__width'][:] + mg.at_node['channel_sed__width'][:]) / 2
#     
#     #Caluclate normalized discharge in m2/s
#     mg.at_node['normalized__discharge'][:] = mg.at_node['surface_water__discharge'][:] / mg.at_node['depth_averaged__width'][:]
# 
#     #print(mg.at_node['normalized__discharge'])
# 
#     #Calculate Fw,shear stress partitioning between bed and banks
#     mg.at_node['shear_stress__partitioning'][:] = 1.78 * (mg.at_node['water_surface__width'][:] / mg.at_node['flow__depth'][:] * np.sin(thetarad) - 2 * np.cos(thetarad) + 1.5) ** (-1.4)
#       
#     
#     #Next calculate width coefficients, Psi bed and Psi bank
#     #psibed = rhow * g * (1 - Fw) / 2 * (1 + wws * np.tan(thetarad) / (wws * np.tan(thetarad) - 2 * h))
#     mg.at_node['psi_bed'][:] = rhow * g * (1 - mg.at_node['shear_stress__partitioning'][:]) / 2 * (1 + mg.at_node['water_surface__width'][:] * np.tan(thetarad) / (mg.at_node['water_surface__width'][:] * np.tan(thetarad) - 2 * mg.at_node['flow__depth'][:]))
#     
#     #psibank = rhow * g * Fw / 2 * (wws / h * np.sin(thetarad) - np.cos(thetarad))
#     mg.at_node['psi_bank'][:] = rhow * g * mg.at_node['shear_stress__partitioning'][:] / 2 * (mg.at_node['water_surface__width'][:] / mg.at_node['flow__depth'][:] * np.sin(thetarad) - np.cos(thetarad))
#     
#     
#     #Multiply erodibilities by width coefficient
#     space.K_br = mg.at_node['K_sp'][:] * mg.at_node['psi_bed'][:]
#     space.K_sed = mg.at_node['K_sed'][:] * mg.at_node['psi_bank'][:]
#     
#     #erode with space
#     _ = space.run_one_step(dt=space_dt_sec)
#     
#     #Calculate bank erosion
#     mg.at_node['bank_erosion__rate'][:] = mg.at_node['K_bank'][:] * mg.at_node['psi_bank'][:] * mg.at_node['normalized__discharge'][:] * (mg.at_node['topographic__steepest_slope'][:]**n_sp)- omegacbank
#     
#     #Bedrock erosion rate from space
#     #mg.at_node['bedrock_erosion__rate'][:] = space._Er.reshape(mg.shape[0], mg.shape[1])
#     
#     mg.at_node['bedrock_erosion__rate'][:] = space._Er
#     
#     #Calculate change in channel bedrock width at each node
#     dwrdt = (mg.at_node['bank_erosion__rate'][:] / math.sin(thetarad) - mg.at_node['bedrock_erosion__rate'][:]/math.tan(thetarad)) * 2 
#     mg.at_node['channel_bedrock__width'][:] += dwrdt * space_dt_sec
#     
#     #print(mg.at_node['bedrock_erosion__rate'])
#     
#     if  mg.at_node['channel_bedrock__width'].any() < wr_min:
#         raise Exception("Channel is too narrow")
#     
#     if  mg.at_node['channel_bedrock__width'].any() > dx:
#         raise Exception("Channel width is greater than one grid cell")
#     
#     #Update channel sediment width, ws
#     mg.at_node['channel_sed__width'][:] = mg.at_node['channel_bedrock__width'][:] + 2 * mg.at_node['soil__depth'][:] / np.tan(thetarad)
#     
#     #uplift
#     mg.at_node['topographic__elevation'][:] += dz_ad
#     
# 
#     if elapsed_time %save_interval== 0:
# 
#         
#         ds_ind = int((elapsed_time/save_interval))
#         
#         time_diff = timeit.default_timer() - starttime
#         time_diff_minutes = np.round(time_diff/60, decimals=2)
#         
#         if save_output == True:
#     
#             for of in out_fields:
#                 ds[of][ds_ind, :, :] = mg['node'][of].reshape(mg.shape)
#                 ds.to_netcdf(ds_file_out)
#         
#         #print(elapsed_time, ds_ind)
#         print(elapsed_time, ds_ind, time_diff_minutes)
#         
#     elapsed_time += space_dt
# 
#     
# 
#     
#     
# #%%
# 
# plt.figure()
# imshow_grid(mg, 'channel_bedrock__width', colorbar_label='Channel Width(m)')    
# 
# #%%
# 
# plt.figure()
# imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')    
#     
# #%%
# 
# # =============================================================================
# # print(wr[5])
# # print(mg.at_node['channel_bedrock__width'][5])
# # print(mg.at_node['topographic__elevation'][5])
# # print(mg.at_node['flow__depth'])
# # =============================================================================
# 
# sed_flux_in = mg.at_node['sediment__influx']
# 
# #%%
# 
# test_depth = ds.isel()
# =============================================================================
# =============================================================================
# 
# =============================================================================
