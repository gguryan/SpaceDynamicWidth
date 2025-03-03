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
                                SpaceLargeScaleEroder,
                                NormalFault)

#%%inputs and outputs

#inputs = load_params('normal_fault_inputs.txt')
#inputs = load_params('C:/Users/gjg882/Desktop/Code/SpaceDynamicWidth/dynamic_w_inputs.txt')

ds_init = xr.open_dataset( 'ModelOutput/SDW_20x20_e-14_2_highU_4.nc')

#path to save netcdf file to 
ds_file_out = 'ModelOutput/SDW_20x20_NormalFault_2.nc'

ds_attrs = ds_init.attrs

#%%




#for converting units throughout - todo update to account for leap days, not important rn 
sec_per_yr =  60 * 60 * 24 * 365

#Model time in years
space_runtime = 100000
space_runtime_sec = space_runtime * sec_per_yr


#grid dimensions

dx = inputs['dx'] #meters
nx=inputs['nx'] 
ny=inputs['ny']


space_dt = inputs['dt'] #years
space_dt_sec = space_dt * sec_per_yr

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
V_mperyr = inputs["V_mperyr"]
phi = inputs["porosity"]
Ff = inputs["Ff"]


Upliftrate_mperyr = inputs['space_uplift']
space_uplift_sec = Upliftrate_mperyr / sec_per_yr

v_seconds = V_mperyr / sec_per_yr



# Other variables, generally won't change:
rhow = 1000  # Water density, kg/m^3
rhos = 2500  # Sediment density, kg/m^3
g = 9.81  # Gravitational acceleration, m/s^2







#%%

#Make a landlab model grid
mg = RasterModelGrid((nx, ny), dx)


#create model grid fields

H_soil = mg.add_zeros('node', 'soil__depth') #Create a grid field for soil depth
mg.at_node['soil__depth'][:] = H_init

wr = np.zeros((nx, ny)) 
wr = mg.add_field("channel_bedrock__width", wr, at="node")

#ws = np.zeros((nx, ny)) 
ws = mg.add_zeros("channel_sed__width", at="node")

psi_bed = mg.add_zeros('psi_bed', at='node')
psi_bank = mg.add_zeros('psi_bank', at='node')

bank_er = mg.add_zeros('bank_erosion__rate', at='node') 
bed_er = mg.add_zeros('bedrock_erosion__rate', at='node')
wws = mg.add_zeros('water_surface__width', at='node')
w_avg = mg.add_zeros('depth_averaged__width', at='node')

Fw = mg.add_zeros('shear_stress__partitioning', at='node')

Q_sec = mg.add_zeros('discharge__seconds', at='node')

q_norm = mg.add_zeros('normalized__discharge_sec', at='node') #in m2/sec (NOT YEARS, conversion is already done)

K_sp_arr = np.ones((nx, ny)) * K_br
mg.add_field("K_sp", K_sp_arr, at="node")

K_sed_arr = np.ones((nx, ny)) * K_sed
mg.add_field("K_sed", K_sed_arr, at="node")

K_bank_arr = np.ones((nx, ny)) * K_bank
mg.add_field("K_bank", K_bank_arr, at="node")


flow_depth = np.ones((nx, ny)) * h_initguess
h = mg.add_field('flow__depth', flow_depth, at = 'node') 



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


#imshow_grid(mg, mg.status_at_node, color_for_closed='blue')

#%%

plt.figure()
imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')
plt.title('Initial Topo')



#%%Use FastscapeEroder component to develop initial drainage network


nf = NormalFault(mg, fault_trace = {'x1': 0,
                                      'y1': 2000, 
                                      'y2': 100, 
                                      'x2': 3000}, include_boundaries=True)

plt.figure()
imshow_grid(mg, nf.faulted_nodes.astype(int), cmap='viridis')       


#%%


fa = FlowAccumulator(mg,  flow_director='D8')
fa.run_one_step()

fsc_uplift = .001 #m/yr
fsc = FastscapeEroder(mg, K_sp=1e-4)
fsc_dt = 100
fsc_time = 0

fsc_nts = 2000

for i in range (fsc_nts):
    
    
    dz_ad = np.zeros(mg.size('node'))
    dz_ad[mg.core_nodes] = fsc_uplift * fsc_dt
    mg.at_node['topographic__elevation'] += dz_ad
    
    fa.run_one_step()
    
    fsc.run_one_step(dt=fsc_dt)
    fsc_time += fsc_dt
    

        

#%%

plt.figure()
imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')
plt.title('Topo after FSC')



#%%Set up soil for SPACE



mg.at_node["bedrock__elevation"] = mg.at_node["topographic__elevation"][:] - mg.at_node["soil__depth"][:]



#%%


Qw = mg.at_node['surface_water__discharge'] #m3/yr
Q_sec_init = Qw / sec_per_yr



S = mg.at_node['topographic__steepest_slope']


#print('initial slopes:', S)
#%%


plt.figure()
imshow_grid(mg, S, colorbar_label='Slope')
plt.title('Initial Slope')




#%%

fa = PriorityFloodFlowRouter(mg)


space = SpaceLargeScaleEroder(mg,
            K_sed =K_sed,
            K_br = K_br,
            F_f = Ff,
            phi = phi,
            H_star = H_star,
            v_s = v_seconds,
            m_sp = m_sp,
            n_sp = n_sp,
            sp_crit_sed = sp_crit_sed,
            sp_crit_br = sp_crit_br,
            discharge_field='surface_water__discharge')

#instantitate space
# =============================================================================
# space = Space(mg,
#            K_sed =K_sed,
#            K_br = K_br,
#            F_f = Ff,
#            phi = phi,
#            H_star = H_star,
#            v_s = v_seconds,
#            m_sp = m_sp,
#            n_sp = n_sp,
#            sp_crit_sed = sp_crit_sed,
#            sp_crit_br = sp_crit_br,
#            discharge_field='surface_water__discharge')
# =============================================================================


#%%

t = np.arange(0, space_runtime_sec+space_dt_sec, space_dt_sec)            
nts = len(t)

t_yrs = t/sec_per_yr

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
        
        'soil__depth' : (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': 'meters',
                'long_name': 'sediment thickness',
                
            }),   
        
        'surface_water__discharge' : (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': 'm**2/s',
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
              'channel_bedrock__width', 
              'soil__depth']
             

#%%Define functions


#function to guess discharge based on flow depth h, manning, width of sediment in channel, previous discharge 
#model uses root finding function to find value of h that gives this function the smallest Q_error
def Qwg(h, manning_n, ws, thetarad, S, Qw_target):
    
    #guess flow depth
    Qwg = (1 / manning_n) * ((h * (ws + h / math.tan(thetarad))) ** (5 / 3)) * (
                (ws + 2 * h / math.sin(thetarad)) ** (-2 / 3)) * S ** 0.5

    Q_error = (Qwg - Qw_target) #For Brent solver 
    
    return Q_error


#function calculate new width of sediment based on soil depth, bank angle, and bedrock width
def calc_ws(mg, thetarad):
    mg.at_node['channel_sed__width'][:] = mg.at_node['channel_bedrock__width'][:] 
    + 2 * mg.at_node['soil__depth'][:] / np.tan(thetarad)
    
    
def change_dir(E_bed, E_bank, thetarad):
    ratio = E_bank / np.cos(thetarad)
    statuses = np.where(E_bed > ratio, 'narrowing', np.where(E_bed < ratio, 'widening', 'static'))
    return statuses

#%%Calculate initial channel geometry

#start w width scaled to drainage area
#TODO for development purposes ONLY
#Convert drainage area to KM sqared for this

da_scaling = 10000

wr[:] = (mg.at_node['drainage_area'] / da_scaling )**0.5

#%%

#calculate initial sediment width
calc_ws(mg, thetarad)

         
#calculate width at water surface
wws[:] = mg.at_node['channel_sed__width'][:] + 2 * h_initguess / np.tan(thetarad)
    
    
#Calculate depth-averaged width
w_avg[:] = (wws + mg.at_node['channel_sed__width'][:]) / 2



#%%

#Save model initial condition/time zero to xarray dataset
for of in out_fields:
    ds[of][0, :, :] = mg['node'][of].reshape(mg.shape)


    

#%% Main model loop

#runoff_manual = 10000 #manual multiplier on discharge 20x20
runoff_manual = 100 #manual multiplier on discharge

elapsed_time = 0
elapsed_time_yrs= 0
total_U = 0
starttime = timeit.default_timer()

#Upper and lower bounds on flow depth - used for root finding function
lower_bound = 0
upper_bound = dx-1 #should probably be lower, doesn't matter in current parameter space

out_node = nx + 1



#for i in range(10000):
for i in range(nts):
    
    nf.run_one_step(dt=space_dt)
    
    #flow routing    
    fa.run_one_step()
    
    
    #convert discharge from m3/yr to m3/sec, manually add in "runoff rate" / multiplier to get reasonable discharge in m3/sec
    Q_sec[:] = Qw[:] / sec_per_yr * runoff_manual
    
    #calculate normalized discharge in m2/sec by dividing by avg channel width
    mg.at_node['normalized__discharge_sec'][:] = (Q_sec[:] / w_avg[:])
    
        
    #iterate through nodes, upstream to downstream to calculate flow depth
    #not actually sure if order is important here
    nodes_ordered = mg.at_node["flow__upstream_node_order"]


    for j in range(len(nodes_ordered) - 1, -1, -1):

        #inputs for function to iterate for flow depth
        func_args = (manning_n, ws[j], thetarad, S[j], Q_sec[j])

        
        if mg.status_at_node[j] == 0: #operate on core nodes only
            
            
            if S[j] == 0: #don't calculate flow depth for nodes with zero slope - it breaks the function 
                #print('S=0 at', i, j)
                pass
            
            
            else:
                
                #calculate flow depth [m]
                mg.at_node['flow__depth'][j] = scipy.optimize.brentq(Qwg, a=lower_bound, b=upper_bound, args=func_args, disp=True, maxiter=100)

        
    #calculate width at water surface
    mg.at_node['water_surface__width'][:] = mg.at_node['channel_sed__width'][:] + 2 * mg.at_node['flow__depth'][:] / np.tan(thetarad)
    
    
    #Calculate depth-averaged width
    mg.at_node['depth_averaged__width'][:] = (mg.at_node['water_surface__width'][:] + mg.at_node['channel_sed__width'][:]) / 2
    

    #Calculate Fw, term for shear stress partitioning between bed and banks
    mg.at_node['shear_stress__partitioning'][:] = 1.78 * (mg.at_node['water_surface__width'][:] / mg.at_node['flow__depth'][:] * np.sin(thetarad) - 2 * np.cos(thetarad) + 1.5) ** (-1.4)
     
    
    #Next calculate width coefficient for bed
    mg.at_node['psi_bed'][:] = rhow * g * (1 - mg.at_node['shear_stress__partitioning'][:]) / 2 * (1 + mg.at_node['water_surface__width'][:] * np.tan(thetarad) / (mg.at_node['water_surface__width'][:] * np.tan(thetarad) - 2 * mg.at_node['flow__depth'][:]))
    

    #calculate width coefficient for channel bank
    mg.at_node['psi_bank'][:] = rhow * g * mg.at_node['shear_stress__partitioning'][:] / 2 * (mg.at_node['water_surface__width'][:] / mg.at_node['flow__depth'][:] * np.sin(thetarad) - np.cos(thetarad))
    
    
    #Multiply erodibilities by width coefficient
    space.K_br = mg.at_node['K_sp'][:] * mg.at_node['psi_bed'][:]
    space.K_sed = mg.at_node['K_sed'][:] * mg.at_node['psi_bed'][:]
    
    
    #Update discharge field to m2/s before calculation erosion with space
    mg.at_node['surface_water__discharge'][:] = mg.at_node['normalized__discharge_sec'][:] 
    
    #erode with space
    space.run_one_step(dt=space_dt_sec)
    
    
    #Calculate bank erosion rate
    mg.at_node['bank_erosion__rate'][:] = mg.at_node['K_bank'][:] * mg.at_node['psi_bank'][:] * mg.at_node['normalized__discharge_sec'][:] * (mg.at_node['topographic__steepest_slope'][:]**n_sp)- omegacbank
    
    
    #Use getter to update bedrock erosion rate from space 
    mg.at_node['bedrock_erosion__rate'][:] = space._Er[:]
    
    #Calculate change in channel bedrock width at each node
    #dwrdt = (mg.at_node['bank_erosion__rate'][:] / math.sin(thetarad) - mg.at_node['bedrock_erosion__rate'][:]/math.tan(thetarad)) * 2 
    dwrdt = 2 * (mg.at_node['bank_erosion__rate'][:] / math.sin(thetarad)) - (mg.at_node['bedrock_erosion__rate'][:]/math.tan(thetarad))
    
    #update channel bedrock width
    mg.at_node['channel_bedrock__width'][:] += dwrdt * space_dt_sec
    
    
    if  mg.at_node['channel_bedrock__width'].any() < wr_min:
        raise Exception("Channel is too narrow")
    
    if (wr > dx).any():
        raise Exception("Channel width is greater than one grid cell")
        
    if (mg.at_node['channel_bedrock__width'] < wr_min).any():
       raise Exception("Channel is too narrow")
    
    #Update channel sediment width, ws 
    mg.at_node['channel_sed__width'][:] = mg.at_node['channel_bedrock__width'][:] + 2 * mg.at_node['soil__depth'][:] / np.tan(thetarad)
    
    
    #uplift the landscape 
    dz_ad = np.zeros(mg.size('node'))
    dz_ad[mg.core_nodes] = space_uplift_sec * space_dt_sec
    mg.at_node['bedrock__elevation'] += dz_ad
    
  
    
    #save output, check how long code has been running
    if elapsed_time_yrs %save_interval== 0:
        
        time_diff = timeit.default_timer() - starttime
        time_diff_minutes = np.round(time_diff/60, decimals=2)
        
        ds_ind = int((elapsed_time_yrs/save_interval))
    
        for of in out_fields:
            ds[of][ds_ind, :, :] = mg['node'][of].reshape(mg.shape)
            
        print(elapsed_time_yrs, time_diff_minutes )
        print ("wr=","{0:0.2f}".format(round(wr[out_node], 2)), "h=", "{0:0.2f}".format(round(h[out_node], 2)), "wwa=", "{0:0.2f}".format(round(w_avg[out_node], 2)))
        
        #write output to netcdf file
        ds.to_netcdf(ds_file_out)
        
    
    #update elapsed time
    elapsed_time += space_dt_sec
    elapsed_time_yrs += space_dt_sec / sec_per_yr
    

    
#%%

plt.figure()
imshow_grid(mg, 'channel_bedrock__width', colorbar_label='Channel Width(m)')   
plt.title('Final Channel Width') 

#%%

plt.figure()
imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')   
plt.title('Final Topo') 
    

#%%

plt.figure()
imshow_grid(mg, 'soil__depth', colorbar_label='Sed Thickness (m)')   
plt.title('Final Sediment Thickness') 

#%%

width_status = change_dir(bed_er, bank_er, thetarad)
