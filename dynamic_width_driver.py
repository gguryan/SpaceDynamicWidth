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

#%%inputs and outputs

#inputs = load_params('dynamic_w_inputs_10x10_gjg.txt')
inputs = load_params('dynamic_w_inputs_layers.txt')
#inputs = load_params("C:/Users/grace/Desktop/Projects/SpaceDynamicWidth/dynamic_w_inputs_10x10_gjg.txt")
#inputs = load_params('C:/Users/gjg882/Desktop/Code/SpaceDynamicWidth/dynamic_w_inputs.txt')

#path to save netcdf file to 
#ds_file_out = 'C:/Users/gjg882/Desktop/Projects/SDW_Output/ModelOutput/Qcalc_test_threshold2.nc'
#ds_file_out = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_kbank2x_nx100.nc'

ds_file_out = 'C:/Users/gjg882/Desktop/Projects/SDW_Output/ModelOutput/nx50_test.nc'

#ds_file_out = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_Ke-13.nc'''

#ds_file_out = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed_nx100_5e-12.nc'

#TODO - try model run with thresholds from original lague model

#%%Input variables

#for converting units throughout - todo update to account for leap days, not important rn 
sec_per_yr =  60 * 60 * 24 * 365

#Model time in years
#space_runtime = 1600000
space_runtime = 200000 #just for testing purposes
space_runtime_sec = space_runtime * sec_per_yr


#grid dimensions

#dx = inputs['dx'] #meters
dx = 100

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

wr_init = inputs["wr_init"] # Bedrock bed width, initial, in m. Not currently used for anything
#wr_min = inputs['wr_min']  # Minimum bedrock bed width, code just stops if it gets this small.
wr_min = .001

U_initguess = inputs['U_initguess']  # Initial guess of avg velocity, in m/s, just to start the iteration process.

h_initguess = 1.0 #TODO probably want to change this later

K_br = inputs["Kr"]
K_sed = inputs["Ks"]
K_bank = inputs["Kbank"]

#Try making things less erodible
#K_br =  K_br / 2
#K_bank = K_bank / 2


n_sp = inputs["n_sp"]
m_sp = inputs["m_sp"] #THIS MUST BE SET TO ONE, gets subsumed into new K calculation
H_star = inputs["Hstar"]

V_mperyr = inputs["V_mperyr"]
phi = inputs["porosity"]
Ff = inputs["Ff"]


Upliftrate_mperyr = inputs['space_uplift']
space_uplift_sec = Upliftrate_mperyr / sec_per_yr

v_seconds = V_mperyr / sec_per_yr

#thresholds
sp_crit_br = inputs["omegacr"]
sp_crit_sed = inputs["omegacs"]
omegacbank = inputs["omegacbank"]

# sp_crit_br = .03 * .12  #tau x estimated velocity (.12 m3/sec for 20x20x100 grid)
# sp_crit_sed = sp_crit_br 
# omegacbank = sp_crit_br * 1.15 #From phillips et al 2022


# Other variables, generally won't change:
rhow = 1000  # Water density, kg/m^3
rhos = 2500  # Sediment density, kg/m^3
g = 9.81  # Gravitational acceleration, m/s^2


#Make a landlab model grid
mg = RasterModelGrid((nx, ny), dx)


#create model grid fields

H_soil = mg.add_zeros('node', 'soil__depth') #Create a grid field for soil depth
mg.at_node['soil__depth'][:] = H_init

wr = np.zeros((nx, ny)) 
wr = mg.add_field("channel_bedrock__width", wr, at="node")

#ws = np.zeros((nx, ny)) 
ws = mg.add_zeros("channel_sediment__width", at="node")

psi_bed = mg.add_zeros('psi_bed', at='node')
psi_bank = mg.add_zeros('psi_bank', at='node')

bank_er = mg.add_zeros('bank_erosion__rate', at='node') 
bed_er = mg.add_zeros('bedrock_erosion__rate', at='node')
wws = mg.add_zeros('water_surface__width', at='node')
w_avg = mg.add_zeros('depth_averaged__width', at='node')

Fw = mg.add_zeros('shear_stress__partitioning', at='node')

#Q_sec = mg.add_zeros('discharge__seconds', at='node')

q_norm = mg.add_zeros('normalized__discharge_sec', at='node') #in m2/sec (NOT YEARS, conversion is already done)

K_sp_arr = np.ones((nx, ny)) * K_br
mg.add_field("K_sp", K_sp_arr, at="node")

K_sed_arr = np.ones((nx, ny)) * K_sed
mg.add_field("K_sed", K_sed_arr, at="node")

K_bank_arr = np.ones((nx, ny)) * K_bank
mg.add_field("K_bank", K_bank_arr, at="node")


flow_depth = np.ones((nx, ny)) * h_initguess

#h = mg.add_field('flow__depth', flow_depth, at = 'node') 
h = mg.add_zeros('flow__depth', at='node')



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
    
    fa1.run_one_step()
    
    fsc.run_one_step(dt=fsc_dt)
    fsc_time += fsc_dt
    

plt.figure()
imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')
plt.title('Topo after FSC')
plt.show()


#%%Set up soil for SPACE



mg.at_node["bedrock__elevation"] = mg.at_node["topographic__elevation"][:] - mg.at_node["soil__depth"][:]


DA = mg.at_node['drainage_area']

#Coefficients calculated from regression line for bedrock rivers on fig 2E of Buckley et al 2024
#Q_calc = 0.25 * (DA[nx+1]**0.6)
#Runoff rate = Q/drainage area, DA * runoff = Q
#Runoff rate is in m/second
#runoff_calc = Q_calc / DA[nx+1]

runoff_mperyr = 3
runoff_calc = runoff_mperyr/sec_per_yr

fa = PriorityFloodFlowRouter(mg, runoff_rate=runoff_calc) 
fa.run_one_step()

#Initialize new flow accumulator for SPACE, discharge in m3/sec


Qw = mg.at_node['surface_water__discharge'] #just drainage area

#Estimate intial width/depth ratio from Buckley
 

S = mg.at_node['topographic__steepest_slope']


wdr = np.zeros(mg.size('node'))
wdr[mg.core_nodes] = 3 * (mg.at_node['surface_water__discharge'][mg.core_nodes]**0.15) 


#%%


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
            
        'channel_sediment__width':
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
              'channel_sediment__width',
              'bank_erosion__rate',
              'bedrock_erosion__rate',
              'flow__depth',
              'surface_water__discharge',
              'topographic__steepest_slope',
              'channel_bedrock__width', 
              'soil__depth']
             


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
    mg.at_node['channel_sediment__width'][mg.core_nodes] = mg.at_node['channel_bedrock__width'][mg.core_nodes] 
    + ( 2 * mg.at_node['soil__depth'][mg.core_nodes] / np.tan(thetarad))
    


#%%Calculate initial channel geometry

#Start with an initial width
#Exponent from Buckley et al 2024 fig 2A
#their numbers are for bankfull width (not bedrock) - fine as starting point
wr[mg.core_nodes] =  ((mg.at_node['drainage_area']/1e6)[mg.core_nodes] ** 0.5) #channel is wider than dx if anything greater than .28 is used


# if (wr > dx).any():
#     raise Exception("Initial bedrock width is greater than one grid cell")
    
# if (wr < wr_min).any():
#     raise Exception("Initial bedrock width is too narrow")


#Check for nodes that are too narrow
too_narrow = mg.core_nodes[wr[mg.core_nodes] < wr_min]
if too_narrow.size > 0:
    raise Exception(f"Initial bedrock width is too narrow at indices: {too_narrow}")
    
#Check for nodes that are too wide
too_wide = mg.core_nodes[wr[mg.core_nodes] > dx]
if too_wide.size > 0:
    raise Exception(f"Intiial bedrock width is too wide at indices: {too_wide}")


#calculate initial sediment width
calc_ws(mg, thetarad)


#estimate initial depth by calculating bedrock width over w/d ratio
#Ignore cells where w/d is zero/there is no discharge
#depth_init = np.divide(wr[mg.core_nodes], wdr[mg.core_nodes], out=np.zeros_like(wr, dtype=float), where=wdr != 0) 



#%%

#depth_init = 7.28*10e3 * (wr**-1.4) #From Veneditti et al 2022, gives very deep channels
h[mg.core_nodes] = wr[mg.core_nodes] / wdr[mg.core_nodes]   


#calculate width at water surface
#wws[:] = mg.at_node['channel_sediment__width'][:] + 2 * h_initguess / np.tan(thetarad)
wws[mg.core_nodes] = mg.at_node['channel_sediment__width'][mg.core_nodes] + ((2 * h[mg.core_nodes]) / np.tan(thetarad))
#wws[:] = mg.at_node['channel_sediment__width'][:] + ((2 * h[:]) / np.tan(thetarad))


if (wws > dx).any():
    raise Exception("Water surface width is greater than one grid cell")
    
#Calculate depth-averaged width
w_avg[mg.core_nodes] = (wws[mg.core_nodes] + mg.at_node['channel_sediment__width'][mg.core_nodes]) / 2


#Qw = velocity * width * depth
#Velocity = Qw/(width*depth)
#vel_est = Qw / (wr * depth_init)

#Cross sectional area
Ax = np.zeros(mg.size('node'))
Ax = wr * h

vel_est = np.zeros(mg.size('node'))
vel_est[mg.core_nodes] = (Qw[mg.core_nodes]/Ax[mg.core_nodes])


#vel_est = np.divide(Qw[mg.core_nodes], Ax, out=np.zeros_like(wr[mg.core_nodes], dtype=float), where=Ax != 0)
#%%
imshow_grid(mg, 'surface_water__discharge')
plt.title('Initial Q, m3/sec')
plt.show()

imshow_grid(mg, 'channel_bedrock__width')
plt.title('Initial Channel Bedrock Width')
plt.show()

#%%

imshow_grid(mg, 'flow__depth')
plt.title('Initial Flow Depth')
plt.show()




#%%

#Save model initial condition/time zero to xarray dataset
for of in out_fields:
    ds[of][0, :, :] = mg['node'][of].reshape(mg.shape)


ds.attrs.update({"runoff_mperyr": runoff_mperyr})
    

#%% Main model loop

if nx == ny:
    outlet=nx+1

elapsed_time = 0
elapsed_time_yrs= 0
total_U = 0
starttime = timeit.default_timer()

#Upper and lower bounds on flow depth - used for root finding function
lower_bound = 0
upper_bound = dx-1 #should probably be lower, doesn't matter in current parameter space



#for i in range(10000):
for i in range(nts):
#for i in range(1):


    
    #flow routing    
    fa.run_one_step()
    

    #calculate normalized discharge in m2/sec by dividing by avg channel width
    q_norm[mg.core_nodes] = (Qw[mg.core_nodes] / w_avg[mg.core_nodes])
    
        
    #iterate through nodes, upstream to downstream to calculate flow depth
    #not actually sure if order is important here
    nodes_ordered = mg.at_node["flow__upstream_node_order"]


    for j in range(len(nodes_ordered) - 1, -1, -1):

        #inputs for function to iterate for flow depth
        func_args = (manning_n, ws[j], thetarad, S[j], Qw[j])

        
        if mg.status_at_node[j] == 0: #operate on core nodes only
            
            
            if S[j] == 0: #don't calculate flow depth for nodes with zero slope - it breaks the function 
                print('S=0 at', i, j)
                pass
            
            
            else:
                
                #calculate flow depth [m]
                h[j] = scipy.optimize.brentq(Qwg, a=lower_bound, b=upper_bound, args=func_args, disp=True, maxiter=100)

        
    #calculate width at water surface
    wws[mg.core_nodes] = mg.at_node['channel_sediment__width'][mg.core_nodes] + 2 * h[mg.core_nodes] / np.tan(thetarad)
    
    
    #Calculate depth-averaged width
    w_avg[mg.core_nodes] = (wws[mg.core_nodes] + mg.at_node['channel_sediment__width'][mg.core_nodes]) / 2
    

    #Calculate Fw, term for shear stress partitioning between bed and banks
    Fw[mg.core_nodes] = 1.78 * (wws[mg.core_nodes] / mg.at_node['flow__depth'][mg.core_nodes] * np.sin(thetarad) - 2 * np.cos(thetarad) + 1.5) ** (-1.4)
     
    
    #Next calculate width coefficient for bed
    mg.at_node['psi_bed'][mg.core_nodes] = rhow * g * (1 - Fw[mg.core_nodes]) / 2 * (1 + mg.at_node['water_surface__width'][mg.core_nodes] * np.tan(thetarad) / wws[mg.core_nodes] * np.tan(thetarad) - 2 * h[mg.core_nodes])
    

    #calculate width coefficient for channel bank
    mg.at_node['psi_bank'][mg.core_nodes] = (rhow * g *  (Fw[mg.core_nodes] / 2)) * (((wws[mg.core_nodes] / h[mg.core_nodes]) * np.sin(thetarad)) - np.cos(thetarad))
    
    
    #Multiply erodibilities by width coefficient
    space.K_br = mg.at_node['K_sp'][:] * mg.at_node['psi_bed'][:]
    space.K_sed = mg.at_node['K_sed'][:] * mg.at_node['psi_bed'][:]
    
    
    
    #Update discharge field to m2/s before calculation erosion with space 
    mg.at_node['surface_water__discharge'][:] = q_norm[:] 

    #erode with space
    space.run_one_step(dt=space_dt_sec)
    
    
    #Calculate bank erosion rate in m/sec
    bank_er[mg.core_nodes] = (mg.at_node['K_bank'][mg.core_nodes] * mg.at_node['psi_bank'][mg.core_nodes] * q_norm[mg.core_nodes] * (S[mg.core_nodes]**n_sp)) - omegacbank
    
    
    #Use getter to update bedrock erosion rate from space 
    bed_er[mg.core_nodes] = space._Er[mg.core_nodes]
    
    #Calculate change in channel bedrock width at each node
    #dwrdt = (mg.at_node['bank_erosion__rate'][:] / math.sin(thetarad) - mg.at_node['bedrock_erosion__rate'][:]/math.tan(thetarad)) * 2 
    dwrdt = 2 * ((bank_er[mg.core_nodes] / math.sin(thetarad)) - (bed_er[mg.core_nodes]/math.tan(thetarad)))
    
    
    
    #print('width before updating', wr)
    #temp = dwrdt*space_dt_sec
    
    #TODO - problem is that dwrdt is way too big
    
    #update channel bedrock width
    mg.at_node['channel_bedrock__width'][mg.core_nodes] += dwrdt * space_dt_sec
    
    #print('width after updating', wr)
    
    # #Check for nodes that are too wide
    # if (wr[mg.core_nodes] > dx).any():
    #     raise Exception("Channel width is greater than one grid cell")
        
    # if (wr[mg.core_nodes] < wr_min).any():
    #    raise Exception("Channel is too narrow")

    #Check for nodes that are too narrow
    too_narrow = mg.core_nodes[wr[mg.core_nodes] < wr_min]
    if too_narrow.size > 0:
        raise Exception(f"Channel is too narrow at indices: {too_narrow}")
        
    #Check for nodes that are too wide
    too_wide = mg.core_nodes[wr[mg.core_nodes] > dx]
    if too_wide.size > 0:
        raise Exception(f"Channel is too wide at indices: {too_wide}")
    
    #Update channel sediment width, ws 
    mg.at_node['channel_sediment__width'][mg.core_nodes] = mg.at_node['channel_bedrock__width'][mg.core_nodes] + 2 * mg.at_node['soil__depth'][mg.core_nodes] / np.tan(thetarad)
    
    
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
        print ("wr=","{0:0.2f}".format(round(wr[outlet], 2)), "h=", "{0:0.2f}".format(round(h[outlet], 2)), "wwa=", "{0:0.2f}".format(round(w_avg[outlet], 2)), "H=", "{0:0.2f}".format(round(mg.at_node["soil__depth"][outlet], 2)))
        print('mean elev', np.mean(z[mg.core_nodes]))
        
        #write output to netcdf file
        ds.to_netcdf(ds_file_out)
        
    
    #update elapsed time
    elapsed_time += space_dt_sec
    elapsed_time_yrs += space_dt_sec / sec_per_yr
    


    
#%%

narrow_x = mg.x_of_node[too_narrow]
narrow_y = mg.y_of_node[too_narrow]



#%%

plt.figure()
imshow_grid(mg, 'channel_bedrock__width', colorbar_label='Channel Width(m)')   
plt.plot(narrow_x, narrow_y, 'rs')
plt.title('Final Channel Width') 
plt.show()

#%%

plt.figure()
imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')   
plt.title('Final Topo') 
    
plt.show()
#%%

plt.figure()
imshow_grid(mg, 'soil__depth', colorbar_label='Sed Thickness (m)')   
plt.title('Final Sediment Thickness') 
plt.show()

#%%

plt.figure()
imshow_grid(mg, 'topographic__steepest_slope', colorbar_label='Slope', cmap='viridis')   
plt.title('Final Slope') 
plt.show()

#%%

zero_slope = np.where(mg.at_node['topographic__steepest_slope'] == 0)[0]
flat_x = mg.x_of_node[zero_slope]
flat_y = mg.y_of_node[zero_slope]


from matplotlib.colors import ListedColormap

# Create a custom colormap based on 'viridis'
viridis = plt.cm.get_cmap('viridis', 256)
new_colors = viridis(np.linspace(0, 1, 256))

# Set the color for zero values to red
new_colors[0] = [1, 0, 0, 1]  # RGBA for red (fully opaque)

# Create a new colormap
custom_cmap = ListedColormap(new_colors)


plt.figure()
imshow_grid(mg, 'topographic__steepest_slope', colorbar_label='Slope', cmap=custom_cmap)  
#plt.plot(flat_x, flat_y, 'rs') 
plt.title('Final Slope') 
plt.show()



#%%

wdr[mg.core_nodes] = wr[mg.core_nodes] / h[mg.core_nodes]

#%%


topo_mean = ds['topographic__elevation'].mean(dim=["x", "y"])
plt.plot(topo_mean["time"], topo_mean)
plt.title('Mean Elevation over Time')
plt.show()