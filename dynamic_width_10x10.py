# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:21:43 2023

@author: gjg882
"""

import numpy as np
import math
from matplotlib import pyplot as plt

import time

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
inputs = load_params('dynamic_w_inputs_10x10.txt')
#inputs = load_params('C:/Users/gjg882/Desktop/Code/SpaceDynamicWidth/dynamic_w_inputs.txt')

#grid dimensions
dx = inputs['dx']
nx=inputs['nx']
ny=inputs['ny']

space_uplift=inputs['space_uplift']

theta_deg = inputs['theta_deg']
thetarad = math.radians(theta_deg) #convert to radians

manning_n = inputs['mannings_n'] #from Lague
rho_w = inputs['rho_w'] #Density of water, kg/m3
rho_sed = inputs["rho_sed"] # Density of sediment, kg/m^3

H_init = inputs["H_init"] #Initial sediment depth on bed, in m

wr_init = inputs["wr_init"] # Bedrock bed width, initial, in m.
wr_min = inputs['wr_min']  # Minimum bedrock bed width, code just stops if it gets this small.

U_initguess = inputs['U_initguess']  # Initial guess of avg velocity, in m/s, just to start the iteration process.

#h_initguess = 60 #TODO probably want to change this later

K_br = inputs["Kr"]
K_sed = inputs["Ks"]
K_bank = inputs["Kbank"]
n_sp = inputs["n_sp"]
m_sp = inputs["m_sp"] #NOTE THAT THIS IS CURRENTLY SET TO ONE
H_star = inputs["Hstar"]
sp_crit_br = inputs["omegacr"]
sp_crit_sed = inputs["omegacs"]
omegacbank = inputs["omegacbank"]
v_s = inputs["V_mperyr"]
phi = inputs["porosity"]
Ff = inputs["Ff"]


V_mperyr = v_s

timestep_s = 31536000  # Timestep duration, in seconds. Note that SPACE model uses 1 year.
maxitnum = 2000000 - 10  # For now, # of iterations of model to do

distdownstrtozeroelev_m = 10000  # Distance downstream to baselevel fixed to zero, e.g. sea level.
#   ASSUMING channel slope "hinges" around this point, i.e. downcutting
#   decreases slope, and slope stays constant this distance down to zero
#   elev.
# Code will calculate initial bedrock elevation (R_init) based on
# distdownstrtozeroelev_m and S_init.
Upliftrate_mmperyr = 1  # Uplift rate in mm/year, converted below to m/s

# Other variables, generally won't change:
rhow = 1000  # Water density, kg/m^3
rhos = 2500  # Sediment density, kg/m^3
g = 9.81  # Gravitational acceleration, m/s^2

error_tolfraction = 0.001  # Error tolerance, given as a fraction, i.e. 0.01 would
# be 1%, meaning that code would stop running if the difference
# between calculated and desired Q, divided by desired Q, was less than 1%.
# dampingfactor = 0.8  # Used in iterations for depth and velocity
dampingfactor = 0.5  # Used in iterations for depth and velocity

# Calculate, convert:
# Calculate ws_init, initial sediment bed width:
V = V_mperyr * 1 / (60 * 60 * 24 * 365)  # Convert effective settling velocity V to meters per second.
Upliftrate_mpers = Upliftrate_mmperyr / (1000 * 60 * 60 * 24 * 365)
#R_init = distdownstrtozeroelev_m * S_init - H_init  # Initial elevation (m above sea level)
timestep_yrs = timestep_s / (60 * 60 * 24 * 365)



#%%

#Make a landlab model grid

mg = RasterModelGrid((nx, ny), dx)



mg.add_zeros('node', 'soil__depth') #Create a grid field for soil depth
mg.at_node['soil__depth'][:] = H_init

wr = np.zeros((nx, ny)) 
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


#h = np.ones((nx, ny)) * h_initguess
h = np.zeros((nx, ny))
mg.add_field('flow__depth', h, at = 'node') 

#%%


#Random initial roughness
np.random.seed(seed = 200) #constant seed for constant random roughness


z = mg.node_x * .1 + mg.node_y * .1
mg.add_field('topographic__elevation',z, at='node')

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

fr = PriorityFloodFlowRouter(mg, flow_metric='D8', suppress_out = True)
fr.run_one_step()

Qw = mg.at_node['surface_water__discharge']
S = mg.at_node['topographic__steepest_slope']
wws = mg.at_node['water_surface__width']

#Calculate an initial width:

#Manning is currently in seconds, need to convert to years



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


space_runtime = 400000
space_dt = 10

t = np.arange(0, space_runtime+space_dt, space_dt)
nts = len(t)

#%%Define a funciton to calculate estimated discharge for a given flow depth, h, then compare estimated discharge to actual

tan_thetar = np.tan(thetarad)
sin_thetar = np.sin(thetarad)

'''
def Qwg(h, manning_n, ws, thetarad, S, Qw):
    
    Qwg = (1 / manning_n) * ((h * (ws + h / tan_thetar)) ** (5 / 3)) * (
        (ws + 2 * h / sin_thetar) ** (-(2 / 3))) * (S ** 0.5)

    Q_error = np.abs(Qw - Qwg)
    
    return Q_error
'''

def Qwg(h, manning_n, ws, thetarad, S, Qw):

    #TODO - need to convert discharge from m3/s to m3/year
    Qwg = (1 / manning_n) * ((h * (ws + h / tan_thetar)) ** (5 / 3)) * (
            (ws + 2 * h / sin_thetar) ** (-2 / 3)) * S ** 0.5
    
    Qwg_years = Qwg * 3.154e7
    Q_error = np.abs(Qw - Qwg_years)
    
    return Q_error


def calc_ws(mg, thetarad):
    mg.at_node['channel_sed__width'][:] = mg.at_node['channel_bedrock__width'][:] + 2 * mg.at_node['soil__depth'][:] / np.tan(thetarad)



#%%

#idea - run 100 timesteps of fsc to develop slope/initial drainage

fsc = FastscapeEroder(mg)
fsc_dt = 100

fa = FlowAccumulator(mg)

for i in range (100):
    fa.run_one_step()
    fsc.run_one_step(fsc_dt)
    dz_ad = np.zeros(mg.size('node'))
    dz_ad[mg.core_nodes] = .001 * fsc_dt
    mg.at_node['topographic__elevation'][:] += dz_ad

    

#%%
plt.figure()
imshow_grid(mg, 'topographic__elevation')
plt.title("Topo after FSC")

#%%
#calculate intiial width based on discharge
mg.at_node['channel_bedrock__width'][:] =  mg.at_node['surface_water__discharge'][:]**0.5 * (10**-2.5)

plt.figure()
imshow_grid(mg, wr)   
plt.title('Init Channel Br Width')

#calculate initial sediment width
calc_ws(mg, thetarad)

#%%

start_time = time.time()
elapsed_time = 0



#Main model loop
for i in range(nts):
    
    #flow routing
    fr.run_one_step()

    #iterate through nodes, upstream to downstream to calculate flow depth
    nodes_ordered = mg.at_node["flow__upstream_node_order"]
    
    #for j in range(len(nodes_ordered) - 1, -1, -1):
    for j in range(len(nodes_ordered)):
        
        #todo give topography an initial slope then give it random noise after that - give it some slope like .001
        
        #print(j, S[j])
        
        print(i, j)

        #if S[j] > 0:
        if mg.status_at_node[j] == 0 and  S[j] > 0: #don't operate on boundary nodes
            
            #if i == 0:
                #h_initguess =  ( ((Qw[j] / 3.154e7) * manning_n * (S[j]**-0.5)) / ws[j] ) **(3/5) #stopped at first node
            #else: h_initguess = mg.at_node['flow__depth'][j]
            h_initguess =  ( ((Qw[j] / 3.154e7) * manning_n * (S[j]**-0.5)) / ws[j] ) **(3/5)
            
        
            #h_initguess = ws[j] / 10 #stopped at node 21
            
            #wide channel assumption: let
            

            func_args = (manning_n, ws[j], thetarad, S[j], Qw[j])
            
            #print('h guess=', h_initguess)
        
            #print('func args=', func_args)
        
           
            h_out = scipy.optimize.newton(Qwg, x0=h_initguess, args=func_args, disp=True, full_output=True, tol=0.01, maxiter=100)
            
            #print(h_out)
            
            mg.at_node['flow__depth'][j] = h_out[0]
            
            #plt.figure()
            #imshow_grid(mg, h)  
    
    plt.figure() 
    imshow_grid(mg, 'flow__depth', colorbar_label=f'Flow Depth, t={i}')
    plt.show()
    plt.close()

    #TODO after optimizing for h, need to calculate hydraulic radius and velocity?
    #Joel's code does this, but rh and U don't actually get used elsewhere in model - just for looking at output?
    

    
    #calculate width at water surface
    mg.at_node['water_surface__width'][:] = mg.at_node['channel_sed__width'][:] + 2 * mg.at_node['flow__depth'][:] / np.tan(thetarad)
    
    #Calculate depth-averaged width
    #mg.at_node['depth_averaged__width'][:] = (mg.at_node['water_surface__width'][:] + mg.at_node['channel_sed__width'][:]) / 2
    
    #Caluclate normalized discharge
    #mg.at_node['normalized__discharge'][:] = mg.at_node['surface_water__discharge'][:] / mg.at_node['depth_averaged__width'][:]

    #Calculate Fw,shear stress partitioning between bed and banks
    #mg.at_node['shear_stress__partitioning'][:] = 1.78 * (mg.at_node['water_surface__width'][:] / mg.at_node['flow__depth'][:] * np.sin(thetarad) - 2 * np.cos(thetarad) + 1.5) ** (-1.4)
    mg.at_node['shear_stress__partitioning'][mg.core_nodes] = 1.78 * (mg.at_node['water_surface__width'][mg.core_nodes] / mg.at_node['flow__depth'][mg.core_nodes] * np.sin(thetarad) - 2 * np.cos(thetarad) + 1.5) ** (-1.4)
    
    #Next calculate width coefficients, Psi bed and Psi bank
    #psibed = rhow * g * (1 - Fw) / 2 * (1 + wws * np.tan(thetarad) / (wws * np.tan(thetarad) - 2 * h))
    #mg.at_node['psi_bed'][:] = rhow * g * (1 - mg.at_node['shear_stress__partitioning'][:]) / 2 * (1 + mg.at_node['water_surface__width'][:] * np.tan(thetarad) / (mg.at_node['water_surface__width'][:] * np.tan(thetarad) - 2 * mg.at_node['flow__depth'][:]))
    mg.at_node['psi_bed'][mg.core_nodes] = rhow * g * (1 - mg.at_node['shear_stress__partitioning'][mg.core_nodes]) / 2 * (1 + mg.at_node['water_surface__width'][mg.core_nodes] * np.tan(thetarad) / (mg.at_node['water_surface__width'][mg.core_nodes] * np.tan(thetarad) - 2 * mg.at_node['flow__depth'][mg.core_nodes]))
    
    #psibank = rhow * g * Fw / 2 * (wws / h * np.sin(thetarad) - np.cos(thetarad))
    #mg.at_node['psi_bank'][:] = rhow * g * mg.at_node['shear_stress__partitioning'][:] / 2 * (mg.at_node['water_surface__width'][:] / mg.at_node['flow__depth'][:] * np.sin(thetarad) - np.cos(thetarad))
    mg.at_node['psi_bank'][mg.core_nodes] = rhow * g * mg.at_node['shear_stress__partitioning'][mg.core_nodes] / 2 * (mg.at_node['water_surface__width'][mg.core_nodes] / mg.at_node['flow__depth'][mg.core_nodes] * np.sin(thetarad) - np.cos(thetarad))
    
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
    mg.at_node['bedrock__elevation'][:] += dz_ad


    #Recalculate topographic elevation to account for rock uplift
    mg.at_node['topographic__elevation'][:] = \
        mg.at_node['bedrock__elevation'][:] + mg.at_node['soil__depth'][:]
     
    plt.figure()    
    imshow_grid(mg, wr, colorbar_label=f'Channel Width, t={i}')
    plt.show()       
    plt.close()
    
    elapsed_time += space_dt
    
    if elapsed_time %1000 == 0:
        
        loop_time = round(((time.time()) - start_time) / 60)
        print(elapsed_time, loop_time)
        
        imshow_grid(mg, wr)   
        

    
    
#%%

plt.figure()
imshow_grid(mg, 'surface_water__discharge')    
plt.title('Discharge')
    
#%%

mg.at_node['channel_bedrock__width'][0] == np.nan


#%%

fr.run_one_step()

plt.figure()
imshow_grid(mg, wr)   
plt.title('Channel Br Width')


#imshow_grid(mg, h)   


#%%

#imshow_grid(mg, 'topographic__steepest_slope')
#imshow_grid(mg, 'flood_status_code')

# =============================================================================
# print(wr[5])
# print(mg.at_node['channel_bedrock__width'][5])
# print(mg.at_node['topographic__elevation'][5])
# print(mg.at_node['flow__depth'])
# =============================================================================

#sed_flux_in = mg.at_node['sediment__influx']