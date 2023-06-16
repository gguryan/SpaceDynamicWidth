# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:21:43 2023

@author: gjg882
"""

import numpy as np
import math
from matplotlib import pyplot as plt

from scipy import optimize

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

inputs = load_params('C:/Users/gjg882/Desktop/Code/SpaceDynamicWidth/dynamic_w_inputs.txt')

#grid dimensions
dx = inputs['dx']
nx=inputs['nx']
ny=inputs['ny']

theta_deg = inputs['theta_deg']
theta_rad = math.radians(theta_deg) #convert to radians

mannings_n = inputs['mannings_n'] #from Lague
rho_w = inputs['rho'] #Density of water, kg/m3
rho_sed = inputs["rho_sed"] # Density of sediment, kg/m^3

H_init = inputs["H_init"] #Initial sediment depth on bed, in m

wr_init = inputs["wr_init"] # Bedrock bed width, initial, in m.
wrmin = inputs['wr_min']  # Minimum bedrock bed width, code just stops if it gets this small.

U_initguess = inputs['U_initguess']  # Initial guess of avg velocity, in m/s, just to start the iteration process.


K_br = inputs["Kr"]
K_sed = inputs["Ks"]
Kbank = inputs["Kbank"]
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

nx = 3
ny = 4

mg = RasterModelGrid((nx, ny), dx)

z = np.zeros((nx, ny))
z = mg.add_field("topographic__elevation", z, at="node")

mg.add_zeros('node', 'soil__depth') #Create a grid field for soil depth
mg.at_node['soil__depth'][:] = H_init

wr = np.ones((nx, ny)) * wr_init
wr = mg.add_field("channel__bedrock_width", wr, at="node")


ws = w = np.ones((nx, ny)) 
ws = mg.add_ones("channel__sed_width", ws, at="node")

mg.add_ones('Psi_bed', at='node')
mg.add_ones('Psi_bank', at='node')

mg.add_zeros('bank__erosion', at='node') #TODO

mg.add_zeros('flow__depth', at = 'node') #this is h #todo should there be an initial value/guess?


z[5] = 5.0
z[6] = 4.0

mg.set_watershed_boundary_condition_outlet_id(7, 
                                              mg.at_node['topographic__elevation'],
                                              -9999.)


imshow_grid(mg, mg.status_at_node, color_for_closed='blue')

#%%

fr = PriorityFloodFlowRouter(mg, flow_metric='D8', suppress_out = True)
fr.run_one_step()
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


space_runtime = 100
space_dt = 100

t = np.arange(0, space_runtime+space_dt, space_dt)
nts = len(t)

#%%

print('m=', m_sp)

#Main model loop

for i in range(nts):
    
    #iteration to calculate channel geometry will go here
    #TODO
    #scipy.optimize.root_scalar
    
    #Next calculate width coefficients, Psi bed and Psi bank
    #TODO
    
    fa.run_one_step()
    
    #erode with space
    
    #Multiply erodibility by width coefficient
    space.K_br = mg.at_node['K_sp'] * mg.at_node['Psi_bed']
    space.K_sed = mg.at_node['K_sed'] * mg.at_node['Psi_bed']
    
    _ = space.run_one_step(dt=space_dt)
    
    #Calculate bank erosion
    #TODO - calculate Ebank
    mg.at_node['bank__erosion'] = mg.at_node['K_bank'] * mg.at_node['Psi_bank'] + TODO #TODO need to add q*s^n - threshold term
    
    #Pull out bed erosion from space  
    E_r_term = space._Er
    mg.at_node['bedrock__erosion'] = E_r_term.reshape(mg.shape[0], mg.shape[1])
    
    
    #Calculate change in channel width at each node
    mg.at_node['channel__width'] += (mg.at_node['bank__erosion'] / math.sin(theta) - mg.at_node['bedrock__erosion']/math.tan(theta)) * 2 * space_dt
    
    
    
    
