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

#set initial conditions/variables

#my inputs
theta_deg = 60 #From Lague - bank angle in degrees
theta = math.radians(theta_deg) #convert to radians

mannings_n = .05 #from Lague

dx = 100 #m

rho_w = 1000 #Density of water, kg/m3

rho_sed = 2500 #Density of sediment, kg/m^3

wb_init = 30 #Initial channel width, m

wr_init = 30 #Initial channel bottom width

S_init = .02 #Initial slope

H_init = 0.68  # Initial sediment depth on bed, in m.

#%%
#Joel's inputs

Qw = 100  # Water discharge, in m3/s
Qs = 0.1  # Sediment load, in m3/s (volume supply rate)
# S_init = 0.02  # Reach slope, m/m
S_init = 0.01527  # Reach slope, m/m

thetadeg = 60  # Sidewall (bank) angle, measured from horizontal. Lague(2010) used 60.
# wr_init = 10  # Bedrock bed width, initial, in m.
wr_init = 27.66  # Bedrock bed width, initial, in m.
# H_init = 0.5  # Initial sediment depth on bed, in m.
H_init = 0.68  # Initial sediment depth on bed, in m.
manningsn = 0.05  # Manning's n, same value as Lague used. s/m^(1/3)
wrmin = 0.1  # Minimum bedrock bed width, code just stops if it gets this small.

U_initguess = 2.75  # Initial guess of avg velocity, in m/s, just to start the iteration process.

# SPACE model variables:
Kr = 0.000000000001  # Rock erodibility, units 1/m. NOTE: Shobe et al. 2017 used 0.001 (typically), but psibed stream power coefficient likely should change this value.
Ks = 0.00000000001  # Sediment entrainability, units 1/m
Kbank = 0.000000000001  # Try order of magnitude smaller than Kr
n = 1.5  # Exponent on slope! Not Manning's n
Hstar = 1  # e-folding length for cover effect, in m
omegacr = 0  # Unit stream power threshold for bedrock erosion. Note that Shobe et al. (2017) have units as m/year, I'm using m/s.
omegacs = 0  # Unit stream power threshold for sediment entrainment
omegacbank = 0  # Unit stream power threshold for bank erosion
V_mperyr = 10  # Grace uses 1, 3, 5. Convert to m/s below
porosity = 0  # phi in SPACE model paper
Ff = 0  # Fraction fines

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
R_init = distdownstrtozeroelev_m * S_init - H_init  # Initial elevation (m above sea level)
timestep_yrs = timestep_s / (60 * 60 * 24 * 365)

    


    

#%%

#Need to use manning to iteratively calculate flow depth
#Try scipy root_scalar(method='newton')





#%%
D=100

stress_vars = Stress_Funcs(wb_init, D, S_init, theta, rho_w)

Fw = stress_vars.calc_Fw()

psi_bank = stress_vars.calc_psi_bank(Fw)
psi_bed = stress_vars.calc_psi_bed(Fw)

#%%

#Make a landlab model grid

nx = 3
ny = 4

mg = RasterModelGrid((nx, ny), dx)

z = np.zeros((nx, ny))
z = mg.add_field("topographic__elevation", z, at="node")

mg.add_zeros('node', 'soil__depth') #Create a grid field for soil depth
mg.at_node['soil__depth'][:] = H_init

w = np.ones((nx, ny)) * wr_init
w = mg.add_field("channel__bottom_width", w, at="node")


ws = w = np.ones((nx, ny)) * ws_init #TODO what is ws_init??
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

imshow_grid(mg, 'channel__width')

#%%

fa = FlowAccumulator(mg, flow_director='D8') #TODO Remove?
fa.run_one_step()

#%%

space = SpaceLargeScaleEroder(mg)

space_runtime = 100
space_dt = 100

t = np.arange(0, space_runtime+space_dt, space_dt)
nts = len(t)

#%%

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
    
    
    
    
