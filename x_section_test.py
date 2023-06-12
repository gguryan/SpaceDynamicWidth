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


theta = 60 #From Lague

manning_n = .05 #from Lague

dx = 100 #m

rho_w = 1000 #Density of water, kg/m3

W_init = 30 #Initial channel width, m

S_init = .02 #Initial slope

    

#%%

#Need to use manning to iteratively calculate flow depth
#Try scipy root_scalar(method='newton')

#For now, let's just say we have depth and velocity already

D = 5 #Flow depth #Calculate this for real later
U = 50 #Flow velocity #Also need a real calculation for this

#%%

stress_vars = Stress_Funcs(W_init, D, S_init, theta, rho_w)

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

w = np.ones((nx, ny))
w = mg.add_field("channel__width", w, at="node")

mg.add_ones('Psi_bed', at='node')
mg.add_ones('Psi_bank', at='node')

mg.add_zeros('bank__erosion', at='node')



mg.at_node['channel__width'][:] = mg.at_node['channel__width'] * W_init

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
    
    
    
    
