# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:56:49 2023

@author: gjg882
"""

import numpy as np
from matplotlib import pyplot as plt
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

dx = 50
nx = 3
ny = 4

H_init = 0.68  # Initial sediment depth on bed, in m.

mg = RasterModelGrid((nx, ny), dx)

z = np.zeros((nx, ny))
z = mg.add_field("topographic__elevation", z, at="node")

mg.add_zeros('node', 'soil__depth') #Create a grid field for soil depth
mg.at_node['soil__depth'][:] = H_init

z[5] = 5.0
z[6] = 4.0

mg.set_watershed_boundary_condition_outlet_id(7, 
                                              mg.at_node['topographic__elevation'],
                                              -9999.)

mg2 = mg.copy()

w = np.zeros((nx, ny))
w = mg2.add_field("channel__width", w, at="node")


imshow_grid(mg, mg.status_at_node, color_for_closed='blue')

#%%

K1 = 1e-4
K3 = 5e-5 #TODO CHANGE THIS
K2 = K1*K3

m1 = 0.5
n1 = 1.0

m2 = 1.0
n2 = 2.0

