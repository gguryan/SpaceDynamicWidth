# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 08:16:59 2023

@author: gjg882
"""

import xarray as xr

from os import chdir
from pathlib import Path

import numpy as np
from landlab import RasterModelGrid
from landlab import load_params
from landlab.io.netcdf import read_netcdf
from landlab.io.netcdf import write_netcdf
from landlab.plot import imshow_grid
from landlab.components import (FlowAccumulator, 
                                DepressionFinderAndRouter,
                                FastscapeEroder,
                                ChannelProfiler,
                                SteepnessFinder,
                                ChiFinder)
#%%

#convert time slice from xarray DS into landlab model grid

#def xr_to_mg_width(ds, plot_time, has_erosion, has_soil, has_width):
    
def xr_to_mg_width(ds, plot_time, out_fields):
    nx = ds.attrs['nx']
    ny = ds.attrs['ny']
    dx = ds.attrs['dx']


    #make model grid using topographic elevation at desired time
    mg = RasterModelGrid((nx, ny), dx)

    for field in out_fields:
    
    
        


    z = ds.topographic__elevation.sel(time=plot_time)
    z = mg.add_field("topographic__elevation", z, at="node")

    #Setting Node 0 (bottom left corner) as outlet node 
    mg.set_watershed_boundary_condition_outlet_id(0, mg.at_node['topographic__elevation'],-9999.)


    #Add bedrock erosion term
    Er = ds.bedrock__erosion.sel(time=plot_time)
    mg.has_field('node', 'bedrock__erosion')
    Er is mg.add_field('node', 'bedrock__erosion', Er, dtype=float)
    
    #Add sediment erosion term
    #Es = ds.sediment__erosion.sel(time=plot_time)
    #mg.has_field('node', 'sediment__erosion')
    #Es is mg.add_field('node', 'sediment__erosion', Es, dtype=float)


    #Add soil depth to model grid
    sed_depth = ds.soil__depth.sel(time=plot_time)
    mg.has_field('node', 'soil__depth')
    sed_depth is mg.add_field('node', 'soil__depth', sed_depth, dtype=float)
    
    #Add soil depth to model grid
    flow_depth = ds.flow__depth.sel(time=plot_time)
    mg.has_field('node', 'flow__depth')
    flow_depth is mg.add_field('node', 'flow__depth', flow_depth, dtype=float)
        
    
    #Add channel width
    wr = ds.channel_bedrock__width.sel(time=plot_time)
    mg.has_field('node', 'channel_bedrock__width')
    wr is mg.add_field('node', 'channel_bedrock__width', wr, dtype=float)
        
    
    return mg

#%%

