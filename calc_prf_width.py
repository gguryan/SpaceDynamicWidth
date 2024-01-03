# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:08:58 2023

@author: gjg882

"""


import numpy as np
import pandas as pd
import xarray as xr
from landlab import RasterModelGrid
from landlab.components import (FlowAccumulator, 
                                DepressionFinderAndRouter,
                                ChannelProfiler,
                                SteepnessFinder)

#%%

#Define function to run channel profiler, calculate steepness, and calculate chi for a given time in xarray dataset

def calc_prf_width(ds, plot_time, sf_min_DA):
    
    nx = ds.attrs['nx']
    ny = ds.attrs['ny']
    dx = ds.attrs['dx']
    
    m_sp = ds.attrs['m_sp']
    n_sp = ds.attrs['n_sp']
    theta = m_sp / n_sp
    
    #make model grid using topographic elevation at desired time
    mg = RasterModelGrid((nx, ny), dx)
    z = ds.topographic__elevation.sel(time=plot_time)
    z = mg.add_field("topographic__elevation", z, at="node")
    
    #Setting Node 0 (bottom left corner) as outlet node 
    mg.set_watershed_boundary_condition_outlet_id(0, mg.at_node['topographic__elevation'],-9999.)
    
    #Add rock type at each node to model grid
    sed_width = ds.channel_sed__width.sel(time=plot_time)
    mg.has_field('node', 'channel_sed__width')
    sed_width is mg.add_field('node', 'channel_sed__width', sed_width, dtype=int)
    
    
    #Add bedrock erosion term
    Er = ds.bedrock_erosion__rate.sel(time=plot_time)
    mg.has_field('node', 'bedrock_erosion__rate')
    Er is mg.add_field('node', 'bedrock_erosion__rate', Er, dtype=float)
    
    #bank_erosion__rate
    
    #Add bank erosion term
    E_bank = ds.bank_erosion__rate.sel(time=plot_time)
    mg.has_field('node', 'bank_erosion__rate')
    E_bank is mg.add_field('node', 'bank_erosion__rate', E_bank, dtype=float)
    
    
    #Add soil depth to model grid
    #sed_depth = ds.soil__depth.sel(time=plot_time)
    #mg.has_field('node', 'soil__depth')
    #sed_depth is mg.add_field('node', 'soil__depth', sed_depth, dtype=float)
    
    
    #Add sediment erosion term
    #Es = ds.sediment_erosion__rate.sel(time=plot_time)
    #mg.has_field('node', 'sediment_erosion__rate')
    #Es is mg.add_field('node', 'sediment_erosion__rate', Es, dtype=float)

    
    #Run flow accumulator
    fa = FlowAccumulator(mg, flow_director='D8')
    fa.run_one_step()

    #Run channel profiler to find main channel nodes
    prf = ChannelProfiler(mg, main_channel_only=True,minimum_channel_threshold=sf_min_DA)
    prf.run_one_step()
    
    #Calculate Channel Steepness
    sf = SteepnessFinder(mg, reference_concavity=theta, min_drainage_area=sf_min_DA)
    sf.calculate_steepnesses()    

    #Get nodes that define main channel
    prf_keys = list(prf.data_structure[0])

    channel_dist = prf.data_structure[0][prf_keys[0]]['distances']
    channel_dist_ids = prf.data_structure[0][prf_keys[0]]['ids']
    channel_elev = mg.at_node["topographic__elevation"][channel_dist_ids]
    channel_ksn = mg.at_node["channel__steepness_index"][channel_dist_ids]
    drainage_area = mg.at_node["drainage_area"][channel_dist_ids]
    channel_slope = mg.at_node['topographic__steepest_slope'][channel_dist_ids]
    



    df = pd.DataFrame({'channel_dist': channel_dist,
                   'channel_elev': channel_elev,
                   'channel_ksn' : channel_ksn,
                   'channel_slope' : channel_slope,
                   'drainage_area' : drainage_area})
    
    df["uplift"] = .001
    
    channel_Er = mg.at_node['bedrock_erosion__rate'][channel_dist_ids] 
    df['channel_Er'] = channel_Er

    #add sediment data for SPACE model runs
    #channel_sed_depth = mg.at_node['soil__depth'][channel_dist_ids]
    #df['channel_sed_depth'] = channel_sed_depth
    
    channel_E_bank = mg.at_node['bank_erosion__rate'][channel_dist_ids]  
    df['channel_E_bank'] = channel_E_bank
        
        
    return mg, df