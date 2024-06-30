# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 15:47:12 2023

@author: gjg882
"""

import numpy as np
import pandas as pd
import xarray as xr
from landlab import RasterModelGrid
from landlab.components import (FlowAccumulator, 
                                DepressionFinderAndRouter,
                                ChannelProfiler,
                                SteepnessFinder,
                                ChiFinder)



#%%

#Define function to run channel profiler, calculate steepness, and calculate chi for a given time in xarray dataset

def calc_prf(ds, plot_time, sf_min_DA, cf_min_DA):
    
    nx = ds.attrs['nx']
    ny = ds.attrs['ny']
    dx = ds.attrs['dx']
    
    m_sp = ds.attrs['m_sp']
    n_sp = ds.attrs['n_sp']
    theta = m_sp / n_sp
    
    model_name = ds.attrs['model_name']
    
    #make model grid using topographic elevation at desired time
    mg = RasterModelGrid((nx, ny), dx)
    z = ds.topographic__elevation.sel(time=plot_time)
    z = mg.add_field("topographic__elevation", z, at="node")
    
    #Setting Node 0 (bottom left corner) as outlet node 
    mg.set_watershed_boundary_condition_outlet_id(0, mg.at_node['topographic__elevation'],-9999.)
    
    #Add rock type at each node to model grid
    rock = ds.rock_type__id.sel(time=plot_time)
    mg.has_field('node', 'rock_type__id')
    rock is mg.add_field('node', 'rock_type__id', rock, dtype=int)
    
    var_list = list(ds.keys())
    if 'bedrock__erosion' in var_list:
    
        #Add bedrock erosion term
        Er = ds.bedrock__erosion.sel(time=plot_time)
        mg.has_field('node', 'bedrock__erosion')
        Er is mg.add_field('node', 'bedrock__erosion', Er, dtype=float)
    
    if model_name == 'Mixed':
        
        #Add soil depth to model grid
        sed_depth = ds.soil__depth.sel(time=plot_time)
        mg.has_field('node', 'soil__depth')
        sed_depth is mg.add_field('node', 'soil__depth', sed_depth, dtype=float)
        
        #Add sediment erosion term
        Es = ds.sediment__erosion.sel(time=plot_time)
        mg.has_field('node', 'sediment__erosion')
        Es is mg.add_field('node', 'sediment__erosion', Es, dtype=float)

    
    #Run flow accumulator
    fa = FlowAccumulator(mg, flow_director='D8')
    fa.run_one_step()

    #Run channel profiler to find main channel nodes
    prf = ChannelProfiler(mg, main_channel_only=True,minimum_channel_threshold=sf_min_DA)
    prf.run_one_step()
    
    #Calculate Channel Steepness
    sf = SteepnessFinder(mg, reference_concavity=theta, min_drainage_area=sf_min_DA)
    sf.calculate_steepnesses()    

    #Calculate Chi
    cf = ChiFinder(mg, min_drainage_area=cf_min_DA,reference_concavity=theta,use_true_dx=True)
    cf.calculate_chi()
    
    #Get nodes that define main channel
    prf_keys = list(prf.data_structure[0])

    channel_dist = prf.data_structure[0][prf_keys[0]]['distances']
    channel_dist_ids = prf.data_structure[0][prf_keys[0]]['ids']
    channel_elev = mg.at_node["topographic__elevation"][channel_dist_ids]
    channel_chi = mg.at_node["channel__chi_index"][channel_dist_ids]
    channel_ksn = mg.at_node["channel__steepness_index"][channel_dist_ids]
    drainage_area = mg.at_node["drainage_area"][channel_dist_ids]
    channel_slope = mg.at_node['topographic__steepest_slope'][channel_dist_ids]
    


    channel_rock_ids = mg.at_node['rock_type__id'][channel_dist_ids]
    channel_rock_ids = channel_rock_ids.astype(int)
    

    df = pd.DataFrame({'channel_dist': channel_dist,
                   'channel_elev': channel_elev,
                   'channel_rock_id': channel_rock_ids, 
                   'channel_ksn' : channel_ksn,
                   'channel_chi' : channel_chi,
                   'channel_slope' : channel_slope,
                   'drainage_area' : drainage_area})
    
    df["uplift"] = .001
    
    if 'bedrock__erosion' in var_list:
        channel_Er = mg.at_node['bedrock__erosion'][channel_dist_ids] 
        df['channel_Er'] = channel_Er
    
    if model_name == 'Mixed':
        
        #add sediment data for SPACE model runs
        channel_sed_depth = mg.at_node['soil__depth'][channel_dist_ids]
        df['channel_sed_depth'] = channel_sed_depth
        
        channel_Es = mg.at_node['sediment__erosion'][channel_dist_ids]  
        df['channel_Es'] = channel_Es
        
        
    return mg, df