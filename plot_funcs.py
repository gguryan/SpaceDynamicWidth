# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 07:53:13 2025

@author: grace
"""


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



def xr_to_mg(ds, plot_time):
    
    ds_attrs = ds.attrs
    
    nx = ds_attrs['nx']
    ny = ds_attrs['ny']
    dx = ds_attrs['dx']
    
    data_vars_dict = ds.data_vars

    data_vars_list = list(data_vars_dict)
    
    mg = RasterModelGrid((nx, ny), dx)
    
    for var_name in data_vars_list:
        
                
        #z = ds.topographic__elevation.sel(time=plot_time)
        #z = mg.add_field("topographic__elevation", z, at="node")
        
        var_data = ds[var_name].sel(time=plot_time, method='nearest')
        mg.add_field(var_name, var_data, at="node")
    
    #reset outlet elevation to zero
    mg.at_node['topographic__elevation'][0] = 0
    
    fa1 = PriorityFloodFlowRouter(mg)
    fa1.run_one_step()
    
    return mg


def plot_channel_prf(ds, plot_time, model_name):
    
    '''
    ds: xarray dataset with model output
    plot_time : int, plot time in years, must be in intervals of 1000 years
    model name : string,  model name to use on plot titles, ('ie no sediment')  
    '''
    
    ds_attrs = ds.attrs

    mg = xr_to_mg(ds, plot_time)

    nx = ds.attrs['nx']
    
    #run flow accumulator for channel profiler purposes
    fa_temp = PriorityFloodFlowRouter(mg) 
    fa_temp.run_one_step()


    prf = ChannelProfiler(
        mg,
        main_channel_only=True,
    )

    prf.run_one_step()

    ordered_dict = prf.data_structure
    
    #get the first watershed from prf ordered dict
    #watershed is a tuple with two elements: outlet ID and a nested dict 
    watershed = next(iter(ordered_dict.items()))
    
    #watershed data is the nested dict
    outlet_ID, watershed_data = watershed
    
    
    #nested_dict = watershed_data
    #channel_key = next(iter(nested_dict))
    channel_dict = watershed_data
    channel_key = next(iter(channel_dict))
    
    
    #get node ids of all nodes in the main channel
    ids_array = channel_dict[channel_key]['ids']


    #Get outlet, midpoint, and upstream end of main channel
    first_value = ids_array[1]
    middle_value = ids_array[len(ids_array) // 2]
    last_value = ids_array[-5]

    x_coords = mg.x_of_node[[first_value, middle_value, last_value]]
    y_coords = mg.y_of_node[[first_value, middle_value, last_value]]

    xy_coords = list(zip(x_coords, y_coords))

    fig = plt.figure()
    #imshow_grid(mg1, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')   
    prf.plot_profiles_in_map_view()
    plt.title(f'Main Channel, {model_name}, Time={plot_time/1000} kyr') 
    plt.plot(x_coords[0], y_coords[0], 'o', color='tab:blue', markersize=3)  
    plt.plot(x_coords[1], y_coords[1], 'o', color='orange', markersize=3)  
    plt.plot(x_coords[2], y_coords[2], 'o', color='green', markersize=3)  

    plt.show()
    

    return fig, xy_coords, mg, middle_value



def calc_channel_dims(ds):
    
    #Calculate additional channel dimensions based on geometric relationships 
    
    sec_per_yr = 3.154e+7
    
    thetadeg = ds.attrs['theta_deg']
    thetarad = np.radians(thetadeg)
   
    #width depth ratio
    wdr = ds['channel_bedrock__width'] / ds['flow__depth']
    ds['width_depth__ratio'] = wdr
    ds['width_depth__ratio'].attrs['long_name'] = 'Bedrock Width/Depth Ratio'

    ds['water_surface__width'] = ds['channel_sediment__width'] + (2 * (ds['flow__depth'] / np.tan(thetarad)))
    ds['water_surface__width'].attrs['long_name'] = 'Water Surface Width'
    ds['flow__depth'].attrs['long_name'] = 'Flow Depth'
    
    mean_er = ds['bedrock_erosion__rate'].mean()
    
    order_of_magnitude = math.floor(math.log10(abs(mean_er)))
    
    # Check if order of magnitude is around -11
    # if order_of_magnitude == -11:
    #     ds['bedrock_erosion__rate'] *= sec_per_yr
    #     ds['bank_erosion__rate'] *= sec_per_yr
    
    return ds



def plot_xy_timeseries_multi(dataset, variables, xy_coords):
    """
    Plot time series for specified variables at multiple xy locations.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        The input dataset
    variables : list
        List of variable names to plot
    xy_coords : list of tuples
        List of (x, y) coordinate pairs to extract and plot
    """
    
    font = 14
    
    # sec_per_year = 3.154e7
    # dataset['bedrock_erosion__rate'] *= sec_per_year
    # dataset['bank_erosion__rate'] *= sec_per_year
    
    Kr = dataset.attrs['Kr']
    Kbank = dataset.attrs['Kbank']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 5*len(variables)), 
                              sharex=True, dpi=300)
    
    plt.rcParams['font.size'] = font  # Set the general font size
    

    
    # If only one variable, convert axes to list for consistent indexing
    if len(variables) == 1:
        axes = [axes]
    
    # Plot each variable
    for i, var_name in enumerate(variables):
        # Get long name for title (use var_name as fallback)
        long_name = dataset[var_name].attrs.get('long_name', var_name)
        
        # Plot time series for each coordinate
        for x_coord, y_coord in xy_coords:
            # Extract data at specific xy location
            var_data = dataset[var_name].sel(x=x_coord, y=y_coord, method='nearest')
            
            # Plot the time series with a unique label
            var_data.plot(ax=axes[i], label=f'({x_coord}, {y_coord})')
        
        axes[i].set_title(f'{long_name}, Kbank={Kbank}, Kbr={Kr}')
        axes[i].legend()
        
    plt.show()
    
    return fig, axes



    
