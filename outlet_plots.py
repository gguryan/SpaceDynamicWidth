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


#ds_file = 'C:/Users/grace/Desktop/Projects/output/newQ_200kyr_lowKbank2.nc'

#ds_file = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_lowKbank2.nc'


#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_lowKbank2.nc'
#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_lowKbank.nc'
#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sameK.nc'

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed3.nc'
ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_kbank2x.nc'

ds = xr.open_dataset(ds_file) 

#%%

plot_time = 100000

ds_attrs = ds.attrs

dx = ds.attrs['dx']

outlet = [100, 100] 



thetarad = np.radians(ds.attrs['theta_deg'])

#%%

def plot_xy_timeseries(dataset, variables, x_coord, y_coord):
    """
    Plot time series for specified variables at a specific xy location.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        The input dataset
    variables : list
        List of variable names to plot
    x_coord : float
        X coordinate for extraction
    y_coord : float
        Y coordinate for extraction
    """
    
    Kr = ds.attrs['Kr']
    Kbank = ds.attrs['Kbank']
    
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(variables), 1, figsize=(10, 4*len(variables)), 
                              sharex=True)
    
    # If only one variable, convert axes to list for consistent indexing
    if len(variables) == 1:
        axes = [axes]
    
    # Plot each variable
    for i, var_name in enumerate(variables):
        # Extract data at specific xy location
        var_data = dataset[var_name].sel(x=x_coord, y=y_coord, method='nearest')
        
        # Get long name for title (use var_name as fallback)
        long_name = dataset[var_name].attrs.get('long_name', var_name)
        
        # Plot the time series
        var_data.plot(ax=axes[i], label=long_name)
        axes[i].set_title(f'{long_name} at (x={x_coord}, y={y_coord}), Kbank={Kbank}, Kbr={Kr}')
        axes[i].legend()
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

#%%


   
#wws[mg.core_nodes] = mg.at_node['channel_sediment__width'][mg.core_nodes] + ((2 * h[mg.core_nodes]) / np.tan(thetarad))
wws = ds['channel_sediment__width'] + (2 * (ds['flow__depth'] / np.tan(thetarad)))
wdr = ds['channel_bedrock__width'] / ds['flow__depth']

ds['width_depth__ratio'] = wdr
ds['width_depth__ratio'].attrs['long_name'] = 'Bedrock Width/Depth Ratio'

ds['water_surface__width'] = wws
ds['water_surface__width'].attrs['long_name'] = 'Water Surface Width'
ds['flow__depth'].attrs['long_name'] = 'Flow Depth'

data_vars_m = ['channel_bedrock__width', 'flow__depth',  'width_depth__ratio', 'topographic__elevation', 'soil__depth' ]
data_vars_rates = ['bank_erosion__rate', 'bedrock_erosion__rate']
    
plot_xy_timeseries(ds, variables=data_vars_m, x_coord=dx, y_coord=dx)

#%%

z = ds.topographic__elevation.sel(time=plot_time)
z.plot()
plt.show()

#%%

#plot_xy_timeseries(ds, variables=data_vars_m, x_coord=500, y_coord=500)

#plot_xy_timeseries(ds, variables=data_vars_m, x_coord=1200, y_coord=1200)


#%%

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
    
    Kr = dataset.attrs['Kr']
    Kbank = dataset.attrs['Kbank']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(variables), 1, figsize=(10, 4*len(variables)), 
                              sharex=True)
    
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
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
    
#%%

xy_coords = [(100, 100), (600, 600), (1200, 1200)]

plot_xy_timeseries_multi(ds, data_vars_m, xy_coords)

#%%

Kbr = ds.attrs['Kr']
Kbank = ds.attrs['Kbank']

check = (Kbank>Kbr)
print('K bank is bigger than Kbr =', check)