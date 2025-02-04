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

ds_file = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_lowKbank2.nc'

ds = xr.open_dataset(ds_file) 

#%%

plot_time = 200000

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

ds['water_surface__width'] = wws
ds['water_surface__width'].attrs['long_name'] = 'Water Surface Width'

data_vars_m = ['channel_bedrock__width', 'flow__depth', 'topographic__elevation', 'channel_sediment__width', 'water_surface__width' ]
data_vars_rates = ['bank_erosion__rate', 'bedrock_erosion__rate']
    
plot_xy_timeseries(ds, variables=data_vars_m, x_coord=dx, y_coord=dx)

#%%

z = ds.topographic__elevation.sel(time=plot_time)
z.plot()
plt.show()

#%%

plot_xy_timeseries(ds, variables=data_vars_m, x_coord=500, y_coord=500)