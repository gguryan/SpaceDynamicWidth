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
import os


#ds_file = 'C:/Users/grace/Desktop/Projects/output/newQ_200kyr_lowKbank2.nc'

#ds_file = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_lowKbank2.nc'


#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_lowKbank2.nc'
#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_lowKbank.nc'
#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sameK.nc'

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed3.nc'
#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_kbank2x.nc'

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_kbank2x_nx100.nc'
#ds_file='C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_test_kr2_150_nx50_2.nc' #didn't work

#ds_file = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed3.nc'


#ds_file = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed_nx100_1e-11.nc'

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed_nx100_5e-12.nc' #Failed
#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed_nx100_5e-12.nc' #Failed

ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_sed_nx100_1e-11.nc' #THIS ONE WORKS


#ds_file='C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_test_kr15_nx50_ctrl.nc'

#ds_file = "C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_kbank2x_nx100.nc"

#%%


#ds_file = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_ratio_05.nc'

#ds_file = "C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_kbank2x.nc" #20x20, 100kyr


ds_file = 'C:/Users/grace/Box/UT/Research/Dynamic Width/ModelOuptut/newQ_200kyr_lowKbank.nc' #"default" - kbank = 1.5e-12, Kbr = 1e-12
ds1 = xr.open_dataset(ds_file)

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_ratio_05_sed.nc'
#ds2 = xr.open_dataset(ds_file) 

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_ratio_05_sed_v3.nc'
#ds1 = xr.open_dataset(ds_file)

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_test_kr15_nx50_ctrl_4.nc'
ds1 = xr.open_dataset(ds_file)

#ds_file = 'C:/Users/gjg882/Box/UT/Research/Dynamic Width/ModelOuptut/SDW_litholayers_ratio_05.nc'
#ds1 = xr.open_dataset(ds_file) 


#%%

# plot_time = 500000

# 

# dx = ds1.attrs['dx']
# nx=ds1.attrs['nx']

# outlet = [100, 100] 

# outlet_id = nx+1



# thetarad = np.radians(ds1.attrs['theta_deg'])

#%%

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
    
    return mg

#%%

plot_time = 500000



# plt.figure()
# imshow_grid(mg1, 'channel_bedrock__width', colorbar_label='Channel Width(m)')   
# plt.title(f'Channel Width at {plot_time}') 
# plt.show()




#%%

def plot_channel_prf(ds, plot_time, model_name):
    
    '''
    ds: xarray dataset with model output
    plot_time : int, plot time in years, must be in intervals of 1000 years
    model name : string,  model name to use on plot titles, ('ie no sediment')  
    '''
    
    ds_attrs = ds.attrs

    mg = xr_to_mg(ds, plot_time)

    nx = ds1.attrs['nx']
    
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



#%%define functions plot_xy_timeseries and calc_channel_dims

def plot_xy_timeseries(ds, variables, x_coord, y_coord):
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
                              sharex=True, dpi=300)
    
    # If only one variable, convert axes to list for consistent indexing
    if len(variables) == 1:
        axes = [axes]
    
    # Plot each variable
    for i, var_name in enumerate(variables):
        # Extract data at specific xy location
        var_data = ds[var_name].sel(x=x_coord, y=y_coord, method='nearest')
        
        # Get long name for title (use var_name as fallback)
        long_name = ds[var_name].attrs.get('long_name', var_name)
        
        # Plot the time series
        var_data.plot(ax=axes[i], label=long_name)
        axes[i].set_title(f'{long_name} at (x={x_coord}, y={y_coord}), Kbank={Kbank}, Kbr={Kr}')
        axes[i].legend()
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

#%%

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
    if order_of_magnitude == -11:
        ds['bedrock_erosion__rate'] *= sec_per_yr
        ds['bank_erosion__rate'] *= sec_per_yr
    
    return ds

#%%

#Calculate additional channel dimensions
var_list1 = list(ds1.variables)
ds1 = calc_channel_dims(ds1)



# var_list2 = list(ds1.variables)
# ds2 = calc_channel_dims(ds2)



#%%

data_vars_m = ['channel_bedrock__width', 'flow__depth',  'width_depth__ratio', 'topographic__elevation', 'soil__depth' ]
data_vars_rates = ['bank_erosion__rate', 'bedrock_erosion__rate']
    
#plot_xy_timeseries(ds1, variables=data_vars_m, x_coord=dx+2, y_coord=dx+2)

#%%


# plt.figure()

# z = ds1.topographic__elevation.sel(time=plot_time)
# z.plot(cmap='pink')


# # for x_i, y_i, c in zip(x, y, colors):
# #     plt.plot(x_i, y_i, 'o', color=c, markersize=1)  # This is the correct syntax
    
    
# # Plot points individually for different colors
# plt.plot(x_coords[0], y_coords[0], 'o', color='tab:blue', markersize=3)  
# plt.plot(x_coords[1], y_coords[1], 'o', color='orange', markersize=3)  
# plt.plot(x_coords[2], y_coords[2], 'o', color='green', markersize=3)  

# #plt.plot(x, y, 'rs', markersize=1)
# plt.show()



#%%

#plot_xy_timeseries(ds, variables=data_vars_m, x_coord=500, y_coord=500)

#plot_xy_timeseries(ds, variables=data_vars_m, x_coord=1200, y_coord=1200)


#%%define function plot_xy_timeseries_multi

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
    
    sec_per_year = 3.154e7
    dataset['bedrock_erosion__rate'] *= sec_per_year
    dataset['bank_erosion__rate'] *= sec_per_year
    
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

    
    return fig, axes
    
#%%


fig, axes = plot_xy_timeseries_multi(ds1, data_vars_rates, xy_coords)

for ax in axes:
        ax.axvline(x=200000, color='red', linestyle='--')

#for v0 and v1
# axes[0].set_ylim([0, 5])
# axes[1].set_ylim([0, 0.75])
# axes[2].set_ylim([0, 9])
# axes[3].set_ylim([0, 80])
#plot_xy_timeseries_multi(ds2, data_vars_m, xy_coords2)

# axes[0].set_ylim([0, 6])
# axes[1].set_ylim([0, 0.75])
# axes[2].set_ylim([0, 16])
# axes[3].set_ylim([0, 80])

#%%




#%%

def generate_summary(ds):
    
    max_time = ds.time.max().values
    
    
    summary_dict = {}
    
    Kbr = ds.attrs['Kr']
    Kbank = ds.attrs['Kbank']
    Ksed = ds.attrs['Ks']
    grid_area = ds.attrs['nx'] * ds.attrs['ny'] * ds.attrs['dx']
    Kb_ratio = Kbank/Kbr
    V = ds.attrs['V_mperyr']
    
    runoff_rate = ds.attrs['runoff_mperyr']
    
    
    #data variables to average
    vars_to_avg = ['topographic__elevation', 
                   'bank_erosion__rate' 
                   'bedrock_erosion__rate',
                   'soil__depth',
                   'channel_bedrock__width',
                   'flow__depth']
  
    for var in vars_to_avg:
        
        da = ds[var].sel(time=max_time)
        
        var_mean = da.mean()
        
        summary_dict.update({ var : var_mean})
    
    wdr = ds['channel_bedrock__width'].sel(time=max_time) / ds['flow__depth'].sel(time=max_time) 
        

#%%

xy900 = ds1.sel(x=900.0, y=900.0, method='nearest')


var_list_900 = list(xy900.variables)

#what is drainage area??

max_time = ds1.time.max().item()
slope900 = xy900.topographic__steepest_slope.sel(time=max_time).values 


fig, xy_coords, mg, ref_node_ind = plot_channel_prf(ds1, max_time, 'No Sediment')

#%%

slope_ref = mg.at_node['topographic__steepest_slope'][ref_node_ind]
DA_ref = mg.at_node['drainage_area'][ref_node_ind]
bed_er_ref = mg.at_node['bedrock_erosion__rate'][ref_node_ind] #erosion rate in m/yr
bank_er_ref = mg.at_node['bank_erosion__rate'][ref_node_ind]  #erosion rate in m/yr

print('slope ref,', slope_ref)
print('DA ref,', DA_ref)
print('bed_er_ref,', bed_er_ref)
print('bank_er_ref,', bank_er_ref)

Am = DA_ref ** 0.5
Sn = slope_ref ** 1

K_ref = bed_er_ref / (Am * Sn)

#%%

mg = xr_to_mg(ds1, plot_time=0)

#All boundaries are closed except outlet node
mg.set_closed_boundaries_at_grid_edges(bottom_is_closed=True,
                                       left_is_closed=True,
                                       right_is_closed=True,
                                       top_is_closed=True)

#Setting Node 0 (bottom left corner) as outlet node 
mg.set_watershed_boundary_condition_outlet_id(0, 
                                              mg.at_node['topographic__elevation'],
                                              -9999.)


fa1 = PriorityFloodFlowRouter(mg)
fa1.run_one_step()
fsc_uplift = .001 #m/yr
fsc = FastscapeEroder(mg, K_sp=(K_ref/1.13))
fsc_time = 0
fsc_dt = 20

fsc_nts = int(500000/20)

for i in range (fsc_nts):
    
    dz_ad = np.zeros(mg.size('node'))
    dz_ad[mg.core_nodes] = fsc_uplift * fsc_dt
    mg.at_node['topographic__elevation'] += dz_ad
    
    fa1.run_one_step()
    
    fsc.run_one_step(dt=fsc_dt)
    fsc_time += fsc_dt
    
    if fsc_time % 10000 == 0:
        mean_z = np.mean(mg.at_node['topographic__elevation'])
        print(fsc_time, ', mean z= ', mean_z)
    
#%%
plt.figure()
imshow_grid(mg, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')
plt.title('Final Topo')
plt.show()

slope_ref2 = mg.at_node['topographic__steepest_slope'][ref_node_ind]
DA_ref2 = mg.at_node['drainage_area'][ref_node_ind]


print('slope ref,', slope_ref2)
print('DA ref,', DA_ref2)


#%%

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
last_value = ids_array[-3]

x_coords = mg.x_of_node[[first_value, middle_value, last_value]]
y_coords = mg.y_of_node[[first_value, middle_value, last_value]]

xy_coords = list(zip(x_coords, y_coords))

fig = plt.figure()
#imshow_grid(mg1, 'topographic__elevation', colorbar_label='Topographic Elevation (m)')   
prf.plot_profiles_in_map_view()
#plt.title(f'Main Channel, {model_name}, Time={plot_time/1000} kyr') 
plt.plot(x_coords[0], y_coords[0], 'o', color='tab:blue', markersize=3)  
plt.plot(x_coords[1], y_coords[1], 'o', color='orange', markersize=3)  
plt.plot(x_coords[2], y_coords[2], 'o', color='green', markersize=3)  

plt.show()

#%%
mg2 = xr_to_mg(ds1, plot_time=500000)
z_avg_2 = np.mean(mg2.at_node['topographic__elevation'])
print('SDW avg z=', z_avg_2)
