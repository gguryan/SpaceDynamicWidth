
from xmovie import Movie
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from landlab.io.native_landlab import load_grid, save_grid


from landlab.plot import imshow_grid

#import matplotlib.animation as animation
#from IPython.display import HTML
#%%

ds1 = xr.open_dataset('C:/Users/gjg882/Desktop/Projects/SpaceDynamicWidth/NormalFaultSims/Faulted_no_sed.nc')
ds2 = xr.open_dataset('C:/Users/gjg882/Desktop/Projects/SpaceDynamicWidth/NormalFaultSims/Faulted_w_sed.nc')

#%%

topo1 = ds1.topographic__elevation
topo2 = ds2.topographic__elevation

s1 = ds1.topographic__steepest_slope
s2 = ds2.topographic__steepest_slope

w1 = ds1.channel_bedrock__width
w2 = ds2.channel_bedrock__width

#%%

ds1.topographic__steepest_slope.sel(time=800000).plot()

#%%


mg_1 = load_grid('NormalFaultSims/nx50_normal_fault_FromSS_NoSed_MG.nc.grid')
imshow_grid(mg_1, 'channel_bedrock__width')


#%%
# =============================================================================
# def custom_plotfunc(da_slope, da_width, tt):
#     
#     nf_x = [0, 3500]
#     nf_y= [2000, 100]
#     
#     step = 10000
#     
#     fig, axs = plt.subplots(2, 1, figsize=(6, 12), dpi=300)
# 
#     da_slope.sel(time=slice(0,tt+1, step=step)).plot(cmap='jet', ax = axs[0], vmax=3)
#     axs[0].set_title('Slope')
#     axs[0].plot(nf_x, nf_y, color='red')
#     
#     
#     da_width.sel(time=tt).plot(cmap='pink', ax=axs[1], vmax = 30)
#     axs[1].set_title('Channel Width')
#     axs[1].plot(nf_x, nf_y, color='red')
#     
#     return None, None
# 
# =============================================================================


#%%

plot_time = ds2.time.max().values
print(plot_time)

step=100000

plot_times = np.arange(0, plot_time+1000, step=step)


nf_x = [0, 3500]
nf_y= [2000, 100]


fig, axes = plt.subplots(figsize=(8,6))

s2_cbar = ds2.isel(time=-1).topographic__steepest_slope.plot.contourf(cmap='jet', ax=axes)

s2.sel(time=plot_time).plot(ax=axes, add_colorbar=False)
axes.plot(nf_x, nf_y, color='red')


#%%
cbar= fig.colorbar(s2_cbar)
cbar.set_label('Slope')


def animate_slope(plot_time):
    axes.clear()
    
    s2.sel(time=plot_time).plot(ax=axes, add_colorbar=False)
    axes.plot(nf_x, nf_y, color='red')

ani=animation.FuncAnimation(fig, animate_slope, frames=plot_times )


ani.save('animation_slope.gif', writer='Pillow', fps = 4) #Save animation as gif-file

#%%

