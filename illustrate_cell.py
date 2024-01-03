# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 06:51:28 2023

@author: gjg882
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import xarray as xr
import math

#%%



w_br = 6 #width of channel bedrock
H = 1.0  #sediment thickness
h = 2.0 #flow depth
theta_deg = 60 #bank angle in degrees

def draw_channel(w_br, H, h, theta_deg):

    thetarad = math.radians(theta_deg)
    
    w_sed = w_br + 2 * H / np.tan(thetarad)
    w_ws = w_sed + 2 * h / np.tan(thetarad)
    
    
    b1 = (w_ws - w_sed) / 2
    b2 = (w_sed - w_br) / 2
    b_tot = b1 + b2
    
    x1 = 1
    x2 = b1 + 1
    x3 = b_tot + 1
    x4 = b_tot + w_br + 1
    x5 = x4 + b2 
    x6 = w_ws + 1
    
    y1 = h + H + 1
    y2 = H + 1
    y3 = 1
    
    points_water = [[x1, y1], [x2, y2], [x5, y2], [x6, y1]]
    points_water.append(points_water[0]) #close the polygon
    
    points_sed = [[x2, y2], [x3, y3], [x4, y3], [x5, y2]]
    points_sed.append(points_sed[0]) #close the polygon
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set_xlim(0, w_ws+2)
    ax.set_ylim(0, H+h+2)
    
    sed_trap = plt.Polygon(points_sed,  fill=True, facecolor='sandybrown', edgecolor='dimgrey')
    water_trap = plt.Polygon(points_water,  fill=True, facecolor='lightblue', edgecolor='dimgrey')
    
    
    ax.add_patch(water_trap)
    ax.add_patch(sed_trap)

#%%

draw_channel( w_br, H, h, theta_deg)

