# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:50:36 2023

@author: gjg882
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import time


#%%INPUTS
#Inputs for single cross section of Lague/SPACE dynamic width
#Converted Joel's MATLAB code into python

# inputfile_lagueinspace
# Contains input variables for lagueinspace_xs.py
# Model is just a single cross section, so Qw, Qs, S don't evolve.


start_time = time.time() #Track model runtime


Qw = 100  # Water discharge, in m3/s
Qs = 0.1  # Sediment load, in m3/s (volume supply rate)
# S_init = 0.02  # Reach slope, m/m
S_init = 0.01527  # Reach slope, m/m

thetadeg = 60  # Sidewall (bank) angle, measured from horizontal. Lague(2010) used 60.
# wr_init = 10  # Bedrock bed width, initial, in m.
wr_init = 27.66  # Bedrock bed width, initial, in m.
# H_init = 0.5  # Initial sediment depth on bed, in m.
H_init = 0.68  # Initial sediment depth on bed, in m.
manningsn = 0.05  # Manning's n, same value as Lague used. s/m^(1/3)
wrmin = 0.1  # Minimum bedrock bed width, code just stops if it gets this small.

U_initguess = 2.75  # Initial guess of avg velocity, in m/s, just to start the iteration process.

# SPACE model variables:
Kr = 0.000000000001  # Rock erodibility, units 1/m. NOTE: Shobe et al. 2017 used 0.001 (typically), but psibed stream power coefficient likely should change this value.
Ks = 0.00000000001  # Sediment entrainability, units 1/m
Kbank = 0.000000000001  # Try order of magnitude smaller than Kr
n = 1.5  # Exponent on slope! Not Manning's n
Hstar = 1  # e-folding length for cover effect, in m
omegacr = 0  # Unit stream power threshold for bedrock erosion. Note that Shobe et al. (2017) have units as m/year, I'm using m/s.
omegacs = 0  # Unit stream power threshold for sediment entrainment
omegacbank = 0  # Unit stream power threshold for bank erosion
V_mperyr = 10  # Grace uses 1, 3, 5. Convert to m/s below
porosity = 0  # phi in SPACE model paper
Ff = 0  # Fraction fines

timestep_s = 31536000  # Timestep duration, in seconds. Note that SPACE model uses 1 year.
maxitnum = 2000000 - 10  # For now, # of iterations of model to do

distdownstrtozeroelev_m = 10000  # Distance downstream to baselevel fixed to zero, e.g. sea level.
#   ASSUMING channel slope "hinges" around this point, i.e. downcutting
#   decreases slope, and slope stays constant this distance down to zero
#   elev.
# Code will calculate initial bedrock elevation (R_init) based on
# distdownstrtozeroelev_m and S_init.
Upliftrate_mmperyr = 1  # Uplift rate in mm/year, converted below to m/s

# Other variables, generally won't change:
rhow = 1000  # Water density, kg/m^3
rhos = 2500  # Sediment density, kg/m^3
g = 9.81  # Gravitational acceleration, m/s^2

error_tolfraction = 0.001  # Error tolerance, given as a fraction, i.e. 0.01 would
# be 1%, meaning that code would stop running if the difference
# between calculated and desired Q, divided by desired Q, was less than 1%.
# dampingfactor = 0.8  # Used in iterations for depth and velocity
dampingfactor = 0.5  # Used in iterations for depth and velocity

# Calculate, convert:
thetarad = math.radians(thetadeg)  # Bank angle in radians
# Calculate ws_init, initial sediment bed width:
V = V_mperyr * 1 / (60 * 60 * 24 * 365)  # Convert effective settling velocity V to meters per second.
Upliftrate_mpers = Upliftrate_mmperyr / (1000 * 60 * 60 * 24 * 365)
R_init = distdownstrtozeroelev_m * S_init - H_init  # Initial elevation (m above sea level)
timestep_yrs = timestep_s / (60 * 60 * 24 * 365)


#%%DRIVER

ws_init = wr_init + 2 * H_init / math.tan(thetarad)
h_initguess = Qw / (U_initguess * ws_init)  # in meters, initial h guess
h = h_initguess
H = H_init
ws = ws_init
S = S_init  # But for now, keeping slope constant
R = R_init
wr = wr_init
hall = [np.nan]  # tracking temporarily, don't have value yet so NaN
Uall = [np.nan]
Hall = [H]
wrall = [wr]
wsall = [ws]
Erall = [np.nan]  # tracking temporarily, don't have value yet so NaN
Esall = [np.nan]  # tracking temporarily, don't have value yet so NaN
Ebankall = [np.nan]  # tracking temporarily, don't have value yet so NaN
Dsall = [np.nan]
Rall = [R]  # bedrock bed elevation, m above sea level (0)
Sall = [S]
dQsdxall = [np.nan]
dQsdx_origall = [np.nan]

# Deposition rate; in this formulation (only) stays constant since Qs and Qw stay constant.
Ds = Qs / Qw * V  # Note that this is the only place Qs goes into the calculation, which seems odd to me.


#maxitnum = 100  # Set the maximum number of iterations

for itnum in range(maxitnum):  # iterate
    Qwerrorfraction = 10  # just a dummy initial value to start iterating for depth, reset before each iteration
    manning_itnum = 0
    htmpall = []

    # At 200,000 yrs, triple the water discharge
    if itnum == 200000:  # essentially equilibrated from initial conditions by this point
        Qw = Qw * 3
        Ds = Qs / Qw * V

    # At 1 million years, double the sediment supply rate
    if itnum == 1000000:  # essentially equilibrated from initial conditions by this point
        Qs = Qs * 1.5
        Ds = Qs / Qw * V

    # Loop to iterate for depth and velocity from Manning's equation
    while Qwerrorfraction > error_tolfraction:
        # calculate estimate of discharge
        Qwg = (1 / manningsn) * ((h * (ws + h / math.tan(thetarad))) ** (5 / 3)) * (
                    (ws + 2 * h / math.sin(thetarad)) ** (-2 / 3)) * S ** 0.5  # Qwg for Qw guess, ie calculation
        Qwerrorfraction = np.abs((Qw - Qwg) / Qw)
        manning_itnum += 1
        if Qwerrorfraction <= error_tolfraction:
            rh = h * (ws + h / np.tan(thetarad)) / (ws + 2 * h / math.sin(thetarad))  # hydraulic radius
            U = (1 / manningsn) * (rh ** (2 / 3)) * S ** 0.5  # Calculate velocity, done!
        else:
            # modify h guess, proportionally
            htmpall.append(h)
            cf = (Qw / Qwg - 1) * dampingfactor + 1
            h *= cf  # Correction factor that I made up; I know there are ways to converge faster and more reliably!
            if manning_itnum % 100 == 0:  # check in every 100 iterations
                print('depth iterations, lagueinspace_xs.m')
                input("Press Enter to continue...")

    hall.append(h)
    Uall.append(U)
    print(itnum)

    # Calculate width at water surface, and depth-averaged wetted width
    wws = ws + 2 * h / np.tan(thetarad)  # ws = width of sediment bed
    wwa = (ws + wws) / 2  # depth-averaged width, used for normalizing discharge

    q = Qw / wwa

    # Calculate stream power partitioning between bed and banks
    Fw = 1.78 * (wws / h * np.sin(thetarad) - 2 * np.cos(thetarad) + 1.5) ** (-1.4)
    # using psi (because it kinda looks like a w for width) as a correction
    # factor or coefficient (kind of subsumed into Kr, Ks) for calculating Er and Es
    psibank = rhow * g * Fw / 2 * (wws / h * np.sin(thetarad) - np.cos(thetarad))
    psibed = rhow * g * (1 - Fw) / 2 * (1 + wws * np.tan(thetarad) / (wws * np.tan(thetarad) - 2 * h))

    # Calculate Er and Es
    Er = (Kr * psibed * q * S ** n - omegacr) * np.exp(-H / Hstar)  # bedrock BED erosion rate beneath cover. in m/s in this
    Es = (Ks * psibed * q * S ** n - omegacs) * (1 - np.exp(-H / Hstar))  # sediment entrainment rate from bed.
    Ebank = Kbank * psibank * q * S ** n - omegacbank

    # Now calculate SPACE model changing variables
    # Calculate change in sed layer thickness
    dHdt = (Ds - Es) / (1 - porosity)
    H = H + dHdt * timestep_s  # UPDATED sed thickness
    if H < 0:
        print('Oops, H<0, set to zero')
        input("Press Enter to continue...")
        H = 0
    Hall.append(H)  # record sed thickness in each timestep

    # change in width of bedrock bed
    dwrdt = 2 * (Ebank / np.sin(thetarad) - Er / np.tan(thetarad))
    wr = wr + dwrdt * timestep_s  # UPDATED bedrock width
    if wr < wrmin:
        print('Oops, wr too small')
        input("Press Enter to continue...")
    ws = wr + 2 * H / np.tan(thetarad)  # width of sediment bed, in meters

    R = R + (Upliftrate_mpers - Er) * timestep_s  # new bedrock bed elevation
    S = (R + H) / distdownstrtozeroelev_m  # new slope, rise/run

    dQsdx = ws * (Es + (1 - Ff) * (Er * wr / ws + 2 * Ebank * (h + H) / (ws * np.sin(thetarad))) - Ds)  # TOTAL change in sed flux, not per unit width
    dQsdx_orig = ws * (Es + (1 - Ff) * Er * wr / ws - Ds)

    wsall.append(ws)
    wrall.append(wr)
    Erall.append(Er)
    Esall.append(Es)
    Ebankall.append(Ebank)
    Dsall.append(Ds)
    Sall.append(S)
    dQsdxall.append(dQsdx)
    dQsdx_origall.append(dQsdx_orig)

end_time = time.time()
model_time = round((end_time - start_time) / 60)
print('Model run time =', model_time, 'minutes')

# =============================================================================
# lagueinspaceout['Hall'] = Hall
# lagueinspaceout['hall'] = hall
# lagueinspaceout['Uall'] = Uall
# lagueinspaceout['wrall'] = wrall
# lagueinspaceout['Erall'] = Erall
# lagueinspaceout['Esall'] = Esall
# lagueinspaceout['Ebankall'] = Ebankall
# =============================================================================

#%%PLOT MODEL OUTPUT


yrs = np.arange(itnum + 2 ) * timestep_yrs

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(yrs, Hall, 'b-')
plt.plot(yrs, hall, 'r-')
plt.plot(yrs, Uall, 'g-')
plt.legend(['H, sed thickness', 'h, water depth', 'U, water vel.'])
plt.xlabel('years')
plt.ylabel('m, m, m/s')
plt.title('lagueinspace_xs.m: Starts~equilib, 3x Qw increase at 0.2 myr, 1.5x Qs increase at 1 myr')


plt.subplot(2, 2, 2)
plt.plot(yrs, wrall, 'k-')
plt.plot(yrs, wsall, 'r-')
plt.legend(['w_r, bedrock bed width', 'w_s, sed bed width'])
plt.ylabel('width, m')
plt.xlabel('years')

plt.subplot(2, 2, 3)
plt.plot(yrs, Erall, 'b-')
plt.plot(yrs, Esall, 'r-')
plt.plot(yrs, Dsall, 'k-')
plt.plot(yrs, Ebankall, 'g-')
plt.legend(['Er', 'Es', 'Ds', 'Ebank'])
plt.ylabel('m/s')
plt.xlabel('years')

plt.subplot(2, 2, 4)
plt.plot(yrs, Sall, 'b-')
plt.ylabel('Reach slope')
plt.xlabel('years')

#print('Can make plots etc from within the running program. type "dbquit" to end')
#input("Press Enter to continue...")
