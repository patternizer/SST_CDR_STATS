#!/usr/bin/env python

# PROGRAM: ocean_area.py
# ----------------------------------------------------------------------------------
# Version 0.1
# 5 June, 2019
# michael.taylor AT reading DOT ac DOT uk

import numpy as np
import xarray
import seaborn as sns; sns.set(style="darkgrid")
import matplotlib
import matplotlib.pyplot as plt; plt.close("all")

#
# Calc lat_vec and lat_fraction
#

ds = xarray.open_dataset('landsea_mask.nc')
x = ds.lon
y = ds.lat
z = ds.mask
water = z==1
land = z==2
water_ice = z==9

f = 1 - (np.sum(land[0,:,:],axis=1) / len(x)*1.)
lat_fraction = np.array(f)

dlat = 0.05
lat_vec = np.array(y)

#
# Calc land_vec
#

ocean_area = 361900000.0
R = 6371.0088 # km

# Formula for the area of the Earth between a line of latitude and the north pole (the area of a spherical cap): A = 2*pi*R*h where R is the radius of the earth and h is the perpendicular distance from the plane containing the line of latitude to the pole. We can calculate h using trigonometry: h = R*(1-sin(lat)). The area north of a line of latitude is therefore: A = 2*pi*R^2(1-sin(lat)).

# The area between two lines of latitude is the difference between the area north of one latitude and the area north of the other latitude: A = |2*pi*R^2(1-sin(lat2)) - 2*pi*R^2(1-sin(lat1)) = 2*pi*R^2 |sin(lat1) - sin(lat2)

# The area of a lat-long rectangle is proportional to the difference in the longitudes. The area I just calculated is the area between longitude lines differing by 360 degrees. Therefore the area we seek is: A = 2*pi*R^2 |sin(lat1)-sin(lat2)| |lon1-lon2|/360 = (pi/180)R^2 |sin(lat1)-sin(lat2)| |lon1-lon2|

A = []
N = len(lat_vec)
for i in range(N):

    dA = 2. * np.pi * R**2.0 * np.absolute( np.sin(np.pi/180 * (lat_vec[i]+dlat/2)) - np.sin(np.pi/180 * (lat_vec[i]-dlat/2)))
    A.append(dA)

surface_vec = np.array(A)
ocean_vec = surface_vec * lat_fraction

fig, ax = plt.subplots()
plt.plot(lat_vec, surface_vec, label='surface area')
plt.plot(lat_vec, ocean_vec, label='ocean')
plt.legend()
plt.xlabel('Latitude / degrees')
plt.ylabel(r'Area / $km^{2}$')
title_str = "ETOPO1 ocean_area=" + "{0:.3e}".format(ocean_area) + " calculated=" + "{0:.3e}".format(np.sum(ocean_vec))
file_str = "ocean_area.png"
plt.title(title_str)
fig.tight_layout()
plt.savefig(file_str)

FPE = 100. * (1.0 - np.sum(ocean_vec) / ocean_area)

print('FPE=',FPE)
print('** END')

