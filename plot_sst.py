#!/usr/bin/env python

# PROGRAM: plot_sst.py
# ----------------------------------------------------------------------------------
# Version 0.9
# 13 February, 2019
# michael.taylor AT reading DOT ac DOT uk

# PYTHON DEBUGGER CONTROL:
#------------------------
# import os; os._exit(0)
# import ipdb
# ipdb.set_trace()

import os.path
import optparse
from  optparse import OptionParser 
import sys
import numpy as np
import xarray
import pandas as pd
from pandas import Series, DataFrame, Panel
import seaborn as sns; sns.set(style="darkgrid")
import datetime
import matplotlib
import matplotlib.pyplot as plt; plt.close("all")

from typhon.plots import plot_bitfield
cmap = 'tab20c' # https://matplotlib.org/users/colormaps
    
def calc_median(counts,bins):
    """
    # -------------------------------
    # CALCULATE MEDIUM FROM HISTOGRAM
    # -------------------------------
    # M_estimated ~ L_m + [ ( N/2 - F_{m-1} ) /  f_m]  * c
    #
    # where,
    #
    # L_m =lower limit of the median bar
    # N = is the total number of observations
    # F_{m-1} = cumulative frequency (total number of observations) in all bars below the median bar
    # f_m = frequency of the median bar
    # c = median bar width
    """
    M = 0
    counts_cumsum = counts.values.cumsum()
    counts_half = counts_cumsum[-1]/2.0    
    for i in np.arange(0,bins.shape[0]-1):
        counts_l = counts_cumsum[i]
        counts_r = counts_cumsum[i+1]
        if (counts_half >= counts_l) & (counts_half < counts_r):
            c = bins[1]-bins[0]
            L_m = bins[i+1]
            F_m_minus_1 = counts_cumsum[i]
            f_m = counts.values[i+1]
            M = L_m + ( (counts_half - F_m_minus_1) / f_m ) * c            
    return M

def plot_n_sst(times,n_sst_q3,n_sst_q4,n_sst_q5):
    """
    # ---------------------------------------
    # PLOT CUMULATIVE SST OBSERVATION DENSITY
    # ---------------------------------------
    """    
    ocean_area = 361900000.0
    t = np.array(times, dtype=np.datetime64)
    years = (t[-1] - t[0]).astype('timedelta64[D]') / np.timedelta64(1, 'D') / 365.0
    Q3 = pd.Series(n_sst_q3, index=times).fillna(0) / ocean_area / years
    Q4 = pd.Series(n_sst_q4, index=times).fillna(0) / ocean_area / years
    Q5 = pd.Series(n_sst_q5, index=times).fillna(0) / ocean_area / years
    df = pd.DataFrame({'QL=3':Q3, 'QL=4':Q4, 'QL=5':Q5})
    df['QL=4 & 5'] = df['QL=4'] + df['QL=5']  
    df = df.mask(np.isinf(df))

    fig = plt.figure()
    plt.plot(times,df['QL=4 & 5'].cumsum(), drawstyle='steps')
    plt.plot(times,df['QL=3'].cumsum(), drawstyle='steps')
    plt.tick_params(labelsize=12)
    plt.ylabel("Observation density / $\mathrm{km^{-2} \ yr^{-1}}$", fontsize=12)
    title_str = ' ' + 'QL=3:max=' + "{0:.2f}".format(df['QL=3'].cumsum().max()) + ' ' + 'QL=4 & 5:max=' + "{0:.2f}".format(df['QL=4 & 5'].cumsum().max())
    plt.title(title_str, fontsize=10)
    plt.legend(loc='best')
    plt.savefig('n_sst.png')    

def plot_n_sst_lat(lat_vec,n_sst_q3_lat,n_sst_q4_lat,n_sst_q5_lat):
    """
    # ------------------------------------------
    # PLOT SST OBSERVATION DENSITY WITH LATITUDE
    # ------------------------------------------
    """     
    Q3 = 180.0 * pd.Series(n_sst_q3_lat)
    Q4 = 180.0 * pd.Series(n_sst_q4_lat)
    Q5 = 180.0 * pd.Series(n_sst_q5_lat)
    df = pd.DataFrame({'QL=3':Q3, 'QL=4':Q4, 'QL=5':Q5})
    df['QL=4 & 5'] = df['QL=4'] + df['QL=5']    
    df['QL=3 & 4 & 5'] = df['QL=3'] + df['QL=4'] + df['QL=5']    
    df = df.mask(np.isinf(df))

    fig = plt.figure()
#    plt.plot(df['QL=3'], lat_vec, drawstyle='steps', label='QL=3')
#    plt.plot(df['QL=4 & 5'], lat_vec, drawstyle='steps', label='QL=4 & 5')
    plt.fill_between(lat_vec, df['QL=4 & 5'],  step="pre", alpha=1.0, label='QL=4 & 5')
    plt.fill_between(lat_vec, df['QL=3'], step="pre", alpha=1.0, label='QL=3')
    ax = plt.gca()    
    ax.set_xlim([-85,85])
    ticks = ax.get_xticks()
    ax.set_xticks(np.linspace(-90, 90, 7))
    plt.tick_params(labelsize=12)
    plt.xlabel("Latitude / $\mathrm{\degree N}$", fontsize=12)
    plt.ylabel("Observation density / $\mathrm{km^{-2} \ yr^{-1}}$", fontsize=12)
    plt.legend(loc='best')
    plt.savefig('n_sst_lat.png')

def plot_histogram_sst(sst_midpoints,sst_q3_hist,sst_q4_hist,sst_q5_hist):
    """
    # ------------------------------
    # PLOT HISTOGRAM OF SST + MEDIAN
    # ------------------------------
    """     
    interpolation = np.arange(270,311,1) # 10x original resolution
    Q3 = 10.0 * pd.Series(np.interp(interpolation,sst_midpoints,sst_q3_hist), index=interpolation)
    Q4 = 10.0 * pd.Series(np.interp(interpolation,sst_midpoints,sst_q4_hist), index=interpolation)
    Q5 = 10.0 * pd.Series(np.interp(interpolation,sst_midpoints,sst_q5_hist), index=interpolation)
    df = pd.DataFrame({'QL=3':Q3, 'QL=4':Q4, 'QL=5':Q5})
    df['QL=4 & 5'] = 0.5 * (df['QL=4'] + df['QL=5'])   
    df = df.mask(np.isinf(df))

    fig = plt.figure()
    plt.fill_between(interpolation,df['QL=4 & 5'], step="pre", alpha=0.4)
    plt.fill_between(interpolation,df['QL=3'], step="pre", alpha=0.4)
    plt.plot(interpolation,df['QL=4 & 5'], drawstyle='steps')
    plt.plot(interpolation,df['QL=3'], drawstyle='steps')
    ax = plt.gca()
    ax.set_xlim([270,310])
    plt.tick_params(labelsize=12)
    plt.xlabel("SST / $\mathrm{K}$", fontsize=12)
    plt.ylabel("Frequency / $\mathrm{\% \ K^{-1}}$", fontsize=12)
 
    M3 = calc_median(df['QL=3'],interpolation)
    M4_5 = calc_median(df['QL=4 & 5'],interpolation)
#    title_str = 'QL=3:sum=' + "{0:.2f}".format(df['QL=3'].sum()) + ' ' + 'QL=4 & 5:sum=' + "{0:.2f}".format(df['QL=4 & 5'].sum())
    title_str = 'QL=3:median=' + "{0:.2f}".format(M3) + ' ' + 'QL=4 & 5:median=' + "{0:.2f}".format(M4_5)
    plt.title(title_str, fontsize=10)
    plt.legend(loc='best')
    plt.savefig('hist_sst.png')
    
def plot_histogram_sensitivity(sensitivity_midpoints,sensitivity_q3_hist,sensitivity_q4_hist,sensitivity_q5_hist):
    """
    # ------------------------------------------------
    # PLOT HISTOGRAM OF RETRIEVAL SENSITIVITY + MEDIAN
    # ------------------------------------------------
    """     
    interpolation = np.arange(0.0,2.01,0.01) # 10x original resolution
    Q3 = pd.Series(np.interp(interpolation,sensitivity_midpoints,sensitivity_q3_hist), index=interpolation)
    Q4 = pd.Series(np.interp(interpolation,sensitivity_midpoints,sensitivity_q4_hist), index=interpolation)
    Q5 = pd.Series(np.interp(interpolation,sensitivity_midpoints,sensitivity_q5_hist), index=interpolation)

    df = pd.DataFrame({'QL=3':Q3, 'QL=4':Q4, 'QL=5':Q5})
    df['QL=4 & 5'] = 0.5 * (df['QL=4'] + df['QL=5'])
    df = df.mask(np.isinf(df))
   
    fig = plt.figure()
    plt.fill_between(100.0*interpolation,df['QL=4 & 5'], step="pre", alpha=0.4)
    plt.fill_between(100.0*interpolation,df['QL=3'], step="pre", alpha=0.4) 
    plt.plot(100.0*interpolation,df['QL=4 & 5'], drawstyle='steps')
    plt.plot(100.0*interpolation,df['QL=3'], drawstyle='steps')
    ax = plt.gca()
    ax.set_xlim([88,110])
    plt.tick_params(labelsize=12)
    plt.xlabel("Retrieval sensitivity / $\mathrm{\%}$", fontsize=12)
    plt.ylabel("Frequency / $\mathrm{\% \ {\%}^{-1} }$", fontsize=12)

    M3 = calc_median(df['QL=3'],interpolation)
    M4_5 = calc_median(df['QL=4 & 5'],interpolation)
#    title_str = 'QL=3:sum=' + "{0:.2f}".format(df['QL=3'].sum()) + ' ' + 'QL=4 & 5:sum=' + "{0:.2f}".format(df['QL=4 & 5'].sum())
    title_str = 'QL=3:median=' + "{0:.2f}".format(M3) + ' ' + 'QL=4 & 5:median=' + "{0:.2f}".format(M4_5)
    plt.title(title_str, fontsize=10)
    plt.legend(loc='best')
    plt.savefig('hist_sensitivity.png')

def plot_histogram_total_uncertainty(total_uncertainty_midpoints,total_uncertainty_q3_hist,total_uncertainty_q4_hist,total_uncertainty_q5_hist):
    """
    # --------------------------------------------
    # PLOT HISTOGRAM OF TOTAL UNCERTAINTY + MEDIAN
    # --------------------------------------------
    """     
    interpolation = np.arange(0.00,4.00,0.01) 
    Q3 = pd.Series(np.interp(interpolation,total_uncertainty_midpoints,total_uncertainty_q3_hist), index=interpolation)
    Q4 = pd.Series(np.interp(interpolation,total_uncertainty_midpoints,total_uncertainty_q4_hist), index=interpolation)
    Q5 = pd.Series(np.interp(interpolation,total_uncertainty_midpoints,total_uncertainty_q5_hist), index=interpolation)

    df = pd.DataFrame({'QL=3':Q3, 'QL=4':Q4, 'QL=5':Q5})
    df['QL=4 & 5'] = 0.5 * (df['QL=4'] + df['QL=5'])
    df = df.mask(np.isinf(df))
 
    fig = plt.figure()
    plt.fill_between(total_uncertainty_midpoints,df['QL=4 & 5'], step="pre", alpha=0.4)
    plt.fill_between(total_uncertainty_midpoints,df['QL=3'], step="pre", alpha=0.4)
    plt.plot(total_uncertainty_midpoints,df['QL=4 & 5'], drawstyle='steps')
    plt.plot(total_uncertainty_midpoints,df['QL=3'], drawstyle='steps')
    ax = plt.gca()
    ax.set_xlim([0.0,1.2])
    plt.tick_params(labelsize=12)
    plt.xlabel("Total uncertainty / $\mathrm{K}$", fontsize=12)
    plt.ylabel("Frequency / $\mathrm{\% \ cK^{-1}}$", fontsize=12)

    M3 = calc_median(df['QL=3'],interpolation)
    M4_5 = calc_median(df['QL=4 & 5'],interpolation)
#    title_str = 'QL=3:sum=' + "{0:.2f}".format(df['QL=3'].sum()) + ' ' + 'QL=4 & 5:sum=' + "{0:.2f}".format(df['QL=4 & 5'].sum())
    title_str = 'QL=3:median=' + "{0:.2f}".format(M3) + ' ' + 'QL=4 & 5:median=' + "{0:.2f}".format(M4_5)
    plt.title(title_str, fontsize=10)
    plt.legend(loc='best')
    plt.savefig('hist_total_uncertainty.png')

def plot_n_sst_timeseries(satellites):
    """
    # -------------------------------------------------------
    # PLOT TIMESERIES OF DAILY OBSERVATION DENSITY PER SENSOR
    # -------------------------------------------------------
    """     
    ocean_area = 361900000.0
#    labels = ['NOAA06','NOAA07','NOAA08','NOAA09','NOAA10','NOAA11','NOAA12','NOAA14','NOAA15','NOAA16','NOAA17','NOAA18','NOAA19','METOPA','AATSR','ATSR1','ATSR2']
#    satellites = ['AVHRR06_G','AVHRR07_G','AVHRR08_G','AVHRR09_G','AVHRR10_G','AVHRR11_G','AVHRR12_G','AVHRR14_G','AVHRR15_G','AVHRR16_G','AVHRR17_G','AVHRR18_G','AVHRR19_G','AVHRRMTA_G','AATSR','ATSR1','ATSR2']

#    labels = ['NOAA07','NOAA09','NOAA11','NOAA12','NOAA14','NOAA15','NOAA16','NOAA17','NOAA18','NOAA19','METOPA','AATSR','ATSR1','ATSR2']
#    satellites = ['AVHRR07_G','AVHRR09_G','AVHRR11_G','AVHRR12_G','AVHRR14_G','AVHRR15_G','AVHRR16_G','AVHRR17_G','AVHRR18_G','AVHRR19_G','AVHRRMTA_G','AATSR','ATSR1','ATSR2']

    labels = ['NOAA07','NOAA09','NOAA11']
    satellites = ['AVHRR07_G','AVHRR09_G','AVHRR11_G']

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    N = 0
    lab = []
    for i in range(0,len(satellites)):
        filename = satellites[i] + '_summary.nc'
        ds = xarray.open_dataset(filename)
        dates = ds['time']
        idx = np.argsort(dates, axis=0) 
        t = np.array(dates)[idx]
        days = (t[-1] - t[0]).astype('timedelta64[D]') / np.timedelta64(1, 'D')
        years = days/365.0
        times_duplicates = pd.Series(t)
        times = times_duplicates.drop_duplicates()
        Q4_duplicates = pd.Series(ds['n_sst_q4'].values[idx], index=t)
        Q5_duplicates = pd.Series(ds['n_sst_q5'].values[idx], index=t)
        n_sst_q4 = 100.0 * Q4_duplicates.groupby(Q4_duplicates.index).sum() / ocean_area / years
        n_sst_q5 = 100.0 * Q5_duplicates.groupby(Q5_duplicates.index).sum() / ocean_area / years
        df = DataFrame({'Q4' : n_sst_q4, 'Q5' : n_sst_q5}) 
        df['Sum'] = df['Q4'].fillna(0) + df['Q5'].fillna(0)
        df['Sum_mean'] = df['Sum'].resample("1d").sum().fillna(0).rolling(window=31, min_periods=1).median()
        N += df['Sum_mean'].shape[0]
        lab.append(labels[i])
        df['Sum_mean'].plot(ax=ax1)
    plt.tick_params(labelsize=12)
    title_str = 'QL=4 & 5'
    ax1.set_title(title_str, fontsize=10)

    N = 0
    lab = []
    for i in range(0,len(satellites)):
        filename = satellites[i] + '_summary.nc'
        ds = xarray.open_dataset(filename)
        dates = ds['time']
        idx = np.argsort(dates, axis=0) 
        t = np.array(dates)[idx]
        days = (t[-1] - t[0]).astype('timedelta64[D]') / np.timedelta64(1, 'D')
        years = days/365.0
        times_duplicates = pd.Series(t)
        times = times_duplicates.drop_duplicates()
        Q3_duplicates = pd.Series(ds['n_sst_q3'].values[idx], index=t)
        n_sst_q3 = 100.0 * Q3_duplicates.groupby(Q3_duplicates.index).sum() / ocean_area / years
        df = DataFrame({'Q3' : n_sst_q3})
        df['Q3_mean'] = df['Q3'].resample("1d").sum().rolling(window=31, min_periods=1).median()
        N += df['Q3_mean'].shape[0]
        lab.append(labels[i])
        df['Q3_mean'].plot(ax=ax2)
    plt.tick_params(labelsize=12)
    title_str = 'QL=3'
    ax2.set_title(title_str, fontsize=10)

    fig.legend(lab, fontsize=8, loc=7)
    fig.subplots_adjust(right=0.8)  
    fig.text(0.01, 0.5, 'Observation density / $\mathrm{100 \ km^{-2} \ yr^{-1}}$', va='center', rotation='vertical')
    plt.savefig('n_sst_timeseries.png')

def calc_lat_fraction():
    """
    # ---------------------------------------------------------------
    # EXTRACT OCEAN FRACTION WITH LATITUDE FROM L4 OSTIA LANDSEA MASK
    # ---------------------------------------------------------------
    """     
    # mask:source = "NAVOCEANO_landmask_v1.0 EUMETSAT_OSI-SAF_icemask ARCLake_lakemask"
    # mask:comment = "water land lake ice"
    # mask:flag_masks = 1b, 2b, 4b, 8b, 16b 
    # mask:summary = "OSTIA L4 product from the ESA SST CCI project, produced using OSTIA reanalysis sytem v3.0" 

    ds = xarray.open_dataset('landsea_mask.nc')
    x = ds.lon
    y = ds.lat
    z = ds.mask     
    land = z==2   
    f = 1 - (np.sum(land[0,:,:],axis=1) / len(x)*1.)
    lat_vec = y
    lat_fraction = f

#    dlat = 0.05
#    lat_vec = np.arange(-90,90,dlat)+0.025
#    lat_fraction = np.zeros(len(lat_vec))
#    for i in range(len(lat_vec)):     
#        lat_fraction[i]=f.values[i] 

    exec(open('plot_landsea_mask.py').read())

    return lat_fraction

def load_data(lat_fraction):

    ocean_area = 361900000.0
#    satellites = ['AVHRR06_G','AVHRR07_G','AVHRR08_G','AVHRR09_G','AVHRR10_G','AVHRR11_G','AVHRR12_G','AVHRR14_G','AVHRR15_G','AVHRR16_G','AVHRR17_G','AVHRR18_G','AVHRR19_G','AVHRRMTA_G','AATSR','ATSR1','ATSR2']
#    satellites = ['AVHRR07_G','AVHRR09_G','AVHRR11_G','AVHRR12_G','AVHRR14_G','AVHRR15_G','AVHRR16_G','AVHRR17_G','AVHRR18_G','AVHRR19_G','AVHRRMTA_G','AATSR','ATSR1','ATSR2']
    satellites = ['AVHRR07_G','AVHRR09_G','AVHRR11_G']
    df = []
    for i in range(0,len(satellites)):
        filename = satellites[i] + '_summary.nc'
        dsi = xarray.open_dataset(filename)
        df.append(dsi)
    ds = xarray.concat(df, dim='time')
    dates = ds['time']
    idx = np.argsort(dates, axis=0) 
    t = np.array(dates)[idx]
    days = (t[-1] - t[0]).astype('timedelta64[D]') / np.timedelta64(1, 'D')
    years = days/365.0
    times_duplicates = pd.Series(t)
    times = times_duplicates.drop_duplicates()

    Q3_duplicates = pd.Series(ds['n_sst_q3'].values[idx], index=t)
    Q4_duplicates = pd.Series(ds['n_sst_q4'].values[idx], index=t)
    Q5_duplicates = pd.Series(ds['n_sst_q5'].values[idx], index=t)
    n_sst_q3 = Q3_duplicates.groupby(Q3_duplicates.index).sum()
    n_sst_q4 = Q4_duplicates.groupby(Q4_duplicates.index).sum()
    n_sst_q5 = Q5_duplicates.groupby(Q5_duplicates.index).sum()

    lat_vec = ds['lat_vec']
    n_sst_q3_lat = np.sum(ds['n_sst_q3_lat'],axis=0) / (lat_fraction * ocean_area) / years
    n_sst_q4_lat = np.sum(ds['n_sst_q4_lat'],axis=0) / (lat_fraction * ocean_area) / years
    n_sst_q5_lat = np.sum(ds['n_sst_q5_lat'],axis=0) / (lat_fraction * ocean_area) / years

    sst_midpoints = ds['sst_midpoints']
    sst_q3_hist = 100.0 * np.sum(ds['sst_q3_hist'],axis=0) / np.sum(np.sum(ds['sst_q3_hist'],axis=0))
    sst_q4_hist = 100.0 * np.sum(ds['sst_q4_hist'],axis=0) / np.sum(np.sum(ds['sst_q4_hist'],axis=0))
    sst_q5_hist = 100.0 * np.sum(ds['sst_q5_hist'],axis=0) / np.sum(np.sum(ds['sst_q5_hist'],axis=0))

    sensitivity_midpoints = ds['sensitivity_midpoints']
    sensitivity_q3_hist = 100.0 * np.sum(ds['sensitivity_q3_hist'],axis=0) / np.sum(np.sum(ds['sensitivity_q3_hist'],axis=0)) 
    sensitivity_q4_hist = 100.0 * np.sum(ds['sensitivity_q4_hist'],axis=0) / np.sum(np.sum(ds['sensitivity_q4_hist'],axis=0)) 
    sensitivity_q5_hist = 100.0 * np.sum(ds['sensitivity_q5_hist'],axis=0) / np.sum(np.sum(ds['sensitivity_q5_hist'],axis=0)) 

    total_uncertainty_midpoints = ds['total_uncertainty_midpoints']
    total_uncertainty_q3_hist = 100.0 * np.sum(ds['total_uncertainty_q3_hist'],axis=0) / np.sum(np.sum(ds['total_uncertainty_q3_hist'],axis=0))
    total_uncertainty_q4_hist = 100.0 * np.sum(ds['total_uncertainty_q4_hist'],axis=0) / np.sum(np.sum(ds['total_uncertainty_q4_hist'],axis=0))
    total_uncertainty_q5_hist = 100.0 * np.sum(ds['total_uncertainty_q5_hist'],axis=0) / np.sum(np.sum(ds['total_uncertainty_q5_hist'],axis=0))
    
    plot_histogram_sst(sst_midpoints,sst_q3_hist,sst_q4_hist,sst_q5_hist)
    plot_histogram_sensitivity(sensitivity_midpoints,sensitivity_q3_hist,sensitivity_q4_hist,sensitivity_q5_hist)
    plot_histogram_total_uncertainty(total_uncertainty_midpoints,total_uncertainty_q3_hist,total_uncertainty_q4_hist,total_uncertainty_q5_hist)
    plot_n_sst_lat(lat_vec,n_sst_q3_lat,n_sst_q4_lat,n_sst_q5_lat)
    plot_n_sst(times,n_sst_q3,n_sst_q4,n_sst_q5)
    plot_n_sst_timeseries(satellites)
 
if __name__ == "__main__":

    lat_fraction = calc_lat_fraction()
    load_data(lat_fraction)


