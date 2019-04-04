#!/usr/bin/env python

# PROGRAM: plot_sst.py
# ----------------------------------------------------------------------------------
# Version 0.14
# 27 March, 2019
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

#from typhon.plots import plot_bitfield
#cmap = 'tab20c' # https://matplotlib.org/users/colormaps
    
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
    counts_cumsum = counts.cumsum()
    counts_half = counts_cumsum[-1]/2.0    
    for i in np.arange(0,bins.shape[0]-1):
        counts_l = counts_cumsum[i]
        counts_r = counts_cumsum[i+1]
        if (counts_half >= counts_l) & (counts_half < counts_r):
            c = bins[1]-bins[0]
            L_m = bins[i+1]
            F_m_minus_1 = counts_cumsum[i]
            f_m = counts[i+1]
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
    title_str = ' ' + 'QL=3:max=' + "{0:.5f}".format(df['QL=3'].cumsum().max()) + ' ' + 'QL=4 & 5:max=' + "{0:.5f}".format(df['QL=4 & 5'].cumsum().max())
    print(title_str)
    plt.legend(loc='best')
    plt.savefig('n_sst.png', dpi=600)    
#    plt.savefig(file.png, edgecolor='black', dpi=400, facecolor='white', transparent=True)

def plot_n_sst_lat(lat_vec,n_sst_q3_lat,n_sst_q4_lat,n_sst_q5_lat):
    """
    # ------------------------------------------
    # PLOT SST OBSERVATION DENSITY WITH LATITUDE
    # ------------------------------------------
    """     

    interpolation = np.arange(-90,90,1)
    multiplier = 1.0
    Q3 = multiplier * pd.Series(np.interp(interpolation,lat_vec,n_sst_q3_lat), index=interpolation)
    Q4 = multiplier * pd.Series(np.interp(interpolation,lat_vec,n_sst_q4_lat), index=interpolation)
    Q5 = multiplier * pd.Series(np.interp(interpolation,lat_vec,n_sst_q5_lat), index=interpolation)
    df = pd.DataFrame({'QL=3':Q3, 'QL=4':Q4, 'QL=5':Q5})
    df['QL=4 & 5'] = df['QL=4'] + df['QL=5']    
    df['QL=3 & 4 & 5'] = df['QL=3'] + df['QL=4'] + df['QL=5']    
    df = df.mask(np.isinf(df))

    fig = plt.figure()
    plt.fill_between(interpolation, df['QL=4 & 5'],  step="post", alpha=0.4)
    plt.fill_between(interpolation, df['QL=3'], step="post", alpha=0.4)
    plt.plot(interpolation, df['QL=4 & 5'], drawstyle='steps-post', label='QL=4 & 5')
    plt.plot(interpolation, df['QL=3'], drawstyle='steps-post', label='QL=3') 
    ax = plt.gca()    
    ax.set_xlim([-90,90])
    ticks = ax.get_xticks()
    ax.set_xticks(np.linspace(-90, 90, 7))
    plt.tick_params(labelsize=12)
    plt.xlabel("Latitude / $\mathrm{\degree N}$", fontsize=12)
    plt.ylabel("Observation density / $\mathrm{km^{-2} \ yr^{-1}}$", fontsize=12)
    plt.legend(loc='best')
    plt.savefig('n_sst_lat.png', dpi=600)

def plot_histogram_sst(sst_midpoints,sst_q3_hist,sst_q4_hist,sst_q5_hist):
    """
    # ------------------------------
    # PLOT HISTOGRAM OF SST + MEDIAN
    # ------------------------------
    """     
#    interpolation = np.arange(260.05,319.95,0.1) # original bin midpoints    
    i = np.arange(260,320,0.1) # bin edges
    n = len(i)
    m = 1.0
    q3 = m * pd.Series(np.interp(i,sst_midpoints,sst_q3_hist), index=i) 
    q4 = m * pd.Series(np.interp(i,sst_midpoints,sst_q4_hist), index=i)
    q5 = m * pd.Series(np.interp(i,sst_midpoints,sst_q5_hist), index=i)
    dq = pd.DataFrame({'QL=3':q3, 'QL=4':q4, 'QL=5':q5})
    dq['QL=4 & 5'] = 0.5 * (dq['QL=4'] + dq['QL=5'])   
#    dq = dq.mask(np.isinf(df))
    M3 = calc_median(dq['QL=3'].values,i[0:n])
    M4_5 = calc_median(dq['QL=4 & 5'].values,i[0:n])
    
    interpolation = np.arange(260,320,1) # 10x original resolution
    n = len(interpolation)
    multiplier = 10.0
    Q3 = multiplier * pd.Series(np.interp(interpolation,sst_midpoints,sst_q3_hist), index=interpolation) 
    Q4 = multiplier * pd.Series(np.interp(interpolation,sst_midpoints,sst_q4_hist), index=interpolation)
    Q5 = multiplier * pd.Series(np.interp(interpolation,sst_midpoints,sst_q5_hist), index=interpolation)
    df = pd.DataFrame({'QL=3':Q3, 'QL=4':Q4, 'QL=5':Q5})
    df['QL=4 & 5'] = 0.5 * (df['QL=4'] + df['QL=5'])   
#    df = df.mask(np.isinf(df))

    fig = plt.figure()
    plt.fill_between(interpolation,df['QL=4 & 5'], step="post", alpha=0.4)
    plt.fill_between(interpolation,df['QL=3'], step="post", alpha=0.4)
    plt.plot(interpolation,df['QL=4 & 5'], drawstyle='steps-post')
    plt.plot(interpolation,df['QL=3'], drawstyle='steps-post')
    ax = plt.gca()
    ax.set_xlim([260,310])
    plt.tick_params(labelsize=12)
    plt.xlabel("SST / $\mathrm{K}$", fontsize=12)
    plt.ylabel("Frequency / $\mathrm{\% \ K^{-1}}$", fontsize=12)

    title_str = 'SST: QL=3:median=' + "{0:.5f}".format(M3) + ' ' + 'QL=4 & 5:median=' + "{0:.5f}".format(M4_5)
    print(title_str)
    plt.legend(loc='best')
    plt.savefig('hist_sst.png', dpi=600)
    
def plot_histogram_sensitivity(sensitivity_midpoints,sensitivity_q3_hist,sensitivity_q4_hist,sensitivity_q5_hist):
    """
    # ------------------------------------------------
    # PLOT HISTOGRAM OF RETRIEVAL SENSITIVITY + MEDIAN
    # ------------------------------------------------
    """     
#    interpolation = np.arange(0.005,1.995,0.01) # original bin midpoints
    interpolation = np.arange(0,2,0.01)
    n = len(interpolation)
    multiplier = 1.0
    Q3 = multiplier * pd.Series(np.interp(interpolation,sensitivity_midpoints,sensitivity_q3_hist), index=interpolation)
    Q4 = multiplier * pd.Series(np.interp(interpolation,sensitivity_midpoints,sensitivity_q4_hist), index=interpolation)
    Q5 = multiplier * pd.Series(np.interp(interpolation,sensitivity_midpoints,sensitivity_q5_hist), index=interpolation)
    df = pd.DataFrame({'QL=3':Q3, 'QL=4':Q4, 'QL=5':Q5})
    df['QL=4 & 5'] = 0.5 * (df['QL=4'] + df['QL=5'])
#    df = df.mask(np.isinf(df))
    M3 = calc_median(df['QL=3'].values,interpolation[0:n])
    M4_5 = calc_median(df['QL=4 & 5'].values,interpolation[0:n])
   
    fig = plt.figure()
    plt.fill_between(100.0*interpolation,df['QL=4 & 5'], step="post", alpha=0.4)
    plt.fill_between(100.0*interpolation,df['QL=3'], step="post", alpha=0.4) 
    plt.plot(100.0*interpolation,df['QL=4 & 5'], drawstyle='steps-post')
    plt.plot(100.0*interpolation,df['QL=3'], drawstyle='steps-post')
    ax = plt.gca()
    ax.set_xlim([85,110])
    plt.tick_params(labelsize=12)
    plt.xlabel("Retrieval sensitivity / $\mathrm{\%}$", fontsize=12)
    plt.ylabel("Frequency / $\mathrm{\% \ {\%}^{-1} }$", fontsize=12)

    title_str = 'Sensitivity: QL=3:median=' + "{0:.5f}".format(M3) + ' ' + 'QL=4 & 5:median=' + "{0:.5f}".format(M4_5)
    print(title_str)
    plt.legend(loc='best')
    plt.savefig('hist_sensitivity.png', dpi=600)

def plot_histogram_total_uncertainty(total_uncertainty_midpoints,total_uncertainty_q3_hist,total_uncertainty_q4_hist,total_uncertainty_q5_hist):
    """
    # --------------------------------------------
    # PLOT HISTOGRAM OF TOTAL UNCERTAINTY + MEDIAN
    # --------------------------------------------
    """     
#    interpolation = np.arange(0.005,3.995+0.01,0.01) # original bin midpoints
    interpolation = np.arange(0,4,0.01)
    n = len(interpolation)
    multiplier = 1.0
    Q3 = multiplier * pd.Series(np.interp(interpolation,total_uncertainty_midpoints,total_uncertainty_q3_hist), index=interpolation)
    Q4 = multiplier * pd.Series(np.interp(interpolation,total_uncertainty_midpoints,total_uncertainty_q4_hist), index=interpolation)
    Q5 = multiplier * pd.Series(np.interp(interpolation,total_uncertainty_midpoints,total_uncertainty_q5_hist), index=interpolation)
    df = pd.DataFrame({'QL=3':Q3, 'QL=4':Q4, 'QL=5':Q5})
    df['QL=4 & 5'] = 0.5 * (df['QL=4'] + df['QL=5'])
#    df = df.mask(np.isinf(df))
    M3 = calc_median(df['QL=3'].values,interpolation[0:n])
    M4_5 = calc_median(df['QL=4 & 5'].values,interpolation[0:n])
 
    fig = plt.figure()
    plt.fill_between(total_uncertainty_midpoints,df['QL=4 & 5'], step="post", alpha=0.4)
    plt.fill_between(total_uncertainty_midpoints,df['QL=3'], step="post", alpha=0.4)
    plt.plot(total_uncertainty_midpoints,df['QL=4 & 5'], drawstyle='steps-post')
    plt.plot(total_uncertainty_midpoints,df['QL=3'], drawstyle='steps-post')
    ax = plt.gca()
    ax.set_xlim([0.0,1.25])
    plt.tick_params(labelsize=12)
    plt.xlabel("Total uncertainty / $\mathrm{K}$", fontsize=12)
    plt.ylabel("Frequency / $\mathrm{\% \ cK^{-1}}$", fontsize=12)

    title_str = 'Uncertainty: QL=3:median=' + "{0:.5f}".format(M3) + ' ' + 'QL=4 & 5:median=' + "{0:.5f}".format(M4_5)
    print(title_str)
    plt.legend(loc='best')
    plt.savefig('hist_total_uncertainty.png', dpi=600)

def plot_histogram_total_uncertainty2(total_uncertainty_midpoints,total_uncertainty_q3_hist_avhrr,total_uncertainty_q4_hist_avhrr,total_uncertainty_q5_hist_avhrr,total_uncertainty_q3_hist_atsr,total_uncertainty_q4_hist_atsr,total_uncertainty_q5_hist_atsr):
    """
    # --------------------------------------------------------------
    # PLOT HISTOGRAM OF TOTAL UNCERTAINTY + MEDIAN FOR AVHRR VS ATSR
    # --------------------------------------------------------------
    """     
#    interpolation = np.arange(0.005,3.995,0.01) # original bin midpoints
    interpolation = np.arange(0,4,0.01)
    n = len(interpolation)
    multiplier = 1.0

    Q3_avhrr = multiplier * pd.Series(np.interp(interpolation,total_uncertainty_midpoints,total_uncertainty_q3_hist_avhrr), index=interpolation)
    Q4_avhrr = multiplier * pd.Series(np.interp(interpolation,total_uncertainty_midpoints,total_uncertainty_q4_hist_avhrr), index=interpolation)
    Q5_avhrr = multiplier * pd.Series(np.interp(interpolation,total_uncertainty_midpoints,total_uncertainty_q5_hist_avhrr), index=interpolation)
    df_avhrr = pd.DataFrame({'QL=3':Q3_avhrr, 'QL=4':Q4_avhrr, 'QL=5':Q5_avhrr})
#    df_avhrr['QL=4 & 5'] = 0.5 * (df_avhrr['QL=4'] + df_avhrr['QL=5'])
    df_avhrr['QL=4 & 5'] = df_avhrr['QL=5']
#    df_avhrr = df_avhrr.mask(np.isinf(df_avhrr))

    Q3_atsr = multiplier * pd.Series(np.interp(interpolation,total_uncertainty_midpoints,total_uncertainty_q3_hist_atsr), index=interpolation)
    Q4_atsr = multiplier * pd.Series(np.interp(interpolation,total_uncertainty_midpoints,total_uncertainty_q4_hist_atsr), index=interpolation)
    Q5_atsr = multiplier * pd.Series(np.interp(interpolation,total_uncertainty_midpoints,total_uncertainty_q5_hist_atsr), index=interpolation)
    df_atsr = pd.DataFrame({'QL=3':Q3_atsr, 'QL=4':Q4_atsr, 'QL=5':Q5_atsr})
    df_atsr['QL=4 & 5'] = 0.5 * (df_atsr['QL=4'] + df_atsr['QL=5'])
#    df_atsr = df_atsr.mask(np.isinf(df_atsr))

    fig = plt.figure()    
    plt.fill_between(total_uncertainty_midpoints,df_avhrr['QL=4 & 5'], step="post", alpha=0.4)
    plt.fill_between(total_uncertainty_midpoints,df_avhrr['QL=3'], step="post", alpha=0.4)
    plt.plot(total_uncertainty_midpoints,df_avhrr['QL=4 & 5'], drawstyle='steps-post')
    plt.plot(total_uncertainty_midpoints,df_avhrr['QL=3'], drawstyle='steps-post')
    ax = plt.gca()
    ax.set_xlim([0.0,1.25])
    plt.tick_params(labelsize=12)
    plt.xlabel("Total uncertainty / $\mathrm{K}$", fontsize=12)
    plt.ylabel("Frequency / $\mathrm{\% \ cK^{-1}}$", fontsize=12)
    M3 = calc_median(df_avhrr['QL=3'].values,interpolation[0:n])
    M4_5 = calc_median(df_avhrr['QL=4 & 5'].values,interpolation[0:n])
    title_str = 'AVHRR: QL=3:median=' + "{0:.5f}".format(M3) + ' ' + 'QL=4 & 5:median=' + "{0:.5f}".format(M4_5)
    print(title_str)
    plt.legend(loc='best')
    plt.savefig('hist_total_uncertainty_avhrr.png', dpi=600)

    fig = plt.figure()    
    plt.fill_between(total_uncertainty_midpoints,df_atsr['QL=4 & 5'], step="post", alpha=0.4)
    plt.fill_between(total_uncertainty_midpoints,df_atsr['QL=3'], step="post", alpha=0.4)
    plt.plot(total_uncertainty_midpoints,df_atsr['QL=4 & 5'], drawstyle='steps-post')
    plt.plot(total_uncertainty_midpoints,df_atsr['QL=3'], drawstyle='steps-post')
    ax = plt.gca()
    ax.set_xlim([0.0,1.25])
    plt.tick_params(labelsize=12)
    plt.xlabel("Total uncertainty / $\mathrm{K}$", fontsize=12)
    plt.ylabel("Frequency / $\mathrm{\% \ cK^{-1}}$", fontsize=12)
    M3 = calc_median(df_atsr['QL=3'].values,interpolation[0:n])
    M4_5 = calc_median(df_atsr['QL=4 & 5'].values,interpolation[0:n])
    title_str = 'ATSR: QL=3:median=' + "{0:.5f}".format(M3) + ' ' + 'QL=4 & 5:median=' + "{0:.5f}".format(M4_5)
    print(title_str)
    plt.legend(loc='best')
    plt.savefig('hist_total_uncertainty_atsr.png', dpi=600)

def calc_n_sst_timeseries(satellites):
    """
    # ---------------------------------------------------------------
    # CALC MEAN OF TIMESERIES OF DAILY OBSERVATION DENSITY PER SENSOR
    # ---------------------------------------------------------------
    """     
    ocean_area = 361900000.0
    labels = ['ATSR1','ATSR2','AATSR','NOAA07','NOAA09','NOAA11','NOAA12','NOAA14','NOAA15','NOAA16','NOAA17','NOAA18','NOAA19','METOPA']

    satellites = ['ATSR1','ATSR2','AATSR','AVHRR07_G','AVHRR09_G','AVHRR11_G','AVHRR12_G','AVHRR14_G','AVHRR15_G','AVHRR16_G','AVHRR17_G','AVHRR18_G','AVHRR19_G','AVHRRMTA_G']
    df_all = pd.DataFrame()
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
        Q4_duplicates = pd.Series(ds['n_sst_q4'].values[idx], index=t)
        Q5_duplicates = pd.Series(ds['n_sst_q5'].values[idx], index=t)
        n_sst_q3 = 365.0 * Q3_duplicates.groupby(Q3_duplicates.index).sum() / ocean_area 
        n_sst_q4 = 365.0 * Q4_duplicates.groupby(Q4_duplicates.index).sum() / ocean_area 
        n_sst_q5 = 365.0 * Q5_duplicates.groupby(Q5_duplicates.index).sum() / ocean_area 
        df = DataFrame({'Q3' : n_sst_q3, 'Q4' : n_sst_q4, 'Q5' : n_sst_q5}) 
        df['Sum'] = df['Q4'] + df['Q5']
        df_all = df_all.append(df,ignore_index=True) 

    satellites_avhrr = ['AVHRR07_G','AVHRR09_G','AVHRR11_G','AVHRR12_G','AVHRR14_G','AVHRR15_G','AVHRR16_G','AVHRR17_G','AVHRR18_G','AVHRR19_G','AVHRRMTA_G']
    df_avhrr = pd.DataFrame()
    for i in range(0,len(satellites_avhrr)):
        filename = satellites_avhrr[i] + '_summary.nc'
        ds = xarray.open_dataset(filename)
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
        n_sst_q3 = 365.0 * Q3_duplicates.groupby(Q3_duplicates.index).sum() / ocean_area 
        n_sst_q4 = 365.0 * Q4_duplicates.groupby(Q4_duplicates.index).sum() / ocean_area 
        n_sst_q5 = 365.0 * Q5_duplicates.groupby(Q5_duplicates.index).sum() / ocean_area 
        df = DataFrame({'Q3' : n_sst_q3, 'Q4' : n_sst_q4, 'Q5' : n_sst_q5}) 
        df['Sum'] = df['Q4'] + df['Q5']
        df_avhrr = df_avhrr.append(df,ignore_index=True)

    satellites_atsr = ['AATSR','ATSR1','ATSR2']
    df_atsr = pd.DataFrame()
    for i in range(0,len(satellites_atsr)):
        filename = satellites_atsr[i] + '_summary.nc'
        ds = xarray.open_dataset(filename)
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
        n_sst_q3 = 365.0 * Q3_duplicates.groupby(Q3_duplicates.index).sum() / ocean_area 
        n_sst_q4 = 365.0 * Q4_duplicates.groupby(Q4_duplicates.index).sum() / ocean_area 
        n_sst_q5 = 365.0 * Q5_duplicates.groupby(Q5_duplicates.index).sum() / ocean_area 
        df = DataFrame({'Q3' : n_sst_q3, 'Q4' : n_sst_q4, 'Q5' : n_sst_q5}) 
        df['Sum'] = df['Q4'] + df['Q5']
        df_atsr = df_atsr.append(df,ignore_index=True) 

    return df_all, df_avhrr, df_atsr

def plot_n_sst_timeseries(satellites):
    """
    # -------------------------------------------------------
    # PLOT TIMESERIES OF DAILY OBSERVATION DENSITY PER SENSOR
    # -------------------------------------------------------
    """     
    ocean_area = 361900000.0
    labels = ['ATSR1','ATSR2','AATSR','NOAA07','NOAA09','NOAA11','NOAA12','NOAA14','NOAA15','NOAA16','NOAA17','NOAA18','NOAA19','METOPA']
    satellites = ['ATSR1','ATSR2','AATSR','AVHRR07_G','AVHRR09_G','AVHRR11_G','AVHRR12_G','AVHRR14_G','AVHRR15_G','AVHRR16_G','AVHRR17_G','AVHRR18_G','AVHRR19_G','AVHRRMTA_G']

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    lab = []
    ncolors = len(satellites)
    ax1.set_prop_cycle('color',[plt.cm.gnuplot2(j) for j in np.linspace(0, 1, ncolors)])
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
        n_sst_q4 = 365.0 * Q4_duplicates.groupby(Q4_duplicates.index).sum() / ocean_area 
        n_sst_q5 = 365.0 * Q5_duplicates.groupby(Q5_duplicates.index).sum() / ocean_area 
        df = DataFrame({'Q4' : n_sst_q4, 'Q5' : n_sst_q5}) 
        df['Sum'] = df['Q4'] + df['Q5']
#        df['Sum'] = df['Q4'].fillna(0) + df['Q5'].fillna(0)
#        df['Sum_mean'] = df['Sum'].resample("1d").sum().fillna(0).rolling(window=31, min_periods=1).median()
#        df['Sum_mean'].plot(ax=ax1)

        lab.append(labels[i])
        ax1.plot(times, df['Sum'], '.', markersize=0.2)
        ax1.set_ylim([0,18])
        print(labels[i] + "," + str(df['Sum'].mean()) + "," + str(df['Sum'].shape[0]))

    plt.tick_params(labelsize=12)
    title_str = 'QL=4 & 5'
    ax1.set_title(title_str, fontsize=10)

    lab = []
    ncolors = len(satellites)
    ax2.set_prop_cycle('color',[plt.cm.gnuplot2(j) for j in np.linspace(0, 1, ncolors)])
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
        n_sst_q3 = 365.0 * Q3_duplicates.groupby(Q3_duplicates.index).sum() / ocean_area
        df = DataFrame({'Q3' : n_sst_q3})
#        df['Q3_mean'] = df['Q3'].resample("1d").sum().rolling(window=31, min_periods=1).median()
#        df['Q3_mean'].plot(ax=ax2)

        lab.append(labels[i])
        ax2.plot(times, df['Q3'], '.', markersize=0.2)
        ax2.set_ylim([0,18])
        print(labels[i] + "," + str(df['Q3'].mean()) + "," + str(df['Q3'].shape[0]))

    plt.tick_params(labelsize=12)
    title_str = 'QL=3'
    ax2.set_title(title_str, fontsize=10)

    fig.legend(lab, fontsize=8, loc=7, markerscale=20, scatterpoints=5)
    fig.subplots_adjust(right=0.8)  
    fig.text(0.01, 0.5, 'Observation density / $\mathrm{km^{-2} \ yr^{-1}}$', va='center', rotation='vertical')
    plt.savefig('n_sst_timeseries.png', dpi=600)

def plot_n_sst_boxplots(satellites):
    """
    # --------------------------------------------------------------
    # PLOT YEARLY BOXPLOTS FROM DAILY OBSERVATION DENSITY PER SENSOR
    # --------------------------------------------------------------
    """     
    ocean_area = 361900000.0
    labels = ['ATSR1','ATSR2','AATSR','NOAA07','NOAA09','NOAA11','NOAA12','NOAA14','NOAA15','NOAA16','NOAA17','NOAA18','NOAA19','METOPA']
    satellites = ['ATSR1','ATSR2','AATSR','AVHRR07_G','AVHRR09_G','AVHRR11_G','AVHRR12_G','AVHRR14_G','AVHRR15_G','AVHRR16_G','AVHRR17_G','AVHRR18_G','AVHRR19_G','AVHRRMTA_G']

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
        n_sst_q4 = 365.0 * Q4_duplicates.groupby(Q4_duplicates.index).sum() / ocean_area 
        n_sst_q5 = 365.0 * Q5_duplicates.groupby(Q5_duplicates.index).sum() / ocean_area 
        df = DataFrame({'Q4' : n_sst_q4, 'Q5' : n_sst_q5}) 
        df['Sum'] = df['Q4'] + df['Q5']
        
        fig, ax = plt.subplots(figsize=(12,5))        
        ts = pd.Series(df['Sum'].values, index=times)
        sns.boxplot(ts.index.month, ts, ax=ax)
        title_str = 'QL=4 & 5:' + labels[i]
        ax.set_ylabel('Observation density / $\mathrm{km^{-2} \ yr^{-1}}$')
        ax.set_title(title_str, fontsize=10)
        file_str = 'n_sst_boxplot_' + labels[i] + '_QL4_5' '.png'
        plt.savefig(file_str, dpi=600)
        plt.close("all")        

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
        n_sst_q3 = 365.0 * Q3_duplicates.groupby(Q3_duplicates.index).sum() / ocean_area
        df = DataFrame({'Q3' : n_sst_q3})

        fig, ax = plt.subplots(figsize=(12,5))
        ts = pd.Series(df['Q3'].values, index=times)
        sns.boxplot(ts.index.month, ts, ax=ax)
        title_str = 'QL=3:' + labels[i]
        ax.set_ylabel('Observation density / $\mathrm{km^{-2} \ yr^{-1}}$')
        ax.set_title(title_str, fontsize=10)
        file_str = 'n_sst_boxplot_' + labels[i] + '_QL3' '.png'
        plt.savefig(file_str, dpi=600)
        plt.close("all")        

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
    water = z==1   
    land = z==2   
    water_ice = z==9   

    # water only = 52.42%
    # land only = 33.67%
    # water + ice = 13.91%

    f = 1 - (np.sum(land[0,:,:],axis=1) / len(x)*1.)    
    lat_vec = y
    lat_fraction = f
    
#    exec(open('plot_landsea_mask.py').read())

    return lat_vec, lat_fraction

def load_data(lat_vec, lat_fraction):

    ocean_area = 361900000.0
    satellites = ['AVHRR07_G','AVHRR09_G','AVHRR11_G','AVHRR12_G','AVHRR14_G','AVHRR15_G','AVHRR16_G','AVHRR17_G','AVHRR18_G','AVHRR19_G','AVHRRMTA_G','AATSR','ATSR1','ATSR2']
    df = []
    for i in range(0,len(satellites)):
        filename = satellites[i] + '_summary.nc'
        dsi = xarray.open_dataset(filename)
        df.append(dsi)
        dsi = []
    ds = xarray.concat(df, dim='time')
    df = []
    dates = ds['time']
    idx = np.argsort(dates, axis=0)
    t = np.array(dates)[idx]
    days = (t[-1] - t[0]).astype('timedelta64[D]') / np.timedelta64(1, 'D')
    years = days/365.0
    times_duplicates = pd.Series(t)

    times = times_duplicates.drop_duplicates()
    dates = []
    days = []
    times_duplicates = []

    Q3_duplicates = pd.Series(ds['n_sst_q3'].values[idx], index=t)
    Q4_duplicates = pd.Series(ds['n_sst_q4'].values[idx], index=t)
    Q5_duplicates = pd.Series(ds['n_sst_q5'].values[idx], index=t)
    idx = []

    n_sst_q3 = Q3_duplicates.groupby(Q3_duplicates.index).sum()
    n_sst_q4 = Q4_duplicates.groupby(Q4_duplicates.index).sum()
    n_sst_q5 = Q5_duplicates.groupby(Q5_duplicates.index).sum()
    Q3_duplicates = []
    Q4_duplicates = []
    Q5_duplicates = []

#    lat_vec = ds['lat_vec']
    n_sst_q3_lat = np.sum(ds['n_sst_q3_lat'],axis=0)[0:3600,] / np.array((lat_fraction * ocean_area) / years)
    n_sst_q4_lat = np.sum(ds['n_sst_q4_lat'],axis=0)[0:3600,] / np.array((lat_fraction * ocean_area) / years)
    n_sst_q5_lat = np.sum(ds['n_sst_q5_lat'],axis=0)[0:3600,] / np.array((lat_fraction * ocean_area) / years)
    
    sst_midpoints = ds['sst_midpoints']
    sst_q3_hist = 100.0 * np.sum(ds['sst_q3_hist'],axis=0) / np.sum(np.sum(ds['sst_q3_hist'],axis=0))
    sst_q4_hist = 100.0 * np.sum(ds['sst_q4_hist'],axis=0) / np.sum(np.sum(ds['sst_q4_hist'],axis=0))
    sst_q5_hist = 100.0 * np.sum(ds['sst_q5_hist'],axis=0) / np.sum(np.sum(ds['sst_q5_hist'],axis=0))

    # water only = 52.42%
    # land only = 33.67%
    # water + ice = 13.91%
    # non-land = 66.33%
    n_ocean = (0.5242 + 0.1391) * 3600 * 7200 * len(t)
    n_q3 = np.sum(n_sst_q3)
    n_q4 = np.sum(n_sst_q4)
    n_q5 = np.sum(n_sst_q5)
    clearsky_q3 = n_q3 / n_ocean
    clearsky_q4 = n_q4 / n_ocean
    clearsky_q5 = n_q5 / n_ocean

    sensitivity_midpoints = ds['sensitivity_midpoints']
    sensitivity_q3_hist = 100.0 * np.sum(ds['sensitivity_q3_hist'],axis=0) / np.sum(np.sum(ds['sensitivity_q3_hist'],axis=0)) 
    sensitivity_q4_hist = 100.0 * np.sum(ds['sensitivity_q4_hist'],axis=0) / np.sum(np.sum(ds['sensitivity_q4_hist'],axis=0)) 
    sensitivity_q5_hist = 100.0 * np.sum(ds['sensitivity_q5_hist'],axis=0) / np.sum(np.sum(ds['sensitivity_q5_hist'],axis=0)) 

    total_uncertainty_midpoints = ds['total_uncertainty_midpoints']
    total_uncertainty_q3_hist = 100.0 * np.sum(ds['total_uncertainty_q3_hist'],axis=0) / np.sum(np.sum(ds['total_uncertainty_q3_hist'],axis=0))
    total_uncertainty_q4_hist = 100.0 * np.sum(ds['total_uncertainty_q4_hist'],axis=0) / np.sum(np.sum(ds['total_uncertainty_q4_hist'],axis=0))
    total_uncertainty_q5_hist = 100.0 * np.sum(ds['total_uncertainty_q5_hist'],axis=0) / np.sum(np.sum(ds['total_uncertainty_q5_hist'],axis=0))
    
    satellites_avhrr = ['AVHRR07_G','AVHRR09_G','AVHRR11_G','AVHRR12_G','AVHRR14_G','AVHRR15_G','AVHRR16_G','AVHRR17_G','AVHRR18_G','AVHRR19_G','AVHRRMTA_G']
    df = []
    for i in range(0,len(satellites_avhrr)):
        filename = satellites_avhrr[i] + '_summary.nc'
        dsi = xarray.open_dataset(filename)
        df.append(dsi)
        dsi = []
    ds_avhrr = xarray.concat(df, dim='time')
    total_uncertainty_q3_hist_avhrr = 100.0 * np.sum(ds_avhrr['total_uncertainty_q3_hist'],axis=0) / np.sum(np.sum(ds_avhrr['total_uncertainty_q3_hist'],axis=0))
    total_uncertainty_q4_hist_avhrr = 100.0 * np.sum(ds_avhrr['total_uncertainty_q4_hist'],axis=0) / np.sum(np.sum(ds_avhrr['total_uncertainty_q4_hist'],axis=0))
    total_uncertainty_q5_hist_avhrr = 100.0 * np.sum(ds_avhrr['total_uncertainty_q5_hist'],axis=0) / np.sum(np.sum(ds_avhrr['total_uncertainty_q5_hist'],axis=0))

    satellites_atsr = ['AATSR','ATSR1','ATSR2']
    df = []
    for i in range(0,len(satellites_atsr)):
        filename = satellites_atsr[i] + '_summary.nc'
        dsi = xarray.open_dataset(filename)
        df.append(dsi)
        dsi = []
    ds_atsr = xarray.concat(df, dim='time')
    total_uncertainty_q3_hist_atsr = 100.0 * np.sum(ds_atsr['total_uncertainty_q3_hist'],axis=0) / np.sum(np.sum(ds_atsr['total_uncertainty_q3_hist'],axis=0))
    total_uncertainty_q4_hist_atsr = 100.0 * np.sum(ds_atsr['total_uncertainty_q4_hist'],axis=0) / np.sum(np.sum(ds_atsr['total_uncertainty_q4_hist'],axis=0))
    total_uncertainty_q5_hist_atsr = 100.0 * np.sum(ds_atsr['total_uncertainty_q5_hist'],axis=0) / np.sum(np.sum(ds_atsr['total_uncertainty_q5_hist'],axis=0))

    plot_histogram_sst(sst_midpoints,sst_q3_hist,sst_q4_hist,sst_q5_hist)
    plot_histogram_sensitivity(sensitivity_midpoints,sensitivity_q3_hist,sensitivity_q4_hist,sensitivity_q5_hist)
    plot_histogram_total_uncertainty(total_uncertainty_midpoints,total_uncertainty_q3_hist,total_uncertainty_q4_hist,total_uncertainty_q5_hist)
    plot_histogram_total_uncertainty2(total_uncertainty_midpoints,total_uncertainty_q3_hist_avhrr,total_uncertainty_q4_hist_avhrr,total_uncertainty_q5_hist_avhrr,total_uncertainty_q3_hist_atsr,total_uncertainty_q4_hist_atsr,total_uncertainty_q5_hist_atsr)
    plot_n_sst(times,n_sst_q3,n_sst_q4,n_sst_q5)
    plot_n_sst_timeseries(satellites)
    plot_n_sst_boxplots(satellites)
    plot_n_sst_lat(lat_vec,n_sst_q3_lat,n_sst_q4_lat,n_sst_q5_lat)

    df_all, df_avhrr, df_atsr = calc_n_sst_timeseries(satellites)
    print(df_all.mean())
    print(df_avhrr.mean())
    print(df_atsr.mean())
 
if __name__ == "__main__":

    lat_vec, lat_fraction = calc_lat_fraction()
    load_data(lat_vec, lat_fraction)


