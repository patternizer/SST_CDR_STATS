#!/usr/bin/env python

# PROGRAM: sst_lat_l3u_l4.py
# ----------------------------------------------------------------------------------
# Version 0.3
# 13 July, 2019
# michael.taylor AT reading DOT ac DOT uk 

import os
import os.path
import glob
from  optparse import OptionParser
import numpy as np
import xarray
import seaborn as sns; sns.set(style="darkgrid")
import matplotlib
import matplotlib.pyplot as plt; plt.close("all")
from statsmodels import robust as rb

def plot_sst_lat(lat,sst_q3_lat,sst_q5_lat,plotfile):

    file_str = plotfile + ".png"

    fig, ax = plt.subplots()
    plt.plot(sst_q3_lat, lat, '.', color='g', alpha=0.2, label='ql=3 (L3U)')
    plt.plot(sst_q5_lat, lat, '.', color='r', alpha=0.2, label='ql=4&5 (L3U)')
    ax = plt.gca()
    ax.set_ylim([-91,90])
    ticks = ax.get_yticks()
    ax.set_yticks(np.linspace(-90, 90, 7))
    plt.legend(loc=1, fontsize=8)
    plt.xlabel(r'mean SST / $K$')
    plt.ylabel(r'Latitude / $degrees$, N')
    plt.title(file_str, fontsize=8)
    fig.tight_layout()
    plt.savefig(file_str)
    plt.close('all')

    return

def plot_sst_lat_diff(lat,sst_q3_lat_diff,sst_q5_lat_diff,plotfile):

    gd_q3 = np.isfinite(sst_q3_lat_diff)
    gd_q5 = np.isfinite(sst_q5_lat_diff)
    q3_ave = np.mean(sst_q3_lat_diff[gd_q3])
    q3_med = np.median(sst_q3_lat_diff[gd_q3])
    q3_std = np.std(sst_q3_lat_diff[gd_q3])
    q3_rsd = rb.mad(sst_q3_lat_diff[gd_q3])
    q5_ave = np.mean(sst_q5_lat_diff[gd_q5])
    q5_med = np.median(sst_q5_lat_diff[gd_q5])
    q5_std = np.std(sst_q5_lat_diff[gd_q5])
    q5_rsd = rb.mad(sst_q5_lat_diff[gd_q5])

    q3str = 'ql=3:'+'Mean='+"{0:.3f}".format(q3_ave)+' Median='+"{0\
:.3f}".format(q3_med)+' SD='+"{0:.3f}".format(q3_std)+' RSD='+"{0:.\
3f}".format(q3_rsd)
    q5str = 'ql=4&5:'+'Mean='+"{0:.3f}".format(q5_ave)+' Median='+"\
{0:.3f}".format(q5_med)+' SD='+"{0:.3f}".format(q5_std)+' RSD='+"{0\
:.3f}".format(q5_rsd)
    file_str = plotfile + "_diff" + ".png"

    fig, ax = plt.subplots()
    plt.plot(sst_q3_lat_diff, lat, '.', color='g', alpha=0.2, label='ql=3')
    plt.plot(sst_q5_lat_diff, lat, '.', color='r', alpha=0.2, label='ql=4&5')
    ax = plt.gca()
    ax.set_ylim([-91,90])
    ticks = ax.get_yticks()
    ax.set_yticks(np.linspace(-90, 90, 7))
    plt.legend(loc=1, fontsize=8)
    plt.xlabel(r'mean SST difference (L3U-L4 analysis) / $K$')
    plt.ylabel(r'Latitude / $degrees$, N')
    plt.title(file_str, fontsize=8)
    fig.tight_layout()
    plt.savefig(file_str)
    plt.close('all')

    return

if __name__ == "__main__":

    #----------------------------------------------
    parser = OptionParser("usage: %prog instrument year month day")
    (options, args) = parser.parse_args()
    try:
        instrument = args[0]
        year = args[1]
        month = args[2]
        day = args[3]
    except:
        instrument = 'AVHRRMTA_G'
        year = 2010
        month = 10
        day = 10

    FLAG_plot = 1

#    path_in = '/Users/michaeltaylor/Desktop/REPOS/AVHRR_SST/'
#    path_l4 = '/Users/michaeltaylor/Desktop/REPOS/AVHRR_SST/DATA/L4/'    
#    path_in = '/gws/nopw/j04/fiduceo/Users/mtaylor/avhrr_sst/'
#    path_l4 = '/gws/nopw/j04/fiduceo/Users/mtaylor/avhrr_sst/DATA/L4/'     

    path_in = os.getcwd()
#    path_l4 = path_in + "/" + "DATA/L4/" + str(year) + "/" + str(month) + "/" + str(day)
    path_l4 = "/gws/nopw/j04/esacci_sst/output/CDR2.1_release/Analysis/L4/v2.1/" + str(year) + "/" + str(month) + "/" + str(day) + "/"

    # LEVEL-4 FILES:
    # -------------
    # /gws/nopw/j04/esacci_sst/output/CDR2.1_release/Analysis/L4/v2.1/

#    file_l4 = '20040404120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc' # NOAA-17, -16
#    file_l4 = '20101010120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc'
    file_l4 = str(year)+str(month)+str(day)+'120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc'

    print('file_l4=', file_l4)

    # LEVEL-3U FILES: 
    # --------------
    # /gws/nopw/j04/fiduceo/Data/CDR/AVHRR_SST/v2.10.2/

    path_l3u = "/gws/nopw/j04/fiduceo/Data/CDR/AVHRR_SST/v2.10.2/" + instrument + "/" + str(year) + "/" + str(month) + "/" + str(day) + "/"
#    path_l3u = path_in + "/" "DATA" + "/" + instrument + "/" + str(year) + "/" + str(month) + "/" + str(day) + "/"

    print('path_l3u=', path_l3u)

    # HARMONISATION FILES:
    # -------------------
    # /gws/nopw/j04/fiduceo/Users/jmittaz/FCDR/Mike/FCDR_AVHRR/GBCS/dat_cci/

    file_in_l4 = os.path.join(path_l4,file_l4)
    dl4 = xarray.open_dataset(file_in_l4)
    sst_l4 = np.array(dl4['analysed_sst'][0,:,:])
    nlat = 3600

    if os.path.isdir(path_in):
        nclist = os.path.join(path_l3u,'*.nc')
        filelist = glob.glob(nclist)

        sst_q3_lat_total = np.zeros(nlat)
        sst_q5_lat_total = np.zeros(nlat)
        sst_q3_lat_nvals = np.zeros(nlat)
        sst_q5_lat_nvals = np.zeros(nlat)
        sst_q3_lat_mean = np.ones(nlat)*np.nan
        sst_q5_lat_mean = np.ones(nlat)*np.nan
        sst_q3_lat_mean_diff = np.ones(nlat)*np.nan
        sst_q5_lat_mean_diff = np.ones(nlat)*np.nan
        sst_l4_lat_mean = np.ones(nlat)*np.nan

        for ifile in range(len(filelist)):

            print('file_in=',str(ifile))
            file_in = str(filelist[ifile])
            ds = xarray.open_dataset(file_in)
            if instrument == 'AVHRRMTA_G':
                file_out = file_in[-80:-3] # filename without path # MTA has one extra char
            else:
                file_out = file_in[-79:-3] # filename without path
            lat = np.array(ds['lat'])
            lon = np.array(ds['lon'])
            sst = np.array(ds['sea_surface_temperature'][0,:,:])
            ql = np.array(ds['quality_level'][0,:,:])
            flags = np.array(ds['l2p_flags'][0,:,:])

            q3 = ql==3
            q5 = ql==5
 
            sst_q3_lat = []
            sst_q5_lat = []
            sst_q3_lat_l4 = []
            sst_q5_lat_l4 = []
            sst_q3_lat_sum = np.zeros(len(lat))
            sst_q5_lat_sum = np.zeros(len(lat))
            sst_q3_lat_n = np.zeros(len(lat))
            sst_q5_lat_n = np.zeros(len(lat))

            for i in range(sst.shape[0]):

                sst_q3_lat.append(np.mean(sst[i,np.where(q3[i,:])]))
                sst_q3_lat_l4.append(np.mean(sst_l4[i,np.where(q3[i,:])]))
                sst_q5_lat.append(np.mean(sst[i,np.where(q5[i,:])]))
                sst_q5_lat_l4.append(np.mean(sst_l4[i,np.where(q5[i,:])]))
                sst_q3_lat_sum[i] = np.sum(sst[i,np.where(q3[i,:])])
                sst_q5_lat_sum[i] = np.sum(sst[i,np.where(q5[i,:])])
                sst_q3_lat_n[i] = sst[i,np.where(q3[i,:])].shape[1]
                sst_q5_lat_n[i] = sst[i,np.where(q5[i,:])].shape[1]

            sst_q3_lat_diff = np.array(sst_q3_lat) - np.array(sst_q3_lat_l4)
            sst_q5_lat_diff = np.array(sst_q5_lat) - np.array(sst_q5_lat_l4)

            if FLAG_plot:

                plotfile = instrument + '_' + file_out[:13]
                plot_sst_lat(lat,sst_q3_lat,sst_q5_lat,plotfile)
                plot_sst_lat_diff(lat,sst_q3_lat_diff,sst_q5_lat_diff,plotfile)

            sst_q3_lat_total += sst_q3_lat_sum
            sst_q5_lat_total += sst_q5_lat_sum 
            sst_q3_lat_nvals += sst_q3_lat_n
            sst_q5_lat_nvals += sst_q5_lat_n

        gd = np.isfinite(sst_q3_lat_nvals)
        sst_q3_lat_mean[gd] = sst_q3_lat_total[gd] / sst_q3_lat_nvals[gd]
        gd = np.isfinite(sst_q5_lat_nvals)
        sst_q5_lat_mean[gd] = sst_q5_lat_total[gd] / sst_q5_lat_nvals[gd]        
        sst_l4_lat_mean = np.mean(sst_l4,axis=1) 
        gd = np.isfinite(sst_q3_lat_mean)
        for k in range(nlat):
            nonan = np.where(sst_l4[k,:] > 0.0)
            sst_l4_lat_mean[k] = np.mean(sst_l4[k,nonan])
        sst_q3_lat_mean_diff[gd] = sst_q3_lat_mean[gd] - sst_l4_lat_mean[gd]
        gd = np.isfinite(sst_q5_lat_mean)
        sst_q5_lat_mean_diff[gd] = sst_q5_lat_mean[gd] - sst_l4_lat_mean[gd]

        if FLAG_plot:

            plotfile = instrument + '_' + file_out[:8]
            plot_sst_lat(lat,sst_q3_lat_mean,sst_q5_lat_mean,plotfile)
            plot_sst_lat_diff(lat,sst_q3_lat_mean_diff,sst_q5_lat_mean_diff,plotfile)

    print('** END')


