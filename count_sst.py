#!/usr/bin/env python

# PROGRAM: count_sst.py

# call as: python count_sst.py instrument year month day
# degug: python -mipdb count_sst.py instrument year month day

# COUNT SST obs density + uncertainty histograms for each q-level and zonal latitude
# ----------------------------------------------------------------------------------
# Version 0.10
# 13 December, 2018
# michael.taylor AT reading DOT ac DOT uk

import os
import os.path
import glob
import optparse
from  optparse import OptionParser 
import sys
import numpy as np
import numpy.ma as ma
import xarray
import datetime

class switch(object):
    value = None
    def __new__(class_, value):
        class_.value = value
        return True

def case(*args):
    return any((arg == switch.value for arg in args))

def calc_bins(x, y, bins):
    """ peform n-D binning (credit: G.Holl) """
    if x.size == y.size == 0:
        return [y[()] for b in bins]

    digits = np.digitize(x, bins)
    binned = [y[digits == i, ...] for i in range(len(bins))]
    return binned

def calc_total(large_scale_correlated_uncertainty,synoptically_correlated_uncertainty,uncorrelated_uncertainty):    
    """ calculate the total uncertainty from the norm of the large-scale and synoptically-correlated uncertainty and the uncorrelated uncertainty """
    total_uncertainty = (large_scale_correlated_uncertainty**2 + \
synoptically_correlated_uncertainty**2 + uncorrelated_uncertainty**2)**0.5
    return total_uncertainty
                   
def load_daily_data(path_in):

    if os.path.isdir(path_in):
        nclist = os.path.join(path_in,'*.nc')
        filelist = glob.glob(nclist)
        df = []
        for i in range(len(filelist)):
            file_in = str(filelist[i])
            ds = xarray.open_dataset(file_in, decode_cf=True)
            ds = ds[['time','ni','nj','lat','lon','sea_surface_temperature','sensitivity','large_scale_correlated_uncertainty','synoptically_correlated_uncertainty','uncorrelated_uncertainty','quality_level','l2p_flags']]
            df.append(ds.isel(time=0))        
        dataframe = xarray.concat(df, dim='nj')
        
        return dataframe

def run(instrument,year,month,day):
    file_stem = instrument + str("%04d" %year) + str("%02d" %month) + str("%02d" %day)
#    path = "/gws/nopw/j04/esacci_sst/output/CDR2.0/l2p/"
    path = "/group_workspaces/cems2/esacci_sst/output/CDR2.0/l2p/"
    path_in = path + instrument + "/"+str("%04d" %year) + "/"+str("%02d" %month) + "/"+str("%02d" %day)+"/"
    path_out = "/group_workspaces/cems2/fiduceo/Users/mtaylor/sst/"
    file_out = path_out + instrument + "/"+str("%04d" %year) + "/"+str("%02d" %month) + "/"+str("%02d" %day)+"/" + file_stem + ".nc"
#    file_out = path_out + file_stem + ".nc"

    if os.path.exists(file_out):
        os.remove(file_out)
    ds = load_daily_data(path_in)
    time = np.unique(ds['time'].values.astype("M8[D]"))[0]
    x = ds['ni']
    y = ds['nj']
    lat = ds['lat']
    lon = ds['lon']
    sst = ds['sea_surface_temperature']
    sensitivity = ds['sensitivity']
    large_scale_correlated_uncertainty = ds['large_scale_correlated_uncertainty']
    synoptically_correlated_uncertainty = ds['synoptically_correlated_uncertainty']
    uncorrelated_uncertainty = ds['uncorrelated_uncertainty']
    quality_level = ds['quality_level']
    l2p_flags = ds['l2p_flags']
    total_uncertainty = calc_total(large_scale_correlated_uncertainty,synoptically_correlated_uncertainty,uncorrelated_uncertainty)

# FILTERING (quality_level, l2p_flags, latitude)

    l2p_flags_int = l2p_flags.astype(int)
    l2p_flags_unique = np.unique(l2p_flags).astype(int)
    land_flag_logical = [np.binary_repr(x,9)[1] for x in l2p_flags_unique]  # bit 2 is land
    ice_flag_logical = [np.binary_repr(x,9)[2] for x in l2p_flags_unique]   # bit 3 is ice
    lake_flag_logical = [np.binary_repr(x,9)[3] for x in l2p_flags_unique]  # bit 4 is lake
    river_flag_logical = [np.binary_repr(x,9)[4] for x in l2p_flags_unique] # bit 5 is river
    land_flag_idx = [i for i, j in enumerate(land_flag_logical) if j == '1']
    ice_flag_idx = [i for i, j in enumerate(ice_flag_logical) if j == '1']
    lake_flag_idx = [i for i, j in enumerate(lake_flag_logical) if j == '1']
    river_flag_idx = [i for i, j in enumerate(river_flag_logical) if j == '1']
    if not ((land_flag_idx) or (ice_flag_idx) or (lake_flag_idx) or (river_flag_idx)):
        gd_q0 = (np.isfinite(sst)) & (quality_level==0)
        gd_q1 = (np.isfinite(sst)) & (quality_level==1)
        gd_q2 = (np.isfinite(sst)) & (quality_level==2)
        gd_q3 = (np.isfinite(sst)) & (quality_level==3)
        gd_q4 = (np.isfinite(sst)) & (quality_level==4)
        gd_q5 = (np.isfinite(sst)) & (quality_level==5)
    else:
        all_true = (l2p_flags_int!=99999)
        sea_only = all_true
        if land_flag_idx:
            for i in range(0,len(land_flag_idx)-1):
                land_flag = l2p_flags_unique[land_flag_idx][i]
                no_land = (l2p_flags_int!=land_flag)                
                sea_only = ((sea_only) & (no_land))
        if ice_flag_idx:
            for i in range(0,len(ice_flag_idx)-1):
                ice_flag = l2p_flags_unique[ice_flag_idx][i]
                no_ice = (l2p_flags_int!=ice_flag)
                sea_only = ((sea_only) & (no_ice))
        if lake_flag_idx:
            for i in range(0,len(lake_flag_idx)-1):
                lake_flag = l2p_flags_unique[lake_flag_idx][i]
                no_lake = (l2p_flags_int!=lake_flag)
                sea_only = ((sea_only) & (no_lake))
        if river_flag_idx:
            for i in range(0,len(river_flag_idx)-1):
                river_flag = l2p_flags_unique[river_flag_idx][i]
                no_river = (l2p_flags_int!=river_flag)
                sea_only = ((sea_only) & (no_river))

        gd_q0 = (np.isfinite(sst)) & (quality_level==0) & (sea_only)
        gd_q1 = (np.isfinite(sst)) & (quality_level==1) & (sea_only)
        gd_q2 = (np.isfinite(sst)) & (quality_level==2) & (sea_only)
        gd_q3 = (np.isfinite(sst)) & (quality_level==3) & (sea_only)
        gd_q4 = (np.isfinite(sst)) & (quality_level==4) & (sea_only)
        gd_q5 = (np.isfinite(sst)) & (quality_level==5) & (sea_only)

    dlat = 1
    lat_vec = np.arange(-90,90+dlat,dlat)
    sst_lat = calc_bins(lat.values.ravel(),sst.values.ravel(),lat_vec)
    sensitivity_lat = calc_bins(lat.values.ravel(),sensitivity.values.ravel(),lat_vec)
    total_uncertainty_lat = calc_bins(lat.values.ravel(),total_uncertainty.values.ravel(),lat_vec)

    gd_q0_lat = calc_bins(lat.values.ravel(),gd_q0.values.ravel(),lat_vec)
    gd_q1_lat = calc_bins(lat.values.ravel(),gd_q1.values.ravel(),lat_vec)
    gd_q2_lat = calc_bins(lat.values.ravel(),gd_q2.values.ravel(),lat_vec)
    gd_q3_lat = calc_bins(lat.values.ravel(),gd_q3.values.ravel(),lat_vec)
    gd_q4_lat = calc_bins(lat.values.ravel(),gd_q4.values.ravel(),lat_vec)
    gd_q5_lat = calc_bins(lat.values.ravel(),gd_q5.values.ravel(),lat_vec)

# OBSERVATION DENSITIES

    n_sst_q0 = len(sst.values[gd_q0.values])
    n_sst_q1 = len(sst.values[gd_q1.values])
    n_sst_q2 = len(sst.values[gd_q2.values])
    n_sst_q3 = len(sst.values[gd_q3.values])
    n_sst_q4 = len(sst.values[gd_q4.values])
    n_sst_q5 = len(sst.values[gd_q5.values])

    n_sensitivity_q0 = len(sensitivity.values[gd_q0.values])
    n_sensitivity_q1 = len(sensitivity.values[gd_q1.values])
    n_sensitivity_q2 = len(sensitivity.values[gd_q2.values])
    n_sensitivity_q3 = len(sensitivity.values[gd_q3.values])
    n_sensitivity_q4 = len(sensitivity.values[gd_q4.values])
    n_sensitivity_q5 = len(sensitivity.values[gd_q5.values])

    n_total_uncertainty_q0 = len(total_uncertainty.values[gd_q0.values])
    n_total_uncertainty_q1 = len(total_uncertainty.values[gd_q1.values])
    n_total_uncertainty_q2 = len(total_uncertainty.values[gd_q2.values])
    n_total_uncertainty_q3 = len(total_uncertainty.values[gd_q3.values])
    n_total_uncertainty_q4 = len(total_uncertainty.values[gd_q4.values])
    n_total_uncertainty_q5 = len(total_uncertainty.values[gd_q5.values])

    n_sst_q0_lat = [len(s[q]) for (s, q) in zip(sst_lat, gd_q0_lat)]
    n_sst_q1_lat = [len(s[q]) for (s, q) in zip(sst_lat, gd_q1_lat)]
    n_sst_q2_lat = [len(s[q]) for (s, q) in zip(sst_lat, gd_q2_lat)]
    n_sst_q3_lat = [len(s[q]) for (s, q) in zip(sst_lat, gd_q3_lat)]
    n_sst_q4_lat = [len(s[q]) for (s, q) in zip(sst_lat, gd_q4_lat)]
    n_sst_q5_lat = [len(s[q]) for (s, q) in zip(sst_lat, gd_q5_lat)]

    n_sensitivity_q0_lat = [len(s[q]) for (s, q) in zip(sensitivity_lat, gd_q0_lat)]
    n_sensitivity_q1_lat = [len(s[q]) for (s, q) in zip(sensitivity_lat, gd_q1_lat)]
    n_sensitivity_q2_lat = [len(s[q]) for (s, q) in zip(sensitivity_lat, gd_q2_lat)]
    n_sensitivity_q3_lat = [len(s[q]) for (s, q) in zip(sensitivity_lat, gd_q3_lat)]
    n_sensitivity_q4_lat = [len(s[q]) for (s, q) in zip(sensitivity_lat, gd_q4_lat)]
    n_sensitivity_q5_lat = [len(s[q]) for (s, q) in zip(sensitivity_lat, gd_q5_lat)]

    n_total_uncertainty_q0_lat = [len(s[q]) for (s, q) in zip(total_uncertainty_lat, gd_q0_lat)]
    n_total_uncertainty_q1_lat = [len(s[q]) for (s, q) in zip(total_uncertainty_lat, gd_q1_lat)]
    n_total_uncertainty_q2_lat = [len(s[q]) for (s, q) in zip(total_uncertainty_lat, gd_q2_lat)]
    n_total_uncertainty_q3_lat = [len(s[q]) for (s, q) in zip(total_uncertainty_lat, gd_q3_lat)]
    n_total_uncertainty_q4_lat = [len(s[q]) for (s, q) in zip(total_uncertainty_lat, gd_q4_lat)]
    n_total_uncertainty_q5_lat = [len(s[q]) for (s, q) in zip(total_uncertainty_lat, gd_q5_lat)]

# HISTOGRAMS

    nbins = 600
    sst_min = 260
    sst_max = 320

    sst_q0_hist, sst_q0_bins = np.histogram(sst.values[gd_q0.values], nbins, range=[sst_min,sst_max], density=False)
    sst_q1_hist, sst_q1_bins = np.histogram(sst.values[gd_q1.values], nbins, range=[sst_min,sst_max], density=False)
    sst_q2_hist, sst_q2_bins = np.histogram(sst.values[gd_q2.values], nbins, range=[sst_min,sst_max], density=False)
    sst_q3_hist, sst_q3_bins = np.histogram(sst.values[gd_q3.values], nbins, range=[sst_min,sst_max], density=False)
    sst_q4_hist, sst_q4_bins = np.histogram(sst.values[gd_q4.values], nbins, range=[sst_min,sst_max], density=False)
    sst_q5_hist, sst_q5_bins = np.histogram(sst.values[gd_q5.values], nbins, range=[sst_min,sst_max], density=False)
    sst_midpoints = 0.5*(sst_q0_bins[1:]+sst_q0_bins[:-1])

    nbins = 50
    sensitivity_min = 0.5
    sensitivity_max = 1

    sensitivity_q0_hist, sensitivity_q0_bins = np.histogram(sensitivity.values[gd_q0.values], nbins, range=[sensitivity_min,sensitivity_max], density=False)
    sensitivity_q1_hist, sensitivity_q1_bins = np.histogram(sensitivity.values[gd_q1.values], nbins, range=[sensitivity_min,sensitivity_max], density=False)
    sensitivity_q2_hist, sensitivity_q2_bins = np.histogram(sensitivity.values[gd_q2.values], nbins, range=[sensitivity_min,sensitivity_max], density=False)
    sensitivity_q3_hist, sensitivity_q3_bins = np.histogram(sensitivity.values[gd_q3.values], nbins, range=[sensitivity_min,sensitivity_max], density=False)
    sensitivity_q4_hist, sensitivity_q4_bins = np.histogram(sensitivity.values[gd_q4.values], nbins, range=[sensitivity_min,sensitivity_max], density=False)
    sensitivity_q5_hist, sensitivity_q5_bins = np.histogram(sensitivity.values[gd_q5.values], nbins, range=[sensitivity_min,sensitivity_max], density=False)
    sensitivity_midpoints = 0.5*(sensitivity_q0_bins[1:]+sensitivity_q0_bins[:-1])

    nbins = 400
    total_uncertainty_min = 0
    total_uncertainty_max = 4

    total_uncertainty_q0_hist, total_uncertainty_q0_bins = np.histogram(total_uncertainty.values[gd_q0.values], nbins, range=[total_uncertainty_min,total_uncertainty_max], density=False)
    total_uncertainty_q1_hist, total_uncertainty_q1_bins = np.histogram(total_uncertainty.values[gd_q1.values], nbins, range=[total_uncertainty_min,total_uncertainty_max], density=False)
    total_uncertainty_q2_hist, total_uncertainty_q2_bins = np.histogram(total_uncertainty.values[gd_q2.values], nbins, range=[total_uncertainty_min,total_uncertainty_max], density=False)
    total_uncertainty_q3_hist, total_uncertainty_q3_bins = np.histogram(total_uncertainty.values[gd_q3.values], nbins, range=[total_uncertainty_min,total_uncertainty_max], density=False)
    total_uncertainty_q4_hist, total_uncertainty_q4_bins = np.histogram(total_uncertainty.values[gd_q4.values], nbins, range=[total_uncertainty_min,total_uncertainty_max], density=False)
    total_uncertainty_q5_hist, total_uncertainty_q5_bins = np.histogram(total_uncertainty.values[gd_q5.values], nbins, range=[total_uncertainty_min,total_uncertainty_max], density=False)
    total_uncertainty_midpoints = 0.5*(total_uncertainty_q0_bins[1:]+total_uncertainty_q0_bins[:-1])

# WRITE NETCDF

    data_out = xarray.Dataset({"time": (("time",), np.atleast_1d(time))})
    data_out["lat_vec"] = (("lat_vec",), lat_vec)
    data_out["sst_midpoints"] = (("sst_midpoints",), sst_midpoints)
    data_out["sensitivity_midpoints"] = (("sensitivity_midpoints",), sensitivity_midpoints)
    data_out["total_uncertainty_midpoints"] = (("total_uncertainty_midpoints",), total_uncertainty_midpoints)

    data_out["n_sst_q0"] = (("time",), np.atleast_1d(n_sst_q0))
    data_out["n_sst_q1"] = (("time",), np.atleast_1d(n_sst_q1))
    data_out["n_sst_q2"] = (("time",), np.atleast_1d(n_sst_q2))
    data_out["n_sst_q3"] = (("time",), np.atleast_1d(n_sst_q3))
    data_out["n_sst_q4"] = (("time",), np.atleast_1d(n_sst_q4))
    data_out["n_sst_q5"] = (("time",), np.atleast_1d(n_sst_q5))

    data_out["n_sensitivity_q0"] = (("time",), np.atleast_1d(n_sensitivity_q0))
    data_out["n_sensitivity_q1"] = (("time",), np.atleast_1d(n_sensitivity_q1))
    data_out["n_sensitivity_q2"] = (("time",), np.atleast_1d(n_sensitivity_q2))
    data_out["n_sensitivity_q3"] = (("time",), np.atleast_1d(n_sensitivity_q3))
    data_out["n_sensitivity_q4"] = (("time",), np.atleast_1d(n_sensitivity_q4))
    data_out["n_sensitivity_q5"] = (("time",), np.atleast_1d(n_sensitivity_q5))

    data_out["n_total_uncertainty_q0"] = (("time",), np.atleast_1d(n_total_uncertainty_q0))
    data_out["n_total_uncertainty_q1"] = (("time",), np.atleast_1d(n_total_uncertainty_q1))
    data_out["n_total_uncertainty_q2"] = (("time",), np.atleast_1d(n_total_uncertainty_q2))
    data_out["n_total_uncertainty_q3"] = (("time",), np.atleast_1d(n_total_uncertainty_q3))
    data_out["n_total_uncertainty_q4"] = (("time",), np.atleast_1d(n_total_uncertainty_q4))
    data_out["n_total_uncertainty_q5"] = (("time",), np.atleast_1d(n_total_uncertainty_q5))
 
    data_out["n_sst_q0_lat"] = (("time","lat_vec",), np.atleast_2d(n_sst_q0_lat))
    data_out["n_sst_q1_lat"] = (("time","lat_vec",), np.atleast_2d(n_sst_q1_lat))
    data_out["n_sst_q2_lat"] = (("time","lat_vec",), np.atleast_2d(n_sst_q2_lat))
    data_out["n_sst_q3_lat"] = (("time","lat_vec",), np.atleast_2d(n_sst_q3_lat))
    data_out["n_sst_q4_lat"] = (("time","lat_vec",), np.atleast_2d(n_sst_q4_lat))
    data_out["n_sst_q5_lat"] = (("time","lat_vec",), np.atleast_2d(n_sst_q5_lat))

    data_out["n_sensitivity_q0_lat"] = (("time","lat_vec",), np.atleast_2d(n_sensitivity_q0_lat))
    data_out["n_sensitivity_q1_lat"] = (("time","lat_vec",), np.atleast_2d(n_sensitivity_q1_lat))
    data_out["n_sensitivity_q2_lat"] = (("time","lat_vec",), np.atleast_2d(n_sensitivity_q2_lat))
    data_out["n_sensitivity_q3_lat"] = (("time","lat_vec",), np.atleast_2d(n_sensitivity_q3_lat))
    data_out["n_sensitivity_q4_lat"] = (("time","lat_vec",), np.atleast_2d(n_sensitivity_q4_lat))
    data_out["n_sensitivity_q5_lat"] = (("time","lat_vec",), np.atleast_2d(n_sensitivity_q5_lat))

    data_out["n_total_uncertainty_q0_lat"] = (("time","lat_vec",), np.atleast_2d(n_total_uncertainty_q0_lat))
    data_out["n_total_uncertainty_q1_lat"] = (("time","lat_vec",), np.atleast_2d(n_total_uncertainty_q1_lat))
    data_out["n_total_uncertainty_q2_lat"] = (("time","lat_vec",), np.atleast_2d(n_total_uncertainty_q2_lat))
    data_out["n_total_uncertainty_q3_lat"] = (("time","lat_vec",), np.atleast_2d(n_total_uncertainty_q3_lat))
    data_out["n_total_uncertainty_q4_lat"] = (("time","lat_vec",), np.atleast_2d(n_total_uncertainty_q4_lat))
    data_out["n_total_uncertainty_q5_lat"] = (("time","lat_vec",), np.atleast_2d(n_total_uncertainty_q5_lat))

    data_out["sst_q0_hist"] = (("time","sst_midpoints",), np.atleast_2d(sst_q0_hist))
    data_out["sst_q1_hist"] = (("time","sst_midpoints",), np.atleast_2d(sst_q1_hist))
    data_out["sst_q2_hist"] = (("time","sst_midpoints",), np.atleast_2d(sst_q2_hist))
    data_out["sst_q3_hist"] = (("time","sst_midpoints",), np.atleast_2d(sst_q3_hist))
    data_out["sst_q4_hist"] = (("time","sst_midpoints",), np.atleast_2d(sst_q4_hist))
    data_out["sst_q5_hist"] = (("time","sst_midpoints",), np.atleast_2d(sst_q5_hist))

    data_out["sensitivity_q0_hist"] = (("time","sensitivity_midpoints",), np.atleast_2d(sensitivity_q0_hist))
    data_out["sensitivity_q1_hist"] = (("time","sensitivity_midpoints",), np.atleast_2d(sensitivity_q1_hist))
    data_out["sensitivity_q2_hist"] = (("time","sensitivity_midpoints",), np.atleast_2d(sensitivity_q2_hist))
    data_out["sensitivity_q3_hist"] = (("time","sensitivity_midpoints",), np.atleast_2d(sensitivity_q3_hist))
    data_out["sensitivity_q4_hist"] = (("time","sensitivity_midpoints",), np.atleast_2d(sensitivity_q4_hist))
    data_out["sensitivity_q5_hist"] = (("time","sensitivity_midpoints",), np.atleast_2d(sensitivity_q5_hist))

    data_out["total_uncertainty_q0_hist"] = (("time","total_uncertainty_midpoints"), np.atleast_2d(total_uncertainty_q0_hist))
    data_out["total_uncertainty_q1_hist"] = (("time","total_uncertainty_midpoints",), np.atleast_2d(total_uncertainty_q1_hist))
    data_out["total_uncertainty_q2_hist"] = (("time","total_uncertainty_midpoints",), np.atleast_2d(total_uncertainty_q2_hist))
    data_out["total_uncertainty_q3_hist"] = (("time","total_uncertainty_midpoints",), np.atleast_2d(total_uncertainty_q3_hist))
    data_out["total_uncertainty_q4_hist"] = (("time","total_uncertainty_midpoints",), np.atleast_2d(total_uncertainty_q4_hist))
    data_out["total_uncertainty_q5_hist"] = (("time","total_uncertainty_midpoints",), np.atleast_2d(total_uncertainty_q5_hist))

    data_out.to_netcdf(file_out)
    data_out.close()

if __name__ == "__main__":

    parser = OptionParser("usage: %prog instrument year month day")
    (options, args) = parser.parse_args()    
    instrument = args[0]
    year = int(args[1])
    month = int(args[2])
    day = int(args[3])
    run(instrument,year,month,day)

