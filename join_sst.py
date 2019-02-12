#!/usr/bin/env python

# PROGRAM: join_sst.py

# call as: python join_sst.py instrument year month day
# degug: python -mipdb join_sst.py instrument year month day
# ----------------------------------------------------------------------------------
# Version 0.1
# 27 December, 2018
# michael.taylor AT reading DOT ac DOT uk

#import os
import os.path
#import glob
#import optparse
from  optparse import OptionParser 
import sys
import numpy as np
import xarray
#import datetime
                   

def run(instrument,year,month,day):

    file_stem = instrument + str("%04d" %year) + str("%02d" %month) + str("%02d" %day)    
    path_in = "/gws/nopw/j04/fiduceo/Users/mtaylor/sst"
#    path_in = "/group_workspaces/cems2/fiduceo/Users/mtaylor/sst"
#    path_in = "/home/laptop/Desktop/sst"
    file_in = path_in + "/" + instrument + "/" + str("%04d" %year) + "/" + str("%02d" %month) + "/" + str("%02d" %day)+"/" + file_stem + ".nc"
    file_out = path_in + "/" + instrument + "/" + instrument + "_summary.nc"

    if os.path.exists(file_in):
#        ds = xarray.open_dataset(file_in, decode_cf=True)
        ds = xarray.open_dataset(file_in)
        if os.path.exists(file_out):
#            dr = xarray.open_dataset(file_out, decode_cf=True)
            dr = xarray.open_dataset(file_out)
            data_out = xarray.concat([dr,ds], dim='time')
            dr.close()
            
        else:
            data_out = ds
        data_out.to_netcdf(file_out)
        data_out.close()
        ds.close()    

if __name__ == "__main__":

    parser = OptionParser("usage: %prog instrument year month day")
    (options, args) = parser.parse_args()    
    instrument = args[0]
    year = int(args[1])
    month = int(args[2])
    day = int(args[3])
    run(instrument,year,month,day)
    

