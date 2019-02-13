#!/usr/bin/env python                                                                    

# PROGRAM: run_count_sst.py

# Copyright (C) 2019 Michael Taylor, University of Reading                                       
# This code was developed for the EC project "Fidelity and Uncertainty in                 
# Climate Data Records from Earth Observations (FIDUCEO).                                 
# Grant Agreement: 638822                                                                 
#                                                                                          
# This program is free software; you can redistribute it and/or modify it                 
# under the terms of the GNU General Public License as published by the Free              
# Software Foundation; either version 3 of the License, or (at your option)               
# any later version.                                                                      
# This program is distributed in the hope that it will be useful, but WITHOUT             
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or                   
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for                
# more details.                                                                           
#                                                                                          
# A copy of the GNU General Public License should have been supplied along                
# with this program; if not, see http://www.gnu.org/licenses/                            

import numpy as np
import datetime 
import calendar
import os
import os.path
import sys
import uuid
import glob
import stat
from  optparse import OptionParser
import subprocess

def make_shell_command(instrument,year,month,day,directory):
         
    currentdir = os.getcwd()
    outdir = '{0}/{1}/{2:04d}/{3:02d}/{4:02d}'.format(currentdir,instrument,year,month,day)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    os.chdir(outdir)
    try:
        os.symlink('/gws/nopw/j04/fiduceo/Users/mtaylor/sst/count_sst.py','count_sst.py')
    except:
        pass

    job_file = 'run.sh'
    job_log = 'run.log'
    with open(job_file,'w') as fp:
        job_str = 'python count_sst.py {0} {1:04d} {2:02d} {3:02d}\n'.format(instrument,year,month,day)
        fp.write(job_str)   

    os.chmod(job_file,stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    job_name='./'+job_file
    job = ['bsub','-q','short-serial','-W','24:00','-R','rusage[mem=60000]','-M','60000','-oo',job_log,job_name]
    subprocess.call(job)
    os.chdir(currentdir)                    

def run_count_sst(instrument):

    for year in range(1978,2018):
        for month in range(1,13):
            maxday = calendar.monthrange(year,month)[1]
            for day in range(1,maxday+1):                
                if ((instrument == 'AATSR') or (instrument == 'ATSR1') or (instrument == 'ATSR2')):
                    directory = '/gws/nopw/j04/esacci_sst/output/CDR2.1_release/ATSR/L2P/v2.1/{0}/{1:04d}/{2:02d}/{3:02d}'.format(instrument,year,month,day)
                else:
                    directory = '/gws/nopw/j04/esacci_sst/output/CDR2.1_release/AVHRR/L2P/v2.1/{0}/{1:04d}/{2:02d}/{3:02d}'.format(instrument,year,month,day)
                if os.path.isdir(directory):     
                    make_shell_command(instrument,year,month,day,directory)       

if __name__ == "__main__":
    
    parser = OptionParser("usage: %prog instr_name")
    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.error("incorrect number of arguments")

    instrument = args[0]
    run_count_sst(instrument)


