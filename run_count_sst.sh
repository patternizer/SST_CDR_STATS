#!/bin/sh

# PROGRAM: run_count_sst.sh
# Copyright (C) 2019 Michael Taylor, University of Reading                                 
# This code was developed for the EC project "Fidelity and Uncertainty in                 
# Climate Data Records from Earth Observations (FIDUCEO).                                 
# Grant Agreement: 638822                                                                 
#                                                                                          
# This program is free software; you can redistribute it and/or modify it                 
# under the terms of the GNU General Public License as published by the Free              
# Software Foundation; either version 3 of the License, or (at your option)               
# any later version.                                                                      
#
# This program is distributed in the hope that it will be useful, but WITHOUT             
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or                   
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for                
# more details.                                                                           
#                                                                                          
# A copy of the GNU General Public License should have been supplied along                
# with this program; if not, see http://www.gnu.org/licenses/                             

echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.7.sh
echo python run_count_sst.py AVHRR07_G >> run.7.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.9.sh
echo python run_count_sst.py AVHRR09_G >> run.9.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.11.sh
echo python run_count_sst.py AVHRR11_G >> run.11.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.12.sh
echo python run_count_sst.py AVHRR12_G >> run.12.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.14.sh
echo python run_count_sst.py AVHRR14_G >> run.14.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.15.sh
echo python run_count_sst.py AVHRR15_G >> run.15.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.16.sh
echo python run_count_sst.py AVHRR16_G >> run.16.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.17.sh
echo python run_count_sst.py AVHRR17_G >> run.17.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.18.sh
echo python run_count_sst.py AVHRR18_G >> run.18.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.19.sh
echo python run_count_sst.py AVHRR19_G >> run.19.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.MA.sh
echo python run_count_sst.py AVHRRMTA_G >> run.MA.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.AATSR.sh
echo python run_count_sst.py AATSR >> run.AATSR.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ATSR1.sh
echo python run_count_sst.py ATSR1 >> run.ATSR1.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ATSR2.sh
echo python run_count_sst.py ATSR2 >> run.ATSR2.sh

bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.7.log < run.7.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.9.log < run.9.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.11.log < run.11.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.12.log < run.12.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.14.log < run.14.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.15.log < run.15.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.16.log < run.16.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.17.log < run.17.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.18.log < run.18.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.19.log < run.19.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.MA.log < run.MA.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.AATSR.log < run.AATSR.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ATSR1.log < run.ATSR1.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ATSR2.log < run.ATSR2.sh

