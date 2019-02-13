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

echo '. /group_workspaces/cems2/fiduceo/Users/mtaylor/anaconda3/bin/activate mike_cems' > run.2.sh
echo python run_count_sst.py AVHRR07_G >> run.2.sh
echo '. /group_workspaces/cems2/fiduceo/Users/mtaylor/anaconda3/bin/activate mike_cems' > run.4.sh
echo python run_count_sst.py AVHRR09_G >> run.4.sh
echo '. /group_workspaces/cems2/fiduceo/Users/mtaylor/anaconda3/bin/activate mike_cems' > run.6.sh
echo python run_count_sst.py AVHRR11_G >> run.6.sh
echo '. /group_workspaces/cems2/fiduceo/Users/mtaylor/anaconda3/bin/activate mike_cems' > run.7.sh
echo python run_count_sst.py AVHRR12_G >> run.7.sh
echo '. /group_workspaces/cems2/fiduceo/Users/mtaylor/anaconda3/bin/activate mike_cems' > run.8.sh
echo python run_count_sst.py AVHRR14_G >> run.8.sh
echo '. /group_workspaces/cems2/fiduceo/Users/mtaylor/anaconda3/bin/activate mike_cems' > run.9.sh
echo python run_count_sst.py AVHRR15_G >> run.9.sh
echo '. /group_workspaces/cems2/fiduceo/Users/mtaylor/anaconda3/bin/activate mike_cems' > run.10.sh
echo python run_count_sst.py AVHRR16_G >> run.10.sh
echo '. /group_workspaces/cems2/fiduceo/Users/mtaylor/anaconda3/bin/activate mike_cems' > run.11.sh
echo python run_count_sst.py AVHRR17_G >> run.11.sh
echo '. /group_workspaces/cems2/fiduceo/Users/mtaylor/anaconda3/bin/activate mike_cems' > run.12.sh
echo python run_count_sst.py AVHRR18_G >> run.12.sh
echo '. /group_workspaces/cems2/fiduceo/Users/mtaylor/anaconda3/bin/activate mike_cems' > run.13.sh
echo python run_count_sst.py AVHRR19_G >> run.13.sh
echo '. /group_workspaces/cems2/fiduceo/Users/mtaylor/anaconda3/bin/activate mike_cems' > run.14.sh
echo python run_count_sst.py AVHRRMTA_G >> run.14.sh
echo '. /group_workspaces/cems2/fiduceo/Users/mtaylor/anaconda3/bin/activate mike_cems' > run.15.sh
echo python run_count_sst.py AATSR >> run.15.sh
echo '. /group_workspaces/cems2/fiduceo/Users/mtaylor/anaconda3/bin/activate mike_cems' > run.16.sh
echo python run_count_sst.py ATSR1 >> run.16.sh
echo '. /group_workspaces/cems2/fiduceo/Users/mtaylor/anaconda3/bin/activate mike_cems' > run.17.sh
echo python run_count_sst.py ATSR2 >> run.17.sh

bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.2.log < run.2.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.4.log < run.4.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.6.log < run.6.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.7.log < run.7.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.8.log < run.8.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.9.log < run.9.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.10.log < run.10.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.11.log < run.11.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.12.log < run.12.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.13.log < run.13.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.14.log < run.14.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.15.log < run.15.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.16.log < run.16.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.16.log < run.17.sh
