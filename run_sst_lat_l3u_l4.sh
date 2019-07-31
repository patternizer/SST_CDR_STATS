#!/bin/sh

# PROGRAM: run_sst_lat.sh
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

echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.sst_lat_l3u_l4.sh
#echo python sst_lat_l3u_l4.py AVHRRMTA_G 2010 10 10 >> run.sst_lat_l3u_l4.sh
#echo python sst_lat_l3u_l4.py AVHRR19_G 2010 10 10 >> run.sst_lat_l3u_l4.sh
#echo python sst_lat_l3u_l4.py AVHRR18_G 2010 10 10 >> run.sst_lat_l3u_l4.sh
#echo python sst_lat_l3u_l4.py AVHRR17_G 2004 04 04 >> run.sst_lat_l3u_l4.s
#echo python sst_lat_l3u_l4.py AVHRR16_G 2004 04 04 >> run.sst_lat_l3u_l4.sh
#echo python sst_lat_l3u_l4.py AVHRR15_G 2004 04 04 >> run.sst_lat_l3u_l4.sh
#echo python sst_lat_l3u_l4.py AVHRR14_G 1998 08 08 >> run.sst_lat_l3u_l4.sh
echo python sst_lat_l3u_l4.py AVHRR12_G 1998 08 08 >> run.sst_lat_l3u_l4.sh
#echo python sst_lat_l3u_l4.py AVHRR11_G 1994 04 04 >> run.sst_lat_l3u_l4.sh

bsub -q short-serial -W10:00 -R rusage[mem=5000] -M 5000 -oo run.1.log < run.sst_lat_l3u_l4.sh



