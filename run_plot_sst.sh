#!/bin/sh

# PROGRAM: run_plot_sst.sh
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

echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.plot_sst.sh
echo python plot_sst.py >> run.plot_sst.sh

bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.1.log < run.plot_sst.sh


